
import traceback

import sys
import logging

import importlib
import numpy as np
import os
import time
import subprocess
import socket
import json


import torch
from contextlib import closing

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client
from carla.driving_benchmark import run_driving_benchmark

from drive import CoILAgent, ECCVGeneralizationSuite, ECCVTrainingSuite, TestT1, TestT2

from testing.unit_tests.test_drive.test_suite import TestSuite
from logger import coil_logger

from logger import monitorer


from configs import g_conf, merge_with_yaml, set_type_of_process

from utils.checkpoint_schedule import  maximun_checkpoint_reach, get_next_checkpoint,\
    is_next_checkpoint_ready, get_latest_evaluated_checkpoint
from utils.general import compute_average_std, get_latest_path


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))



def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_carla_simulator(gpu, town_name, no_screen):

    # Set the outfiles for the process
    carla_out_file = os.path.join('_output_logs',
                      'CARLA_'+ g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out")
    carla_out_file_err = os.path.join('_output_logs',
                      'CARLA_err_'+ g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out")

    # TODO: Add parameters
    mode = 'VGL'
    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

    if no_screen and mode == 'SDL':
        print (" EXECUTING NO SCREEN! ")
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'


    if mode == 'SDL':
        os.environ['SDL_HINT_CUDA_DEVICE'] = str(gpu)

        sp = subprocess.Popen([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + town_name,
                                '-windowed',
                               '-benchmark', '-fps=10', '-world-port='+str(port)], shell=False,
                               stdout=open(carla_out_file, 'w'), stderr=open(carla_out_file_err, 'w'))
    elif mode == 'VGL':
        os.environ['DISPLAY'] =":5"
        sp = subprocess.Popen(['vglrun', '-d', ':7.' + str(gpu),
                                    carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4',
                                    '/Game/Maps/' + town_name, '-windowed', '-benchmark',
                                    '-fps=10', '-world-port='+str(port)],
                               shell=False,
                               stdout=open(carla_out_file, 'w'), stderr=open(carla_out_file_err, 'w'))
    else:
        raise ValueError("Invalid Mode !")


    coil_logger.add_message('Loading', {'CARLA': carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4' 
                           '-windowed'+ '-benchmark'+ '-fps=10'+ '-world-port='+ str(port)})

    return sp, port




# OBS: note, for now carla and carla test are in the same GPU

# TODO: Add all the necessary logging.

# OBS : I AM FIXING host as localhost now
# TODO :  Memory use should also be adaptable with a limit, for now that seems to be doing fine in PYtorch

def execute(gpu, exp_batch, exp_alias, drive_conditions, memory_use=0.2, host='127.0.0.1',
            suppress_output=True, no_screen=False):

    try:


        print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))


        print ("drive cond", drive_conditions)
        exp_set_name, town_name = drive_conditions.split('_')

        if g_conf.USE_ORACLE:
            control_filename = 'control_output_auto.csv'
        else:
            control_filename = 'control_output.csv'



        if exp_set_name == 'ECCVTrainingSuite':
            experiment_set = ECCVTrainingSuite()
            set_type_of_process('drive', drive_conditions)
        elif exp_set_name == 'ECCVGeneralizationSuite':
            experiment_set = ECCVGeneralizationSuite()
            set_type_of_process('drive', drive_conditions)
        elif exp_set_name == 'TestT1':
            experiment_set = TestT1()
            set_type_of_process('drive', drive_conditions)
        elif exp_set_name == 'TestT2':
            experiment_set = TestT2()
            set_type_of_process('drive', drive_conditions)
        else:

            raise ValueError(" Exp Set name is not correspondent to a city")




        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            #sys.stderr = open(os.path.join('_output_logs',
            #                  'err_'+g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
            #                  "a", buffering=1)



        carla_process, port = start_carla_simulator(gpu, town_name, no_screen)

        coil_logger.add_message('Loading', {'Poses': experiment_set.build_experiments()[0].poses})

        coil_logger.add_message('Loading', {'CARLAClient': host + ':' + str(port)})

        # Now actually run the driving_benchmark

        latest = get_latest_evaluated_checkpoint()
        if latest is None:  # When nothing was tested, get latest returns none, we fix that.
            latest = 0



            csv_outfile = open(os.path.join('_logs', exp_batch, exp_alias,
                                            g_conf.PROCESS_NAME + '_csv', control_filename),
                               'w')

            csv_outfile.write("%s,%s,%s,%s,%s,%s,%s,%s\n"
                              % ('step', 'episodes_completion', 'intersection_offroad',
                                 'intersection_otherlane', 'collision_pedestrians',
                                 'collision_vehicles', 'episodes_fully_completed',
                                 'driven_kilometers'))
            csv_outfile.close()


        # Write the header of the summary file used conclusion
        # While the checkpoint is not there

        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):

            try:
                # Get the correct checkpoint
                if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

                    latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
                    checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                                         , 'checkpoints', str(latest) + '.pth'))

                    coil_agent = CoILAgent(checkpoint, town_name)

                    coil_logger.add_message('Iterating', {"Checkpoint": latest}, latest)

                    run_driving_benchmark(coil_agent, experiment_set, town_name,
                                          exp_batch + '_' + exp_alias + '_' + str(latest)
                                          + '_drive_' + control_filename[:-4]
                                          , True, host, port)

                    path = exp_batch + '_' + exp_alias + '_' + str(latest) \
                           + '_' + g_conf.PROCESS_NAME.split('_')[0] + '_' + control_filename[:-4] \
                           + '_' + g_conf.PROCESS_NAME.split('_')[1] + '_' + g_conf.PROCESS_NAME.split('_')[2]


                    print(path)
                    print("Finished")
                    benchmark_json_path = os.path.join(get_latest_path(path), 'metrics.json')
                    with open(benchmark_json_path, 'r') as f:
                        benchmark_dict = json.loads(f.read())


                    averaged_dict = compute_average_std([benchmark_dict],
                                                        experiment_set.weathers,
                                                        len(experiment_set.build_experiments()))
                    print (averaged_dict)
                    csv_outfile = open(os.path.join('_logs', exp_batch, exp_alias,
                                                    g_conf.PROCESS_NAME + '_csv',
                                                    control_filename),
                                       'a')

                    csv_outfile.write("%d,%f,%f,%f,%f,%f,%f,%f\n"
                                % (latest, averaged_dict['episodes_completion'],
                                     averaged_dict['intersection_offroad'],
                                     averaged_dict['intersection_otherlane'],
                                     averaged_dict['collision_pedestrians'],
                                     averaged_dict['collision_vehicles'],
                                     averaged_dict['episodes_fully_completed'],
                                     averaged_dict['driven_kilometers']))

                    csv_outfile.close()

                    # TODO: When you add the message you need to check if the experiment continues properly



                    # TODO: WRITE AN EFICIENT PARAMETRIZED OUTPUT SUMMARY FOR TEST.


                else:
                    time.sleep(0.1)




            except TCPConnectionError as error:
                logging.error(error)
                time.sleep(1)
                carla_process.kill()
                coil_logger.add_message('Error', {'Message': 'TCP serious Error'})
                exit(1)
            except KeyboardInterrupt:
                carla_process.kill()
                coil_logger.add_message('Error', {'Message': 'Killed By User'})
                exit(1)
            except:
                traceback.print_exc()
                carla_process.kill()
                coil_logger.add_message('Error', {'Message': 'Something Happened'})
                exit(1)


        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        traceback.print_exc()
        carla_process.kill()
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()
        carla_process.kill()
        coil_logger.add_message('Error', {'Message': 'Something happened'})

    carla_process.kill()

