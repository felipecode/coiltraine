
import traceback

import sys
import logging

import importlib
import numpy as np
import os
import time
import subprocess
import socket


import torch
from contextlib import closing

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client
from carla.driving_benchmark import run_driving_benchmark

from drive import CoILAgent, ECCVGeneralizationSuite, ECCVTrainingSuite

from testing.unit_tests.test_drive.test_suite import TestSuite
from logger import coil_logger

from logger import monitorer


from configs import g_conf, merge_with_yaml, set_type_of_process

from utils.checkpoint_schedule import  maximun_checkpoint_reach, get_next_checkpoint, is_next_checkpoint_ready



def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))



def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_carla_simulator(gpu, town_name, no_screen):

    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

    if no_screen:
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'


    os.environ['SDL_HINT_CUDA_DEVICE'] = str(gpu)

    #subprocess.call()

    sp = subprocess.Popen([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + town_name,
                            '-windowed',
                           '-benchmark', '-fps=10', '-world-port='+str(port)], shell=False,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    coil_logger.add_message('Loading', {'CARLA': carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4' 
                           '-windowed'+ '-benchmark'+ '-fps=10'+ '-world-port='+ str(port)})

    return sp, port




# OBS: note, for now carla and carla test are in the same GPU

# TODO: Add all the necessary logging.

# OBS : I AM FIXING host as localhost now
# TODO :  Memory use should also be adaptable with a limit, for now that seems to be doing fine in PYtorch

def execute(gpu, exp_batch, exp_alias, exp_set_name, memory_use=0.2, host='127.0.0.1',
            suppress_output=True, no_screen=False):

    try:


        print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))



        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)



        carla_process, port = start_carla_simulator(gpu, exp_set_name, no_screen)




        # TODO we have some external class that control this weather thing.

        try:
            # TODO: REMOVE this part. This is for a newer version of CARLA.
            pass
            #exp_set_builder_module = importlib.import_module('drive.' + exp_set_name)
            #exp_set_builder = getattr(exp_set_builder_module, 'build_' + exp_set_name)
        except:
            carla_process.kill()
            coil_logger.add_message('Error', {'Message': 'Suite name not existent'})
            raise ValueError("Suite name not existent")


        if exp_set_name == 'Town01':

            experiment_set = ECCVTrainingSuite()
            set_type_of_process('drive', 'ECCVTrainingSuite_' + exp_set_name)

        elif exp_set_name == 'Town02':

            experiment_set = ECCVGeneralizationSuite()
            set_type_of_process('drive', 'ECCVGeneralizationSuite_' + exp_set_name)
        else:

            raise ValueError(" Exp Set name is not correspondent to a city")




        coil_logger.add_message('Loading', {'Poses': experiment_set.build_experiments()[0].poses})

        coil_logger.add_message('Loading', {'CARLAClient': host + ':' + str(port)})

        while True:
            try:

                # Now actually run the driving_benchmark

                latest = 0
                # While the checkpoint is not there
                while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):


                    # Get the correct checkpoint
                    if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

                        latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
                        checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                                             , 'checkpoints', str(latest) + '.pth'))

                        coil_agent = CoILAgent(checkpoint)

                        coil_logger.add_message('Iterating', {"Checkpoint": latest}, latest)

                        run_driving_benchmark(coil_agent, experiment_set, exp_set_name,
                                              exp_batch + '_' + exp_alias + '_' + str(latest)
                                              + '_drive'
                                              , True, host, port)



                        # TODO: When you add the message you need to check if the experiment continues properly



                        # TODO: WRITE AN EFICIENT PARAMETRIZED OUTPUT SUMMARY FOR TEST.


                    else:
                        time.sleep(0.1)

                    break


            except TCPConnectionError as error:
                logging.error(error)
                time.sleep(1)
                carla_process.kill()
                break
            except KeyboardInterrupt:
                carla_process.kill()
                coil_logger.add_message('Error', {'Message': 'Killed By User'})
                break
            except:
                traceback.print_exc()
                carla_process.kill()
                coil_logger.add_message('Error', {'Message': 'Something Happened'})
                break
    except KeyboardInterrupt:
        traceback.print_exc()
        carla_process.kill()
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()
        carla_process.kill()
        coil_logger.add_message('Error', {'Message': 'Something happened'})

    carla_process.kill()

