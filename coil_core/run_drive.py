
import traceback

import sys
import logging

import numpy as np
import os
import time
import subprocess
import socket
import json

import torch
from contextlib import closing

from carla.tcp import TCPConnectionError
from carla.driving_benchmark import run_driving_benchmark

from drive import CoILAgent
from logger import coil_logger
from configs import g_conf, merge_with_yaml, set_type_of_process
from utils.checkpoint_schedule import  maximun_checkpoint_reach, get_next_checkpoint,\
    is_next_checkpoint_ready, get_latest_evaluated_checkpoint, validation_stale_point
from utils.general import compute_average_std_separatetasks, get_latest_path, write_header_control_summary,\
     write_data_point_control_summary, camelcase_to_snakecase, unique


def frame2numpy(frame, frame_size):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frame_size[1], frame_size[0], 3))


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]



def start_carla_simulator(gpu, town_name, docker):
    # The out part is only used for docker
    # Set the outfiles for the process
    carla_out_file = os.path.join('_output_logs',
                                  'CARLA_'+ g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out")
    carla_out_file_err = os.path.join('_output_logs',
                                'CARLA_err_'+ g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out")
    port = find_free_port()

    if docker is not None:
        sp = subprocess.Popen(['docker', 'run', '--rm', '-d', '-p', str(port)+'-'+str(port+2)+':'+str(port)+'-'+str(port+2),
                              '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES='+str(gpu), docker,
                               '/bin/bash', 'CarlaUE4.sh', '/Game/Maps/' + town_name, '-windowed',
                               '-benchmark', '-fps=10', '-world-port=' + str(port)], shell=False,
                              stdout=subprocess.PIPE)

        (out, err) = sp.communicate()

    else:  # If you run carla without docker it requires a screen.

        carla_path = os.environ['CARLA_PATH']
        sp = subprocess.Popen([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + town_name,
                               '-windowed',
                               '-benchmark', '-fps=10', '-world-port='+str(port)], shell=False,
                               stdout=open(carla_out_file, 'w'), stderr=open(carla_out_file_err, 'w'))

        out = "0"


    coil_logger.add_message('Loading', {'CARLA':  '/CarlaUE4/Binaries/Linux/CarlaUE4' 
                           '-windowed'+ '-benchmark'+ '-fps=10'+ '-world-port='+ str(port)})

    return sp, port, out


def driving_iteration(checkpoint_number, gpu, town_name, experiment_set, exp_batch, exp_alias, params,
                      control_filename, task_list):

    port = 2006
    try:
        """ START CARLA"""
        #carla_process, port, out = start_carla_simulator(gpu, town_name,
        #                                                 params['docker'])

        checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                             , 'checkpoints', str(checkpoint_number) + '.pth'))

        coil_agent = CoILAgent(checkpoint, town_name)
        print ("Checkpoint ", checkpoint_number)
        coil_logger.add_message('Iterating', {"Checkpoint": checkpoint_number}, checkpoint_number)

        """ MAIN PART, RUN THE DRIVING BENCHMARK """
        run_driving_benchmark(coil_agent, experiment_set, town_name,
                              exp_batch + '_' + exp_alias + '_' + str(checkpoint_number)
                              + '_drive_' + control_filename
                              , True, params['host'], port)

        """ Processing the results to write a summary"""
        path = exp_batch + '_' + exp_alias + '_' + str(checkpoint_number) \
               + '_' + g_conf.PROCESS_NAME.split('_')[0] + '_' + control_filename \
               + '_' + g_conf.PROCESS_NAME.split('_')[1] + '_' + g_conf.PROCESS_NAME.split('_')[2]

        print(path)
        print("Finished")
        benchmark_json_path = os.path.join(get_latest_path(path), 'metrics.json')
        with open(benchmark_json_path, 'r') as f:
            benchmark_dict = json.loads(f.read())

        print(" number of episodes ", len(experiment_set.build_experiments()))
        averaged_dict = compute_average_std_separatetasks([benchmark_dict],
                                                          experiment_set.weathers,
                                                          len(experiment_set.build_experiments()))

        file_base = os.path.join('_logs', exp_batch, exp_alias,
                                 g_conf.PROCESS_NAME + '_csv', control_filename)

        for i in range(len(task_list)):
            write_data_point_control_summary(file_base, task_list[i],
                                             averaged_dict, checkpoint_number, i)

        # plot_episodes_tracks(os.path.join(get_latest_path(path), 'measurements.json'),
        #                     )
        print(averaged_dict)

        carla_process.kill()
        """ KILL CARLA, FINISHED THIS BENCHMARK"""
        subprocess.call(['docker', 'stop', out[:-1]])


    except TCPConnectionError as error:
        logging.error(error)
        time.sleep(1)
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])
        coil_logger.add_message('Error', {'Message': 'TCP serious Error'})
        exit(1)

    except KeyboardInterrupt:
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        exit(1)
    except:
        traceback.print_exc()
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        exit(1)


def execute(gpu, exp_batch, exp_alias, drive_conditions, params):

    try:
        print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))


        print ("drive cond", drive_conditions)
        exp_set_name, town_name = drive_conditions.split('_')

        # Here is some way to diferentiate betwen using the oracle to output acc/brake
        if g_conf.USE_ORACLE:
            control_filename = 'control_output_auto'
        else:
            control_filename = 'control_output'


        experiment_suite_module = __import__('drive.suites.' + camelcase_to_snakecase(exp_set_name) + '_suite',
                                             fromlist=[exp_set_name])
        experiment_suite_module = getattr(experiment_suite_module, exp_set_name)

        experiment_set = experiment_suite_module()

        set_type_of_process('drive', drive_conditions)

        if params['suppress_output']:
            sys.stdout = open(os.path.join('_output_logs',
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        coil_logger.add_message('Loading', {'Poses': experiment_set.build_experiments()[0].poses})

        experiment_list = experiment_set.build_experiments()
        # Get all the uniquely named tasks
        task_list = unique([experiment.task_name for experiment in experiment_list ])
        # Now actually run the driving_benchmark

        latest = get_latest_evaluated_checkpoint(control_filename + '_' + task_list[0])

        if latest is None:  # When nothing was tested, get latest returns none, we fix that.
            latest = 0
            # The used tasks are hardcoded, this need to be improved
            file_base = os.path.join('_logs', exp_batch, exp_alias,
                                     g_conf.PROCESS_NAME + '_csv', control_filename)

            for i in range(len(task_list)):
                # Write the header of the summary file used conclusion
                # While the checkpoint is not there
                write_header_control_summary(file_base, task_list[i])

        """ 
            ######
            Run a single driving iteration specified by the iteration were validation is stale
            ######
        """

        if g_conf.FINISH_ON_VALIDATION_STALE is not None:

            while validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE) is None:
                time.sleep(0.1)

            validation_state_iteration = validation_stale_point(g_conf.FINISH_ON_VALIDATION_STALE)
            driving_iteration(validation_state_iteration, gpu, town_name, experiment_set, exp_batch,
                              exp_alias, params, control_filename, task_list)

        else:
            """
            #####
            Main Loop , Run a benchmark for each specified checkpoint on the "Test Configuration"
            #####
            """
            while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):
                # Get the correct checkpoint
                # We check it for some task name, all of then are ready at the same time
                if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE,
                                            control_filename + '_' + task_list[0]):

                    latest = get_next_checkpoint(g_conf.TEST_SCHEDULE,
                                                 control_filename + '_' + task_list[0])

                    driving_iteration(latest, gpu, town_name, experiment_set, exp_batch,
                                      exp_alias, params, control_filename, task_list)

                else:
                    time.sleep(0.1)

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something happened'})



