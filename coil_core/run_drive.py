
import traceback

import sys
import logging
import json
import datetime


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

from drive import CoILAgent
from drive import ECCVTrainingSuite
from drive import ECCVGeneralizationSuite
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

def start_carla_simulator(gpu, exp_batch, exp_alias, city_name):

    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

    #os.environ['SDL_VIDEODRIVER'] = 'offscreen'
    #os.environ['SDL_HINT_CUDA_DEVICE'] = str(gpu)

    #subprocess.call()

    sp = subprocess.Popen([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + city_name
                           , '-windowed', '-benchmark', '-fps=10', '-world-port='+str(port)], shell=False,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sp, port




# OBS: note, for now carla and carla test are in the same GPU

# TODO: Add all the necessary logging.

# OBS : I AM FIXING host as localhost now
# OBS : Memory use should also be adaptable lets leave it fixed for now

def execute(gpu, exp_batch, exp_alias, city_name='Town01', memory_use=0.2, host='127.0.0.1'):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    #drive_config.city_name = city_name
    # TODO Eliminate drive config.

    print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu



    sys.stdout = open(str(os.getpid()) + ".out", "a", buffering=1)


    #vglrun - d:7.$GPU $CARLA_PATH / CarlaUE4 / Binaries / Linux / CarlaUE4 / Game / Maps /$TOWN - windowed - benchmark - fps = 10 - world - port =$PORT;
    #sleep    100000

    carla_process, port = start_carla_simulator(gpu, exp_batch, exp_alias, city_name)


    merge_with_yaml(os.path.join(exp_batch, exp_alias+'.yaml'))
    set_type_of_process('test')

    #test_agent = CarlaDrive(experiment_name)




    # TODO we have some external class that control this weather thing.

    """
    if city_name == 'Town01':
        experiment_suite = ECCVTrainingSuite()
    else:
        experiment_suite = ECCVGeneralizationSuite()
    """
    experiment_suite = TestSuite()


    while True:
        try:

            with make_carla_client(host, port) as client:


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
                        run_driving_benchmark(coil_agent, experiment_suite, city_name,
                                              exp_batch + '_' + exp_alias + 'iteration', False,
                                              host, port)

                        # Read the resulting dictionary
                        with open(os.path.join('_benchmark_results',
                                               exp_batch+'_'+exp_alias + 'iteration', 'metrics.json')
                                  , 'r') as f:
                            summary_dict = json.loads(f.read())

                        # TODO: When you add the message you need to check if the experiment continues properly
                        coil_logger.add_message({'Running': {"DBSummary": summary_dict}})



                        #test_agent.finish_model()

                        #test_agent.write(results)

                    else:
                        time.sleep(0.1)
                # TODO: is this really needed ??? I believe not.
                #monitorer.export_results(os.path.join('_benchmark_results',
                #                                      exp_batch + '_' +exp_alias +'iteration'))
                break


        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
            carla_process.kill()

        except KeyboardInterrupt:
            carla_process.kill()
        except:
            traceback.print_exc()
            carla_process.kill()

    carla_process.kill()

