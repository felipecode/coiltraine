
import traceback

import sys
import logging

import datetime

import numpy as np
import os
import time
import subprocess


from drive import ECCVTrainingSuite
from drive import ECCVGeneralizationSuite

# TODO: MAKE A SYSTEM TO CONTROL CHeckpoint
from utils.checkpoint_schedule import next_check_point_ready, maximun_checkpoint_reach


from carla.tcp import TCPConnectionError
from carla.client import make_carla_client


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))



def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# TODO: note, for now carla and carla test are in the same GPU

# TODO: Go for that KWARGS stuff .... MAYBE
# TODO: Add all the necessary logging.

# OBS : I AM FIXING host as localhost now
# OBS : Memory use should also be adaptable lets leave it fixed for now

def execute(gpu, exp_batch, exp_alias, city_name='Town01', memory_use=0.2, host='127.0.0.1'):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    #drive_config.city_name = city_name
    # TODO Eliminate drive config.

    print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    #vglrun - d:7.$GPU $CARLA_PATH / CarlaUE4 / Binaries / Linux / CarlaUE4 / Game / Maps /$TOWN - windowed - benchmark - fps = 10 - world - port =$PORT;
    #sleep    100000


    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

    os.environ['SDL_VIDEODRIVER'] = 'offscreen'
    os.environ['SDL_HINT_CUDA_DEVICE'] = str(gpu)

    subprocess.call([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + city_name,
                     '-benchmark', '-fps=10', '-world-port='+str(port)])


    #test_agent = CarlaDrive(experiment_name)




    # TODO we have some external class that control this weather thing.

    if city_name == 'Town01':
        experiment_suite = ECCVTrainingSuite()
    else:
        experiment_suite = ECCVGeneralizationSuite()



    while True:
        try:

            with make_carla_client(host, port) as client:


                # Now actually run the driving_benchmark


                # While the checkpoint is not there
                while not maximun_checkpoint_reach():

                    #TODO DO we redo the agent here ?
                    checkpoint = get_next_checkpoint()
                    coil_agent = CoILAgent(checkpoint)

                    # Get the correct checkpoint
                    if next_check_point_ready():

                        run_driving_benchmark(coil_agent, experiment_suite, city_name,
                                              exp_batch+'_'+exp_alias +'iteration', True,
                                              host, port)


                        test_agent.finish_model()

                        test_agent.write(results)

                    else:
                        time.sleep(0.1)


                test_agent.export_results(str(weather_used))
                break


        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
