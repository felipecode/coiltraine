
import traceback

import sys
import logging

import configparser
import datetime

import numpy as np
import os
import time

from carla import image_converter


# MAKE A SYSTEM TO CONTROL CHeckpoint

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))



def maximun_checkpoint_reach():
    if self._current_checkpoint_number >= len(self._checkpoint_schedule):
        return True
    else:
        return False



def next_check_point_ready():
    """
    Looks at every checkpoint file in the folder. And for each of
    then tries to find the one that matches EXACTLY with the one in the schedule

    :return:
    """

    checkpoint_files = sorted(os.listdir(self._config_input.models_path))
    for f in checkpoint_files:

        match = re.search('model.ckpt-(\d+)', f)
        if match:
            checkpoint_number = match.group(1)

            if int(checkpoint_number) == (self._checkpoint_schedule[self._current_checkpoint_number]):
                self._checkpoint_number_to_test = str(self._checkpoint_schedule[self._current_checkpoint_number])

                return True
    logging.info('Checkpoint Not Found, Will wait for %d' % self._checkpoint_schedule[self._current_checkpoint_number] )
    return False

def get_test_name():

    return str(self._checkpoint_number_to_test)

def finish_model():
    """
    Increment and go to the next model

    :return None:

    """
    self._current_checkpoint_number += 1


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

    print("Running ", __file__, " On GPU ",gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    #vglrun - d:7.$GPU $CARLA_PATH / CarlaUE4 / Binaries / Linux / CarlaUE4 / Game / Maps /$TOWN - windowed - benchmark - fps = 10 - world - port =$PORT;
    #sleep    100000


    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

    os.environ['SDL_VIDEODRIVER'] = 'offscreen'
    os.environ['SDL_HINT_CUDA_DEVICE'] = str(gpu)

    subprocess.call([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + city_name,
                     '-benchmark', '-fps=10', '-world-port='+str(port)])


    test_agent = CarlaDrive(experiment_name)



    # TODO we have some external class that control this weather thing.

    if city_name == 'Town01':
        weather_used = 1
    else:
        weather_used = 14



    while True:
        try:

            with make_carla_client(host, port) as client:



                # While the checkpoint is not there
                while not maximun_checkpoint_reach():

                    coil_agent = CoILAgent(checkpoint)

                    # Get the correct checkpoint
                    if test_agent.next_check_point_ready():

                        if city_name == 'Town01':




                            data_bench = DataBenchmark(city_name=city_name, name_to_save=test_agent.get_test_name()
                                                                                         + '_' + experiment_name+'auto',
                                                       camera_set=drive_config.camera_set, continue_experiment=True,
                                                       )

                        else:
                            data_bench = GeneralizationBenchmark(city_name=city_name,
                                                                 name_to_save=test_agent.get_test_name()
                                                        + '_' + experiment_name+'auto',
                                                       continue_experiment=True,
                                                       )


                        test_agent.load_model()

                        run_driving_benchmark(coil_agent, experiment_suite, args.city_name,
                                              args.log_name, args.continue_experiment,
                                              args.host, args.port)

                        results = data_bench.benchmark_agent(test_agent, client)
                        test_agent.finish_model()

                        test_agent.write(results)

                    else:
                        time.sleep(0.1)



                logging.info(" Maximun Checkpoint reach")
                test_agent.export_results(str(weather_used))
                break


        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
