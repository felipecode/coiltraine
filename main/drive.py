
import traceback

import sys
import logging

import configparser
import datetime

import numpy as np
import os
import time

from carla import image_converter

from drive_interfaces.carla.comercial_cars.test_carla_machine import TestCarlaMachine
from drive_interfaces.carla.comercial_cars.data_benchmark import DataBenchmark
from drive_interfaces.carla.comercial_cars.generalization_benchmark import GeneralizationBenchmark

#from drive_interfaces.carla.comercial_cars.lightbenchmark import LightBenchmark
#from drive_interfaces.carla.comercial_cars.test_benchmark import TestBenchmark

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client


def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))


def maximun_checkpoint_reach():


    return 0




# TODO: Go for that KWARGS stuff.
# TODO: Add all the necessary logging.

def execute(host, port, experiment_name, city_name='Town01', weather_used=1, memory_use=0.2):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    #drive_config.city_name = city_name
    # TODO Eliminate drive config.

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
                while not .maximun_checkpoint_reach():

                    coil_agent = CoILAgent(checkpoint)

                    # Get the correct checkpoint
                    if test_agent.next_check_point_ready():

                        if city_name == 'Town01':




                            data_bench = DataBenchmark(city_name=city_name, name_to_save=test_agent.get_test_name()
                                                                                         + '_' + experiment_name+'auto',
                                                       camera_set=drive_config.camera_set, continue_experiment=True,
                                                       )
                            print "DATA BENCH"
                        else:
                            data_bench = GeneralizationBenchmark(city_name=city_name,
                                                                 name_to_save=test_agent.get_test_name()
                                                        + '_' + experiment_name+'auto',
                                                       continue_experiment=True,
                                                       )
                            print "GEN BENCH"

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
