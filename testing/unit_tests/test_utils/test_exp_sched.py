import os
import numpy as np
import unittest
import shutil
import torch
import random
from logger import coil_logger
from configs import g_conf, merge_with_yaml, set_type_of_process

from utils.experiment_schedule import *




class testExpSched(unittest.TestCase):

    def test_mount_experiment_heap(self):


        path = 'configs'
        folder = 'eccv'
        experiments_list = os.listdir(os.path.join(path, folder))

        validation_datasets = ['SmallTest', 'OtherSmallTest']
        drive_environments = ['Town01', 'Town02']



        heap_of_exp = mount_experiment_heap(folder, experiments_list, True,
                              validation_datasets, drive_environments)



        print ("Number of experiment", len(heap_of_exp))

        print ("Returns")


        print(heap_of_exp)


    def test_gpu_poping(self):

        gpus_list = ['0', '1', '2', '3']
        path = 'configs'
        folder = 'test_exps'
        experiments_list = os.listdir(os.path.join(path, folder))

        experiments_list = [experiment.split('.')[-2] for experiment in experiments_list]

        validation_datasets = ['SmallTest', 'OtherSmallTest']
        drive_environments = ['Town01', 'Town02']
        allocation_parameters = {'gpu_value': 3.5,
                                 'train_cost': 2,
                                 'validation_cost': 1.5,
                                 'drive_cost': 1.5}

        allocated_gpus = {gpu: allocation_parameters['gpu_value'] for gpu in gpus_list}
        executing_processes = []

        free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(allocated_gpus,
                                                                           executing_processes,
                                                                           allocation_parameters)
        print (" Free GPUS, resources on the most free")
        print (free_gpus, resources_on_most_free_gpu)

        print ("Experiments list")
        print (experiments_list)

        tasks_queue = mount_experiment_heap(folder, experiments_list, True,
                                            validation_datasets, drive_environments)

        print ("Tasks queue", tasks_queue)

        executing_processes = []
        while True:

            while resources_on_most_free_gpu > min([allocation_parameters['train_cost'],
                                                 allocation_parameters['validation_cost'],
                                                 allocation_parameters['drive_cost']])\
                    and tasks_queue != []:
                #Allocate all the gpus
                print ("TASKS ", tasks_queue)
                popped_thing = heapq.heappop(tasks_queue)
                process_specs = popped_thing[2]  # To get directly the dict
                print ("process got: ", process_specs)
                print (free_gpus, resources_on_most_free_gpu)

                if process_specs['type'] == 'train' and resources_on_most_free_gpu >= allocation_parameters['train_cost']:
                    free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                                 free_gpus,
                                                                 allocation_parameters['train_cost'])
                    #execute_train(gpu_number, process_specs['folder'], process_specs['experiment'])
                    process_specs.update({'gpu': gpu_number})
                    executing_processes.append(process_specs)

                elif process_specs['type'] == 'validation' and resources_on_most_free_gpu >= allocation_parameters['validation_cost']:
                    free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                                 free_gpus,
                                                                 allocation_parameters['validation_cost'])
                    #execute_validation(gpu_number, process_specs['folder'], process_specs['experiment'],
                    #                        process_specs['dataset'])
                    process_specs.update({'gpu': gpu_number})
                    executing_processes.append(process_specs)

                elif process_specs['type'] == 'drive' and resources_on_most_free_gpu >= allocation_parameters['drive_cost']:

                    free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                                 free_gpus,
                                                                 allocation_parameters['drive_cost'])
                    #execute_drive(gpu_number, process_specs['folder'], process_specs['experiment'],
                    #                   process_specs['environment'])
                    process_specs.update({'gpu': gpu_number})
                    executing_processes.append(process_specs)



            random_process = random.choice(executing_processes)
            print ('random process', random_process)
            fp_name = random_process['experiment']
            g_conf.immutable(False)
            merge_with_yaml('configs/test_exps/' + fp_name + '.yaml')
            # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS

            if random_process['type'] == 'drive':
                set_type_of_process(random_process['type'], random_process['environment'])
            elif random_process['type'] == 'validation':
                set_type_of_process(random_process['type'], random_process['dataset'])
            else:
                set_type_of_process(random_process['type'])

            random_message = random.choice(['Finished', 'Error', 'Iterating'])

            print ('set ', random_process['type'], ' from ', random_process['experiment'], ' to ', random_message  )

            if  random_message == 'Iterating':
                coil_logger.add_message(random_message, {'Iteration': 1}, 1)
                coil_logger.add_message(random_message, {'Iteration': 2}, 2)
            else:
                coil_logger.add_message(random_message, {})


            free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(
                                                                      allocated_gpus,
                                                                      executing_processes,
                                                                      allocation_parameters)

            coil_logger.close()
            if len(executing_processes) == 0:
                break
            print ("Free GPU After  ", free_gpus, resources_on_most_free_gpu)

            print ("WE have ", len(executing_processes), " Running.")

# TODO: Test it should continue experiments that are iterating but stopped.

# You mount the experiment heap just once, then you just eliminated th
