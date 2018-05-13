import os
import numpy as np
import unittest
import shutil
import torch
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



        heap_of_exp = mount_experiment_heap(folder, experiments_list,
                              validation_datasets, drive_environments)



        print ("Number of experiment", len(heap_of_exp))

        print ("Returns")


        print(heap_of_exp)


    def test_gpu_poping(self):

        allocated_gpus = ['0', '1', '2', '3', '4']
        allocated_gpus = [[gpu]*2 for gpu in allocated_gpus]
        path = 'configs'
        folder = 'eccv'
        experiments_list = os.listdir(os.path.join(path, folder))

        validation_datasets = ['SmallTest', 'OtherSmallTest']
        drive_environments = ['Town01', 'Town02']

        executing_processes = []

        free_gpus, number_of_free_gpus = get_free_gpus(allocated_gpus, executing_processes)

        print (free_gpus, number_of_free_gpus)

        tasks_queue = mount_experiment_heap(folder, experiments_list,
                              validation_datasets, drive_environments)

        executing_processes = []


        while len(free_gpus) > 0:
            #Allocate all the gpus

            process_specs = heapq.heappop(tasks_queue)[2]  # To get directly the dict


            if process_specs['type'] == 'train' and number_of_free_gpus >=1:
                free_gpus, number_of_free_gpus, gpu_number = pop_one_gpu(free_gpus)
                #execute_train(gpu_number, process_specs['folder'], process_specs['experiment'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            elif process_specs['type'] == 'validation':
                free_gpus, number_of_free_gpus, gpu_number = pop_half_gpu(free_gpus)

                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            else:  # == test

                free_gpus, number_of_free_gpus, gpu_number = pop_half_gpu(free_gpus)

                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            print (free_gpus)
            print (executing_processes)



        free_gpus = get_free_gpus(allocated_gpus, executing_processes)
        print ("Free GPU Before", free_gpus)

        g_conf.NAME = 'experiment_1'
        # TODO: this merge is weird.
        merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')

        coil_logger.add_message('Iterating', {'Current Loss': 3,
                                 'Best Loss': 1, 'Best Loss Iteration': 1,
                                 'Some Output': 2,
                                 'GroundTruth': 3,
                                 'Error': 4,
                                 'Inputs': 5},
                                2000)


        free_gpus = get_free_gpus(allocated_gpus, executing_processes)
        print ("Free GPU After", free_gpus)

