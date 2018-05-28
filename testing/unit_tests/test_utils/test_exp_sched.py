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

        gpus_list = ['0', '1', '2', '3']
        path = 'configs'
        folder = 'eccv'
        experiments_list = os.listdir(os.path.join(path, folder))

        validation_datasets = ['SmallTest', 'OtherSmallTest']
        drive_environments = ['Town01', 'Town02']
        allocation_parameters = {'gpu_value': 5,
                                 'train_cost': 2,
                                 'validation_cost': 1.5,
                                 'drive_cost': 1.5}

        allocated_gpus = {gpu: allocation_parameters['gpu_value'] for gpu in gpus_list}
        executing_processes = []

        free_gpus, resources_on_most_free_gpu = get_gpu_resources(allocated_gpus, executing_processes,
                                                               allocation_parameters)

        print (free_gpus, resources_on_most_free_gpu)

        tasks_queue = mount_experiment_heap(folder, experiments_list,
                              validation_datasets, drive_environments)

        executing_processes = []


        while resources_on_most_free_gpu > min([allocation_parameters['train_cost'],
                                             allocation_parameters['validation_cost'],
                                             allocation_parameters['drive_cost']]):
            #Allocate all the gpus

            process_specs = heapq.heappop(tasks_queue)[2]  # To get directly the dict
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

        free_gpus, resources_on_most_free_gpu = get_gpu_resources(free_gpus, executing_processes, allocation_parameters)
        print ("Free GPU Before", free_gpus)

        g_conf.NAME = 'experiment_1'
        merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')

        coil_logger.add_message('Iterating',
                                {'Iteration': 2000* g_conf.BATCH_SIZE,
                                 'Current Loss': 3,
                                 'Best Loss': 1, 'Best Loss Iteration': 1,
                                 'Some Output': 2,
                                 'GroundTruth': 3,
                                 'Error': 4,
                                 'Inputs': 5},
                                2000 * g_conf.BATCH_SIZE)


        free_gpus, resources_on_most_free_gpu = get_gpu_resources(allocated_gpus, executing_processes, allocation_parameters)
        print ("Free GPU After", free_gpus)

# TODO: Test it should continue experiments that are iterating but stopped.
