
import multiprocessing
# Import all the test libraries.
import sys
import os
from logger import monitorer
import time

from coil_core import train, validate, run_drive

from utils import get_free_gpus, pop_half_gpu, pop_one_gpu, mount_experiment_heap

import heapq

# You could send the module to be executed and they could have the same interface.



def execute_train(gpu, exp_batch, exp_alias):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """

    p = multiprocessing.Process(target=train.execute, args=(gpu, exp_batch, exp_alias,))
    p.start()

def execute_validation(gpu, exp_batch, exp_alias, dataset):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    #if module_name not in set(["train","drive","evaluate"]):
    #    raise ValueError("Invalid module to execute")



    # The difference between train and validation is the
    p = multiprocessing.Process(target=validate.execute, args=(gpu, exp_batch, exp_alias, dataset))
    p.start()


#TODO: set before the dataset path as environment variables

def execute_drive(gpu, exp_batch, exp_alias, city_name):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """

    p = multiprocessing.Process(target=run_drive.execute, args=(gpu, exp_batch, exp_alias, city_name,))
    #p.daemon = True
    p.start()




def folder_execute(folder, allocated_gpus, param=None):
    """
    On this mode the training software keeps all
    It forks a process to run the monitor over the training logs.
    Arguments
        param, prioritize training, prioritize test, prioritize
    """


    #TODO: it is likely that the monitorer classes is not actually necessary.

    experiments_list = os.listdir(os.path.join('configs', folder))

    # Each gpu has maximun 2 slots

    allocated_gpus = [[gpu] * 2 for gpu in allocated_gpus]

    validation_datasets = ['SmallTest', 'OtherSmallTest']
    drive_environments = ['Town01', 'Town02']


    # TODO: for now the number of gpus used per process is hardcoded, train 1, val 0.5, drive 0.5

    executing_processes = []

    free_gpus, number_of_free_gpus = get_free_gpus(allocated_gpus, executing_processes)

    # Is a queue of tasks to be executed. The priority is always train.
    # then test then val.
    # TODO: change the priority to test the ones that have already been trained.
    tasks_queue = mount_experiment_heap(folder, experiments_list,
                                        validation_datasets, drive_environments)

    # No process is executing right now.


    while True:
        #        if not done or executing  get to the list
        while len(free_gpus) > 0:
            #Allocate all the gpus

            process_specs = heapq.heappop(tasks_queue)[2]  # To get directly the dict


            if process_specs['type'] == 'train' and number_of_free_gpus >=1:
                free_gpus, number_of_free_gpus, gpu_number = pop_one_gpu(free_gpus)
                execute_train(gpu_number, process_specs['folder'], process_specs['experiment'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            elif process_specs['type'] == 'validation':
                free_gpus, number_of_free_gpus, gpu_number = pop_half_gpu(free_gpus)
                execute_validation(gpu_number, process_specs['folder'], process_specs['experiment'],
                                   process_specs['dataset'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            else:  # == test

                free_gpus, number_of_free_gpus, gpu_number = pop_half_gpu(free_gpus)
                execute_drive(gpu_number, process_specs['folder'], process_specs['experiment'],
                                   process_specs['environment'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

        else:
            time.sleep(1)

        # Check allocated process, and look which ones finished.
        free_gpus = get_free_gpus(allocated_gpus, executing_processes)
        #Check



if __name__ == '__main__':


    execute_train("0", "eccv", "experiment_1")
    #execute_validation("0", "eccv", "experiment_1","SmallTest")
    #execute_drive("0", "eccv", "experiment_1", 'Town02')
    #folder_execute('eccv', "0,1,2")
