import argparse
import multiprocessing
# Import all the test libraries.
import sys
import os
from logger import monitorer
import time

from coil_core import train, validate, run_drive

from utils.experiment_schedule import get_gpu_resources, allocate_gpu_resources, mount_experiment_heap


#pop_half_gpu, pop_one_gpu, mount_experiment_heap

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




def folder_execute(params=None):
    """
    On this mode the training software keeps all
    It forks a process to run the monitor over the training logs.
    Arguments
        param, prioritize training, prioritize test, prioritize
    """

    # TODO: BUG, the experiments dont continue when they stoped. IT NEED TO MARK ERROR.

    folder = params['folder']
    allocated_gpus = params['gpus']
    validation_datasets = params['validation_datasets']
    driving_environments = params['driving_environments']
    allocation_parameters = params['allocation_parameters']

    experiments_list = os.listdir(os.path.join('configs', folder))
    experiments_list = [experiment.split('.')[-2] for experiment in experiments_list]

    # Each gpu has maximun 2 slots

    allocated_gpus = {gpu: allocation_parameters['gpu_value']   for gpu in allocated_gpus}

    print (allocated_gpus)

    # TODO: for now the number of gpus used per process is hardcoded, train 1, val 0.5, drive 0.5

    executing_processes = []

    free_gpus, resources_on_most_free_gpu = get_gpu_resources(allocated_gpus, executing_processes,
                                                           allocation_parameters)

    # Is a queue of tasks to be executed. The priority is always train.
    # then test then val.
    # TODO: change the priority to test the ones that have already been trained.
    tasks_queue = mount_experiment_heap(folder, experiments_list, params['is_training'],
                                        validation_datasets, driving_environments)

    # No process is executing right now.

    print (tasks_queue)


    while True:
        #        if not done or executing  get to the list
        # If amount of resources is smaller than a threshold.
        while resources_on_most_free_gpu >= min([allocation_parameters['train_cost'],
                                            allocation_parameters['validation_cost'],
                                            allocation_parameters['drive_cost']]):
            #Allocate all the gpus

            process_specs = heapq.heappop(tasks_queue)[2]  # To get directly the dict
            print ("process ", process_specs)
            print (free_gpus, resources_on_most_free_gpu)
            if process_specs['type'] == 'train' and resources_on_most_free_gpu >= allocation_parameters['train_cost']:
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                             free_gpus,
                                                             allocation_parameters['train_cost'])
                execute_train(gpu_number, process_specs['folder'], process_specs['experiment'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            elif process_specs['type'] == 'validation' and resources_on_most_free_gpu >= allocation_parameters['validation_cost']:
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                             free_gpus,
                                                             allocation_parameters['validation_cost'])
                execute_validation(gpu_number, process_specs['folder'], process_specs['experiment'],
                                        process_specs['dataset'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            elif process_specs['type'] == 'drive' and resources_on_most_free_gpu >= allocation_parameters['drive_cost']:

                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                                             free_gpus,
                                                             allocation_parameters['drive_cost'])
                execute_drive(gpu_number, process_specs['folder'], process_specs['experiment'],
                                   process_specs['environment'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)




        time.sleep(5)


        monitorer.plot_folder_summaries(folder,
                                        params['is_training'],
                                        validation_datasets,
                                        driving_environments)
        # Check allocated process, and look which ones finished.
        free_gpus = get_gpu_resources(allocated_gpus, executing_processes,allocation_parameters)
        #Check



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)


    argparser.add_argument(
        '--single_process',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val_datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive_envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )

    args = argparser.parse_args()




    for gpu in args.gpus:
        try:
            int(gpu)
        except:
            raise ValueError(" Gpu is not a valid int number")



    # Obs this is like a fixed parameter, how much a validation and a train and drives ocupies

    # TODO: of course this change from gpu to gpu , but for now we just assume at least a K40
    # Maybe the latest voltas will be underused


    # TODO: Divide maybe into different executables.
    if args.single_process is not None:
        if args.single_process == 'train':
            execute_train("0", "eccv", "experiment_1")

        if args.single_process == 'validation':
            execute_validation("0", "eccv", "experiment_1", "SmallTest")

        if args.single_process == 'drive':
            execute_drive("0", "eccv", "experiment_1", 'Town02')


    else:
        allocation_parameters = {'gpu_value': 3.5,
                                 'train_cost': 2,
                                 'validation_cost': 1.5,
                                 'drive_cost': 1.5}

        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': list(args.driving_environments),
            'allocation_parameters': allocation_parameters
        }

        print (params)

        folder_execute(params)
