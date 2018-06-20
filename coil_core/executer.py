import os
import time
import multiprocessing
import heapq

from utils.experiment_schedule import get_gpu_resources, allocate_gpu_resources, \
    mount_experiment_heap, get_remainig_exps
from utils.general import create_exp_path
from logger import printer, monitorer

from . import train, validate, run_drive


def execute_train(gpu, exp_batch, exp_alias, suppress_output=True):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    create_exp_path(exp_batch, exp_alias)
    p = multiprocessing.Process(target=train.execute,
                                args=(gpu, exp_batch, exp_alias, suppress_output))
    p.start()


def execute_validation(gpu, exp_batch, exp_alias, dataset, suppress_output=True):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    # if module_name not in set(["train","drive","evaluate"]):
    #    raise ValueError("Invalid module to execute")

    create_exp_path(exp_batch, exp_alias)
    # The difference between train and validation is the
    p = multiprocessing.Process(target=validate.execute,
                                args=(gpu, exp_batch, exp_alias, dataset, suppress_output))
    p.start()


def execute_drive(gpu, exp_batch, exp_alias, exp_set_name, suppress_output=True, no_screen=False):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """

    create_exp_path(exp_batch, exp_alias)
    p = multiprocessing.Process(target=run_drive.execute,
                                args=(gpu, exp_batch, exp_alias, exp_set_name,
                                      0.2, "127.0.0.1", suppress_output,
                                      no_screen))

    p.start()


def folder_execute(params=None):
    """
    On this mode the training software keeps all
    It forks a process to run the monitor over the training logs.
    Arguments
        param, prioritize training, prioritize test, prioritize
    """

    folder = params['folder']
    allocated_gpus = params['gpus']
    validation_datasets = params['validation_datasets']
    driving_environments = params['driving_environments']
    allocation_parameters = params['allocation_parameters']

    experiments_list = os.listdir(os.path.join('configs', folder))
    experiments_list = [experiment.split('.')[-2] for experiment in experiments_list]

    # Each gpu has maximun 2 slots

    allocated_gpus = {gpu: allocation_parameters['gpu_value'] for gpu in allocated_gpus}

    print(allocated_gpus)

    executing_processes = []

    free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(allocated_gpus,
                                                                                   executing_processes,
                                                                                   allocation_parameters)

    # Is a queue of tasks to be executed. The priority is always train.
    # then test then val.
    # TODO: change the priority to test the ones that have already been trained.
    tasks_queue = mount_experiment_heap(folder, experiments_list, params['is_training'],
                                        validation_datasets, driving_environments)

    # No process is executing right now.

    print(tasks_queue)

    # TODO: the while should go outside, so the monitorer process is independent of the type of execution

    while True:
        #        if not done or executing  get to the list
        # If amount of resources is smaller than a threshold.

        while resources_on_most_free_gpu >= min([allocation_parameters['train_cost'],
                                                 allocation_parameters['validation_cost'],
                                                 allocation_parameters['drive_cost']]) \
                and tasks_queue != []:
            # Allocate all the gpus
            popped_thing = heapq.heappop(tasks_queue)
            process_specs = popped_thing[2]  # To get directly the dict


            # Get the train status, that will affect in scheduling a validation or drive process
            train_status = monitorer.get_status(folder, process_specs['experiment'], 'train')[0]

            if process_specs['type'] == 'train' and resources_on_most_free_gpu >= \
                    allocation_parameters['train_cost']:
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                    free_gpus,
                    allocation_parameters['train_cost'])

                execute_train(gpu_number, process_specs['folder'], process_specs['experiment'])
                process_specs.update({'gpu': gpu_number})

                executing_processes.append(process_specs)

            elif process_specs['type'] == 'validation' and resources_on_most_free_gpu >= \
                    allocation_parameters['validation_cost'] \
                    and (train_status == 'Iterating' or train_status == 'Loading' or
                         train_status == 'Finished'):
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                        free_gpus, allocation_parameters['validation_cost'])
                execute_validation(gpu_number, process_specs['folder'], process_specs['experiment'],
                                   process_specs['dataset'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)

            elif process_specs['type'] == 'drive' and resources_on_most_free_gpu >= \
                    allocation_parameters['drive_cost'] \
                    and (train_status == 'Iterating' or train_status == 'Loading' or
                         train_status == 'Finished'):
                free_gpus, resources_on_most_free_gpu, gpu_number = allocate_gpu_resources(
                                            free_gpus, allocation_parameters['drive_cost'])
                execute_drive(gpu_number, process_specs['folder'], process_specs['experiment'],
                              process_specs['environment'], no_screen=params['no_screen'])
                process_specs.update({'gpu': gpu_number})
                executing_processes.append(process_specs)





        tasks_queue = mount_experiment_heap(folder, experiments_list, params['is_training'],
                                            validation_datasets, driving_environments, False)



        printer.plot_folder_summaries(folder,
                                      params['is_training'],
                                      validation_datasets,
                                      driving_environments)
        # Check allocated process, and look which ones finished.
        free_gpus, resources_on_most_free_gpu, executing_processes = get_gpu_resources(
            allocated_gpus,
            executing_processes,
            allocation_parameters)

        if len(tasks_queue) == 0 and len(executing_processes) == 0:
            break
        #print ("Task queue", tasks_queue)
        #print ("exec proc", executing_processes)
        print("resources", free_gpus)
        time.sleep(10)

    print("ALL EXPERIMENTS EXECUTED")
