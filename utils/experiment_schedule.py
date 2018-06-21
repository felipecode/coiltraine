
from logger import monitorer
import heapq


def get_remainig_exps(executing_processes, experiment_list):

    executing_list = []
    for process in executing_processes:
        executing_list.append(process['experiment'])

    return list(set(experiment_list)- set(executing_list))


def get_gpu_resources(gpu_resources, executing_processes, allocation_params):

    """

    Args:
        allocated_gpus:
        executing_processes:
        allocation_params:

    Returns:

    """
    still_executing_processes = []
    for process_specs in executing_processes:
        # Make process name:
        if process_specs['type'] == 'drive':

            name = 'drive_' + process_specs['environment']

        elif process_specs['type'] == 'validation':
            name = 'validation_' + process_specs['dataset']
        else:
            name = process_specs['type']

        status = monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                     name)[0]

        if status == "Finished" or status == 'Error':

            gpu_resources[process_specs['gpu']] += allocation_params[process_specs['type']+'_cost']

        else:
            still_executing_processes.append(process_specs)


    return gpu_resources, max(gpu_resources.values()), still_executing_processes


def allocate_gpu_resources(gpu_resources, amount_to_allocate):
    """
        On GPU management allocate gpu resources considering a dictionary with resources
        for each gpu
    Args:
        gpu_resources:
        amount_to_allocate:

    Returns:

    """

    for gpu, resource in gpu_resources.items():
        if resource >= amount_to_allocate:
            gpu_resources[gpu] -= amount_to_allocate
            return gpu_resources, max(gpu_resources.values()), gpu

    raise ValueError("Not enough gpu resources to allocate")


# TODO: function need severe refactoring  !!! !
def mount_experiment_heap(folder, experiments_list, is_training, executing_processes, old_tasks_queue,
                          validation_datasets, drive_environments, restart_error=True):


    tasks_queue = []
    for experiment in experiments_list:
        if experiment in executing_processes:

            continue


        # Train is always priority. # TODO: some system to check priority depending on iterations
        task_to_add = None
        # TODO: One thing is error other thing is stop. However at a first step we can try to restart all error things
        if is_training:
            if monitorer.get_status(folder, experiment, 'train')[0] == "Not Started":

                task_to_add = (0, experiment + '_train',
                                         {'type': 'train', 'folder': folder,
                                          'experiment': experiment})

            elif restart_error and monitorer.get_status(folder, experiment, 'train')[0] \
                            == "Error":

                task_to_add = (0, experiment + '_train',
                               {'type': 'train', 'folder': folder,
                                'experiment': experiment})
        if task_to_add is not None and task_to_add not in old_tasks_queue:
            heapq.heappush(tasks_queue, task_to_add)

        task_to_add = None

        for val_data in validation_datasets:
            if monitorer.get_status(folder, experiment, 'validation_' + val_data)[0] == "Not Started":
                task_to_add = (2, experiment + '_validation_' + val_data,
                                             {'type': 'validation', 'folder': folder,
                                              'experiment': experiment, 'dataset': val_data})

            elif restart_error and monitorer.get_status(folder, experiment, 'validation_'
                                                                + val_data)[0] == "Error":
                task_to_add = (2, experiment + '_validation_' + val_data,
                                             {'type': 'validation', 'folder': folder,
                                              'experiment': experiment, 'dataset': val_data})
        if task_to_add is not None and task_to_add not in old_tasks_queue:
            heapq.heappush(tasks_queue, task_to_add)

        task_to_add = None

        for drive_env in drive_environments:
            if monitorer.get_status(folder, experiment, 'drive_' + drive_env)[0] == "Not Started":
                task_to_add = (1, experiment + '_drive_' + drive_env,
                                             {'type': 'drive', 'folder': folder,
                                              'experiment': experiment, 'environment': drive_env})

            elif restart_error and monitorer.get_status(folder, experiment, 'drive_' + drive_env)\
                                                        [0] == "Error":
                task_to_add = (1, experiment + '_drive_' + drive_env,
                               {'type': 'drive', 'folder': folder,
                                'experiment': experiment, 'environment': drive_env})


        if task_to_add is not None and task_to_add not in old_tasks_queue:
            heapq.heappush(tasks_queue, task_to_add)

    return tasks_queue