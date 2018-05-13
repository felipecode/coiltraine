
from logger import monitorer
import heapq


def get_free_gpus(allocated_gpus, executing_processes):
    free_gpus = allocated_gpus

    for process_specs in executing_processes:

        if monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                process_specs['type']) == "Error" or \
                monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                     process_specs['type']) == "Finished":
            free_gpus.append([process_specs['gpu'], process_specs['gpu']])

    return free_gpus, sum(len(row) for row in free_gpus) / 2.0


def pop_half_gpu(free_gpus):
    for gpu in free_gpus:
        if len(gpu) > 1:
            free_gpus[free_gpus.index(gpu)] = [free_gpus[free_gpus.index(gpu)][0]]
            return free_gpus, sum(len(row) for row in free_gpus) / 2.0, gpu[0]
        else:
            del free_gpus[free_gpus.index(gpu)]
            return free_gpus, sum(len(row) for row in free_gpus) / 2.0, gpu[0]


def pop_one_gpu(free_gpus):
    for gpu in free_gpus:
        if len(gpu) > 1:
            del free_gpus[free_gpus.index(gpu)]
            return free_gpus, sum(len(row) for row in free_gpus) / 2.0, gpu[0]
        # Otherwise does not pop. maybe raise
        else:
            raise ValueError("No full gpu available")


def mount_experiment_heap(folder, experiments_list, validation_datasets, drive_environments):


    tasks_queue = []
    for experiment in experiments_list:


        # Train is always priority. # TODO: some system to check priority depending on iterations
        if monitorer.get_status(folder, experiment, 'train')[0] == "Not Started" or \
                monitorer.get_status(folder, experiment, 'train')[0] == "Error":

            heapq.heappush(tasks_queue, (0,  experiment+'_train' ,
                                         {'type': 'train', 'folder': folder,
                                             'experiment': experiment}))


        for val_data in validation_datasets:
            if monitorer.get_status(folder, experiment, 'validation_' + val_data)[0] == "Not Started" or \
                    monitorer.get_status(folder, experiment, 'validation_'+ val_data)[0] == "Error":


                heapq.heappush(tasks_queue, (2, experiment+'_validation_' + val_data,
                                             {'type': 'validation', 'folder': folder,
                                                 'experiment': experiment, 'dataset': val_data}))

        for drive_env in drive_environments:
            if monitorer.get_status(folder, experiment, 'drive_' + drive_env)[0] == "Not Started" or \
                    monitorer.get_status(folder, experiment, 'drive_' + drive_env)[0] == "Error":


                heapq.heappush(tasks_queue, (1, experiment+'_drive_' + drive_env,
                                            {'type': 'drive', 'folder': folder,
                                                 'experiment': experiment, 'environment': drive_env}))

    return tasks_queue