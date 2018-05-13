
from logger import monitorer


def get_free_gpus(alocated_gpus, executing_processes):
    free_gpus = alocated_gpus

    for process_specs in executing_processes:

        if monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                process_specs['type']) == "Error" or \
                monitorer.get_status(process_specs['folder'], process_specs['experiment'],
                                     process_specs['type']) == "Finished":
            free_gpus.append([process_specs['gpu'], process_specs['gpu']])

    return free_gpus, sum(sum(free_gpus)) / 2.0


def pop_half_gpu(free_gpus):
    for gpu in free_gpus:
        if len(gpu) > 1:
            free_gpus.erase(free_gpus.index(gpu))
            return free_gpus, sum(sum(free_gpus)) / 2.0, gpu[0]
        else:
            free_gpus[free_gpus.index(gpu)] = [free_gpus[free_gpus.index(gpu)][0]]
            return free_gpus, sum(sum(free_gpus)) / 2.0, gpu[0]


def pop_one_gpu(free_gpus):
    for gpu in free_gpus:
        if len(gpu) > 1:
            free_gpus.erase(free_gpus.index(gpu))
            return free_gpus, sum(sum(free_gpus)) / 2.0, gpu[0]


def mount_experiment_heap(folder, experiments_list, validation_datasets):


    tasks_queue = []
    for experiment in experiments_list:

        # Train is always priority. # TODO: some system to check priority depending on iterations
        if monitorer.get_status(folder, experiment, 'train') == "Not Started" or \
                monitorer.get_status(folder, experiment, 'train') == "Error":

            heapq.heappush(tasks_queue, (0, {'type': 'train', 'folder': folder,
                                             'experiment': experiment}))

        for val_data in validation_datasets:
            if monitorer.get_status(folder, experiment, 'validation_' + val_data) == "Not Started" or \
                    monitorer.get_status(folder, experiment, 'validation_'+ val_data) == "Error":

                heapq.heappush(tasks_queue, (2, {'type': 'validation', 'folder': folder,
                                                 'experiment': experiment, 'dataset': val_data}))

        for drive_env in drive_environments:
            if monitorer.get_status(folder, experiment, 'drive_' + drive_env) == "Not Started" or \
                    monitorer.get_status(folder, experiment, 'drive_' + drive_env) == "Error":

                heapq.heappush(tasks_queue, (1, {'type': 'drive', 'folder': folder,
                                                 'experiment': experiment, 'environment': drive_env}))