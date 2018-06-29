import os
import time

from .monitorer import get_status, get_episode_number, get_number_episodes_completed
from configs import g_conf, merge_with_yaml
from utils.general import sort_nicely, get_latest_path, static_vars
from visualization.data_reading import read_control_csv

"""
COLOR CODINGS
"""
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
ITALIC = '\033[3m'
RED = '\033[91m'
LIGHT_GREEN = '\033[32m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
DARK_BLUE = '\033[94m'
BLUE = '\033[94m'
END = '\033[0m'

def format_name(experiment_name):

    # We start by spliting it by parts

    specs = experiment_name.split('_')

    format_string = ''

    format_string = ' Dataset:', specs[0], \
                    '## Type of Network:', specs[1], \
                    '## Network Size:', specs[2], \
                    '## Regularization:', specs[3], \
                    '## Temporal:', specs[4], \
                    '## Output:', specs[5], \


    return format_string






def print_train_summary(summary):

    if summary == '':
        return
    print ('        SUMMARY:')
    print ('            Iteration: ', BLUE + str(summary['Iteration']) + END)
    print ('            Images/s: ', BOLD + str(summary['Images/s']) + END)
    print ('            Loss: ', UNDERLINE + str(summary['Loss']) + END)
    print ('            Best Loss: ', LIGHT_GREEN + UNDERLINE + str(summary['BestLoss']) + END)
    print ('            Best Loss Iteration: ', BLUE + UNDERLINE + str(summary['BestLossIteration']) + END)
    #print ('            Best Error: ',UNDERLINE + str(summary['BestError']) + END)
    print ('            Outputs: ', UNDERLINE + str(summary['Output']) + END)
    print ('            Ground Truth: ', UNDERLINE + str(summary['GroundTruth']) + END)
    print ('            Error: ', UNDERLINE + str(summary['Error']) + END)



def print_validation_summary(current, latest, verbose):

    if current == '':
        return




    print ('        CHECKPOINT: ', DARK_BLUE + str(current['Checkpoint']) + END)

    print ('        CURRENT: ')
    print ('            Iteration: ', BLUE + str(current['Iteration']) + END)
    if verbose:
        print ('            Mean Error: ', UNDERLINE + str(current['MeanError']) + END)
        print ('            Loss: ', UNDERLINE + str(current['Loss']) + END)
        print ('            Outputs: ', UNDERLINE + str(current['Output']) + END)
        print ('            Ground Truth: ', UNDERLINE + str(current['GroundTruth']) + END)
        print ('            Error: ', UNDERLINE + str(current['Error']) + END)

    if latest == '':
        return

    print ('        LATEST: ')
    print ('            Loss: ', UNDERLINE + str(latest['Loss']) + END)
    print ('            Best Loss: ', LIGHT_GREEN + UNDERLINE + str(latest['BestLoss']) + END)
    print ('            Best Loss Checkpoint: ', BLUE + UNDERLINE + str(latest['BestLossCheckpoint']) + END)
    print ('            Error: ', UNDERLINE + str(latest['Error']) + END)
    print ('            Best Error: ', LIGHT_GREEN + UNDERLINE + str(latest['BestError']) + END)
    print ('            Best Error Checkpoint: ', BLUE + UNDERLINE + str(latest['BestErrorCheckpoint']) + END)


@static_vars(previous_checkpoint=g_conf.TEST_SCHEDULE[0],
             previous_checkpoint_number=None,
             previous_checkpoint_time=0)
def print_drive_summary(path, csv_filename, checkpoint, verbose):




    print ('        CHECKPOINT: ', DARK_BLUE + str(checkpoint) + END)

    # Check if there is already files to check

    if os.path.exists(os.path.join(path, 'summary.csv')):
        print ('        CURRENT: ')
        print ('            Episode: ', BLUE + str(get_episode_number(path)) + END, ' Time: ',
               time.time() - print_drive_summary.previous_checkpoint_time )
        print ('            Completed: ', GREEN + UNDERLINE + str(get_number_episodes_completed(path)) + END)

    if print_drive_summary.previous_checkpoint != checkpoint:
        print_drive_summary.previous_checkpoint = checkpoint

    if get_episode_number(path) != print_drive_summary.previous_checkpoint_number:
        print_drive_summary.previous_checkpoint_number = get_episode_number(path)
        print_drive_summary.previous_checkpoint_time = time.time()

    if checkpoint == g_conf.TEST_SCHEDULE[0]:
        return

    print_drive_summary.previous_checkpoint = g_conf.TEST_SCHEDULE[g_conf.TEST_SCHEDULE.index(checkpoint)-1]
    # TODO: we need to get the previous checkpoint

    averaged_metrics, header = read_control_csv(csv_filename)


    print ('        SUMMARY: ')
    print ('            Average Completion: ', LIGHT_GREEN + UNDERLINE +
           str(averaged_metrics[float(print_drive_summary.previous_checkpoint)]
               [header.index('episodes_fully_completed')-1]) + END)
    #print ('            Kilometers Per Infraction: ', GREEN + UNDERLINE +
    #       str(averaged_metrics[float(print_drive_summary.previous_checkpoint)]
    #           [header.index('collision_pedestrians')-1]) + END)



def plot_folder_summaries(exp_batch, train, validation_datasets, drive_environments, verbose=False):

    # TODO: if train is not running the user should be warned

    os.system('clear')
    process_names = []
    if train:
        process_names.append('train')

    for val in validation_datasets:
        process_names.append('validation' + '_' + val)


    for drive in drive_environments:
        process_names.append('drive' + '_' + drive)




    experiments_list = os.listdir(os.path.join('configs', exp_batch))

    experiments_list = [experiment.split('.')[-2] for experiment in experiments_list]

    print (experiments_list)

    for experiment in experiments_list:
        if experiment == '':
            raise ValueError("Empty Experiment on List")
        g_conf.immutable(False)

        merge_with_yaml(os.path.join('configs', exp_batch, experiment + '.yaml'))

        print (BOLD +  experiment +' : ' + g_conf.EXPERIMENT_GENERATED_NAME + END)

        for process in process_names:
            try:
                output = get_status(exp_batch, experiment, process)
            except:
                import traceback
                traceback.print_exc()

            status = output[0]
            summary = output[1]
            print ('    ', process)

            if status == 'Not Started':

                print ('       STATUS: ', BOLD + status + END)

            elif status == 'Iterating' or status == 'Loading':

                print('        STATUS: ', YELLOW + status + END)

            elif status == 'Finished':

                print('        STATUS: ', GREEN + status + END)

            elif status == 'Error':

                print('        STATUS: ', RED + status + END)


            if status == 'Iterating':
                if 'train' in process:
                    print_train_summary(summary[status])
                if 'validation' in process:
                    if summary[1] != '':   # If it has no summary we don't plot
                        print_validation_summary(summary[0][status], summary[1][status]['Summary'],
                                                 verbose)
                    else:
                        print_validation_summary(summary[0][status], '',
                                                 verbose)
                if 'drive' in process:
                    if 'Agent' not in summary[status]:
                        continue
                    checkpoint = summary[status]['Checkpoint']  # Get the sta
                    # This contain the results from completed iterations
                    if g_conf.USE_ORACLE:
                        control_filename = 'control_output_auto.csv'
                    else:
                        control_filename = 'control_output.csv'

                    csv_file_path = os.path.join('_logs', exp_batch, experiment,
                                                 process + '_csv', control_filename)
                    path = exp_batch + '_' + experiment + '_' + str(checkpoint) \
                           + '_' + process.split('_')[0] + '_' + control_filename[:-4] \
                           + '_' + process.split('_')[1] + '_' + process.split('_')[2]
                    print_drive_summary(get_latest_path(path), csv_file_path, checkpoint, verbose)



def print_folder_process_names(exp_batch):


    experiments_list = os.listdir(os.path.join('configs', exp_batch))
    sort_nicely(experiments_list)


    for experiment in  experiments_list:
        if '.yaml' in experiment:
            g_conf.immutable(False)

            merge_with_yaml(os.path.join('configs', exp_batch, experiment))


            print (experiment.split('.')[-2] + ': ' + g_conf.EXPERIMENT_GENERATED_NAME)



