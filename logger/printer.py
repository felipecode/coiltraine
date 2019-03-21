import os
import time

from .monitorer import get_status, get_episode_number, get_number_episodes_completed
from configs import g_conf, merge_with_yaml
from configs.coil_global import get_names
from coilutils.general import sort_nicely, get_latest_path, static_vars


"""
COLOR CODINGS, USED FOR PRINTING ON THE TERMINAL.
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


def print_train_summary(summary):
    if summary == '':
        return
    print('        SUMMARY:')
    print('            Iteration: ', BLUE + str(summary['Iteration']) + END)
    print('            Images/s: ', BOLD + str(summary['Images/s']) + END)
    print('            Loss: ', UNDERLINE + str(summary['Loss']) + END)
    print('            Best Loss: ', LIGHT_GREEN + UNDERLINE + str(summary['BestLoss']) + END)
    print('            Best Loss Iteration: ',
          BLUE + UNDERLINE + str(summary['BestLossIteration']) + END)
    # print ('            Best Error: ',UNDERLINE + str(summary['BestError']) + END)
    print('            Outputs: ', UNDERLINE + str(summary['Output']) + END)
    print('            Ground Truth: ', UNDERLINE + str(summary['GroundTruth']) + END)
    print('            Error: ', UNDERLINE + str(summary['Error']) + END)


def print_validation_summary(current, latest, verbose):
    if current == '':
        return

    print('        CHECKPOINT: ', DARK_BLUE + str(current['Checkpoint']) + END)

    print('        CURRENT: ')
    print('            Iteration: ', BLUE + str(current['Iteration']) + END)
    if verbose:
        print('            Mean Error: ', UNDERLINE + str(current['MeanError']) + END)
        print('            Loss: ', UNDERLINE + str(current['Loss']) + END)
        print('            Outputs: ', UNDERLINE + str(current['Output']) + END)
        print('            Ground Truth: ', UNDERLINE + str(current['GroundTruth']) + END)
        print('            Error: ', UNDERLINE + str(current['Error']) + END)

    if latest == '':
        return

    print('        LATEST: ')
    print('            Loss: ', UNDERLINE + str(latest['Loss']) + END)
    print('            Best MSE: ', LIGHT_GREEN + UNDERLINE + str(latest['BestMSE']) + END)
    print('            Best MSE Checkpoint: ',
          BLUE + UNDERLINE + str(latest['BestMSECheckpoint']) + END)
    print('            Error: ', UNDERLINE + str(latest['Error']) + END)
    print('            Best Error: ', LIGHT_GREEN + UNDERLINE + str(latest['BestError']) + END)
    print('            Best Error Checkpoint: ',
          BLUE + UNDERLINE + str(latest['BestErrorCheckpoint']) + END)


@static_vars(previous_checkpoint=g_conf.TEST_SCHEDULE[0],
             previous_checkpoint_number=None,
             previous_checkpoint_time=0)
def print_drive_summary(path,  checkpoint ):
    print('        CHECKPOINT: ', DARK_BLUE + str(checkpoint) + END)

    # Check if there is already files to check

    if os.path.exists(os.path.join(path, 'summary.csv')):
        print('        CURRENT: ')
        print('            Episode: ', BLUE + str(get_episode_number(path)) + END, ' Time: ',
              time.time() - print_drive_summary.previous_checkpoint_time)
        print('            Completed: ',
              GREEN + UNDERLINE + str(get_number_episodes_completed(path)) + END)

    if print_drive_summary.previous_checkpoint != checkpoint:
        print_drive_summary.previous_checkpoint = checkpoint

    if get_episode_number(path) != print_drive_summary.previous_checkpoint_number:
        print_drive_summary.previous_checkpoint_number = get_episode_number(path)
        print_drive_summary.previous_checkpoint_time = time.time()

    if checkpoint == g_conf.TEST_SCHEDULE[0]:
        return


def plot_folder_summaries(exp_batch, train, validation_datasets, drive_environments, verbose=False):
    """
        Main plotting function for the folder mode.
    Args:
        exp_batch: The exp batch (folder) being plotted on the screen.
        train: If train process is being printed
        validation_datasets: The validation datasets being computed
        drive_environments: The driving environments/ Benchmarks
        verbose:

    Returns:
        None

    """

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

    names_list = get_names(exp_batch)
    sorted_keys = sorted(range(len(names_list)),
                         key=lambda k: names_list[experiments_list[k] + '.yaml'])

    for key in sorted_keys:
        experiment = experiments_list[key]
        generated_name = names_list[experiment + '.yaml']

        if experiment == '':
            raise ValueError("Empty Experiment on List")

        g_conf.immutable(False)

        merge_with_yaml(os.path.join('configs', exp_batch, experiment + '.yaml'))

        print(BOLD + experiment + ' : ' + generated_name + END)

        for process in process_names:
            try:
                output = get_status(exp_batch, experiment, process)
            except:  # TODO: bad design. But the printing should not stop for any error on reading
                import traceback
                traceback.print_exc()

            status = output[0]
            summary = output[1]
            print('    ', process)

            if status == 'Not Started':

                print('       STATUS: ', BOLD + status + END)

            elif status == 'Loading':

                print('        STATUS: ', YELLOW + status + END, ' - ',  YELLOW + summary + END)

            elif status == 'Iterating':

                print('        STATUS: ', YELLOW + status + END)

            elif status == 'Finished':

                print('        STATUS: ', GREEN + status + END)

            elif status == 'Error':

                print('        STATUS: ', RED + status + END, ' - ',  RED + summary + END)

            if status == 'Iterating':
                if 'train' in process:
                    print_train_summary(summary[status])
                if 'validation' in process:
                    if summary[1] != '':  # If it has no summary we don't plot
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
                        control_filename = 'control_output_auto'
                    else:
                        control_filename = 'control_output'

                    path = exp_batch + '_' + experiment + '_' + str(checkpoint) \
                           + '_' + process.split('_')[0] + '_' + control_filename \
                           + '_' + process.split('_')[1] + '_' + process.split('_')[2]

                    print_drive_summary(get_latest_path(path), checkpoint)


def print_folder_process_names(exp_batch):
    experiments_list = os.listdir(os.path.join('configs', exp_batch))
    sort_nicely(experiments_list)

    for experiment in experiments_list:
        if '.yaml' in experiment:
            g_conf.immutable(False)

            merge_with_yaml(os.path.join('configs', exp_batch, experiment))

            print(experiment.split('.')[-2] + ': ' + g_conf.EXPERIMENT_GENERATED_NAME)
