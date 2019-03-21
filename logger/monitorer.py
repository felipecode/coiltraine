import os
import re
import numpy as np

from logger import json_formatter
from configs import g_conf
from coilutils.general import sort_nicely


from .carla_metrics_parser import get_averaged_metrics
from plotter.data_reading import read_summary_csv

# Check the log and also put it to tensorboard


#### Get things from CARLA benchmark directly to plot as logs #####
def get_episode_number(benchmark_log_name):
    """ Get the current episode"""

    control_dict = read_summary_csv(os.path.join(benchmark_log_name, 'summary.csv'))
    if control_dict is None:
        return None

    # TODO: weird bug on data reading that eats the last letter
    try:
        return len(control_dict['result'])
    except:
        return len(control_dict['resul'])


def get_number_episodes_completed(benchmark_log_name):
    """ Get the number of episodes that where completed"""
    control_dict = read_summary_csv(os.path.join(benchmark_log_name, 'summary.csv'))
    if control_dict is None:
        return None
    try:
        return sum(control_dict['result'])
    except:
        return sum(control_dict['resul'])




def get_latest_output(data):

    # Find the one that has an iteration .........
    for i in range(1, len(data)):
        if 'Iterating' in data[-i] and ('Iteration' in data[-i]['Iterating'] or
                                            'Checkpoint' in data[-i]['Iterating']) and \
                                       'Summary' not in data[-i]['Iterating']:

            return data[-i]



def get_summary(data):

    for i in range(1, len(data)):
        # Find the summary log in the logging file
        if 'Iterating' in data[-i]:  # Test if it is an iterating log
            if 'Summary' in data[-i]['Iterating']:
                return data[-i] # found the summary.
    else:  # NO SUMMARY YET COMPUTED
        return ''

def get_error_summary(data):

    if 'Error' in data[-1]:
        return data[-1]['Error']['Message']  # found the summary.
    else:
        return ''

def get_latest_checkpoint_validation():
    csv_file_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                 g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_csv')

    csv_files = os.listdir(csv_file_path)

    if len(csv_files) == 0:
        return None

    sort_nicely(csv_files)

    csv_file_numbers = set([float(re.findall('\d+', file)[0]) for file in csv_files])

    not_evaluated_logs = list(set(g_conf.TEST_SCHEDULE).difference(csv_file_numbers))

    not_evaluated_logs = sorted(not_evaluated_logs, reverse=False)

    if len(not_evaluated_logs) == 0:  # Just in case that is the last one
        return g_conf.TEST_SCHEDULE[-1]

    if g_conf.TEST_SCHEDULE.index(not_evaluated_logs[0]) == 0:
        return None


    return g_conf.TEST_SCHEDULE[g_conf.TEST_SCHEDULE.index(not_evaluated_logs[0])-1]

def get_latest_checkpoint_drive(control_filename):

    # add csv to the file name

    control_filename = control_filename + '.csv'

    csv_file_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                 g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_csv')

    print (" TESTED PATHJ ", csv_file_path)
    if not os.path.exists(os.path.join(csv_file_path, control_filename)):
        return None


    f = open(os.path.join(csv_file_path, control_filename), "r")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(open(os.path.join(csv_file_path, control_filename), "rb"),
                             delimiter=",", skiprows=1)

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    return float(data_matrix[-1][header.index('step')])


def get_latest_checkpoint(filename):

    if 'validation' in g_conf.PROCESS_NAME:
        return get_latest_checkpoint_validation()
    elif 'drive' in g_conf.PROCESS_NAME:
        return get_latest_checkpoint_drive(filename)
    else:
        raise ValueError("The process name is not producing checkpoints")


def get_status(exp_batch, experiment, process_name):

    """

    Args:
        exp_batch: The experiment batch name
        experiment: The experiment name.

    Returns:
        A status that is a vector with two fields
        [ Status, Summary]

        Status is from the set = (Does Not Exist, Not Started, Loading, Iterating, Error, Finished)
        Summary constains a string message summarizing what is happening on this phase.

        * Not existent
        * To Run
        * Running
            * Loading - sumarize position ( Briefly)
            * Iterating  - summarize
        * Error ( Show the error)
        * Finished ( Summarize)

    """

    # Configuration file path
    config_file_path = os.path.join('configs', exp_batch, experiment + '.yaml')

    # The path for log
    log_file_path = os.path.join('_logs', exp_batch, experiment, process_name)

    # First we check if the experiment exist

    if not os.path.exists(config_file_path):

        return ['Does Not Exist', '']

    # The experiment exist ! However, check if the log file exist.

    if not os.path.exists(log_file_path):

        return ['Not Started', '']

    # Read the full json file.
    try:
        data = json_formatter.readJSONlog(open(log_file_path, 'r'))
    except ValueError:
        return ['Loading', 'Writing Logs']

    except Exception:
        import traceback
        traceback.print_exc()
        return ['Error', "Couldn't read the json"]

    if len(data) == 0:
        return ['Not Started', '']

    # Now check if the latest data is loading
    if 'Loading' in data[-1]:
        return ['Loading', '']

    # Then we check if finished or is going on

    if 'Iterating' in data[-1]:

        if 'validation' in process_name:
            return ['Iterating', [get_latest_output(data), get_summary(data)]]
        elif 'train' in process_name:
            return ['Iterating', get_latest_output(data)]
        elif 'drive' in process_name:
            return ['Iterating', get_latest_output(data)]  # We in theory just return
        else:
            raise ValueError("Not Valid Experiment name")

    # TODO: there is the posibility of some race conditions on not having error as last

    if 'Finished' in data[-1]:
        return ['Finished', ' ']
    if 'Error' in data[-1]:
        return ['Error', get_error_summary(data)]


    raise ValueError(" No valid status found")



