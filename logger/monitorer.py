import os
import re

from logger import json_formatter
from configs import g_conf
from utils.general import sort_nicely


from .carla_metrics_parser import get_averaged_metrics
from visualization.data_reading import read_summary_csv

# Check the log and also put it to tensorboard



def get_current_iteration(exp):
    """

    Args:
        exp:

    Returns:
        The number of iterations this experiments has already run in this mode.
        ( Depends on validation etc...

    """
    # TODO:

    pass


#### Get things from CARLA benchmark directly to plot as logs #####
def get_episode_number(benchmark_log_name):
    """ Get the current episode"""
    control_dict = read_summary_csv(os.path.join(benchmark_log_name, 'summary.csv'))
    if control_dict is None:
        return None
    return len(control_dict['result'])


def get_number_episodes_completed(benchmark_log_name):
    """ Get the number of episodes that where completed"""
    control_dict = read_summary_csv(os.path.join(benchmark_log_name, 'summary.csv'))
    if control_dict is None:
        return None
    return sum(control_dict['result'])




def get_latest_output(data):

    # Find the one that has an iteration .........
    for i in range(1, len(data)):
        if 'Iterating' in data[-i] and ('Iteration' in data[-i]['Iterating'] or
                                            'Checkpoint' in data[-i]['Iterating']) and \
                                       'Summary' not in data[-i]:

            return data[-i]



def get_summary(data):


    # IT HAS TO BE ITERATING  ! ! !  ! ! !
    for i in range(1, len(data)):
        # Find the summary log in the logging file
        if 'Iterating' in data[-i]:  # Test if it is an iterating log
            if 'Summary' in data[-i]['Iterating']:
                return data[-i] # found the summary.
    else:  # NO SUMMARY YET COMPUTED
        return ''



def get_latest_checkpoint():


    # The path for log

    csv_file_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                 g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_csv')

    csv_files = os.listdir(csv_file_path)

    if len (csv_files) == 0:
        return None

    sort_nicely(csv_files)

    #data = json_formatter.readJSONlog(open(log_file_path, 'r'))

    return int(re.findall('\d+', csv_files[-1])[0])


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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ['Error', "Couldn't read the json"]


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
        return ['Error', ' ']


    raise ValueError(" No valid status found")



