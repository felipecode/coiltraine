import os
import re

from logger import json_formatter
from configs import g_conf
from utils.general import sort_nicely
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


def get_latest_checkpoint():


    # The path for log

    csv_file_path = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                 g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_csv')

    csv_files = os.listdir(csv_file_path)

    if csv_files == []:
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

    print(config_file_path, log_file_path)
    # First we check if the experiment exist

    if not os.path.exists(config_file_path):

        return ['Does Not Exist', '']


    # The experiment exist ! However, check if the log file exist.

    if not os.path.exists(log_file_path):

        return ['Not Started', '']

    # Read the full json file.
    data = json_formatter.readJSONlog(open(log_file_path, 'r'))

    print (data)

    # Now check if the latest data is loading
    if 'Loading' in data[-1]:
        return ['Loading', '']

    # Then we check if finished or is going on

    if 'Model' in data[-1] or 'Reading' in data[-1] or 'Loss' in data[-1]:

        if list(data[-1].values())[0]['Iteration'] >= g_conf.param.MISC.NUMBER_OF_ITERATIONS:
            return ['Finished', ' ']
        else:
            return ['Iterating', ' ']

    if 'Error' in data[-1]:
        return ['Error', ' ']


    return None

def export_results(benchmark_results_folder):

    """
        Reads some data and export the csv file as a result file somewhere so it
        can be read by the visualization module.
    Returns:

    """

    pass

