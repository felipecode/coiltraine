import re
import os
import smtplib
import numpy as np

from email.mime.text import MIMEText
from PIL import Image



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# TODO: there should be a more natural way to do that
def command_number_to_index(command_vector):

    return command_vector-2


def plot_test_image(image, name):

    image_to_plot = Image.fromarray(image)
    image_to_plot.save(name)

def create_log_folder(exp_batch_name):
    """
        Only the train creates the path. The validation should wait for the training anyway,
        so there is no need to create any path for the logs. That avoids race conditions.
    Returns:

    """
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))


def create_exp_path(exp_batch_name, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(os.path.join(root_path, exp_batch_name, experiment_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, experiment_name))

def get_latest_path(path):
    """ Considering a certain path for experiments, get the latest one."""
    import glob
    print (" PATHS ", os.path.join('_benchmarks_results', path+'*'))

    files_list = glob.glob(os.path.join('_benchmarks_results', path+'*'))
    sort_nicely(files_list)

    return files_list[-1]



def send_email(address, message):
    msg = MIMEText(message)

    msg['Subject'] = 'The experiment is finished '
    msg['From'] = address
    msg['To'] = address


    s = smtplib.SMTP('localhost')
    s.sendmail(address, [address], msg.as_string())
    s.quit()

def compute_average_std(dic_list, weathers, number_of_tasks=1):

    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion'

    ]

    infraction_metrics = [
        'collision_pedestrians',
        'collision_vehicles',
        'collision_other',
        'intersection_offroad',
        'intersection_otherlane'

    ]
    weather_name_dict = {1: 'Clear Noon', 3: 'After Rain Noon',
                         6: 'Heavy Rain Noon', 8: 'Clear Sunset',
                         4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}

    number_of_episodes = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1])

    # The average results between the dictionaries.
    average_results_matrix = {}

    for metric_name in (metrics_to_average+infraction_metrics):
        average_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})

    count_dic_pos = 0
    for metrics_summary in dic_list:


        for metric in metrics_to_average:


            values = metrics_summary[metric]
            #print values

            metric_sum_values = np.zeros(number_of_episodes)
            for weather, tasks in values.items():
                if float(weather) in set(weathers):
                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):

                        if t == []:
                            print('    Metric Not Computed')
                        else:
                            metric_sum_values[count] += (float(sum(t)) / float(len(t))) * 1.0 / float(
                                len(weathers))

                        count += 1

            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]



        for metric in infraction_metrics:
            values_driven = metrics_summary['driven_kilometers']
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)
            summed_driven_kilometers = np.zeros(number_of_episodes)


            # print (zip(values.items(), values_driven.items()))
            for items_metric, items_driven in zip(values.items(), values_driven.items()):
                weather = items_metric[0]
                tasks = items_metric[1]
                tasks_driven = items_driven[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t, t_driven in zip(tasks, tasks_driven):
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))
                            summed_driven_kilometers[count] += t_driven

                        count += 1


            for i in range(len(metric_sum_values)):
                if metric_sum_values[i] == 0:
                    average_results_matrix[metric][i][count_dic_pos] = summed_driven_kilometers[i]
                else:
                    average_results_matrix[metric][i][count_dic_pos] = summed_driven_kilometers[i] \
                                                                       / metric_sum_values[i]



        count_dic_pos += 1

    # TODO: there is likely to be an issue on this part, both are hardcoded
    # TODO: This is just working when there is one weather and one task

    print (metrics_summary)
    print (metrics_summary['average_speed'])
    average_speed_task =  sum(metrics_summary['average_speed'][str(float(list(weathers)[0]))])


    average_results_matrix.update({'driven_kilometers': np.array([summed_driven_kilometers[0]])})
    average_results_matrix.update({'average_speed':np.array([average_speed_task])})
    print (average_results_matrix)


    for metric, vectors in average_results_matrix.items():



        for i in range(len(vectors)):
            average_results_matrix[metric][i] = np.mean(average_results_matrix[metric][i])





    return average_results_matrix