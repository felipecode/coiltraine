import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn;

seaborn.set_style('whitegrid')
import copy
from matplotlib import rcParams


def sldist(c1, c2): return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def split_episodes(meas_file, exp_id=-1):
    """
        The idea is to split the positions assumed by the ego vehicle on every episode.
    Args:
        meas_file: the file containing the measurements.

    Returns:
        a matrix where each vector is a vector of points from the episodes.
        a vector with the travelled distance on each episode

    """
    f = open(meas_file, "rU")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-1]
    f.close()

    # print (header_details)

    details_matrix = np.loadtxt(open(meas_file, "rb"), delimiter=",", skiprows=1)

    #
    # print (details_matrix)
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                    details_matrix[0, header_details.index('pos_y')]]

    #
    exp_id_vec = details_matrix[:, header_details.index('exp_id')]
    if exp_id != -1:
        exp_id_vec = np.where(exp_id_vec == exp_id)
        print(" Exp id ", len(details_matrix[exp_id_vec[0], :]))
        print(exp_id_vec[0][0:50])
        details_matrix = details_matrix[exp_id_vec[0], :]

    episode_positions_matrix = []
    positions_vector = []
    travelled_distances = []
    travel_this_episode = 0
    previous_start_point = details_matrix[0, header_details.index('start_point')]
    previous_end_point = details_matrix[0, header_details.index('end_point')]
    previous_repetition = details_matrix[0, header_details.index('rep')]

    expected_end_points = []
    expected_start_points = []

    episodes_throttles = []
    throttles = []
    throttles.append(details_matrix[0, header_details.index('throttle')])

    episodes_offroad = []
    offroads = []
    offroads.append(details_matrix[0, header_details.index('intersection_offroad')])

    episodes_otherlane = []
    otherlanes = []
    otherlanes.append(details_matrix[0, header_details.index('intersection_otherlane')])

    for i in range(1, len(details_matrix)):

        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]

        start_point = details_matrix[i, header_details.index('start_point')]
        end_point = details_matrix[i, header_details.index('end_point')]
        repetition = details_matrix[i, header_details.index('rep')]

        throttle = details_matrix[i, header_details.index('throttle')]
        intersection_offroad = details_matrix[i, header_details.index('intersection_offroad')]
        intersection_otherlane = details_matrix[i, header_details.index('intersection_otherlane')]

        positions_vector.append(point)

        if (previous_start_point != start_point or end_point != previous_end_point) or \
                repetition != previous_repetition or i == len(
            details_matrix) - 1:  # "or i == len(details_matrix)-1 ": To conclude the last episode

            travelled_distances.append(travel_this_episode)
            travel_this_episode = 0
            positions_vector.pop()
            episode_positions_matrix.append(positions_vector)
            positions_vector = []

            episodes_throttles.append(throttles)
            throttles = []
            throttles.append(throttle)

            episodes_offroad.append(offroads)
            offroads = []
            offroads.append(intersection_offroad)

            # This is for saving the intersection_otherlane for each episode
            episodes_otherlane.append(otherlanes)
            otherlanes = []
            otherlanes.append(intersection_otherlane)

            # This is for saving the expected start and end points for each episode
            expected_start_points.append(int(previous_start_point))
            expected_end_points.append(int(previous_end_point))

        else:
            throttles.append(throttle)
            offroads.append(intersection_offroad)
            otherlanes.append(intersection_otherlane)

        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point

        previous_start_point = start_point
        previous_end_point = end_point
        previous_repetition = repetition

    return episode_positions_matrix, travelled_distances, expected_start_points, expected_end_points, episodes_throttles, episodes_offroad, episodes_otherlane


def get_results(summary_file):
    """
        To get the results of all eposides.
    Args:
        summary_file: the file containing the summary results.

    Returns:
        the results of all eposides.

    """
    f = open(summary_file, "rU")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-1]
    f.close()

    details_matrix = np.loadtxt(open(summary_file, "rb"), delimiter=",", skiprows=1)

    results = []
    for i in range(len(details_matrix)):
        result = details_matrix[i, header_details.index('result')]
        results.append(int(result))
    return results


def analyze_stopping_problem(batch_folder, exp_id):
    # We build the measurement file used for the benchmarks.
    meas_file = os.path.join(batch_folder,
                             'measurements.csv')
    # We build the summary file used for the benchmarks.
    summary_file = os.path.join(batch_folder,
                                'summary.csv')

    # image_location = map.__file__[:-7]

    # Split the measurements for each of the episodes
    episodes_positions, travelled_distances, \
    expected_start_points, expected_end_points, \
    episodes_throttles, episodes_offroad, episodes_otherlane = split_episodes(meas_file, exp_id)

    episode_results = get_results(summary_file)

    print("Number of episodes ", len(episodes_positions))
    print("Number of episodes ", len(episode_results))

    positions = list(range(len(episodes_positions)))

    fail_count = 0
    fail_on_stopping = 0
    episodes_fail_on_stopping = []
    for i in positions:
        if episode_results[i] == 0:
            fail_count += 1
            offroads = episodes_offroad[i]
            otherlanes = episodes_otherlane[i]
            throttles = episodes_throttles[i]
            if all(num == 0.0 for num in offroads[-50:]) and all(
                    num == 0.0 for num in otherlanes[-50:]):
                if all(num <= 0.1 for num in throttles[-30:]):
                    print(throttles[-50:])
                    episodes_fail_on_stopping.append(i)
                    fail_on_stopping += 1

    print('fail count', fail_count)
    print('fail on stopping problem count', fail_on_stopping)
    print('episodes fail on stopping problem', episodes_fail_on_stopping)
    percentage = fail_on_stopping / fail_count
    print('Percentage', percentage)
    return percentage


def compute_average_std_separatetasks(dic_list, weathers, number_of_tasks=1, number_of_reps=1):
    """
    There are two types of outputs, these come packed in a dictionary

    Success metrics, these are averaged between weathers, is basically the percentage of completion for a
    single task.

    Infractions, these are summed and divided by the total number of driven kilometers


    For this you have the concept of averaging all the weathers from the experiment suite.

    """

    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion',
        'percentage_off_road',
        'percentage_green_lights'

    ]

    metrics_to_sum = [
        'end_pedestrian_collision',
        'end_vehicle_collision',
        'end_other_collision'
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
                         4: 'Cloudy After Rain', 10: ' Rainy after rain',
                         14: 'Soft Rain Sunset'}

    number_of_experiments = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1])
    number_of_episodes = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1][0])

    # The average results between the dictionaries.
    average_results_matrix = {}
    std_results_matrix = {}

    for metric_name in (metrics_to_average + infraction_metrics + metrics_to_sum):
        average_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})
        std_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})

    count_dic_pos = 0
    for metrics_summary in dic_list:

        for metric in metrics_to_average:
            print ("METRIC ", metric)
            values = metrics_summary[metric]

            metric_sum_values = np.zeros((number_of_experiments, number_of_reps))
            for weather, tasks in values.items():

                print (tasks)
                if float(weather) in set(weathers):
                    print("W", weather)
                    count = 0
                    for t in tasks:
                        print (t)
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if len(t) == 0:
                            print('    Metric Not Computed')
                        else:
                            for r in range(number_of_reps):
                                print("Rep ", r)
                                metric_sum_values[count][r] += (float(sum(t[r:-1:number_of_reps])))
                                print("episodes ", range(r, len(t), number_of_reps))
                                print(metric_sum_values[count][r])
                        count += 1

            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = 0
                # We take the average of each rep and them average again
                for r in range(number_of_reps):
                    average_results_matrix[metric][i][count_dic_pos] += metric_sum_values[i][r] / \
                                                                        (number_of_episodes * len(
                                                                            weathers))

                std_results_matrix[metric][i][count_dic_pos] = 0
                for r in range(number_of_reps):
                    std_results_matrix[metric][i][count_dic_pos] += \
                        math.fabs(average_results_matrix[metric][i][count_dic_pos] -
                                  metric_sum_values[i][r] / ((number_of_episodes / number_of_reps) *
                                                             len(weathers))) / number_of_reps

        # For the metrics we sum over all the weathers here, this is to better subdivide the driving
        #  envs. The infraction metrics are divided by the number of kilometers in the end
        for metric in infraction_metrics:
            values_driven = metrics_summary['driven_kilometers']
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_experiments)
            summed_driven_kilometers = np.zeros(number_of_experiments)

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

            # On this part average results matrix basically assume the number of infractions.
            for i in range(len(metric_sum_values)):
                if metric_sum_values[i] == 0:
                    average_results_matrix[metric][i][count_dic_pos] = 1
                else:
                    average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]

        for metric in metrics_to_sum:
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_experiments)

            # print (zip(values.items(), values_driven.items()))
            for items_metric in values.items():
                weather = items_metric[0]
                tasks = items_metric[1]

                if float(weather) in set(weathers):

                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):
                        if t == []:
                            print('Metric Not Computed')
                        else:

                            metric_sum_values[count] += float(sum(t))

                        count += 1

            # On this part average results matrix basically assume the number of infractions.
            print(" metric sum ", metric_sum_values)
            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i] / \
                                                                   (number_of_episodes * len(
                                                                       weathers))

        count_dic_pos += 1

    average_speed_task = sum(metrics_summary['average_speed'][str(float(list(weathers)[0]))])

    average_results_matrix.update({'driven_kilometers': np.array(summed_driven_kilometers)})

    average_results_matrix.update({'average_speed': np.array([average_speed_task])})
    print(average_results_matrix)

    return average_results_matrix, std_results_matrix


def compute_our_res_dict(experiment_name):
    root_path = '/home/felipe/CoIL_TRI/_logs'
    # CVPR SAMPLING GOES INSIDE

    # Add several experiments parts
    experiments = [
        'cvprfinal_valstop_seed1/' + experiment_name,
        'cvprfinal_valstop_seed2/' + experiment_name,
        'cvprfinal_valstop_seed3/' + experiment_name,
        'cvprfinal_valstop_seed4/' + experiment_name,
        'cvprfinal_valstop_seed5/' + experiment_name]



    results_dict = {}

    tasks = ['empty', 'normal', 'cluttered']
    scenarios = ['Long3Training_Town01', 'Long3NewWeather_Town01',
                 'Long3NewTown_Town02', 'Long3NewWeatherTown_Town02']

    out_tasks = {'empty': 'Empty', 'normal': 'Normal', 'cluttered': 'Cluttered'}

    out_scenarios = {'Long3Training_Town01': 'Training', 'Long3NewWeather_Town01': 'NewWeather',
                     'Long3NewTown_Town02': 'NewTown',
                     'Long3NewWeatherTown_Town02': 'NewWeatherTown'}

    check_point = 0

    for t in tasks:
        print("Task ", t)
        scenarios_dict = {}
        for s in scenarios:
            # HERe we do a plot
            print("  Scenario ", s)
            metric_dict = {'episodes_fully_completed': [],
                           'end_pedestrian_collision': [],
                           'end_vehicle_collision': [],
                           'end_other_collision': [],
                           'timeout': [],
                           'stopping': []
                           }

            for exp in experiments:
                print("      exp ", exp)
                f = open(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                      'control_output_' + t + '.csv'), "rU")
                header_details = f.readline()

                header_details = header_details.split(',')
                # header_details[-1] = header_details[-1][:-1]
                f.close()

                # 'episodes_fully_completed': [],
                # 'end_pedestrian_collision':[],
                # 'end_vehicle_collision':[],
                # 'end_other_collision':[],
                # 'timeout':[],
                # 'stopping': []

                success = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                                  'control_output_' + t + '.csv'),
                                     delimiter=",",
                                     skiprows=1,
                                     usecols=([header_details.index('episodes_fully_completed')]))

                print("        success", success)

                if success.shape != (0,)  or  success.shape != ():
                    try:
                        success = success[0]
                        # std = std[0]
                    except IndexError:
                        pass

                    metric_dict['episodes_fully_completed'].append(success)

                end_ped = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                                  'control_output_' + t + '.csv'),
                                     delimiter=",",
                                     skiprows=1,
                                     usecols=(
                                         [header_details.index('end_pedestrian_collision')]))

                print("        end ped", end_ped)

                if end_ped.shape != (0,)  or  end_ped.shape != ():
                    try:
                        end_ped = end_ped[0]
                        # std = std[0]
                    except IndexError:
                        pass

                    metric_dict['end_pedestrian_collision'].append(end_ped)

                end_car = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                                  'control_output_' + t + '.csv'),
                                     delimiter=",",
                                     skiprows=1,
                                     usecols=(
                                         [header_details.index('end_vehicle_collision')]))

                print("        end car", end_car)

                if end_car.shape != (0,)  or  end_car.shape != ():
                    try:
                        end_car = end_car[0]
                        # std = std[0]
                    except IndexError:
                        pass

                    metric_dict['end_vehicle_collision'].append(end_car)

                end_other = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                                    'control_output_' + t + '.csv'),
                                       delimiter=",",
                                       skiprows=1,
                                       usecols=(
                                           [header_details.index('end_other_collision')]))


                if end_other.shape != (0,) or  end_other.shape != ():
                    try:
                        end_other = end_other[0]
                        # std = std[0]
                    except IndexError:
                        pass

                    metric_dict['end_other_collision'].append(end_other)


                print("        end other", end_other)
                print("        end other shape", end_other.shape)
                # Timeout

                metric_dict['timeout'].append(1 - (
                        success + end_car + end_other + end_ped))
                # Stopping problem
                metric_dict['stopping'].append(0.0)


            maximun_value_index = int(np.argmax(metric_dict['episodes_fully_completed']))
            print ("max value index", maximun_value_index)
            minimun_value_index = int(np.argmin(metric_dict['episodes_fully_completed']))

            maximun_metric_dict ={'episodes_fully_completed': [],
                   'end_pedestrian_collision': [],
                   'end_vehicle_collision': [],
                   'end_other_collision': [],
                   'timeout': [],
                   'stopping': []
                   }
            minimun_metric_dict ={'episodes_fully_completed': [],
                   'end_pedestrian_collision': [],
                   'end_vehicle_collision': [],
                   'end_other_collision': [],
                   'timeout': [],
                   'stopping': []
                   }
            for metric in metric_dict.keys():
                maximun_success = metric_dict[metric][maximun_value_index]
                minimun_success = metric_dict[metric][minimun_value_index]
                # maximun_std = variation_values_std[maximun_value_index]
                # minimun_std = variation_values_std[minimun_value_index]
                maximun_metric_dict[metric] = maximun_success * 100
                minimun_metric_dict[metric] = minimun_success * 100


            scenarios_dict.update({out_scenarios[s]: maximun_metric_dict})

        results_dict.update({out_tasks[t]: scenarios_dict})


    return results_dict


def add_data_for_method(compared_results, paths_dict, method_name):
    # Translates from tasks to positions
    task_pos_dict = {'Empty': 0, 'Normal': 1, 'Cluttered': 2}

    scenario_weather = {'NewTown': [1.0, 3.0, 6.0, 8.0], 'Training': [1.0, 3.0, 6.0, 8.0],
                        'NewWeather': [10.0, 14.0], 'NewWeatherTown': [10.0, 14.0]}

    for scenario, path_vecs in paths_dict.items():

        for path in path_vecs:
            print (path)
            benchmark_json_path = os.path.join(path, 'metrics.json')
            with open(benchmark_json_path, 'r') as f:
                benchmark_dict = json.loads(f.read())

            avg_train_metrics, _ = compute_average_std_separatetasks([benchmark_dict],
                                                                     scenario_weather[scenario],
                                                                     number_of_tasks=3,
                                                                     number_of_reps=1)
            print ('AVG ')
            print (avg_train_metrics)

            for task, methods in compared_results.items():
                completed = avg_train_metrics['episodes_fully_completed'][task_pos_dict[task]][0]
                compared_results[task][method_name][scenario]['episodes_fully_completed'].append(
                    completed)

                # End pedestrians
                end_ped = avg_train_metrics['end_pedestrian_collision'][task_pos_dict[task]][0]
                compared_results[task][method_name][scenario]['end_pedestrian_collision'].append(
                    end_ped)

                # End Cars
                end_cars = avg_train_metrics['end_vehicle_collision'][task_pos_dict[task]][0]
                compared_results[task][method_name][scenario]['end_vehicle_collision'].append(
                    end_cars)

                # End Others
                end_others = avg_train_metrics['end_other_collision'][task_pos_dict[task]][0]
                compared_results[task][method_name][scenario]['end_other_collision'].append(
                    end_others)

                # Timeout

                compared_results[task][method_name][scenario]['timeout'].append(1 - (
                        completed + end_cars + end_others + end_ped)
                                                                         )
                # Stopping problem
                compared_results[task][method_name][scenario]['stopping'].append(
                    analyze_stopping_problem(path, task_pos_dict[task])
                )


import json

# ***** main loop *****
if __name__ == "__main__":

    # root path for each case

    scenario_dict = {'episodes_fully_completed': [],
                     'end_pedestrian_collision': [],
                     'end_vehicle_collision': [],
                     'end_other_collision': [],
                     'timeout': [],
                     'stopping': []
                     }

    compared_results = {
        'Empty':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
        'Normal':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
        'Cluttered':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
    }

    MT_root = '/home/felipe/CoIL_TRI/_results_MT/_benchmarks_results'
    # Translate scenarios to weathers

    MT_bench_paths = {
        'Training': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town01')
                     # os.path.join(MT_root, 'test3_test_LongitudinalControl2018_Town01')
                     ],
        'NewWeather': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town01')
                       # os.path.join(MT_root, 'test3_test_LongitudinalControl2018_Town01')
                       ],
        'NewTown': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town02'),
                    os.path.join(MT_root, 'test2_LongitudinalControl2018_Town02'),
                    os.path.join(MT_root, 'test3_LongitudinalControl2018_Town02')
                    ],
        'NewWeatherTown': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town02'),
                           os.path.join(MT_root, 'test2_LongitudinalControl2018_Town02'),
                           os.path.join(MT_root, 'test3_LongitudinalControl2018_Town02')
                           ],
        }

    CAL_root = '/home/felipe/CoIL_TRI/_results_CAL/_benchmarks_results'
    CAL_bench_paths = {'Training': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town01'),
                                    os.path.join(CAL_root,
                                                 'test110_LongitudinalControl2018_Town01'),
                                    os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town01')
                                    ],
                       'NewWeather': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town01'),
                                      os.path.join(CAL_root,
                                                   'test110_LongitudinalControl2018_Town01'),
                                      os.path.join(CAL_root,
                                                   'test320_LongitudinalControl2018_Town01')
                                      ],
                       'NewTown': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town02'),
                                   os.path.join(CAL_root, 'test220_LongitudinalControl2018_Town02'),
                                   os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town02')
                                   ],
                       'NewWeatherTown': [
                           os.path.join(CAL_root, 'test_LongitudinalControl2018_Town02'),
                           os.path.join(CAL_root, 'test220_LongitudinalControl2018_Town02'),
                           os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town02')
                           ],
                       }

    IL_root = '/home/felipe/CoIL_TRI/_results_IL/_benchmarks_results'
    IL_bench_paths = {'Training': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town01'),
                                   os.path.join(IL_root, 'test2_LongitudinalControl2018_Town01')
                                   # os.path.join(IL_root, 'test110_LongitudinalControl2018_Town01'),
                                   # os.path.join(IL_root, 'test320_LongitudinalControl2018_Town01')
                                   ],
                      'NewWeather': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town01'),
                                     os.path.join(IL_root, 'test2_LongitudinalControl2018_Town01')
                                     # os.path.join(IL_root, 'test110_LongitudinalControl2018_Town01'),
                                     # os.path.join(IL_root, 'test320_LongitudinalControl2018_Town01')
                                     ],
                      'NewTown': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town02'),
                                  os.path.join(IL_root, 'test2_LongitudinalControl2018_Town02'),
                                  os.path.join(IL_root, 'test3_LongitudinalControl2018_Town02')
                                  ],
                      'NewWeatherTown': [
                          os.path.join(IL_root, 'test_LongitudinalControl2018_Town02'),
                          os.path.join(IL_root, 'test2_LongitudinalControl2018_Town02'),
                          os.path.join(IL_root, 'test3_LongitudinalControl2018_Town02')
                          ],
                      }


    our_results = compute_our_res_dict('res34-50-lowdropout-imnet')


    # merge both results
    for task in compared_results.keys():
        compared_results[task].update({'Ours':our_results[task]})

    # our_results_2 = compute_our_res_dict('res34-50-lowdropout-imnet-nospeed')

    print (compared_results)

    exit(1)

    add_data_for_method(compared_results, paths_dict=MT_bench_paths, method_name='MT')

    add_data_for_method(compared_results, paths_dict=CAL_bench_paths, method_name='CAL')

    add_data_for_method(compared_results, paths_dict=IL_bench_paths, method_name='CoIL')

    print (compared_results)

    scenarios_list = ["Training", "New Weather", "New Town", "New Weather & Town"]

    print_sd = {'Empty': 'Empty Town', 'Normal': 'Regular Traffic', 'Cluttered': 'Dense Traffic'}



    colors = ['k', 'b', 'y']

    for task, methods in compared_results.items():

        print("Task ", task)
        # The matrix with all std values
        count = 0

        for method, envs in methods.items():
            std_vec = []
            mean_vec = []

            print("  method ", method)
            for env, metrics in envs.items():
                print("    env ", env)
                for metric, runs in metrics.items():
                    print ( " $%.2f$ &" % np.mean(runs), end='')


            # count += 1



    """
    
    exp  cvprfinal_valstop_seed3/res34-50-lowdropout-imnet
        success [0.506667 0.506667 0.506667 0.506667]
        end ped [0. 0. 0. 0.]
        end car [0. 0. 0. 0.]
        end other [0.013333 0.013333 0.013333 0.013333]
      exp  cvprfinal_valstop_seed4/res34-50-lowdropout-imnet
        success [0.453333 0.453333 0.453333 0.453333]
        end ped [0. 0. 0. 0.]
        end car [0. 0. 0. 0.]
        end other [0.146667 0.146667 0.146667 0.146667]
      exp  cvprfinal_valstop_seed5/res34-50-lowdropout-imnet
        success 0.246667
        end ped 0.0
        end car 0.0
        end other 0.06
max value index 5
Traceback (most recent call last):
  File "tools/compute_average_from_benchs.py", line 710, in <module>
    our_results = compute_our_res_dict('res34-50-lowdropout-imnet')
  File "tools/compute_average_from_benchs.py", line 507, in compute_our_res_dict
    maximun_success = metric_dict[metric][maximun_value_index]


    min_mean_vec = []
    min_std_vec = []
    max_mean_vec = []
    max_std_vec = []
    print (our_results[task].items())
    for env, values in our_results[task].items():
        print (env, values)
        # Now we plot for our method
        max_mean_vec.append(our_results[task][env][0])
        max_std_vec.append(our_results[task][env][1])
        min_mean_vec.append(our_results[task][env][2])
        min_std_vec.append(our_results[task][env][3])

    print("min mean ", min_mean_vec)
    print("min std ", min_std_vec)

    ax.errorbar(range(len(scenarios_list)), min_mean_vec,
                yerr=min_std_vec, ecolor='b', color='r',
                fmt='o', capsize=2, markersize=10, label='Ours Min')

    ax.errorbar(range(len(scenarios_list)), max_mean_vec,
                yerr=max_std_vec, ecolor='r', color='g',
                fmt='o', capsize=2, markersize=10, label='Ours Max')

    min_mean_vec = []
    min_std_vec = []
    max_mean_vec = []
    max_std_vec = []
    print (our_results_2[task].items())
    for env, values in our_results_2[task].items():
        print (env, values)
        # Now we plot for our method
        max_mean_vec.append(our_results_2[task][env][0])
        max_std_vec.append(our_results_2[task][env][1])
        min_mean_vec.append(our_results_2[task][env][2])
        min_std_vec.append(our_results_2[task][env][3])

    print("min mean ", min_mean_vec)
    print("min std ", min_std_vec)

    ax.errorbar(range(len(scenarios_list)), min_mean_vec,
                yerr=min_std_vec, ecolor='b', color='m',
                fmt='o', capsize=2, markersize=10, label='Ours 2 Min')

    ax.errorbar(range(len(scenarios_list)), max_mean_vec,
                yerr=max_std_vec, ecolor='r', color='b',
                fmt='o', capsize=2, markersize=10, label='Ours 2 Max')



    ax.locator_params(nbins=4)
    ax.set_ylim([0, 100])
    ax.set_ylabel('Success Rate')

    ax.set_xticklabels(['a'] + scenarios_list)

    ax.legend(fontsize=30)
    for item in ([ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)


    fig.savefig('plot' + task + '.png', orientation='landscape',
                bbox_inches='tight')

    plt.close(fig)
    """

"""
ax.locator_params(axis='x', nbins=len(experiments))
ax.errorbar(range(len(success_values_stds)), success_values_averages,
            yerr=success_values_stds, linewidth=5)
ax.errorbar(range(len(success_values_stds)), success_values_averages,
            yerr=success_values_std_average, ecolor='r', linewidth=5)

ax.set_xticklabels(['a'] + list(experiments.keys()))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
             + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(10)
"""
