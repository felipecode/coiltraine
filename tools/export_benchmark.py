import os,json
import numpy as np



def read_summary_csv(control_csv_file):



    f = open(control_csv_file, "rU")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(control_csv_file, delimiter=",", skiprows=1)
    summary_dict = {}

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    count = 0
    for _ in header:
        summary_dict.update({header[count]: data_matrix[:, count]})
        count += 1



    return summary_dict


def compute_average_std_separatetasks(dic_list, weathers, number_of_tasks=1):
    """
    There are two types of outputs, these come packed in a dictionary

    Success metrics, these are averaged between weathers, is basically the percentage of completion for a
    single task.

    Infractions, these are summed and divided by the total number of driven kilometers


    For this you have the concept of averaging all the weathers from the experiment suite.

    """

    metrics_to_average = [
        'episodes_fully_completed',
        'episodes_completion'

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
                         4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}

    number_of_episodes = len(list(dic_list[0]['episodes_fully_completed'].items())[0][1])

    # The average results between the dictionaries.
    average_results_matrix = {}

    for metric_name in (metrics_to_average + infraction_metrics + metrics_to_sum):
        average_results_matrix.update({metric_name: np.zeros((number_of_tasks, len(dic_list)))})

    count_dic_pos = 0
    for metrics_summary in dic_list:

        for metric in metrics_to_average:

            values = metrics_summary[metric]
            # print values

            metric_sum_values = np.zeros(number_of_episodes)
            for weather, tasks in values.items():
                if float(weather) in set(weathers):
                    count = 0
                    for t in tasks:
                        # if isinstance(t, np.ndarray) or isinstance(t, list):

                        if t == []:
                            print('    Metric Not Computed')
                        else:
                            metric_sum_values[count] += (float(sum(t)))

                        count += 1

            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]/(25*len(weathers))

        # For the metrics we sum over all the weathers here, this is to better subdivide the driving envs
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

            # On this part average results matrix basically assume the number of infractions.
            for i in range(len(metric_sum_values)):
                if metric_sum_values[i] == 0:
                    average_results_matrix[metric][i][count_dic_pos] = 1
                else:
                    average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]

        for metric in metrics_to_sum:
            values = metrics_summary[metric]
            metric_sum_values = np.zeros(number_of_episodes)

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
            print (" metric sum ", metric_sum_values)
            for i in range(len(metric_sum_values)):
                average_results_matrix[metric][i][count_dic_pos] = metric_sum_values[i]/(25*len(weathers))

        count_dic_pos += 1



    average_speed_task = sum(metrics_summary['average_speed'][str(float(list(weathers)[0]))])

    average_results_matrix.update({'driven_kilometers': np.array(summed_driven_kilometers)})

    average_results_matrix.update({'average_speed': np.array([average_speed_task])})
    print(average_results_matrix)


    return average_results_matrix


def write_data_point_control_summary(path, task, averaged_dict, step, pos):

    filename = os.path.join(path + '_' + task + '.csv')

    print (filename)

    if not os.path.exists(filename):
        raise ValueError("The filename does not yet exists")

    csv_outfile = open(filename, 'a')

    csv_outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n"
                      % (step,
                         averaged_dict['episodes_completion'][pos][0],
                         averaged_dict['intersection_offroad'][pos][0],
                         averaged_dict['collision_pedestrians'][pos][0],
                         averaged_dict['collision_vehicles'][pos][0],
                         averaged_dict['episodes_fully_completed'][pos][0],
                         averaged_dict['driven_kilometers'][pos],
                         averaged_dict['end_pedestrian_collision'][pos][0],
                         averaged_dict['end_vehicle_collision'][pos][0],
                         averaged_dict['end_other_collision'][pos][0],
                         averaged_dict['intersection_otherlane'][pos][0]))

    csv_outfile.close()

def write_header_control_summary(path, task):

    filename = os.path.join(path + '_' + task + '.csv')

    print (filename)

    csv_outfile = open(filename, 'w')

    csv_outfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                      % ('step', 'episodes_completion', 'intersection_offroad',
                          'collision_pedestrians', 'collision_vehicles', 'episodes_fully_completed',
                         'driven_kilometers', 'end_pedestrian_collision',
                         'end_vehicle_collision',  'end_other_collision', 'intersection_otherlane' ))
    csv_outfile.close()



def export_csv_separate(exp_batch, variables_to_export, task_list):
    # TODO: add parameter for auto versus auto.

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch))

    # TODO: for now it always takes the position of maximun succes
    if 'episodes_fully_completed' not in set(variables_to_export):

        raise ValueError(" export csv needs the episodes fully completed param on variables")

    # Make the header of the exported csv
    csv_outfile = os.path.join(root_path, exp_batch, 'result.csv')


    with open(csv_outfile, 'w') as f:
        f.write("experiment,environment")
        for variable in variables_to_export:
            f.write(",%s" % variable)

        f.write("\n")




    experiment_list = []
    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
            scenario = []
            for log in experiments_logs:
                dicts_to_write = {}
                for task in task_list:
                    dicts_to_write.update({task: {}})

                for task in task_list:
                    if 'drive' in log and '_csv' in log:
                        csv_file_path = os.path.join(root_path, exp_batch, exp, log, 'control_output_' + task + '.csv')

                        if not os.path.exists(csv_file_path):
                            continue
                        control_csv = read_summary_csv(csv_file_path)
                        if control_csv is None:
                            continue
                        print (control_csv)

                        position_of_max_success = np.argmax(control_csv['episodes_fully_completed'])
                        print (dicts_to_write)


                        for variable in variables_to_export:
                            dicts_to_write[task].update({variable: control_csv[variable][position_of_max_success]})

                scenario.append(dicts_to_write)
            experiment_list.append(scenario)


    print (" FULL DICT")
    print (experiment_list)

    with open(csv_outfile, 'a') as f:


        for exp in experiments:
            print ("EXP ", exp)
            if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
                experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
                count = 0
                for log in experiments_logs:
                    if 'drive' in log and '_csv' in log:

                        f.write("%s,%s" % (exp, log.split('_')[1]))
                        for variable in variables_to_export:


                            f.write(",")
                            for task in task_list:
                                if experiment_list[experiments.index(exp)][count][task]:
                                    f.write("%.2f/" % experiment_list[experiments.index(exp)][count][task][variable])


                        f.write("\n")
                    count += 1



if __name__ == "__main__":

    exp_batch = 'icra_paper'
    exp_alias = 'experiment_1'

    task_list = ['empty', 'normal', 'cluttered']
    variables_to_export = ['episodes_fully_completed', 'end_pedestrian_collision', 'end_vehicle_collision',
                           'end_other_collision', 'driven_kilometers']
    benchmark_json_path = os.path.join('/home/eder/felipecode/imitation-learning/_benchmarks_results/test_LongitudinalControl2018_Town02', 'metrics.json')

    with open(benchmark_json_path, 'r') as f:
        benchmark_dict = json.loads(f.read())

    averaged_dict = compute_average_std_separatetasks([benchmark_dict],
                                                      [1, 3, 6, 8],
                                                      25)

    file_base = os.path.join('_logs', exp_batch, exp_alias,
                              'drive_csv', 'control_output')

    for i in range(len(task_list)):
        write_header_control_summary(file_base, task_list[i])
    # TODO: Number of tasks is hardcoded
    # TODO: Number of tasks is hardcoded

    print("TASK LIST ")
    print(task_list)

    for i in range(len(task_list)):
        # write_data_point_control_summary(file_base, 'empty', averaged_dict, latest, 0)
        # write_data_point_control_summary(file_base, 'normal', averaged_dict, latest, 1)
        write_data_point_control_summary(file_base, task_list[i], averaged_dict, 0, i)



    export_csv_separate(exp_batch, variables_to_export, task_list)