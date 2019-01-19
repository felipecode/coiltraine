import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

from matplotlib import rcParams


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


    results_dict  = {}

    tasks = ['empty', 'normal', 'cluttered']
    scenarios = ['Long3Training_Town01', 'Long3NewWeather_Town01',
                 'Long3NewTown_Town02', 'Long3NewWeatherTown_Town02']

    out_tasks = {'empty': 'Empty', 'normal': 'Normal', 'cluttered': 'Cluttered'}

    out_scenarios = {'Long3Training_Town01':'Training', 'Long3NewWeather_Town01':'NewWeather',
                     'Long3NewTown_Town02':'NewTown', 'Long3NewWeatherTown_Town02':'NewWeatherTown'}

    check_point = 0

    for t in tasks:
        print("Task ", t)
        scenarios_dict = {}
        for s in scenarios:
            # HERe we do a plot
            print("  Scenario ", s)

            variation_values = []
            variation_values_std = []
            for exp in experiments:
                print("      exp ", exp)
                success = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                                  'control_output_' + t + '.csv'),
                                     delimiter=",",
                                     skiprows=1,
                                     usecols=([5]))

                std = np.loadtxt(os.path.join(root_path, exp, 'drive_' + s + '_csv',
                                              'control_output_' + t + '.csv'),
                                 delimiter=",",
                                 skiprows=1,
                                 usecols=([6]))
                print("        succ", success)
                print("        std", std)

                if success.shape != (0,):
                    try:
                        success = success[0]
                        std = std[0]
                    except IndexError:
                        pass

                    variation_values.append(success)
                    variation_values_std.append(std)

            maximun_value_index = int(np.argmax(variation_values))
            print ("max value index", maximun_value_index)
            minimun_value_index = int(np.argmin(variation_values))

            maximun_success = variation_values[maximun_value_index]
            minimun_success = variation_values[minimun_value_index]
            maximun_std = variation_values_std[maximun_value_index]
            minimun_std = variation_values_std[minimun_value_index]

            scenarios_dict.update({out_scenarios[s]: [maximun_success*100, maximun_std*100,
                                                      minimun_success*100, minimun_std*100]})

        results_dict.update({out_tasks[t]: scenarios_dict})


    return results_dict



def draw_one_of_our_method():
    pass

# ***** main loop *****
if __name__ == "__main__":

    #root path


    compared_results= {
        'Empty':
            {"MT": {"Training": [90, 88],
                     "NewWeather": [88, 90],
                     "NewTown": [52, 52, 54],
                     "NewWeatherTown": [60, 62, 62]},
             "CAL": {"Training": [83, 85, 86],
                     "NewWeather": [86, 89, 92],
                     "NewTown": [50, 45, 49],
                     "NewWeatherTown": [36, 36, 36]},
             "CoIL": {"Training": [85, 81, 72],
                     "NewWeather": [60, 64, 72],
                     "NewTown": [39, 42, 46],
                     "NewWeatherTown": [39, 42, 46]}
             },
        'Normal':
            {"MT": {"Training": [55, 60],
                    "NewWeather": [66, 62],
                    "NewTown": [30, 31, 33],
                    "NewWeatherTown": [38, 38, 32]},
             "CAL": {"Training": [72, 80, 74],
                     "NewWeather": [70, 66, 80],
                     "NewTown": [39, 37, 33],
                     "NewWeatherTown": [16, 20, 20]},
             "CoIL": {"Training": [65, 65, 42],
                      "NewWeather": [36, 50, 42],
                      "NewTown": [24, 23, 25],
                      "NewWeatherTown": [10, 14, 18]}
             },
        'Cluttered':
            {"MT": {"Training": [18, 16],
                    "NewWeather": [8, 16],
                    "NewTown": [13, 14, 18],
                    "NewWeatherTown": [18, 22, 16]},
             "CAL": {"Training": [43, 48, 39],
                     "NewWeather": [30, 34, 38],
                     "NewTown": [12, 18, 9],
                     "NewWeatherTown": [18, 10, 10]},
             "CoIL": {"Training": [19, 25, 12],
                      "NewWeather": [4, 8, 12],
                      "NewTown": [6, 7, 6],
                      "NewWeatherTown": [2, 4, 3]}
             }
    }

    scenarios_list = ["Training", "New Weather", "New Town", "New Weather & Town"]

    print_sd = {'Empty': 'Empty Town', 'Normal': 'Regular Traffic', 'Cluttered': 'Dense Traffic'}


    our_results = compute_our_res_dict('res34-50-lowdropout-imnet')


    our_results_2 = compute_our_res_dict('res34-50-lowdropout-imnet-nospeed')
    print (our_results)

    colors = ['k', 'b', 'y']


    for task, methods in compared_results.items():

        print("tasks ", task)
        fig, ax = plt.subplots(figsize=(16, 14))
        plt.title(print_sd[task], fontsize=40)
        # The matrix with all std values
        count = 0

        for method, envs in methods.items():
            std_vec = []
            mean_vec = []

            print("  method ", method)
            for env, runs in envs.items():
                print ("    env ")
                print (np.mean(runs))
                print (np.std(runs))
                mean_vec.append(np.mean(runs))
                std_vec.append(np.std(runs))

            print("mean ", mean_vec)
            print("std ", std_vec)

            ax.errorbar(range(len(envs)), mean_vec,
                        yerr=std_vec, ecolor='r', color=colors[count],
                        fmt='d', capsize=2, markersize=8, label=method)
            count += 1


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