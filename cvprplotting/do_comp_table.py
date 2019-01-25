import pprint as pp
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
from data_reading import compute_avg_result_dict_ours

def plot_comparison(compared_results, name, metric, colors, formats):

    scenarios_list = ["Training", "New Weather", "New Town", "New Weather & Town"]
    print_sd = {'Empty': 'Empty Town', 'Normal': 'Regular Traffic', 'Cluttered': 'Dense Traffic'}


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
                print (np.mean(runs[metric]))
                print (np.std(runs[metric]))
                mean_vec.append(np.mean(runs[metric]))
                std_vec.append(np.std(runs[metric]))

            print("mean ", mean_vec)
            print("std ", std_vec)

            ax.errorbar(range(len(envs)), mean_vec,
                        yerr=std_vec, ecolor=colors[count], color=colors[count],
                        fmt=formats[count], capsize=2, markersize=8, label=method)
            count += 1

        ax.locator_params(nbins=4)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Success Rate')

        ax.set_xticklabels(['a'] + scenarios_list)

        ax.legend(fontsize=30)
        for item in ([ax.xaxis.label, ax.yaxis.label]
                     + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)

        fig.savefig(name + '_plot' + task + '.png', orientation='landscape',
                    bbox_inches='tight')

        plt.close(fig)


if __name__ == "__main__":


    root_path = '/home/felipe/CoIL_TRI/_logs'

    # Network Arch Comparison
    icra_10 = [
                'valstop_abspeed3_seed1/' + 'icra-10-lowdropout',
                'valstop_abspeed3_seed2/' + 'icra-10-lowdropout',
                'valstop_abspeed3_seed3/' + 'icra-10-lowdropout',
                'valstop_abspeed3_seed4/' + 'icra-10-lowdropout' ]

    res18_10 = [
                'valstop_abspeed2_seed1/' + 'res18-10-lowdropout-imnet',
                'valstop_abspeed2_seed2/' + 'res18-10-lowdropout-imnet',
                'valstop_abspeed2_seed3/' + 'res18-10-lowdropout-imnet',
                'valstop_abspeed2_seed4/' + 'res18-10-lowdropout-imnet' ]



    res34_10 = [
                'valstop_ablationspeed_seed1/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed2/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed3/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed4/' + 'res34-10-lowdropout-imnet' ]
    res50_10 = [
                'valstop_abspeed3_seed1/' + 'res50-10-lowdropout-imnet',
                'valstop_abspeed3_seed2/' + 'res50-10-lowdropout-imnet',
                'valstop_abspeed3_seed3/' + 'res50-10-lowdropout-imnet',
                'valstop_abspeed3_seed4/' + 'res50-10-lowdropout-imnet' ]

    icra_10 = [os.path.join(root_path, name) for name in icra_10]
    res18_10 = [os.path.join(root_path, name) for name in res18_10]
    res34_10 = [os.path.join(root_path, name) for name in res34_10]
    res50_10 = [os.path.join(root_path, name) for name in res50_10]

    icra_10_results_max = compute_avg_result_dict_ours(icra_10, 'Long', maximum=True)
    icra_10_results_min = compute_avg_result_dict_ours(icra_10, 'Long', maximum=False)
    res18_10_results_max = compute_avg_result_dict_ours(res18_10, 'Long', maximum=True)
    res18_10_results_min = compute_avg_result_dict_ours(res18_10, 'Long', maximum=False)
    res34_10_results_max = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=True)
    res34_10_results_min = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=False)
    res50_10_results_max = compute_avg_result_dict_ours(res50_10, 'Long', maximum=True)
    res50_10_results_min = compute_avg_result_dict_ours(res50_10, 'Long', maximum=False)

    compared_results = {'Empty': {},
                        'Normal': {},
                        'Cluttered': {}}

    for task in compared_results.keys():
        compared_results[task].update({'icra_10_max': icra_10_results_max[task]})
        compared_results[task].update({'icra_10_min': icra_10_results_min[task]})
        compared_results[task].update({'res18_10_max': res18_10_results_max[task]})
        compared_results[task].update({'res18_10_min': res18_10_results_min[task]})
        compared_results[task].update({'res34_10_max': res34_10_results_max[task]})
        compared_results[task].update({'res34_10_min': res34_10_results_min[task]})
        compared_results[task].update({'res50_10_max': res50_10_results_max[task]})
        compared_results[task].update({'res50_10_min': res50_10_results_min[task]})

    pp.pprint(compared_results)
    colors = ['r','r','y','y', 'g', 'g', 'b', 'b']

    formats =['o','d','o','d','o','d','o','d']
    plot_comparison(compared_results, 'arch', 'episodes_fully_completed', colors, formats)

    # SPEED based plotting

    speed_10 = [
                'valstop_ablationspeed_seed1/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed2/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed3/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed4/' + 'res34-10-lowdropout-imnet' ]


    trick_10 = [
                'valstop_ablationspeed_seed1/' + 'res34-10-lowdropout-imnet-trick',
                'valstop_ablationspeed_seed2/' + 'res34-10-lowdropout-imnet-trick',
                'valstop_ablationspeed_seed3/' + 'res34-10-lowdropout-imnet-trick',
                'valstop_ablationspeed_seed4/' + 'res34-10-lowdropout-imnet-trick' ]


    ns_10 = [
                'valstop_abspeed3_seed1/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed2/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed3/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed4/' + 'res34-10-lowdropout-imnet-nospeed' ]

    speed_10 = [os.path.join(root_path, name) for name in speed_10]
    trick_10 = [os.path.join(root_path, name) for name in trick_10]
    ns_10 = [os.path.join(root_path, name) for name in ns_10]

    speed_10_results_max = compute_avg_result_dict_ours(speed_10, 'Long3', maximum=True)
    speed_10_results_min = compute_avg_result_dict_ours(speed_10, 'Long3', maximum=False)
    trick_10_results_max = compute_avg_result_dict_ours(trick_10, 'Long3', maximum=True)
    trick_10_results_min = compute_avg_result_dict_ours(trick_10, 'Long3', maximum=False)
    ns_10_results_max = compute_avg_result_dict_ours(ns_10, 'Long', maximum=True)
    ns_10_results_min = compute_avg_result_dict_ours(ns_10, 'Long', maximum=False)

    compared_results = {'Empty': {},
                        'Normal': {},
                        'Cluttered': {}}

    for task in compared_results.keys():
        compared_results[task].update({'speed_10_max': speed_10_results_max[task]})
        compared_results[task].update({'speed_10_min': speed_10_results_min[task]})
        compared_results[task].update({'trick_10_max': trick_10_results_max[task]})
        compared_results[task].update({'trick_10_min': trick_10_results_min[task]})
        compared_results[task].update({'ns_10_max': ns_10_results_max[task]})
        compared_results[task].update({'ns_10_min': ns_10_results_min[task]})

    colors = ['r', 'r', 'y', 'y', 'g', 'g']
    formats =['o', 'd', 'o', 'd', 'o', 'd']

    plot_comparison(compared_results, 'speed', 'episodes_fully_completed', colors, formats)


    # Data Based

    res34_2 = [
        'valstop_abspeed2_seed1/' + 'res34-2-lowdropout-imnet',
        'valstop_abspeed2_seed2/' + 'res34-2-lowdropout-imnet',
        'valstop_abspeed2_seed3/' + 'res34-2-lowdropout-imnet',
        'valstop_abspeed2_seed4/' + 'res34-2-lowdropout-imnet']
    
    res34_10 = [
        'valstop_ablationspeed_seed1/' + 'res34-10-lowdropout-imnet',
        'valstop_ablationspeed_seed2/' + 'res34-10-lowdropout-imnet',
        'valstop_ablationspeed_seed3/' + 'res34-10-lowdropout-imnet',
        'valstop_ablationspeed_seed4/' + 'res34-10-lowdropout-imnet']

    res34_50 = [
        'valstop_ablation_extra/' + 'res34-50-lowdropout-imnet-notrick-s1',
        'valstop_ablation_extra/' + 'res34-50-lowdropout-imnet-notrick-s2',
        'valstop_ablation_extra/' + 'res34-50-lowdropout-imnet-notrick-s3',
        'valstop_ablation_extra/' + 'res34-50-lowdropout-imnet-notrick-s4']

    res34_100 = [
        'cvprfinal_valstop_seed1/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed2/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed3/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed4/' + 'res34-100-lowdropout-imnet']

    res34_2 = [os.path.join(root_path, name) for name in res34_2]
    res34_10 = [os.path.join(root_path, name) for name in res34_10]
    res34_50 = [os.path.join(root_path, name) for name in res34_50]
    res34_100 = [os.path.join(root_path, name) for name in res34_100]

    res34_2_results_max = compute_avg_result_dict_ours(res34_2, 'Long', maximum=True)
    res34_2_results_min = compute_avg_result_dict_ours(res34_2, 'Long', maximum=False)
    res34_10_results_max = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=True)
    res34_10_results_min = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=False)
    res34_50_results_max = compute_avg_result_dict_ours(res34_50, 'Long3', maximum=True)
    res34_50_results_min = compute_avg_result_dict_ours(res34_50, 'Long3', maximum=False)
    res34_100_results_max = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=True)
    res34_100_results_min = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=False)
    
    compared_results = {'Empty': {},
                        'Normal': {},
                        'Cluttered': {}}
    
    for task in compared_results.keys():
        compared_results[task].update({'res34_2_max': res34_2_results_max[task]})
        compared_results[task].update({'res34_2_min': res34_2_results_min[task]})
        compared_results[task].update({'res34_10_max': res34_10_results_max[task]})
        compared_results[task].update({'res34_10_min': res34_10_results_min[task]})
        compared_results[task].update({'res34_50_max': res34_50_results_max[task]})
        compared_results[task].update({'res34_50_min': res34_50_results_min[task]})
        compared_results[task].update({'res34_100_max': res34_100_results_max[task]})
        compared_results[task].update({'res34_100_min': res34_100_results_min[task]})
    
    colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b']
    formats = ['o', 'd', 'o', 'd', 'o', 'd', 'o', 'd']
    
    plot_comparison(compared_results, 'data', 'episodes_fully_completed', colors, formats)

    # Data No image net

    res34_2 = [
        'valstop_abspeed2_seed1/' + 'res34-2-lowdropout',
        'valstop_abspeed2_seed2/' + 'res34-2-lowdropout',
        'valstop_abspeed2_seed3/' + 'res34-2-lowdropout',
        'valstop_abspeed2_seed4/' + 'res34-2-lowdropout']

    res34_10 = [
        'valstop_abspeed2_seed1/' + 'res34-10-lowdropout',
        'valstop_abspeed2_seed2/' + 'res34-10-lowdropout',
        'valstop_abspeed2_seed3/' + 'res34-10-lowdropout',
        'valstop_abspeed2_seed4/' + 'res34-10-lowdropout']

    res34_50 = [
        'cvprfinal_valstop_seed1/' + 'res34-50-lowdropout',
        'cvprfinal_valstop_seed1/' + 'res34-50-lowdropout',
        'cvprfinal_valstop_seed1/' + 'res34-50-lowdropout',
        'cvprfinal_valstop_seed1/' + 'res34-50-lowdropout']

    res34_100 = [
        'cvprfinal_valstop_seed1/' + 'res34-100-lowdropout',
        'cvprfinal_valstop_seed2/' + 'res34-100-lowdropout',
        'cvprfinal_valstop_seed3/' + 'res34-100-lowdropout',
        'cvprfinal_valstop_seed4/' + 'res34-100-lowdropout']

    res34_2 = [os.path.join(root_path, name) for name in res34_2]
    res34_10 = [os.path.join(root_path, name) for name in res34_10]
    res34_50 = [os.path.join(root_path, name) for name in res34_50]
    res34_100 = [os.path.join(root_path, name) for name in res34_100]

    res34_2_results_max = compute_avg_result_dict_ours(res34_2, 'Long', maximum=True)
    res34_2_results_min = compute_avg_result_dict_ours(res34_2, 'Long', maximum=False)
    res34_10_results_max = compute_avg_result_dict_ours(res34_10, 'Long', maximum=True)
    res34_10_results_min = compute_avg_result_dict_ours(res34_10, 'Long', maximum=False)
    res34_50_results_max = compute_avg_result_dict_ours(res34_50, 'Long3', maximum=True)
    res34_50_results_min = compute_avg_result_dict_ours(res34_50, 'Long3', maximum=False)
    res34_100_results_max = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=True)
    res34_100_results_min = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=False)

    compared_results = {'Empty': {},
                        'Normal': {},
                        'Cluttered': {}}

    for task in compared_results.keys():
        compared_results[task].update({'res34_2_max': res34_2_results_max[task]})
        compared_results[task].update({'res34_2_min': res34_2_results_min[task]})
        compared_results[task].update({'res34_10_max': res34_10_results_max[task]})
        compared_results[task].update({'res34_10_min': res34_10_results_min[task]})
        compared_results[task].update({'res34_50_max': res34_50_results_max[task]})
        compared_results[task].update({'res34_50_min': res34_50_results_min[task]})
        compared_results[task].update({'res34_100_max': res34_100_results_max[task]})
        compared_results[task].update({'res34_100_min': res34_100_results_min[task]})

    colors = ['r', 'r', 'y', 'y', 'g', 'g', 'b', 'b']
    formats = ['o', 'd', 'o', 'd', 'o', 'd', 'o', 'd']

    plot_comparison(compared_results, 'data_no_imnet', 'episodes_fully_completed', colors, formats)