import numpy as np
import os
import math
import matplotlib.pyplot as plt

# ***** main loop *****
if __name__ == "__main__":

    #root path

    root_path = '/home/felipe/CoIL_TRI/_logs'
    # CVPR SAMPLING GOES INSIDE

    # Add several experiments parts
    experiments = {
        '50': [
            'cvprfinal_valstop_seed1/res34-50-lowdropout',
            'cvprfinal_valstop_seed2/res34-50-lowdropout',
            'cvprfinal_valstop_seed3/res34-50-lowdropout',
            'cvprfinal_valstop_seed4/res34-50-lowdropout'],
        '50-ns': [
            'cvprfinal_valstop_seed1/res34-50-lowdropout-nospeed',
            'cvprfinal_valstop_seed2/res34-50-lowdropout-nospeed',
            'cvprfinal_valstop_seed3/res34-50-lowdropout-nospeed',
            'cvprfinal_valstop_seed4/res34-50-lowdropout-nospeed'],
        '50-imnet': [
            'cvprfinal_valstop_seed1/res34-50-lowdropout-imnet',
            'cvprfinal_valstop_seed2/res34-50-lowdropout-imnet',
            'cvprfinal_valstop_seed3/res34-50-lowdropout-imnet',
            'cvprfinal_valstop_seed4/res34-50-lowdropout-imnet'],
        '50-imnet-ns': [
            'cvprfinal_valstop_seed1/res34-50-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed2/res34-50-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed3/res34-50-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed4/res34-50-lowdropout-imnet-nospeed'],
        '100': [
            'cvprfinal_valstop_seed1/res34-100-lowdropout',
            'cvprfinal_valstop_seed2/res34-100-lowdropout',
            'cvprfinal_valstop_seed3/res34-100-lowdropout',
            'cvprfinal_valstop_seed4/res34-100-lowdropout'],
        '100-ns': [
            'cvprfinal_valstop_seed1/res34-100-lowdropout-nospeed',
            'cvprfinal_valstop_seed2/res34-100-lowdropout-nospeed',
            'cvprfinal_valstop_seed3/res34-100-lowdropout-nospeed',
            'cvprfinal_valstop_seed4/res34-100-lowdropout-nospeed'],
        '100-imnet': [
                           'cvprfinal_valstop_seed1/res34-100-lowdropout-imnet',
                           'cvprfinal_valstop_seed2/res34-100-lowdropout-imnet',
                           'cvprfinal_valstop_seed3/res34-100-lowdropout-imnet',
                           'cvprfinal_valstop_seed4/res34-100-lowdropout-imnet'],
        '100-imnet-ns': [
            'cvprfinal_valstop_seed1/res34-100-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed2/res34-100-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed3/res34-100-lowdropout-imnet-nospeed',
            'cvprfinal_valstop_seed4/res34-100-lowdropout-imnet-nospeed']

    }

    tasks = ['empty', 'normal', 'cluttered']
    scenarios = ['Long3Training_Town01', 'Long3NewWeather_Town01',
                 'Long3NewTown_Town02', 'Long3NewWeatherTown_Town02']

    check_point = 0

    for t in tasks:
        print ("Task ", t)
        for s in scenarios:
            # HERe we do a plot
            success_values_averages = []
            success_values_std_average = []
            success_values_stds = []
            print("  Scenario ", s)
            for key, values in experiments.items():

                print("    key ", key)
                variation_values = []
                variation_values_std = []
                for exp in values:
                    print ("      exp ", exp)
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
                        variation_values_std.append(std*std)


                success_values_averages.append(np.mean(variation_values))
                success_values_std_average.append(math.sqrt(np.mean(variation_values_std)))
                success_values_stds.append(np.std(variation_values))

            print("avg")
            print(success_values_averages)
            print("std")
            print(success_values_stds)
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.title('Task: ' + t + ' | Scenario: ' + s)

            ax.set_ylim([0, 1])
            ax.set_ylabel('Success Rate')
            ax.locator_params(axis='x', nbins=len(experiments))
            ax.errorbar(range(len(success_values_stds)), success_values_averages,
                        yerr=success_values_stds, linewidth=5)
            ax.errorbar(range(len(success_values_stds)), success_values_averages,
                        yerr=success_values_std_average, ecolor='r', linewidth=5)

            ax.set_xticklabels(['a'] + list(experiments.keys()))

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(10)

            fig.savefig('plot' + t + '_' + s + '.png', orientation='landscape',
                        bbox_inches='tight')

            plt.close(fig)

