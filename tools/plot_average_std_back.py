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
        'vary-sample': ['cvprfixed_sampling_valstop/res34-50-varyseed-s1',
                      'cvprfixed_sampling_valstop/res34-50-varyseed-s2',
                      'cvprfixed_sampling_valstop/res34-50-varyseed-s3',
                      'cvprfixed_sampling_valstop/res34-50-varyseed-s4'],

        'vary-init': ['cvprfixed_sampling_valstop/res34-50-varyinit-s1',
                      'cvprfixed_sampling_valstop/res34-50-varyinit-s2',
                      'cvprfixed_sampling_valstop/res34-50-varyinit-s3',
                      'cvprfixed_sampling_valstop/res34-50-varyinit-s4'],
        'vary-all': [
                     'cvprfinal_valstop_seed1/res34-50-lowdropout',
                     'cvprfinal_valstop_seed2/res34-50-lowdropout',
                     'cvprfinal_valstop_seed3/res34-50-lowdropout',
                     'cvprfinal_valstop_seed4/res34-50-lowdropout'],
        'vary-all-imnet': [
                           'cvprfinal_valstop_seed1/res34-50-lowdropout-imnet',
                           'cvprfinal_valstop_seed2/res34-50-lowdropout-imnet',
                           'cvprfinal_valstop_seed3/res34-50-lowdropout-imnet',
                           'cvprfinal_valstop_seed4/res34-50-lowdropout-imnet'],
        'vary-all-imnet-ns': [
                           'cvprfinal_valstop_seed1/res34-50-lowdropout-imnet-nospeed',
                           'cvprfinal_valstop_seed2/res34-50-lowdropout-imnet-nospeed',
                           'cvprfinal_valstop_seed3/res34-50-lowdropout-imnet-nospeed',
                           'cvprfinal_valstop_seed4/res34-50-lowdropout-imnet-nospeed']
    }

    tasks = ['empty', 'normal', 'cluttered']
    scenarios = ['Long3Training_Town01', 'Long3NewWeather_Town01',
                 'Long3NewTown_Town02', 'Long3NewWeatherTown_Town02']

    check_point = 0
    task_avg = [0] * 3
    task_std = [0] * 3
    task_cv = [0] * 3

    for t in tasks:
        print ("tasks   ", t)
        absce_avg = [0] * 5
        absce_std = [0] * 5
        absce_cv = [0] * 5



        for s in scenarios:
            # HERe we do a plot
            success_values_averages = []
            success_values_std_average = []
            success_values_stds = []

            print ("scenarios ", s)
            count = 0
            for key, values in experiments.items():

                print (key)
                variation_values = []
                variation_values_std = []
                for exp in values:
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
                    if success.shape != (0,):
                        try:
                            success = success[0]
                            std = std[0]
                        except IndexError:
                            pass

                        variation_values.append(success)
                        variation_values_std.append(std * std)

                success_values_averages.append(np.mean(variation_values))
                success_values_std_average.append(math.sqrt(np.mean(variation_values_std)))
                success_values_stds.append(np.std(variation_values))
                absce_avg[count] += np.mean(variation_values) * 0.2
                absce_std[count] += math.sqrt(np.mean(variation_values_std)) * 0.2
                absce_cv[count] += (math.sqrt(np.mean(variation_values_std)) /
                                    np.mean(variation_values)) * 0.2
                count += 1



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
                item.set_fontsize(14)

            fig.savefig('plot' + t + '_' + s + '.png', orientation='landscape',
                        bbox_inches='tight')

            plt.close(fig)

        print ("Task  ", t)
        print ("Average Task ", absce_avg)
        print ("Average Task STD ", absce_std)
        print ("Average Task CV ", absce_cv)

        task_avg = [t+a*0.333 for t, a in zip(task_avg, absce_avg)]
        task_cv = [t+a*0.333 for t, a in zip(task_cv, absce_cv)]
        task_std = [t+a*0.333 for t, a in zip(task_std, absce_std)]

    print (" Overal average ")

    print ("Average  ", task_avg)
    print ("Average  STD ", task_std)
    print ("Average  CV ", task_cv)
