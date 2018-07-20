
import argparse
import os
import json
import numpy as np

if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--folder',
        metavar='E',
        default=None,
        help='The folder of experiments you want to plot')


    argparser.add_argument('-c',
        '--control',
        metavar='C',
        default='auto',
        help='IP of the host server (default: localhost)')

    argparser.add_argument('-f',
        '--folder-name',
        metavar='F',
        default='eccv',
        help='IP of the host server (default: localhost)')

    argparser.add_argument('-p',
        '--plot-folder',
        metavar='P',
        default='',
        help='Params module (default: USERNAME)')

    argparser.add_argument('-t',
        '--towns',
        metavar='T',
        nargs='+',
        type=int,
        default=[],
        help='Params module (default: USERNAME_params)')

    argparser.add_argument('-s',
       '--strings-to-contain',
       metavar='S',
       default=None,
       help='IP of the host server (default: localhost)')


    args = argparser.parse_args()

    root_path = '_logs'
    all_metrics_dict_json_path = os.path.join(root_path, args.folder_name, 'plots',
                                               args.plot_folder, 'all_metricsTown02.json')
    with open(all_metrics_dict_json_path, 'r') as f:
        all_metrics_dict = json.loads(f.read())


    # Experiments with absolute error < 0.05 and success < 0.32
    absolute_error_name  = 'Steering absolute error'
    count_01             = 'Counting realtive 0.1 (Alexey)'
    count_003             = 'Counting realtive 0.03 (Alexey)'
    success_rate_name    = 'Success rate'
    found_experiments    = []


    experiments_to_print = ['experiment_59', 'experiment_22']


    name_keys = []
    for key in all_metrics_dict.keys():
        name_keys.append( key[(key.index(':')+1):])

    sorted_keys = sorted(name_keys)
    with open("tested_experiments.csv", 'w') as f:

        for key in sorted_keys:
            first = True
            for i in key.split('_'):
                if first:
                    f.write(i)
                    first = False
                else:
                    f.write(',' +i)
            f.write('\n')


    for key, value in all_metrics_dict.items():

        out = False
        for experiment in experiments_to_print:
            if experiment in key:
                out = True
                break
        if out:
            continue

        print (key)
        for i in  range(len(all_metrics_dict[key][absolute_error_name])):
            print ('%3f' % all_metrics_dict[key][absolute_error_name][i], end=' ')

        print ()
        for i in range(len(all_metrics_dict[key][absolute_error_name])):
            print ('%3f' % all_metrics_dict[key][count_01][i], end=' ')
        print ()
        for i in range(len(all_metrics_dict[key][absolute_error_name])):
            print ('%3f' % all_metrics_dict[key][success_rate_name][i], end=' ')
        print ()

        print (np.array(all_metrics_dict[key][absolute_error_name]) < 0.05)
        condition = (np.array(all_metrics_dict[key][absolute_error_name]) < 0.05) & \
                    (np.array(all_metrics_dict[key][success_rate_name]) < 0.15)
        print (condition)
        if sum(condition) > 0:
            found_experiments.append(key)


    print (found_experiments)