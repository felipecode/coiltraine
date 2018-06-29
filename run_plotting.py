

#  Generate the plots on the images folder that are going to be linked

import argparse
import os
import importlib
import sys


from visualization.scatter_plotter import plot_scatter
from utils.general import erase_wrong_plotting_summaries


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
        default='test',
        help='IP of the host server (default: localhost)')

    argparser.add_argument('-p',
        '--params-file',
        metavar='P',
        default='sample_plot',
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
    argparser.add_argument(
        '-ebv', '--erase-bad-validations',
        action='store_true',
        dest='erase_bad_validations',
        help='Set to carla to run offscreen'
    )

    argparser.add_argument('--ignore-lstm', action='store_true')

    argparser.add_argument('--add-noise', action='store_true')



    """
    You can either generate images for a experiment list

    Right now the plots that are enabled are two,
        * Different bins related plot versus control
        * The full plot for loss versus control

    """

    args = argparser.parse_args()
    if args.control == 'auto' or args.control == '_auto':
        control = '_auto'
    else:
        control = ''

    if args.folder:
        list_of_experiments = os.listdir(os.path.join('configs', args.folder))

        # TODO: the ignore lstm goes after getting exps names.
        #if args.ignore_lstm:
        #    list_of_experiments = [l for l in list_of_experiments if 'lstm' not in l]

    else:
        list_of_experiments = []



    if args.strings_to_contain is not None:
        final_list_of_experiments = []
        sub_strings = args.strings_to_contain.split(',')
        for experiment in list_of_experiments:
            if all(sub in experiment for sub in sub_strings):
                final_list_of_experiments.append(experiment)
    else:
        final_list_of_experiments = list_of_experiments

    if args.towns:
        town_to_name = {1: 'Town01_1', 2: 'Town02_14'}
        towns = [town_to_name[x] for x in args.towns]

    print(args)

    # Import the parameters of what and how to plot
    sys.path.append('visualization/plotting_params')
    params_module = importlib.import_module(args.params_file)

    data_params = params_module.data_params

    if args.towns:
        data_params['towns'] = towns

    if hasattr(params_module, 'list_of_experiments'):
        assert (not (final_list_of_experiments and params_module.list_of_experiments)), 'List of experiments should either be given by flags or in the param file, not both'
        final_list_of_experiments = params_module.list_of_experiments

    print('final_list_experiments', final_list_of_experiments)
    print('data params', data_params)
    print('process params', params_module.processing_params)
    print('plot params', params_module.plot_params)


    if args.erase_bad_validations:
        validations = ['Town01W1Noise', 'Town02W14Noise']
        erase_wrong_plotting_summaries(args.folder, validations)


    plot_scatter(args.folder, final_list_of_experiments, data_params, params_module.processing_params,
                 params_module.plot_params, out_folder=args.folder_name)
