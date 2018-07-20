

#  Generate the plots on the images folder that are going to be linked

import argparse
import os
import importlib
import sys


from visualization.exporter import export_csv
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




    if args.erase_bad_validations:
        validations = ['Town01W1Noise', 'Town02W14Noise', 'Town01W1', 'Town02W14']
        erase_wrong_plotting_summaries(args.folder, validations)

    #final_list_of_experiments = ['experiment_29.yaml']
    variables_to_export = ['episodes_fully_completed', 'collision_pedestrians', 'driven_kilometer']

    # TODO: for now it basically will just export the best
    export_csv(args.folder, variables_to_export)
