#  Generate the plots on the images folder that are going to be linked

import argparse
import os
import importlib
import sys


from plotter import plot_scatter
from coilutils.general import erase_wrong_plotting_summaries


if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--folder',
        metavar='E',
        default='eccv',
        help='The folder of experiments you want to plot')

    argparser.add_argument('-c',
        '--control',
        metavar='C',
        default='auto',
        help='IP of the host server (default: localhost)')

    argparser.add_argument('-p',
        '--params-file',
        metavar='P',
        default='sample_plot',
        help='Params module (default: USERNAME)')


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
    else:
        list_of_experiments = []

    # Import the parameters of what and how to plot
    sys.path.append('plotter/plotting_params')
    params_module = importlib.import_module(args.params_file)

    data_params = params_module.data_params

    if hasattr(params_module, 'list_of_experiments'):
        final_list_of_experiments = params_module.list_of_experiments
    else:
        final_list_of_experiments = list_of_experiments

    print('final_list_experiments', final_list_of_experiments)
    print('data params', data_params)
    print('process params', params_module.processing_params)
    print('plot params', params_module.plot_params)

    if args.erase_bad_validations:
        validations = ['Town01W1Noise', 'Town02W14Noise', 'Town01W1', 'Town02W14']
        erase_wrong_plotting_summaries(args.folder, validations)

    #if check_csv_ground_truths
    # Check if the validation folders already have the

    plot_scatter(args.folder, final_list_of_experiments, data_params, params_module.processing_params,
                 params_module.plot_params, out_folder=args.params_file)
