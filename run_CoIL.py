import argparse
import logging
import resource


from coil_core import execute_train, execute_validation, execute_drive, folder_execute
from utils.general import create_log_folder, create_exp_path, erase_logs, fix_driving_environments

from visualization import plot_scatter

# You could send the module to be executed and they could have the same interface.





if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)


    argparser.add_argument(
        '--single-process',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val-datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    argparser.add_argument(
        '-ns', '--no-screen',
        action='store_true',
        dest='no_screen',
        help='Set to carla to run offscreen'
    )

    args = argparser.parse_args()



    for gpu in args.gpus:
        try:
            int(gpu)
        except:
            raise ValueError(" Gpu is not a valid int number")

    if args.folder is None:
        raise ValueError(" You should set a folder name where the experiments are placed")


    # We increase the number of files available.
    resource.setrlimit(
        resource.RLIMIT_CORE,
        (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    # Obs this is like a fixed parameter, how much a validation and a train and drives ocupies





    create_log_folder(args.folder)
    erase_logs(args.folder)
    

    if args.single_process is not None:

        if args.exp is None:
            raise ValueError(" You should set the exp alias when using single process")

        create_exp_path(args.folder, args.exp)

        if args.single_process == 'train':
            # TODO make without position, increases the legibility.
            execute_train("0", args.folder, args.exp, False)

        elif args.single_process == 'validation':
            execute_validation("0", args.folder, args.exp, args.validation_datasets[0], False)

        elif args.single_process == 'drive':
            execute_drive("0", args.folder, args.exp, args.driving_environments[0], False,
                          no_screen=args.no_screen)
        else:

            raise (" Invalid name for single process, chose from (train, validation, test)")


    else:

        # TODO: of course this change from gpu to gpu , but for now we just assume at least a K40

        # Maybe the latest voltas will be underused
        # OBS: This usage is also based on my tensorflow experiences, maybe pytorch allows more.
        allocation_parameters = {'gpu_value': 3.5,
                                 'train_cost': 2,
                                 'validation_cost': 1.5,
                                 'drive_cost': 1.5}
        # TODO: temporary function until carla map change feature is fixed.

        driving_environments = fix_driving_environments(list(args.driving_environments))
        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': driving_environments,
            'allocation_parameters': allocation_parameters,
            'no_screen': args.no_screen,
        }

        folder_execute(params)

        plot_scatter()