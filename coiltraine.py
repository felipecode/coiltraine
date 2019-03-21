import argparse

from coil_core import execute_train, execute_validation, execute_drive, folder_execute
from coilutils.general import create_log_folder, create_exp_path, erase_logs,\
                          erase_wrong_plotting_summaries, erase_validations

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
    argparser.add_argument(
        '-ebv', '--erase-bad-validations',
        action='store_true',
        dest='erase_bad_validations',
        help='erase the bad validations (Incomplete)'
    )
    argparser.add_argument(
        '-rv', '--restart-validations',
        action='store_true',
        dest='restart_validations',
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-gv',
        '--gpu-value',
        dest='gpu_value',
        type=float,
        default=3.5
    )
    argparser.add_argument(
        '-nw',
        '--number-of-workers',
        dest='number_of_workers',
        type=int,
        default=12
    )
    argparser.add_argument(
        '-dk', '--docker',
        dest='docker',
        default='carlasim/carla:0.8.4',
        type=str,
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '-rc', '--record-collisions',
        action='store_true',
        dest='record_collisions',
        help='Set to run carla using docker'
    )
    args = argparser.parse_args()

    # Check if the vector of GPUs passed are valid.
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:  # Reraise a meaningful error.
            raise ValueError("GPU is not a valid int number")

    # Check if the mandatory folder argument is passed
    if args.folder is None:
        raise ValueError("You should set a folder name where the experiments are placed")

    # Check if the driving parameters are passed in a correct way
    if args.driving_environments is not None:
        for de in list(args.driving_environments):
            if len(de.split('_')) < 2:
                raise ValueError("Invalid format for the driving environments should be Suite_Town")

    # This is the folder creation of the
    create_log_folder(args.folder)
    erase_logs(args.folder)
    if args.erase_bad_validations:
        erase_wrong_plotting_summaries(args.folder, list(args.validation_datasets))
    if args.restart_validations:
        erase_validations(args.folder, list(args.validation_datasets))

    # The definition of parameters for driving
    drive_params = {
        "suppress_output": True,
        "no_screen": args.no_screen,
        "docker": args.docker,
        "record_collisions": args.record_collisions
    }
    # There are two modes of execution
    if args.single_process is not None:
        ####
        # MODE 1: Single Process. Just execute a single experiment alias.
        ####

        if args.exp is None:
            raise ValueError(" You should set the exp alias when using single process")

        create_exp_path(args.folder, args.exp)

        if args.single_process == 'train':
            execute_train(gpu="0", exp_batch=args.folder, exp_alias=args.exp,
                          suppress_output=False, number_of_workers= args.number_of_workers)

        elif args.single_process == 'validation':
            execute_validation(gpu="0", exp_batch=args.folder, exp_alias=args.exp,
                               dataset=args.validation_datasets[0], suppress_output=False)

        elif args.single_process == 'drive':
            drive_params['suppress_output'] = False
            execute_drive("0", args.folder, args.exp, list(args.driving_environments)[0], drive_params)

        else:
            raise Exception("Invalid name for single process, chose from (train, validation, test)")

    else:
        ####
        # MODE 2: Folder execution. Execute train/validation/drive for all experiments on
        #         a certain training folder
        ####
        # We set by default that each gpu has a value of 3.5, allowing a training and
        # a driving/validation
        allocation_parameters = {'gpu_value': args.gpu_value,
                                 'train_cost': 1.5,
                                 'validation_cost': 1.0,
                                 'drive_cost': 1.5}

        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': list(args.driving_environments),
            'driving_parameters': drive_params,
            'allocation_parameters': allocation_parameters,
            'number_of_workers': args.number_of_workers

        }

        folder_execute(params)
        print("SUCCESSFULLY RAN ALL EXPERIMENTS")
