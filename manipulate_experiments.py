import argparse
import logging
import resource

from logger import printer
from utils.general import create_log_folder, create_exp_path, erase_logs, fix_driving_environments, \
                            get_validation_datasets, get_driving_environments

from visualization import plot_scatter

# You could send the module to be executed and they could have the same interface.


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--check-status',
        action='store_true',
        dest='check_status',
    )
    argparser.add_argument(
        '--folder',
        default='eccv',
        type=str
    )
    argparser.add_argument(
        '--erase-experiments',
        nargs='+',
        dest='gpus',
        type=str
    )

    args = argparser.parse_args()

    # Obs this is like a fixed parameter, how much a validation and a train and drives ocupies


    if args.check_status:
        validation_datasets = get_validation_datasets(args.folder)
        drive_environments = get_driving_environments(args.folder)
        printer.plot_folder_summaries(args.folder, True, validation_datasets, drive_environments,
                              verbose=False)

    if args.erase_experiments:
        pass

