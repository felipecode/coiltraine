import argparse
import logging
import resource

from logger import printer

from utils.general import create_log_folder, create_exp_path, erase_logs, fix_driving_environments, \
                            get_validation_datasets, get_driving_environments


from visualization.exporter import export_csv, export_status, export_csv_separate



# You could send the module to be executed and they could have the same interface.


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--export-status',
        action='store_true',
        dest='export_status',
    )
    argparser.add_argument(
        '--export-results',
        action='store_true',
        dest='export_results',
    )
    argparser.add_argument(
        '--folder',
        default='eccv',
        type=str
    )
    argparser.add_argument(
        '--erase-experiments',
        nargs='+',
        dest='erase_experiments',
        type=str
    )

    args = argparser.parse_args()

    # Obs this is like a fixed parameter, how much a validation and a train and drives ocupies

    if args.export_status:
        validation_datasets = get_validation_datasets(args.folder)
        drive_environments = get_driving_environments(args.folder)
        export_status(args.folder, validation_datasets, drive_environments)

    if args.export_results:
        variables_to_export = ['episodes_fully_completed', 'end_pedestrian_collision',
                               'end_vehicle_collision',
                               'end_other_collision', 'driven_kilometers']

        # TODO: for now it basically will just export the best
        export_csv_separate(args.folder, variables_to_export,
                            ['empty', 'cluttered'])

    if args.erase_experiments:
        pass
