import os
import numpy as np
from .data_reading import read_control_csv, read_summary_csv

def export_csv(exp_batch, variables_to_export):
    # TODO: add parameter for auto versus auto.

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch))

    # TODO: for now it always takes the position of maximun succes
    if 'episodes_fully_completed' not in set(variables_to_export):
        raise ValueError( " export csv needs the episodes fully completed param on variables")

    # Make the header of the exported csv
    csv_outfile = os.path.join(root_path, exp_batch, 'result.csv')

    with open (csv_outfile, 'w') as f:
        f.write("experiment,environment")
        for variable in variables_to_export:
            f.write(",%s" % variable)

        f.write("\n")


    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch, exp))
            for log in experiments_logs:
                if 'drive' in log and '_csv' in log:
                    csv_file_path = os.path.join(root_path, exp_batch, exp, log, 'control_output.csv')
                    control_csv = read_summary_csv(csv_file_path)

                    print (control_csv)
                    with open(csv_outfile, 'a') as f:
                        f.write("%s,%s" % (exp, log.split('_')[1]) )
                        print (' var', variable)
                        print (control_csv[variable])

                        position_of_max_success = np.argmax(control_csv['episodes_fully_completed'])
                        for variable in variables_to_export:
                            f.write(",%f" % control_csv[variable][position_of_max_success])

                        f.write("\n")



