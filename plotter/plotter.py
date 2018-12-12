import json
import numpy as np
import os
import time
import shutil
import logging


import pprint
import collections

from configs.coil_global import get_names

from . import scatter_plotter
from . import data_reading
from . import metrics


def read_data(exp_batch, experiment, val_dataset, drive_folder, data_params):

    """
    Read the data. That correspond on reading all the csv produced by the validation module
    Also read the control csv produced by the tests performed in CARLA
    Args:
        exp_batch: The experiment folder to compute the plots
        experiment: The experiment to read
        val_dataset: The name of the validation dataset
        drive_folder: The name of the driving benchmark folder
        data_params: The params for reading the data. ( If the control is auto and the root path)

    Returns:

    """
    # read the data
    data = {}
    # Add town information to the data
    data['town'] = drive_folder.split('_')[-1]
    data['experiment'] = experiment
    # Set the control dataset.
    full_path_control = os.path.join(data_params['root_path'], exp_batch, experiment,
                                     'drive_' + drive_folder + '_csv')

    control_data = data_reading._read_control_data(full_path_control, data_params['control'])
    if control_data is None:
        print("control is none")
        return None

    # We get the path for the validation csvs
    full_path_validation = os.path.join(data_params['root_path'], exp_batch, experiment,
                                        'validation_' + val_dataset + '_csv')

    # Based on the control data, we read the rest of the data
    values = data_reading._read_data(full_path_validation, control_data)

    if values is None:
        return None
    else:
        data['values'] = values

    return data


def filter_data(data, filter_param, val_dataset):

    """
    Filters the data to get just a a subpart of it before applying the algorithms
    Args:
        data:
        filter_param:
        val_dataset:

    Returns:

    """
    if filter_param:
        if 'camera' not in filter_param:
            raise ValueError("Filter params should contain cameras")
        print (" GOING TO FILTER CENTER ", filter_param)
        # prepare the mask
        camera_name_to_label = {'central': 1, 'left': 0, 'right': 2}
        camera_labels = data_reading.get_camera_labels(val_dataset)
        mask = np.where(camera_labels == camera_name_to_label[filter_param['camera']])
        # actually filter
        keys_to_filter = ['speed_input', 'steer_gt', 'steer_pred']
        data_filtered = {}
        data_filtered['values'] = collections.OrderedDict()
        for step, values_item in data['values'].items():
            data_filtered['values'][step] = {}
            for key in keys_to_filter:
                data_filtered['values'][step][key] = values_item[key][mask]

        print("len key after filtering ", len(data_filtered['values'][2000.0]['steer_gt']))
    else:
        data_filtered = data
    return data_filtered


def compute_metric(metric_name, data, param):
    """
    Compute a offline metric on a certain dataset. Should return a single number
    Args:
        metric_name: The name of the metric to be computed. It must exist on the metrics.py module
        data: The data on which the metric os going
        param: the params necessary for this metric computation

    Returns:

    """
    metric_func = getattr(metrics, 'compute_' + metric_name)
    if metric_name in ['id', 'step', 'experiment']:
        metric_results = metric_func(data, param)
    else:
        metric_results = metrics.compute_and_aggregate(metric_func, data, param)
    return metric_results


def process_data(data, processing_params, val_dataset):
    """
    Process all the data following the processing params specified on the configuration
    python file. Each processing correspond to a filtering of the data(Selecting a wanted subset)
    and a computation of a metric.

    Args:
        data:
        processing_params:
        val_dataset:

    Returns:

    """
    metrics = {}
    for metric_label, metric_param in processing_params.items():
        data_filtered = filter_data(data, metric_param['filter'], val_dataset)
        results = compute_metric(metric_param['metric'], data_filtered, metric_param['params'])
        metrics[metric_label] = results

    return metrics


def plot_scatter(exp_batch, list_of_experiments, data_params,
                 processing_params, plot_params, out_folder=None):
    """
    Creates a scatter plot for the pairs of validation and driving evaluation.
    Computes the average of several offline metric and correlates each
    of these metrics to driving quality metric. The pairs of offline metrics and driving
    quality metrics that are going to be plotted are seton the plotting params files.
    Args:
        exp_batch:
        list_of_experiments:
        data_params:
        processing_params:
        plot_params:
        out_folder:

    Returns:

    """

    # create a folder, in case it is none
    if out_folder is None:
        out_folder = time.strftime("plots_%Y_%m_%d_%H_%M_%S", time.gmtime())

    print ("out path")
    print (data_params['root_path'], exp_batch, 'plots', out_folder)
    out_path = os.path.join(data_params['root_path'], exp_batch, 'plots', out_folder)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        shutil.rmtree(out_path)
        os.makedirs(out_path)

    # save the parameters
    print ("Out path", out_path)
    with open(os.path.join(out_path, 'params.txt'), 'w') as f:
        f.write('list_of_experiments:\n' + pprint.pformat(list_of_experiments, indent=4))
        f.write('\n\ndata_params:\n' + pprint.pformat(data_params, indent=4))
        f.write('\n\nprocessing_params:\n' + pprint.pformat(processing_params, indent=4))
        f.write('\n\nplot_params:\n' + pprint.pformat(plot_params, indent=4))

    list_of_exps_names = get_names(exp_batch)
    list_of_experiments = [experiment.split('.')[-2] for experiment in list_of_experiments]

    # The all metrics is a dictionary containing all the computed metrics for each pair
    # validation/drive to be computed.
    all_metrics = {}
    # Lets cache the data to improve read speed # TODO cache

    # TODO: add some loadbar here.
    for experiment in list_of_experiments:

        # The only thing that matters
        for validation, drive in data_params['validation_driving_pairs'].items():
            print('\n === Experiment %s _ %s %s ===\n' % (list_of_exps_names[experiment+'.yaml']
                                                          , validation, drive))
            print('\n ** Reading the data **\n')
            # this reads the data and infers the masks (or offsets) for different cameras
            data = read_data(exp_batch, experiment, validation, drive, data_params)
            if data is None: # This folder did not work out, probably is missing important data
                print('\n ** Missing Data on Folder **\n')
                continue

            # Print data
            logging.debug(" %s_%s " % (validation, drive))
            for step, data_item in data['values'].items():
                logging.debug(" %d" % step)
                for k, v in data_item.items():
                    logging.debug("%s %d" % (k, len(v)))
            print('\n ** Processing the data **\n')
            computed_metrics = process_data(data, processing_params, validation)
            # Compute metrics from the data.
            # Can be multiple metrics, given by the processing_params list.
            # Should be vectorized as much as possible.
            # The output is a list of the same size as processing_params.
            metrics_dict = {experiment + ' : ' + list_of_exps_names[experiment+'.yaml']:
                            computed_metrics}
            if validation + '_' + drive in all_metrics:
                all_metrics[validation + '_' + drive].update(metrics_dict)
            else:
                all_metrics.update({validation + '_' + drive: metrics_dict})
            print (all_metrics)

    # Plot the results

    for validation, drive in data_params['validation_driving_pairs'].items():

        with open(os.path.join(out_path, 'all_metrics' + validation + '_' + drive + '.json'), 'w') as fo:

            fo.write(json.dumps(all_metrics[validation + '_' + drive]))

        print('\n === Plotting the results ===\n')
        for plot_label, plot_param in plot_params.items():
            print(plot_param)
            print('\n ** Plotting %s **' % plot_label)
            if 'analysis' in plot_param:
                scatter_plotter.plot_analysis(all_metrics[validation + '_' + drive], plot_param,
                                              out_file=os.path.join(out_path,
                                                    plot_label + validation + '_' + drive + '.pdf'))
            else:
                scatter_plotter.plot(all_metrics[validation + '_' + drive], plot_param,
                                     out_file=os.path.join(out_path,
                                                    plot_label + validation + '_' + drive + '.pdf'))


