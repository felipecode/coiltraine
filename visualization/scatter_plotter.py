
import numpy as np


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import math
import os
import time

import traceback
import matplotlib.patches as mpatches
from . import data_reading
from . import metrics_module
import pprint
import collections
import matplotlib.cm as cmx
import matplotlib.colors as colors


root_path = 'eccv_results'

camera_labels_1 = np.array(map(int, map(float, open('camera_label_file_Town01_1.txt'))))
camera_labels_1_noise = np.array(map(int, map(float, open('camera_label_file_Town01_1_noise.txt'))))
camera_labels_2 = np.array(map(int, map(float, open('camera_label_file_Town02_14.txt'))))
camera_labels_2_noise = np.array(map(int, map(float, open('camera_label_file_Town02_14_noise.txt'))))


def read_data(experiment, town, noise, data_params):

    #the town path
    full_path = os.path.join(data_params['root_path'],experiment,experiment + '_' + town + noise)
    # read the data
    data = {}

    data['town'] = town
    data['experiment'] = experiment
    values = data_reading._read_town_data(full_path, data_params['control'])
    if values is None:
        return None
    else:
        data['values'] = values

    return data

def filter_data(data, filter_param, noise):
    if filter_param:
        list_cameras = {'Town01_1': 'camera_label_file_Town01_1.txt', 'Town02_14': 'camera_label_file_Town02_14.txt'}
        if 'camera' in filter_param:
            # prepare the mask
            camera_name_to_label = {'central': 1, 'left': 0, 'right': 2}

            if data['town'] == 'Town01_1' and noise == '_noise':
                camera_labels = camera_labels_1_noise
            elif data['town'] == 'Town01_1':
                camera_labels = camera_labels_1
            elif data['town'] == 'Town02_14' and noise == '_noise':
                camera_labels = camera_labels_2_noise
            else:
                camera_labels = camera_labels_2

            mask = np.where(camera_labels == camera_name_to_label[filter_param['camera']])

            # actually filter
            keys_to_filter = ['speed_input', 'steer_gt', 'steer_pred', 'steer_error']
            data_filtered = {}
            data_filtered['town'] = data['town']
            data_filtered['values'] = collections.OrderedDict()
            for step, values_item in data['values'].items():
                data_filtered['values'][step] = {}
                for key in keys_to_filter:
                    data_filtered['values'][step][key] = values_item[key][mask]
    else:
        data_filtered = data
    return data_filtered

# TODO implement. Returns a list? Or a numpy array? With the length equal to the number_of_iterations*2 (train and test)?

def compute_lims(data_x, data_y):
    x_std = max(np.std(data_x), 0.001)
    y_std = max(np.std(data_y), 0.001)
    margin = 0.3
    x_max = np.max(data_x) + margin*x_std
    x_min = np.min(data_x) - margin*x_std
    y_max = np.max(data_y) + margin*y_std
    y_min = np.min(data_y) - margin*y_std
    spread_ratio = ((x_max - x_min)/x_std) / ((y_max-y_min)/y_std)
    if spread_ratio > 1:
        extra = (spread_ratio-1) * (y_max-y_min) / 2
        y_min -= extra
        y_max += extra
    else:
        extra = (1/spread_ratio-1) * (x_max-x_min) / 2
        x_min -= extra
        x_max += extra
    assert(np.abs(((x_max - x_min)/x_std) / ((y_max-y_min)/y_std)) < 1.0001)
    return [x_min, x_max], [y_min, y_max]

def compute_metric(metric_name, data, param):
    metric_func = getattr(metrics_module, 'compute_' + metric_name)
    if metric_name in ['id', 'step','experiment']:
        metric_results = metric_func(data, param)
    else:
        metric_results = metrics_module.compute_and_aggregate(metric_func, data, param)
    return metric_results

def process_data(data, processing_params,noise):
    metrics = {}

    for metric_label,metric_param in processing_params.items():
        data_filtered = filter_data(data, metric_param['filter'],noise)
        results = compute_metric(metric_param['metric'], data_filtered, metric_param['params'])
        metrics[metric_label] = results

    return metrics

def make_scatter_plot_analysis(all_metrics, plot_param, out_file = None):
    if 'Regularization' in plot_param['title']:
        model_to_legend = {'25_nor_no_single_ctrl_bal_regr_all': 'No regularization', '25_nor_ndrop_single_ctrl_bal_regr_all': 'Dropout', '25_nor_saug_single_ctrl_bal_regr_all': 'Dropout + mild aug.', '25_nor_maug_single_ctrl_bal_regr_all': 'Dropout + heavy aug.'}
        model_to_id = {'25_nor_no_single_ctrl_bal_regr_all': 1, '25_nor_ndrop_single_ctrl_bal_regr_all': 2, '25_nor_saug_single_ctrl_bal_regr_all': 3, '25_nor_maug_single_ctrl_bal_regr_all': 4}
    elif 'Data distribution' in plot_param['title']:
        model_to_legend = {'25_nor_ndrop_single_ctrl_bal_regr_all': 'Three cameras with noise', '25_nor_ndrop_single_ctrl_bal_regr_jcen': 'Central camera with noise', '25_nor_ndrop_single_ctrl_bal_regr_nnjc': 'Central camera, no noise', '25_nor_ndrop_single_ctrl_seq_regr_all': 'Three cameras with noise, no balancing'}
        model_to_id = {'25_nor_ndrop_single_ctrl_bal_regr_nnjc': 1, '25_nor_ndrop_single_ctrl_bal_regr_jcen': 2, '25_nor_ndrop_single_ctrl_bal_regr_all': 3, '25_nor_ndrop_single_ctrl_seq_regr_all': 4}
    elif 'Model architecture' in plot_param['title']:
        model_to_legend = {'25_small_ndrop_single_ctrl_bal_regr_all': 'Shallow CNN', '25_nor_ndrop_single_ctrl_bal_regr_all': 'Standard CNN', '25_deep_ndrop_single_ctrl_bal_regr_all': 'Deep CNN', '25_nor_ndrop_lstm_ctrl_bal_regr_all': 'Standard LSTM'}
        model_to_id = {'25_small_ndrop_single_ctrl_bal_regr_all': 1, '25_nor_ndrop_single_ctrl_bal_regr_all': 2, '25_deep_ndrop_single_ctrl_bal_regr_all': 3, '25_nor_ndrop_lstm_ctrl_bal_regr_all': 4}
    else:
        model_to_legend = {'1_nor_maug_single_ctrl_bal_regr_all': '1 hour', '5_nor_maug_single_ctrl_bal_regr_all': '5 hours', '25_nor_maug_single_ctrl_bal_regr_all': '25 hours', '80_nor_maug_single_ctrl_bal_regr_all': '80 hours'}
        model_to_id = {'1_nor_maug_single_ctrl_bal_regr_all': 1, '5_nor_maug_single_ctrl_bal_regr_all': 2, '25_nor_maug_single_ctrl_bal_regr_all': 3, '80_nor_maug_single_ctrl_bal_regr_all': 4}

    town_to_legend = {'Town01_1': 'Town 1', 'Town02_14': 'Town 2'}
    town_to_id = {'Town01_1': 1, 'Town02_14': 2}
    def exp_to_legend_and_idx(exp):
        legend = exp
        idx = []
        for model in model_to_legend:
            if exp.startswith(model):
                legend = legend.replace(model, model_to_legend[model])
                idx.append(model_to_id[model])
                break
        for town in town_to_legend:
            if exp.endswith(town):
                legend = legend.replace(town, town_to_legend[town])
                idx.append(town_to_id[town])
                break
        legend = legend.replace('_', ', ')
        return legend, idx


    #3 Prepare the axes
    fig, ax = plt.subplots(figsize=(8,8))

    # Color map
    plt.set_cmap('jet')
    cm = plt.get_cmap()
    cNorm  = colors.Normalize(vmin=0, vmax=50)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    #  Font size
    plt.rc('font', size=16)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title

    # compute limits so that x and y scaled w.r.t. their std
    print(np.log(plot_param['x_lim']) if plot_param['x']['log'] else plot_param['x_lim'])
    ax.set_xlim(np.log(plot_param['x_lim'])/np.log(10.) if plot_param['x']['log'] else plot_param['x_lim'])
    ax.set_ylim(np.log(plot_param['y_lim'])/np.log(10.) if plot_param['y']['log'] else plot_param['y_lim'])

    # set num ticks
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=8)

    # axes labels and log scaling
    x_label = 'Steering error'
    y_label = 'Success rate'
    if plot_param['x']['log']:
        x_label += ' (log)'
        ax.set_xticklabels(['%.1e' % np.power(10,float(t)) for t in ax.get_xticks()])
    if plot_param['y']['log']:
        y_label += ' (log)'
        ax.set_yticklabels(['%.1e' % np.power(10,float(t)) for t in ax.get_yticks()])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Plotting
    scatter_handles = {}
    for experiment, metrics in all_metrics.items():
        print(experiment)
        data = {'x': [], 'y': [], 'size': [], 'color': []}
        for key in data:
            data[key] = np.array(metrics[plot_param[key]['data']])

        # remove nans
        # if np.any(np.isnan(data['x'])) or np.any(np.isnan(data['y']) or np.any(np.isinf(data['x']) or np.any(np.isinf(data['y']))
        nans = np.logical_or.reduce((np.isnan(data['x']), np.isnan(data['y']), np.isinf(data['x']), np.isinf(data['y'])))
        print('\n ** Removing %d NaNs and infs before log **' % np.sum(nans))
        for key in data:
            data[key] = data[key][np.invert(nans)]

        data_x = np.log(data['x'])/np.log(10) if plot_param['x']['log'] else np.copy(data['x'])
        data_y = np.log(data['y'])/np.log(10) if plot_param['y']['log'] else np.copy(data['y'])

        nans = np.logical_or.reduce((np.isnan(data_x), np.isnan(data_y), np.isinf(data_x), np.isinf(data_y)))
        print('\n ** Removing %d NaNs and infs after log **' % np.sum(nans))
        for key in data:
            data[key] = data[key][np.invert(nans)]
        data_x = data_x[np.invert(nans)]
        data_y = data_y[np.invert(nans)]

        # the actual plotting
        color_val = scalarMap.to_rgba(hash(experiment) % 50)
        color_vec = [color_val] * len(data_x)
        # print('color_vec', color_vec)
        # print('data[\'color\']', data['color'])
        # print(len(data_x))
        scatter_handles[experiment] = ax.scatter(data_x, data_y, s=data['size'], c=color_vec, alpha=0.5)
        ax.plot(data_x, data_y, color=color_val)

    sorted_keys = sorted(scatter_handles.keys(), key=lambda x: exp_to_legend_and_idx(x)[1])
    ax.legend([scatter_handles[k] for k in sorted_keys], [exp_to_legend_and_idx(k)[0] for k in sorted_keys])
    plt.title(plot_param['title'])

    # Save to out_file
    if plot_param['print']:
        fig.savefig(out_file, bbox_inches='tight')


    # red_patch = mpatches.Patch(color='red', label='Generalization')
    #
    # blue_patch = mpatches.Patch(color='blue', label='Training')
    #
    # #ax.set_xlabel('Steer Prediction Error for Threhold')
    # ax.legend(handles=[red_patch, blue_patch])

def make_scatter_plot(all_metrics, plot_param, out_file = None):
    # Rearrange the data
    data = {'x': [], 'y': [], 'size': [], 'color': []}
    for experiment, metrics in all_metrics.items():
        for key in data:
            data[key] += metrics[plot_param[key]['data']]

    # convert to numpy
    for key in data:
        data[key] = np.array(data[key])

    # remove nans
    nans = np.logical_or.reduce((np.isnan(data['x']), np.isnan(data['y']), np.isinf(data['x']), np.isinf(data['y'])))
    print('\n ** Removing %d NaNs and infs before log **' % np.sum(nans))
    for key in data:
        data[key] = data[key][np.invert(nans)]

    # take log if needed
    data_x = np.log(data['x'])/np.log(10) if plot_param['x']['log'] else np.copy(data['x'])
    data_y = np.log(data['y'])/np.log(10) if plot_param['y']['log'] else np.copy(data['y'])

    # Remove nans after log again - these typically come from LSTM TODO may need to make this more principled
    nans = np.logical_or.reduce((np.isnan(data_x), np.isnan(data_y), np.isinf(data_x), np.isinf(data_y)))

    print('\n ** Removing %d NaNs and infs after log **' % np.sum(nans))
    for key in data:
        # print(key, data[key])
        data[key] = data[key][np.invert(nans)]
    data_x = data_x[np.invert(nans)]
    data_y = data_y[np.invert(nans)]

    ###
    ### Plotting
    ###
    fig, ax = plt.subplots(figsize=(8,8))

    # Font size
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.set_cmap('jet')
    plt.rc('font', size=16)          # controls default text sizes
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

    # compute limits so that x and y scaled w.r.t. their std
    x_lim, y_lim = compute_lims(data_x, data_y)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # set num ticks
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=8)

    # axes labels and log scaling
    x_label = plot_param['x']['data']
    y_label = plot_param['y']['data']
    if plot_param['x']['log']:
        x_label += ' (log)'
        ax.set_xticklabels(['%.1e' % np.power(10,float(t)) for t in ax.get_xticks()])
    if plot_param['y']['log']:
        y_label += ' (log)'
        ax.set_yticklabels(['%.1e' % np.power(10,float(t)) for t in ax.get_yticks()])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # the actual plotting
    # print('data[\'color\']', data['color'])
    ax.scatter(data_x, data_y, s=data['size'], c=data['color'], alpha=0.5)

    # Correlation
    corr = np.corrcoef(data_x, data_y)[0,1]
    print('Correlation %f' % corr)
    if 'title' in plot_param:
        title = plot_param['title']
    else:
        title = plot_param['y']['data'] + ' vs ' + plot_param['x']['data']
    plt.title(title + '\ncorrelation %.2f' % corr)

    # Save to out_file
    if plot_param['print']:
        fig.savefig(out_file, bbox_inches='tight')

def plot_scatter(list_of_experiments, data_params, processing_params, plot_params, out_folder=None):

    # create a folder
    if out_folder is None:
        out_folder = time.strftime("plots_%Y_%m_%d_%H_%M_%S", time.gmtime())
    out_path = os.path.join(data_params['root_path'], 'plots', out_folder)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        raise Exception('Folder %s already exists' % out_path)

    # save the parameters
    with open(os.path.join(out_path,'params.txt'),'w') as f:
        f.write('list_of_experiments:\n' + pprint.pformat(list_of_experiments,indent=4))
        f.write('\n\ndata_params:\n' + pprint.pformat(data_params,indent=4))
        f.write('\n\nprocessing_params:\n' + pprint.pformat(processing_params,indent=4))
        f.write('\n\nplot_params:\n' + pprint.pformat(plot_params,indent=4))

    all_metrics = {}
    for experiment in list_of_experiments:

        #if '25_nor_no' in experiment or '5_small_ndrop_single_wp' in experiment:
        #    continue

        for town in data_params['towns']:
            print('\n === Experiment %s _ %s %s ===\n' % (experiment,town,data_params['noise']))
            print('\n ** Reading the data **\n')
            data = read_data(experiment,town, data_params['noise'], data_params) # this reads the data and infers the masks (or offsets) for different cameras
            if data is None: # This folder didnt work out, probably is missing important data
                print('\n ** Missing Data on Folder **\n')
                continue

            # Print data
            print(data['town'])
            for step, data_item in data['values'].items():
                print(step)
                for k,v in data_item.items():
                    print(k, len(v))
            print('\n ** Processing the data **\n')
            metrics = process_data(data, processing_params,data_params['noise']) # Compute metrics from the data. Can be multiple metrics, given by the processing_params list. Should be vectorized as much as possible. The output is a list of the same size as processing_params.
            all_metrics[experiment + '_' + town] = metrics # append to the computed list of metrics to the dictionary of results.

    with open(os.path.join(out_path,'all_metrics.txt'),'w') as f:
        f.write('all_metrics:\n' + pprint.pformat(all_metrics, indent=4))

    # Plot the results
    print('\n === Plotting the results ===\n')
    for plot_label, plot_param in plot_params.items():
        print(plot_param)
        print('\n ** Plotting %s **' % plot_label)
        if 'analysis' in plot_param:
            make_scatter_plot_analysis(all_metrics, plot_param, out_file=os.path.join(out_path, plot_label+'.pdf'))
        else:
            make_scatter_plot(all_metrics, plot_param, out_file=os.path.join(out_path, plot_label+'.pdf'))

# TODO Should we cache the computed metrics? The metrics should have unique names then
