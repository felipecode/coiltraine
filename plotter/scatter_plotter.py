
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors





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


def plot_analysis(all_metrics, plot_param, out_file = None):

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
    c_norm = colors.Normalize(vmin=0, vmax=50)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

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
        ax.set_xticklabels(['%.1e' % np.power(10, float(t)) for t in ax.get_xticks()])
    if plot_param['y']['log']:
        y_label += ' (log)'
        ax.set_yticklabels(['%.1e' % np.power(10, float(t)) for t in ax.get_yticks()])
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
        color_val = scalar_map.to_rgba(hash(experiment) % 50)
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


def plot(all_metrics, plot_param, out_file=None):
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

    if 'plot_best_n_percent' in plot_param and plot_param['plot_best_n_percent']:
        sorting_indices = np.argsort(data['x'])
        selected_indices = sorting_indices[:int(plot_param['plot_best_n_percent']/100.*len(sorting_indices))]
        for key in data:
            data[key] = data[key][selected_indices]

    # take log if needed
    data_x = np.log(data['x'])/np.log(10) if plot_param['x']['log'] else np.copy(data['x'])
    data_y = np.log(data['y'])/np.log(10) if plot_param['y']['log'] else np.copy(data['y'])


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
    fig, ax = plt.subplots(figsize=(8, 8))

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
        ax.set_xticklabels(['%.3f' % np.power(10, float(t)) for t in ax.get_xticks()])
    if plot_param['y']['log']:
        y_label += ' (log)'
        ax.set_yticklabels(['%.2f' % np.power(10, float(t)) for t in ax.get_yticks()])
    else:    
        ax.set_yticklabels(['%.2f' % float(t) for t in ax.get_yticks()])
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
    # NOTE removed by Alexey for the paper 
    # plt.title(title + '\ncorrelation %.2f' % corr)
    plt.title('Correlation %.2f' % corr)

    # Save to out_file
    if plot_param['print']:
        fig.savefig(out_file, bbox_inches='tight')
