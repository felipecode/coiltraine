import collections

data_params = {'control': '_auto', 'root_path': '_logs', 'towns': ['Town01', 'Town02'],
               'noise': @NOISE@,
               'drive_environments': {'Town01': 'ECCVTrainingSuite',
                                      'Town02': 'ECCVGeneralizationSuite'}}   # some parameters for which data to read. May or may not be needed (maybe we always read all data, and filter afterwards?)


data_filter = {@CAMERA@}
processing_params = {'Success rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                     'Average completion':   {'metric': 'control_average_completion', 'filter': {}, 'params': {}},
                     'Km per infraction': {'metric': 'km_per_infraction', 'filter': {}, 'params': {}},
                     'Steering absolute error': {'metric': 'steering_error', 'filter': data_filter,
                                                 'params': {}},
                     'Steering MSE': {'metric': 'steering_avg_mse', 'filter': data_filter,
                                      'params': {}},
                     'Classification error @ 0.001': {'metric': 'steering_classification_error',
                                                 'filter': data_filter,
                                                 'params': {'threshold': 0.001}},
                     'Classification error @ 0.002': {'metric': 'steering_classification_error',
                                                 'filter': data_filter,
                                                 'params': {'threshold': 0.002}},
                     'Classification error @ 0.005': {'metric': 'steering_classification_error',
                                                 'filter': data_filter,
                                                 'params': {'threshold': 0.005}},
                     'Classification error @ 0.01': {'metric': 'steering_classification_error',
                                                'filter': data_filter,
                                                'params': {'threshold': 0.01}},
                     'Classification error @ 0.03': {'metric': 'steering_classification_error',
                                                'filter': data_filter,
                                                'params': {'threshold': 0.03}},
                     # 'Steering absolute error gt > 0.001': {'metric': 'steering_error_filter_gt',
                     #                                        'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.001)}},
                     # 'Steering absolute error gt > 0.01': {'metric': 'steering_error_filter_gt',
                     #                                       'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.01)}},
                     # 'Steering absolute error gt > 0.03': {'metric': 'steering_error_filter_gt',
                     #                                       'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.03)}},
                     # 'Steering absolute error gt > 0.05': {'metric': 'steering_error_filter_gt',
                     #                                       'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.05)}},
                     # 'Steering absolute error gt > 0.1': {'metric': 'steering_error_filter_gt',
                     #                                      'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.1)}},
                     # 'Steering absolute error gt > 0.2': {'metric': 'steering_error_filter_gt',
                     #                                      'filter': data_filter, 'params': {
                     #         'gt_condition': (lambda x: abs(x) > 0.2)}},
                     # 'Steering absolute error gt > 0.001': {'metric': 'steering_error_filter_gt',
                     #                                        'filter': data_filter, 'params': {
                     #        'gt_condition': (lambda x: abs(x) > 0.001)}},

                     'Steering MSE, GT > 0.001': {'metric': 'steering_avg_mse_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.001)}},
                     'Steering MSE, GT > 0.01': {'metric': 'steering_avg_mse_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.01)}},
                     'Steering MSE, GT > 0.03': {'metric': 'steering_avg_mse_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.03)}},
                     'Steering MSE, GT > 0.05': {'metric': 'steering_avg_mse_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.05)}},
                     'Steering MSE, GT > 0.1': {'metric': 'steering_avg_mse_filter_gt',
                                                          'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.1)}},
                     'Steering MSE, GT > 0.2': {'metric': 'steering_avg_mse_filter_gt',
                                                          'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.2)}},

                     'Speed-weighted error': {'metric': 'displacement', 'filter': data_filter,
                                      'params': {'aggregate': {'type': 'mean'}}},
                     'Speed-weighted error 70th percentile': {'metric': 'displacement',
                                                      'filter': data_filter, 'params': {
                             'aggregate': {'type': 'percentile', 'percentile': 70}}},
                     'Speed-weighted error 90th percentile': {'metric': 'displacement',
                                                      'filter': data_filter, 'params': {
                             'aggregate': {'type': 'percentile', 'percentile': 90}}},
                     'Thresholded relative error @ 0.05': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.05)}}},
                     'Thresholded relative error @ 0.05 (Felipe)': {'metric': 'count_errors_weighted',
                                                         'filter': data_filter,
                                                         'params': {'coeff': 0.05}},
                     'Thresholded relative error @ 0.03': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.03)}}},
                     'Thresholded relative error @ 0.01': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.01)}}},
                     'Thresholded relative error @ 0.1': {'metric': 'relative_error_smoothed',
                                                        'filter': data_filter,
                                                        'params': {'steer_smooth': 1e-5,
                                                                   'aggregate': {'type': 'count',
                                                                                 'condition': (
                                                                                 lambda
                                                                                     x: x > 0.1)}}},
                     'Cumulative error, 16 steps': {'metric': 'cumulative_displacement',
                                                          'filter': data_filter,
                                                          'params': {'window': 16, 'timestep': 0.1,
                                                                     'aggregate': {
                                                                         'type': 'mean'}}},
                     'Cumulative error, 64 steps': {'metric': 'cumulative_displacement',
                                                          'filter': data_filter,
                                                          'params': {'window': 16, 'timestep': 0.1,
                                                                     'aggregate': {
                                                                         'type': 'mean'}}},
                     'step': {'metric': 'step', 'filter': {}, 'params': {}},
                     'town_id': {'metric': 'id', 'filter': {}, 'params': {}},
                     'exp': {'metric': 'experiment', 'filter': {}, 'params': {}}
                     }

plot_params = collections.OrderedDict()

plot_params['infr_vs_success'] = {'print': True,
                                  'x': {'data': 'Success rate', 'log': False},
                                  'y': {'data': 'Km per infraction', 'log': True},
                                  'size': {'data': 'step'},
                                  'color': {'data': 'town_id'}
                                  }

plot_params['infr_vs_avg_comp'] = {'print': True,
                                   'x': {'data': 'Average completion', 'log': False},
                                   'y': {'data': 'Km per infraction', 'log': True},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['success_vs_avg_compl'] = {'print': True,
                                       'x': {'data': 'Average completion', 'log': False},
                                       'y': {'data': 'Success rate', 'log': False},
                                       'size': {'data': 'step'},
                                       'color': {'data': 'town_id'}
                                       }

plot_params['ctrl_vs_steer'] = {'print': True,
                                'x': {'data': 'Steering absolute error', 'log': True},
                                'y': {'data': 'Success rate', 'log': False},
                                'size': {'data': 'step'},
                                'color': {'data': 'town_id'}
                                }


plot_params['ctrl_vs_steer_mse'] = {'print': True,
                                    'x': {'data': 'Steering MSE', 'log': True},
                                    'y': {'data': 'Success rate', 'log': False},
                                    'size': {'data': 'step'},
                                    'color': {'data': 'town_id'}
                                    }


plot_params['ctrl_vs_displ'] = {'print': True,
                                'x': {'data': 'Speed-weighted error', 'log': True},
                                'y': {'data': 'Success rate', 'log': False},
                                'size': {'data': 'step'},
                                'color': {'data': 'town_id'}
                                }

plot_params['ctrl_vs_displ70'] = {'print': True,
                                  'x': {'data': 'Speed-weighted error 70th percentile', 'log': True},
                                  'y': {'data': 'Success rate', 'log': False},
                                  'size': {'data': 'step'},
                                  'color': {'data': 'town_id'}
                                  }

plot_params['ctrl_vs_displ90'] = {'print': True,
                                  'x': {'data': 'Speed-weighted error 90th percentile', 'log': True},
                                  'y': {'data': 'Success rate', 'log': False},
                                  'size': {'data': 'step'},
                                  'color': {'data': 'town_id'}
                                  }

plot_params['ctrl_vs_count_alexey_005'] = {'print': True,
                                           'x': {'data': 'Thresholded relative error @ 0.05',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_003'] = {'print': True,
                                           'x': {'data': 'Thresholded relative error @ 0.03',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_001'] = {'print': True,
                                           'x': {'data': 'Thresholded relative error @ 0.01',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_01'] = {'print': True,
                                          'x': {'data': 'Thresholded relative error @ 0.1',
                                                'log': True},
                                          'y': {'data': 'Success rate', 'log': False},
                                          'size': {'data': 'step'},
                                          'color': {'data': 'town_id'}
                                          }

plot_params['ctrl_vs_cum_disp64'] = {'print': True,
                                     'x': {'data': 'Cumulative error, 64 steps', 'log': True},
                                     'y': {'data': 'Success rate', 'log': False},
                                     'size': {'data': 'step'},
                                     'color': {'data': 'town_id'}
                                     }

plot_params['ctrl_vs_cum_disp16'] = {'print': True,
                                     'x': {'data': 'Cumulative error, 16 steps', 'log': True},
                                     'y': {'data': 'Success rate', 'log': False},
                                     'size': {'data': 'step'},
                                     'color': {'data': 'town_id'}
                                     }

plot_params['ctrl_vs_steer_mse_gt_0001'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.001', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_steer_mse_gt_001'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.01', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_steer_mse_gt_003'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.03', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_steer_mse_gt_005'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.05', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_steer_mse_gt_01'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.1', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_steer_mse_gt_02'] = {'print': True,
                                   'x': {'data': 'Steering MSE, GT > 0.2', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_classif_error_0001'] = {'print': True,
                                   'x': {'data': 'Classification error @ 0.001', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_classif_error_0002'] = {'print': True,
                                   'x': {'data': 'Classification error @ 0.002', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_classif_error_0005'] = {'print': True,
                                   'x': {'data': 'Classification error @ 0.005', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_classif_error_001'] = {'print': True,
                                   'x': {'data': 'Classification error @ 0.01', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

plot_params['ctrl_vs_classif_error_003'] = {'print': True,
                                   'x': {'data': 'Classification error @ 0.03', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'}
                                   }

for key in plot_params:
    plot_params[key]['plot_best_n_percent'] = @BEST_N_PERCENT@
