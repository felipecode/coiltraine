import collections

data_params = {'control': '_auto', 'root_path': '_logs', 'towns': ['Town01', 'Town02'],
               'noise': True,
               'drive_environments': {'Town01': 'ECCVTrainingSuite',
                                      'Town02': 'ECCVGeneralizationSuite'}}   # some parameters for which data to read. May or may not be needed (maybe we always read all data, and filter afterwards?)


data_filter = {}
processing_params = {'Success rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                    'Average completion':   {'metric': 'control_average_completion', 'filter': {}, 'params': {}},
                    'Km per infraction': {'metric': 'km_per_infraction', 'filter': {}, 'params': {}},
                     'Steering absolute error': {'metric': 'steering_error', 'filter': data_filter,
                                                 'params': {}},
                     'Steering absolute error gt > 0.001': {'metric': 'steering_error_filter_gt',
                                                            'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.001)}},
                     'Steering absolute error gt > 0.01': {'metric': 'steering_error_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.01)}},
                     'Steering absolute error gt > 0.03': {'metric': 'steering_error_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.03)}},
                     'Steering absolute error gt > 0.05': {'metric': 'steering_error_filter_gt',
                                                           'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.05)}},
                     'Steering absolute error gt > 0.1': {'metric': 'steering_error_filter_gt',
                                                          'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.1)}},
                     'Steering absolute error gt > 0.2': {'metric': 'steering_error_filter_gt',
                                                          'filter': data_filter, 'params': {
                             'gt_condition': (lambda x: abs(x) > 0.2)}},
                     'Steering MSE': {'metric': 'steering_avg_mse', 'filter': data_filter,
                                      'params': {}},
                     'Steering accuracy 0.002': {'metric': 'steering_accuracy',
                                                 'filter': data_filter,
                                                 'params': {'threshold': 0.002}},
                     'Steering accuracy 0.005': {'metric': 'steering_accuracy',
                                                 'filter': data_filter,
                                                 'params': {'threshold': 0.005}},
                     'Steering accuracy 0.01': {'metric': 'steering_accuracy',
                                                'filter': data_filter,
                                                'params': {'threshold': 0.01}},
                     'Steering accuracy 0.03': {'metric': 'steering_accuracy',
                                                'filter': data_filter,
                                                'params': {'threshold': 0.03}},
                     'Steering accuracy 0.005 gt > 0.1': {'metric': 'steering_accuracy_filter_gt',
                                                          'filter': data_filter,
                                                          'params': {'threshold': 0.005,
                                                                     'gt_condition': (
                                                                     lambda x: abs(x) > 0.1)}},
                     'Steering accuracy 0.01 gt > 0.1': {'metric': 'steering_accuracy_filter_gt',
                                                         'filter': data_filter,
                                                         'params': {'threshold': 0.01,
                                                                    'gt_condition': (
                                                                    lambda x: abs(x) > 0.1)}},
                     'Steering accuracy 0.03 gt > 0.1': {'metric': 'steering_accuracy_filter_gt',
                                                         'filter': data_filter,
                                                         'params': {'threshold': 0.03,
                                                                    'gt_condition': (
                                                                    lambda x: abs(x) > 0.1)}},
                     'Displacement': {'metric': 'displacement', 'filter': data_filter,
                                      'params': {'aggregate': {'type': 'mean'}}},
                     'Displacement 70th percentile': {'metric': 'displacement',
                                                      'filter': data_filter, 'params': {
                             'aggregate': {'type': 'percentile', 'percentile': 70}}},
                     'Displacement 90th percentile': {'metric': 'displacement',
                                                      'filter': data_filter, 'params': {
                             'aggregate': {'type': 'percentile', 'percentile': 90}}},
                     'Counting realtive 0.05 (Alexey)': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.05)}}},
                     'Counting realtive 0.05 (Felipe)': {'metric': 'count_errors_weighted',
                                                         'filter': data_filter,
                                                         'params': {'coeff': 0.05}},
                     'Counting realtive 0.03 (Alexey)': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.03)}}},
                     'Counting realtive 0.01 (Alexey)': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.01)}}},
                     'Counting realtive 0.1 (Alexey)': {'metric': 'relative_error_smoothed',
                                                        'filter': data_filter,
                                                        'params': {'steer_smooth': 1e-5,
                                                                   'aggregate': {'type': 'count',
                                                                                 'condition': (
                                                                                 lambda
                                                                                     x: x > 0.1)}}},
                     'Cumulative displacement 16 steps': {'metric': 'cumulative_displacement',
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

plot_params['ctrl_vs_steer_gt_0001'] = {'print': True,
                                        'x': {'data': 'Steering absolute error gt > 0.001',
                                              'log': True},
                                        'y': {'data': 'Success rate', 'log': False},
                                        'size': {'data': 'step'},
                                        'color': {'data': 'town_id'}
                                        }

plot_params['ctrl_vs_steer_gt_001'] = {'print': True,
                                       'x': {'data': 'Steering absolute error gt > 0.01',
                                             'log': True},
                                       'y': {'data': 'Success rate', 'log': False},
                                       'size': {'data': 'step'},
                                       'color': {'data': 'town_id'}
                                       }

plot_params['ctrl_vs_steer_gt_003'] = {'print': True,
                                       'x': {'data': 'Steering absolute error gt > 0.03',
                                             'log': True},
                                       'y': {'data': 'Success rate', 'log': False},
                                       'size': {'data': 'step'},
                                       'color': {'data': 'town_id'}
                                       }

plot_params['ctrl_vs_steer_gt_005'] = {'print': True,
                                       'x': {'data': 'Steering absolute error gt > 0.05',
                                             'log': True},
                                       'y': {'data': 'Success rate', 'log': False},
                                       'size': {'data': 'step'},
                                       'color': {'data': 'town_id'}
                                       }

plot_params['ctrl_vs_steer_gt_01'] = {'print': True,
                                      'x': {'data': 'Steering absolute error gt > 0.1',
                                            'log': True},
                                      'y': {'data': 'Success rate', 'log': False},
                                      'size': {'data': 'step'},
                                      'color': {'data': 'town_id'}
                                      }

plot_params['ctrl_vs_steer_gt_02'] = {'print': True,
                                      'x': {'data': 'Steering absolute error gt > 0.2',
                                            'log': True},
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

plot_params['ctrl_vs_steer_acc_001'] = {'print': True,
                                        'x': {'data': 'Steering accuracy 0.01', 'log': False},
                                        'y': {'data': 'Success rate', 'log': False},
                                        'size': {'data': 'step'},
                                        'color': {'data': 'town_id'}
                                        }

plot_params['ctrl_vs_steer_acc_003'] = {'print': True,
                                        'x': {'data': 'Steering accuracy 0.03', 'log': False},
                                        'y': {'data': 'Success rate', 'log': False},
                                        'size': {'data': 'step'},
                                        'color': {'data': 'town_id'}
                                        }

plot_params['ctrl_vs_steer_acc_0005'] = {'print': True,
                                         'x': {'data': 'Steering accuracy 0.005', 'log': False},
                                         'y': {'data': 'Success rate', 'log': False},
                                         'size': {'data': 'step'},
                                         'color': {'data': 'town_id'}
                                         }

plot_params['ctrl_vs_steer_acc_0002'] = {'print': True,
                                         'x': {'data': 'Steering accuracy 0.002', 'log': False},
                                         'y': {'data': 'Success rate', 'log': False},
                                         'size': {'data': 'step'},
                                         'color': {'data': 'town_id'}
                                         }

plot_params['ctrl_vs_steer_acc_0005_gt_01'] = {'print': True,
                                               'x': {'data': 'Steering accuracy 0.005 gt > 0.1',
                                                     'log': False},
                                               'y': {'data': 'Success rate', 'log': False},
                                               'size': {'data': 'step'},
                                               'color': {'data': 'town_id'}
                                               }

plot_params['ctrl_vs_steer_acc_001_gt_01'] = {'print': True,
                                              'x': {'data': 'Steering accuracy 0.01 gt > 0.1',
                                                    'log': False},
                                              'y': {'data': 'Success rate', 'log': False},
                                              'size': {'data': 'step'},
                                              'color': {'data': 'town_id'}
                                              }

plot_params['ctrl_vs_steer_acc_003_gt_01'] = {'print': True,
                                              'x': {'data': 'Steering accuracy 0.03 gt > 0.1',
                                                    'log': False},
                                              'y': {'data': 'Success rate', 'log': False},
                                              'size': {'data': 'step'},
                                              'color': {'data': 'town_id'}
                                              }

plot_params['ctrl_vs_displ'] = {'print': True,
                                'x': {'data': 'Displacement', 'log': True},
                                'y': {'data': 'Success rate', 'log': False},
                                'size': {'data': 'step'},
                                'color': {'data': 'town_id'}
                                }

plot_params['ctrl_vs_displ70'] = {'print': True,
                                  'x': {'data': 'Displacement 70th percentile', 'log': True},
                                  'y': {'data': 'Success rate', 'log': False},
                                  'size': {'data': 'step'},
                                  'color': {'data': 'town_id'}
                                  }

plot_params['ctrl_vs_displ90'] = {'print': True,
                                  'x': {'data': 'Displacement 90th percentile', 'log': True},
                                  'y': {'data': 'Success rate', 'log': False},
                                  'size': {'data': 'step'},
                                  'color': {'data': 'town_id'}
                                  }

plot_params['ctrl_vs_count_alexey_005'] = {'print': True,
                                           'x': {'data': 'Counting realtive 0.05 (Alexey)',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_003'] = {'print': True,
                                           'x': {'data': 'Counting realtive 0.03 (Alexey)',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_001'] = {'print': True,
                                           'x': {'data': 'Counting realtive 0.01 (Alexey)',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_count_alexey_01'] = {'print': True,
                                          'x': {'data': 'Counting realtive 0.1 (Alexey)',
                                                'log': True},
                                          'y': {'data': 'Success rate', 'log': False},
                                          'size': {'data': 'step'},
                                          'color': {'data': 'town_id'}
                                          }

plot_params['ctrl_vs_count_felipe_005'] = {'print': True,
                                           'x': {'data': 'Counting realtive 0.05 (Felipe)',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }

plot_params['ctrl_vs_cum_disp16'] = {'print': True,
                                     'x': {'data': 'Cumulative displacement 16 steps', 'log': True},
                                     'y': {'data': 'Success rate', 'log': False},
                                     'size': {'data': 'step'},
                                     'color': {'data': 'town_id'}
                                     }
