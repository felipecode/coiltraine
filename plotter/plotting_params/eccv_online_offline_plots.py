import collections

data_params = {'control': '_auto', 'root_path': '_logs', 'towns': ['Town01', 'Town02'],
               'validation_driving_pairs': {'Town01W1': 'ECCVTrainingSuite_Town01',
                                            'Town01W1Noise': 'ECCVTrainingSuite_Town01',
                                            'Town02W14': 'ECCVGeneralizationSuite_Town02',
                                            'Town02W14Noise': 'ECCVGeneralizationSuite_Town02'},

               }

data_filter = {}
# The parameters processed that are going to be used for after plotting
processing_params = {'Success rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                     'Average completion':   {'metric': 'control_average_completion', 'filter': {}, 'params': {}},
                     'Km per infraction': {'metric': 'km_per_infraction', 'filter': {}, 'params': {}},
                     'Steering absolute error': {'metric': 'steering_error', 'filter': data_filter,
                                                 'params': {}},
                     'Steering MSE': {'metric': 'steering_avg_mse', 'filter': data_filter,
                                      'params': {}},

                     'Displacement': {'metric': 'displacement', 'filter': data_filter,
                                      'params': {'aggregate': {'type': 'mean'}}},


                     'Counting realtive 0.1 (Felipe)': {'metric': 'count_errors_weighted',
                                                         'filter': data_filter,
                                                         'params': {'coeff': 0.1}},
                     'Counting realtive 0.03 (Alexey)': {'metric': 'relative_error_smoothed',
                                                         'filter': data_filter,
                                                         'params': {'steer_smooth': 1e-5,
                                                                    'aggregate': {'type': 'count',
                                                                                  'condition': (
                                                                                  lambda
                                                                                      x: x > 0.03)}}},

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



#### Definition of the plots that are going to be made



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

plot_params['ctrl_vs_steer_50'] = {'print': True,
                                   'x': {'data': 'Steering absolute error', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'},
                                   'plot_best_n_percent': 50
                                  }

plot_params['ctrl_vs_steer_mse'] = {'print': True,
                                    'x': {'data': 'Steering MSE', 'log': True},
                                    'y': {'data': 'Success rate', 'log': False},
                                    'size': {'data': 'step'},
                                    'color': {'data': 'town_id'}
                                    }
plot_params['ctrl_vs_steer_mse_50'] = {'print': True,
                                    'x': {'data': 'Steering MSE', 'log': True},
                                    'y': {'data': 'Success rate', 'log': False},
                                    'size': {'data': 'step'},
                                    'color': {'data': 'town_id'},
                                    'plot_best_n_percent': 50
                                    }

plot_params['ctrl_vs_displ'] = {'print': True,
                                'x': {'data': 'Displacement', 'log': True},
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



plot_params['ctrl_vs_count_felipe_01'] = {'print': True,
                                           'x': {'data': 'Counting realtive 0.1 (Felipe)',
                                                 'log': True},
                                           'y': {'data': 'Success rate', 'log': False},
                                           'size': {'data': 'step'},
                                           'color': {'data': 'town_id'}
                                           }
