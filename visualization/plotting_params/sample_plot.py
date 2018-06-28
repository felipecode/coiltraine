import collections

data_params = {'control': '_auto', 'root_path': '_logs', 'towns': ['Town01', 'Town02'],
               'drive_environments': {'Town01': 'ECCVTrainingSuite',
                                      'Town02': 'ECCVGeneralizationSuite'}}   # some parameters for which data to read. May or may not be needed (maybe we always read all data, and filter afterwards?)
processing_params = {'control_success_rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                     'control_average_completion':   {'metric': 'control_average_completion', 'filter': {}, 'params': {}},
                     'km_per_infraction': {'metric': 'km_per_infraction', 'filter': {}, 'params': {}},
                     'steer_error': {'metric': 'steering_error', 'filter': {}, 'params': {}},
                     'steer_error_central': {'metric': 'steering_error', 'filter': {'camera': 'central'}, 'params': {}},
                     'displacement': {'metric': 'displacement', 'filter': {}, 'params': {'aggregate': {'type': 'mean'}}},
                     'displacement_central': {'metric': 'displacement', 'filter': {'camera': 'central'}, 'params': {'aggregate': {'type': 'mean'}}},

                     'correlation_central': {'metric': 'correlation', 'filter': {'camera': 'central'},
                                             'params': {'thresh_steer': 0.05}},
                     'correlation': {'metric': 'correlation', 'filter': {},
                                     'params': {'thresh_steer': 0.05}},

                     'step': {'metric': 'step', 'filter': {}, 'params': {}},
                     'city_id': {'metric': 'id', 'filter': {}, 'params': {}}
                     }


plot_params = collections.OrderedDict()
# central
plot_params['infr_vs_success'] = {'print': True,
                                 'title': 'Km per Infraction vs Control Success Rate',
                                 'x': {'data': 'control_success_rate', 'log': False},
                                 'y': {'data': 'km_per_infraction', 'log': True},
                                 'size': {'data': 'step'},
                                 'color': {'data': 'city_id'}
                                 }

plot_params['infr_vs_avg_compl'] = {'print': True,
                                 'title': 'Km per Infraction vs Control Average Completion',
                                 'x': {'data': 'control_average_completion', 'log': False},
                                 'y': {'data': 'km_per_infraction', 'log': True},
                                 'size': {'data': 'step'},
                                 'color': {'data': 'city_id'}
                                 }


plot_params['ctrl_vs_steer_central'] = {'print': True,
                                     'title': 'Control vs Steering Error Central',
                                     'x': {'data': 'steer_error_central', 'log': True},
                                     'y': {'data': 'control_success_rate', 'log': False},
                                     'size': {'data': 'step'},
                                     'color': {'data': 'city_id'}
                                     }

plot_params['ctrl_vs_disp_central'] = {'print': True,
                                     'title': 'Control vs Displacement Central',
                                     'x': {'data': 'displacement_central', 'log': True, },
                                     'y': {'data': 'control_success_rate', 'log': False},
                                     'size': {'data': 'step'},
                                     'color': {'data': 'city_id'}
                                     }


plot_params['ctrl_vs_steer'] = {'print': True,
                                 'title': 'Control vs Steering Error',
                                 'x': {'data': 'steer_error', 'log': True},
                                 'y': {'data': 'control_success_rate', 'log': False},
                                 'size': {'data': 'step'},
                                 'color': {'data': 'city_id'}
                                 }

plot_params['ctrl_vs_disp'] = {'print': True,
                                    'title': 'Control vs Displacement',
                                    'x': {'data': 'displacement', 'log': True, },
                                    'y': {'data': 'control_success_rate', 'log': False},
                                    'size': {'data': 'step'},
                                    'color': {'data': 'city_id'}
                                    }

