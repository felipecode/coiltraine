import collections

list_of_experiments = [
'25_small_ndrop_single_ctrl_bal_regr_all',
'25_nor_ndrop_single_ctrl_bal_regr_all',
'25_deep_ndrop_single_ctrl_bal_regr_all',
'25_nor_ndrop_lstm_ctrl_bal_regr_all'
]

data_params = {'control': '_auto','root_path': 'eccv_results' ,'towns': [] }
processing_params = {'Success rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                   'Average completion':   {'metric': 'control_average_completion', 'filter': {}, 'params': {}},
                   'Km per infraction': {'metric': 'km_per_infraction', 'filter': {}, 'params': {}},
                   'Steering absolute error': {'metric': 'steering_error', 'filter': {'camera': 'central'}, 'params': {}},
                   'Steering absolute error three cameras': {'metric': 'steering_error', 'filter': {}, 'params': {}},
                   'step': {'metric': 'step', 'filter': {}, 'params': {}},
                   'town_id': {'metric': 'id', 'filter': {}, 'params': {}},
                   'exp': {'metric': 'experiment', 'filter': {}, 'params': {}}
                   }


plot_params = collections.OrderedDict()

plot_params['analysis_architecture_ctrl_vs_steer'] = {'print': True,
                                  'title': 'Model architecture',
                                   'x': {'data': 'Steering absolute error', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'},
                                   'analysis': 123,
                                   'x_lim': [0.007,0.3],
                                   'y_lim': [-0.05, 1.]
                                   }

plot_params['analysis_architecture_ctrl_vs_steer_3cam'] = {'print': True,
                                    'title': 'Model architecture',
                                   'x': {'data': 'Steering absolute error three cameras', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'},
                                   'analysis': 123,
                                   'x_lim': [0.007,0.3],
                                   'y_lim': [-0.05, 1.]
                                   }

plot_params['analysis_architecture_avg_vs_steer'] = {'print': True,
                                  'title': 'Model architecture',
                                   'x': {'data': 'Steering absolute error', 'log': True},
                                   'y': {'data': 'Average completion', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'},
                                   'analysis': 123,
                                   'x_lim': [0.007,0.3],
                                   'y_lim': [-0.2, 1.]
                                   }

plot_params['analysis_architecture_avg_vs_steer_3cam'] = {'print': True,
                                    'title': 'Model architecture',
                                   'x': {'data': 'Steering absolute error three cameras', 'log': True},
                                   'y': {'data': 'Average completion', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'},
                                   'analysis': 123,
                                   'x_lim': [0.007,0.3],
                                   'y_lim': [-0.2, 1.]
                                   }

plot_params['ctrl_vs_steer'] = {'print': True,
                                   'x': {'data': 'Steering absolute error', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'}
                                   }

plot_params['ctrl_vs_steer_3cam'] = {'print': True,
                                   'x': {'data': 'Steering absolute error three cameras', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'exp'}
                                   }
