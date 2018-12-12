import collections

## Set the experiments taht are going to be used to compute the plot
list_of_experiments = ['experiment_64.yaml', 'experiment_67.yaml', 'experiment_68.yaml']
# Set the output validation and driving data
# that is going to be read from each of the experiments
# The plots are made correlating prediction (offline) with driving (online).
# With this the user must define the pairs that are going to be correlated.
# The pairs are in the form ValidationDataset: driving benchmark. The
# validation dataset must exist on the COIL_DATASET_PATH
data_params = {'control': '_auto', 'root_path': '_logs',
               'validation_driving_pairs': {'Town01W1': 'ECCVTrainingSuite_Town01',
                                            'Town01W1Noise': 'ECCVTrainingSuite_Town01',
                                            'Town02W14': 'ECCVGeneralizationSuite_Town02',
                                            'Town02W14Noise': 'ECCVGeneralizationSuite_Town02'},

               }
# There is not data filter
data_filter = {}
# The parameters processed that are going to be used for after plotting
processing_params = {'Success rate':   {'metric': 'control_success_rate', 'filter': {}, 'params': {}},
                     'Steering absolute error': {'metric': 'steering_error', 'filter': data_filter,
                                                 'params': {}},
                     'step': {'metric': 'step', 'filter': {}, 'params': {}},
                     'town_id': {'metric': 'id', 'filter': {}, 'params': {}},
                     'exp': {'metric': 'experiment', 'filter': {}, 'params': {}}
                     }

plot_params = collections.OrderedDict()



#### Definition of the plots that are going to be made

plot_params['ctrl_vs_steer_50'] = {'print': True,
                                   'x': {'data': 'Steering absolute error', 'log': True},
                                   'y': {'data': 'Success rate', 'log': False},
                                   'size': {'data': 'step'},
                                   'color': {'data': 'town_id'},
                                   'plot_best_n_percent': 50
                                  }

