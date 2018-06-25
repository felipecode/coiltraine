

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from utils import AttributeDict
import copy
import numpy as np
import os
import os.path as osp
import yaml


from configs.namer import generate_name
from logger.coil_logger import create_log, add_message



# TODO: NAMing conventions ?

_g_conf = AttributeDict()



"""#### GENERAL CONFIGURATION PARAMETERS ####"""
_g_conf.NUMBER_OF_LOADING_WORKERS = 12
_g_conf.SENSORS = {'rgb': (3, 88, 200)}
_g_conf.MEASUREMENTS = {'targets': (31)}
_g_conf.TARGETS = ['steer', 'throttle', 'brake']
_g_conf.INPUTS = ['speed_module']
_g_conf.BALANCE_DATA = True
_g_conf.STEERING_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
#_g_conf.STEERING_DIVISION = [0.01, 0.02, 0.07, 0.4, 0.4, 0.07, 0.02, 0.01]  # Forcing curves alot
_g_conf.LABELS_DIVISION = [[0, 2, 5], [3], [4]]
_g_conf.BATCH_SIZE = 120

#_g_conf.AUGMENTATION_SUITE = [iag.ToGPU()]#, iag.Add((0, 0)), iag.Dropout(0, 0), iag.Multiply((1, 1.04)),
#                             #iag.GaussianBlur(sigma=(0.0, 3.0)),
#                             iag.ContrastNormalization((0.5, 1.5))
#                             ]


_g_conf.AUGMENTATION = None


#_g_conf.AUGMENTATION_SUITE_CPU = [ ia.Add((0, 0)), ia.Dropout(0, 0),
#                              ia.GaussianBlur(sigma=(0.0, 3.0)),
#                              ia.ContrastNormalization((0.5, 1.5))
#                              ]
_g_conf.DATA_USED = 'all' #  central, all, sides,
_g_conf.USE_NOISE_DATA = True
_g_conf.TRAIN_DATASET_NAME = '1HoursW1-3-6-8'  # We only set the dataset in configuration for training

_g_conf.LOG_SCALAR_WRITING_FREQUENCY = 2   # TODO NEEDS TO BE TESTED ON THE LOGGING FUNCTION ON  CREATE LOG
_g_conf.LOG_IMAGE_WRITING_FREQUENCY = 1000

_g_conf.EXPERIMENT_BATCH_NAME = "eccv"
_g_conf.EXPERIMENT_NAME = "default"
_g_conf.EXPERIMENT_GENERATED_NAME = None
# TODO: not necessarily the configuration need to know about this
_g_conf.PROCESS_NAME = "None"
_g_conf.NUMBER_ITERATIONS = 2000
_g_conf.SAVE_SCHEDULE = range(0, 2000, 200)
_g_conf.NUMBER_FRAMES_FUSION = 1
_g_conf.NUMBER_IMAGES_SEQUENCE = 1
_g_conf.SEQUENCE_STRIDE = 1
_g_conf.TEST_SCHEDULE = range(0, 2000, 200)
_g_conf.SPEED_FACTOR = 40.0
_g_conf.AUGMENT_LATERAL_STEERINGS = 6


"""#### Network Related Parameters ####"""


_g_conf.MODEL_TYPE = 'coil_icra'
_g_conf.MODEL_CONFIGURATION = {}


_g_conf.LEARNING_RATE_DECAY_INTERVAL = 50000
_g_conf.LEARNING_RATE_DECAY_LEVEL = 0.5
#TODO check how to use this part

_g_conf.LEARNING_RATE = 0.0002  # First
_g_conf.BRANCH_LOSS_WEIGHT = [0.95, 0.95, 0.95, 0.95, 0.05]
_g_conf.VARIABLE_WEIGHT = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
_g_conf.LOSS_FUNCTION = 'MSE'




"""#### Simulation Related Parameters ####"""

_g_conf.IMAGE_CUT = [115, 510]  # How you should cut the input image that is received from the server
_g_conf.USE_ORACLE = True


def _check_integrity():


    pass



def merge_with_yaml(yaml_filename):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:

        yaml_file = yaml.load(f)

        yaml_cfg = AttributeDict(yaml_file)



    _merge_a_into_b(yaml_cfg, _g_conf)


    path_parts = os.path.split(yaml_filename)
    _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
    _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    _g_conf.EXPERIMENT_GENERATED_NAME = generate_name(_g_conf)


def get_names(folder):
    #

    alias_in_folder = os.listdir(os.path.join('configs',folder))

    experiments_in_folder = []

    for experiment_alias in alias_in_folder:
        g_conf.immutable(False)
        print (experiment_alias)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.append(g_conf.EXPERIMENT_GENERATED_NAME)

    return experiments_in_folder


# TODO: Make this nicer, now it receives only one parameter
def set_type_of_process(process_type, param=None):
    """
    This function is used to set which is the type of the current process, test, train or val
    and also the details of each since there could be many vals and tests for a single
    experiment.

    NOTE: AFTER CALLING THIS FUNCTION, THE CONFIGURATION CLOSES

    Args:
        type:

    Returns:

    """

    if _g_conf.PROCESS_NAME == "default":
        raise RuntimeError(" You should merge with some exp file before setting the type")

    if process_type == 'train':
        _g_conf.PROCESS_NAME = process_type
    elif process_type == "validation":
        _g_conf.PROCESS_NAME = process_type + '_' + param
    if process_type == "drive":  # FOR drive param is city name.
        _g_conf.CITY_NAME = param.split('_')[-1]
        _g_conf.PROCESS_NAME = process_type + '_' + param

    #else:  # FOr the test case we join with the name of the experimental suite.

    create_log(_g_conf.EXPERIMENT_BATCH_NAME,
               _g_conf.EXPERIMENT_NAME,
               _g_conf.PROCESS_NAME,
               _g_conf.LOG_SCALAR_WRITING_FREQUENCY,
               _g_conf.LOG_IMAGE_WRITING_FREQUENCY)

    if process_type == "train":
        if not os.path.exists(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                            _g_conf.EXPERIMENT_NAME,
                                            'checkpoints') ):
                os.mkdir(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                      _g_conf.EXPERIMENT_NAME,
                                      'checkpoints'))




    if process_type == "validation" or process_type == 'drive':
        if not os.path.exists(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                           _g_conf.EXPERIMENT_NAME,
                                           _g_conf.PROCESS_NAME + '_csv')):
            os.mkdir(os.path.join('_logs', _g_conf.EXPERIMENT_BATCH_NAME,
                                          _g_conf.EXPERIMENT_NAME,
                                           _g_conf.PROCESS_NAME + '_csv'))



    # We assure ourselves that the configuration file added does not kill things
    _check_integrity()



    add_message('Loading', {'ProcessName': _g_conf.EXPERIMENT_GENERATED_NAME,
                            'FullConfiguration': generate_param_dict()})


    _g_conf.immutable(True)





def merge_with_parameters():
    pass



def generate_param_dict():
    # TODO IMPLEMENT ! generate a cool param dictionary USE
    # https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    return _g_conf.TRAIN_DATASET_NAME + 'dict'





def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """

    assert isinstance(a, AttributeDict) or isinstance(a, dict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), 'Argument `b` must be an AttrDict'


    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if is it more than second stack
            if stack is not None:
                b[k] = v_
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts

        if isinstance(v, dict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v



def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects


    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, type(None)):
        value_a = value_a
    elif isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range) and not isinstance(value_a, list):
        value_a = eval(value_a)
    elif isinstance(value_b, range) and isinstance(value_a, list):
        value_a = list(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a





g_conf = _g_conf

