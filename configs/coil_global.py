

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


from logger.coil_logger import create_log, add_message

import imgauggpu as iag
import imgaug.augmenters as ia



# TODO: NAMing conventions ?

_g_conf = AttributeDict()



"""#### GENERAL CONFIGURATION PARAMETERS ####"""
_g_conf.NUMBER_OF_LOADING_WORKERS = 12
_g_conf.SENSORS = {'rgb': (3, 88, 200)}
_g_conf.MEASUREMENTS = {'targets': (31)}
_g_conf.TARGETS = ['steer', 'throttle', 'brake']
_g_conf.INPUTS = ['speed_module']
_g_conf.STEERING_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
#_g_conf.STEERING_DIVISION = [0.01, 0.02, 0.07, 0.4, 0.4, 0.07, 0.02, 0.01]  # Forcing curves alot
_g_conf.LABELS_DIVISION = [[0, 2, 5], [3], [4]]
_g_conf.BATCH_SIZE = 120

_g_conf.AUGMENTATION_SUITE = [iag.ToGPU()]#, iag.Add((0, 0)), iag.Dropout(0, 0), iag.Multiply((1, 1.04)),
                              #iag.GaussianBlur(sigma=(0.0, 3.0)),
 #                             iag.ContrastNormalization((0.5, 1.5))
 #                             ]
#_g_conf.AUGMENTATION_SUITE_CPU = [ ia.Add((0, 0)), ia.Dropout(0, 0),
#                              ia.GaussianBlur(sigma=(0.0, 3.0)),
#                              ia.ContrastNormalization((0.5, 1.5))
#                              ]
_g_conf.TRAIN_DATASET_NAME = '1HoursW1-3-6-8'  # We only set the dataset in configuration for training

_g_conf.LOG_SCALAR_WRITING_FREQUENCY = 2   # TODO NEEDS TO BE TESTED ON THE LOGGING FUNCTION ON  CREATE LOG
_g_conf.LOG_IMAGE_WRITING_FREQUENCY = 15

_g_conf.EXPERIMENT_BATCH_NAME = "eccv"
_g_conf.EXPERIMENT_NAME = "default"
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


_g_conf.MODEL_NAME = 'coil_icra'
_g_conf.TRAINING_SCHEDEULE = [[50000, 0.5], [100000, 0.5 * 0.5], [150000, 0.5 * 0.5 * 0.5],
                              [200000, 0.5 * 0.5 * 0.5 * 0.5],
                              [250000, 0.5 * 0.5 * 0.5 * 0.5 * 0.5]]  # Number of iterations, multiplying factor
#TODO check how to use this part

_g_conf.LEARNING_RATE = 0.0002  # First
_g_conf.BRANCH_LOSS_WEIGHT = [0.95, 0.95, 0.95, 0.95, 0.05]
_g_conf.VARIABLE_WEIGHT = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05, 'Speed': 1.0}




"""#### Simulation Related Parameters ####"""
_g_conf.CITY_NAME = 'Town01'
_g_conf.EXPERIMENTAL_SUITE_NAME = 'TestSuite'
_g_conf.IMAGE_CUT = [115, 510]  # How you should cut the input image that is received from the server
_g_conf.USE_ORACLE = False


def _check_integrity():


    pass



def merge_with_yaml(yaml_filename):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:

        yaml_file = yaml.load(f)

        yaml_cfg = AttributeDict(yaml_file)

    print ("yaml here", yaml_cfg)

    print ("batch size ", yaml_cfg.BATCH_SIZE)

    _merge_a_into_b(yaml_cfg, _g_conf)

    #TODO: Merging is missing

    path_parts = os.path.split(yaml_filename)
    _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
    _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]






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
        _g_conf.CITY_NAME = param
        _g_conf.PROCESS_NAME = process_type + '_' + _g_conf.CITY_NAME + '_' + _g_conf.EXPERIMENTAL_SUITE_NAME

    #else:  # FOr the test case we join with the name of the experimental suite.

    create_log(_g_conf.EXPERIMENT_BATCH_NAME,
               _g_conf.EXPERIMENT_NAME,
               _g_conf.PROCESS_NAME)

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

    add_message('Loading', {'ProcessName': generate_name(),
                            'FullConfiguration': generate_param_dict()})

    _g_conf.immutable(True)





def merge_with_parameters():
    pass

def generate_name():
    # TODO: Make a cool name generator, maybe in another class
    return _g_conf.TRAIN_DATASET_NAME + str(202)

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
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)

    elif isinstance(value_b, str):

        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range):
        value_a = eval(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a





g_conf = _g_conf




"""
# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead


# Miscelaneous configuration
__C.


# Configuration for training
__C.TRAIN = AttributeDict()

MISC:
  SAVE_MODEL_INTERVAL:
  BATCH_SIZE=120
  NUMBER_ITERATIONS:

LOGGING:

INPUT:
  DATASET:


TRAINING:


EVALUATION:
  NUMBER_BATCHES: 1800
  #NUMBER_IMAGES: 1800* # Number of images. ALL DERIVATED METRICS ARE COMPUTED INSIDE THE MODULUES

TEST:

"""



"""


# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set(
    {
        'FINAL_MSG',
        'MODEL.DILATION',
        'ROOT_GPU_ID',
        'RPN.ON',
        'TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED',
        'TRAIN.DROPOUT',
        'USE_GPU_NMS',
        'TEST.NUM_TEST_IMAGES',
    }
)



# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'MODEL.PS_GRID_SIZE': 'RFCN.PS_GRID_SIZE',
    'MODEL.ROI_HEAD': 'FAST_RCNN.ROI_BOX_HEAD',
    'MRCNN.MASK_HEAD_NAME': 'MRCNN.ROI_MASK_HEAD',
    'TRAIN.DATASET': (
        'TRAIN.DATASETS',
        "Also convert to a tuple, e.g., " +
        "'coco_2014_train' -> ('coco_2014_train',) or " +
        "'coco_2014_train:coco_2014_valminusminival' -> " +
        "('coco_2014_train', 'coco_2014_valminusminival')"
    ),
    'TRAIN.PROPOSAL_FILE': (
        'TRAIN.PROPOSAL_FILES',
        "Also convert to a tuple, e.g., " +
        "'path/to/file' -> ('path/to/file',) or " +
        "'path/to/file1:path/to/file2' -> " +
        "('path/to/file1', 'path/to/file2')"
    ),
    'TEST.SCALES': (
        'TEST.SCALE',
        "Also convert from a tuple, e.g. (600, ), " +
        "to a integer, e.g. 600."
    ),
    'TEST.DATASET': (
        'TEST.DATASETS',
        "Also convert from a string, e.g 'coco_2014_minival', " +
        "to a tuple, e.g. ('coco_2014_minival', )."
    ),
    'TEST.PROPOSAL_FILE': (
        'TEST.PROPOSAL_FILES',
        "Also convert from a string, e.g. '/path/to/props.pkl', " +
        "to a tuple, e.g. ('/path/to/props.pkl', )."
    ),
}

"""