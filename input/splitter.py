import sys
import numpy as np
import collections

import torch

from configs import g_conf
from logger import coil_logger
from coilutils.general import softmax

from .coil_sampler import PreSplittedSampler, RandomSampler


def order_sequence(steerings, keys_sequence):
    sequence_average = []

    for i in keys_sequence:
        sampled_sequence = steerings[(i):(i + g_conf.NUMBER_IMAGES_SEQUENCE)]

        sequence_average.append(sum(sampled_sequence) / len(sampled_sequence))

    # sequence_average =  get_average_over_interval_stride(steerings_train,sequence_size,stride_size)
    return [i[0] for i in sorted(enumerate(sequence_average), key=lambda x: x[1])], sequence_average


def partition_keys_by_percentiles(steerings, keys, percentiles):
    iter_index = 0
    quad_pos = 0
    splited_keys = []
    quad_vec = [percentiles[0]]
    for i in range(1, len(percentiles)):
        quad_vec.append(quad_vec[-1] + percentiles[i])

    for i in range(0, len(steerings)):

        if i >= quad_vec[quad_pos] * len(steerings) - 1:
            # We split

            splited_keys.append(keys[iter_index:i])
            if keys[iter_index:i] == []:
                raise RuntimeError("Reach into an empty bin.")
            iter_index = i
            quad_pos += 1
            # THe value of steering splitted
            # The number of keys for this split
            # print ([steerings[i], len(splited_keys)])
            coil_logger.add_message('Loading', {'SplitPoints': [steerings[i], len(splited_keys)]})

    return splited_keys


def select_data_sequence(control, selected_data):
    """
    The policy is to check if the majority of images are of a certain label.

    Args:
        control:
        selected_data:

    Returns:

        The valid keys
    """

    break_sequence = False


    count = 0
    del_pos = []

    while count * g_conf.SEQUENCE_STRIDE <= (
            len(control) - g_conf.NUMBER_IMAGES_SEQUENCE):

        # We count the number of positions not corresponding to a label
        eliminated_positions = 0
        for iter_sequence in range((count * g_conf.SEQUENCE_STRIDE),
                                   (count * g_conf.SEQUENCE_STRIDE) +
                                   g_conf.NUMBER_IMAGES_SEQUENCE):

            #print ("IMAGES SEQUENCE ", g_conf.NUMBER_IMAGES_SEQUENCE )
            # The position is one

            if control[iter_sequence] not in selected_data:
                eliminated_positions += 1


            if eliminated_positions > g_conf.NUMBER_IMAGES_SEQUENCE/2:
                del_pos.append(count * g_conf.SEQUENCE_STRIDE)
                break_sequence = True
                break

        if break_sequence:
            break_sequence = False
            count += 1
            continue

        count += 1

    return del_pos


""" Split the outputs keys with respect to the labels. 
The selected labels represents how it is going to be split """


def label_split(labels, keys, selected_data):
    """

    Args:
        labels:
        keys:
        selected_data:

    Returns:

    """

    keys_for_divison = []  # The set of all possible keys for each division
    sorted_steering_division = []
    if isinstance(selected_data, list):
        selected_data_vec = selected_data

    else:  # for this case we are doing label split based on scalar.
        if not isinstance(selected_data, int):
            raise ValueError(" Invalid type for scalar label selection")

        selected_data_vec = [[1]] + int(100/selected_data -1) * [[0]]

    for j in range(len(selected_data_vec)):

        keys_to_delete = select_data_sequence(labels, selected_data_vec[j])

        keys_for_this_part = list(set(keys) - set(keys_to_delete))
        # If it is empty, kindly ask the user to change the label division
        if not keys_for_this_part:
            raise RuntimeError("No Element found of the key ", selected_data_vec[j],
                               "please select other keys")

        keys_for_divison.append(keys_for_this_part)

    return keys_for_divison


def float_split(output_to_split, keys, percentiles):
    """
    Split data based on the the float value of some variable.
    Everything is splitted with respect to the percentages.

    Arguments :

    """


    # We use this keys to grab the steerings we want... divided into groups
    # TODO: Test the spliting based on median.
    #print ('Start keys ',keys)
    keys_ordered, average_outputs = order_sequence(output_to_split, keys)

    # we get new keys and order steering, each steering group
    sorted_outputs = [average_outputs[j] for j in keys_ordered]
    corresponding_keys = [keys[j] for j in keys_ordered]

    # We split each group...
    if len(keys_ordered) > 0:
        splitted_keys = partition_keys_by_percentiles(sorted_outputs,
                                                      corresponding_keys, percentiles)
    else:
        splitted_keys = []


    return splitted_keys


# READABILITY IS HORRIBLE


def remove_angle_traffic_lights(data, positions_dict):
    # will return all the keys that does not contain the expression.

    return (data['angle'] == positions_dict['angle'] and data['traffic_lights']!=positions_dict['traffic_lights'])


def remove_angle(data, positions_dict):
    # This will remove a list of angles that you dont want
    # Usually used to get just the central camera

    return data['angle'] == positions_dict['angle']


def remove_traffic_lights(data, positions_dict):
    # This will remove a list of angles that you dont want
    # Usually used to get just the central camera

    data = convert_measurements(data)
    keys = np.where(data['traffic_lights'] == 1)[0]
    return keys


####################### SPLITTING FUNCTIONS #########################

def split_sequence(data, var, positions):

    # positions will start as something like 3,9,17
    print (data)
    print (var)
    print (positions)
    keys = [np.where(data[var] <= positions[var][0])[0]]



    for i in range(len(positions[var])-1):
        print (data[var] )
        print ( positions[var][i], positions[var][i+1])
        keys.append(np.where(
            np.logical_and(data[var] > positions[var][i], data[var] <= positions[var][i + 1]))[0])



    keys.append(np.where(data[var] > positions[var][-1])[0])

    return keys


def convert_measurements(measurements):

    conv_measurements = dict.fromkeys(measurements[0].keys())
    conv_measurements = {key: [] for key in conv_measurements}

    for data_point in measurements:

        for key, value in data_point.items():
            conv_measurements[key].append(value)


    for key in conv_measurements.keys():
        conv_measurements[key] = np.array(conv_measurements[key])


    return conv_measurements


def split_brake(data, positions):
    data = convert_measurements(data)
    return split_sequence(data, 'brake', positions)


def split_speed_module(data, positions):
    data = convert_measurements(data)
    return split_sequence(data, 'speed_module', positions)

def split_speed_module_throttle(data, positions_dict):
    data = convert_measurements(data)
    keys = [np.where(np.logical_and(data['speed_module'] < positions_dict['speed_module'][0],
                                                           data['throttle'] > positions_dict['throttle'][0]))[0],
                         np.where(np.logical_or(np.logical_and(data['speed_module'] < positions_dict['speed_module'][0],
                                                           data['throttle'] <= positions_dict['throttle'][0]),
                                                data['speed_module'] >= positions_dict['speed_module'][0]))[0]
             ]

    return keys

def split_pedestrian_vehicle_traffic_lights_move(data, positions_dict):
    data = convert_measurements(data)
    keys = [np.where(np.logical_and(data['pedestrian'] < 1.0,
                                    data['pedestrian'] > 0.))[0],
            np.where(data['pedestrian'] == 0.)[0],
            np.where(data['vehicle'] < 1. )[0],
            np.where(np.logical_and(data['traffic_lights'] < 1.0, data['speed_module'] >= 0.0666))[0],
            np.where(np.logical_and(np.logical_and(data['pedestrian'] == 1.,
                                                   data['vehicle'] == 1.),
                                    np.logical_or(data['traffic_lights'] == 1.,
                                                   np.logical_and(data['traffic_lights'] < 1.0,
                                                                  data['speed_module'] < 0.066)
                                                  )
                                    )
                     )[0]

            ]
    return keys


def split_pedestrian_vehicle_traffic_lights(data, positions_dict):
    data = convert_measurements(data)
    keys = [np.where(np.logical_and(data['pedestrian'] < 1.0,
                                    data['pedestrian'] > 0.))[0],
            np.where(data['pedestrian'] == 0.)[0],
            np.where(data['vehicle'] < 1. )[0],
            np.where(data['traffic_lights'] < 1.0)[0],
            np.where(np.logical_and(np.logical_and(data['pedestrian'] == 1.,
                                                   data['vehicle'] == 1.),
                                     data['traffic_lights'] == 1.))[0]

            ]
    return keys

def split_lateral_noise_longitudinal_noise(data, positions_dict):
    data = convert_measurements(data)


    keys = [np.where(data['steer'] != data['steer_noise'])[0],
            np.where(np.logical_or(data['throttle'] != data['throttle_noise'],
                                   data['brake'] != data['brake_noise']))[0],
            np.where(np.logical_and(np.logical_and(data['steer'] == data['steer_noise'],
                                                   data['throttle'] == data['throttle_noise']),
                                    data['brake'] == data['brake_noise']))[0]
            ]
    return keys


def split_left_central_right(data, positions_dict):
    data = convert_measurements(data)


    keys = [np.where(data['angle'] == -30.)[0],
            np.where(data['angle'] == 0. )[0],
            np.where(data['angle'] == 30.) [0]
            ]
    return keys


##### GET the property so we can perform augmentation later.


def get_boost_pedestrian_vehicle_traffic_lights(data, key, positions_dict):

    boost = 0

    #print (data['pedestrian'][key])
    if 0 < data[key]['pedestrian'] < 1.0:
        boost += positions_dict['boost'][0]

    if data[key]['pedestrian'] == 0.:
        boost += positions_dict['boost'][1]

    if data[key]['vehicle'] < 1.:
        boost +=  positions_dict['boost'][2]

    if data[key]['pedestrian'] == 1.0 and data[key]['vehicle'] == 1. and data[key]['traffic_lights'] == 1. :
        boost += positions_dict['boost'][3]

    return boost




def parse_split_configuration(configuration):
    """
    Turns the configuration line of splitting into a name and a set of params.
    """
    if configuration is None:
        return "None", None
    conf_dict = collections.OrderedDict(configuration)

    name = 'split'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def get_inverse_freq_weights(keys, dataset_size):

    invers_freq_weights = []
    print (" frequency")
    for key_vec in keys:
        print ((len(key_vec)/dataset_size))
        invers_freq_weights.append((len(key_vec)/dataset_size))

    return softmax(np.array(invers_freq_weights))


# TODO: for now is not possible to maybe balance just labels or just steering.
# TODO: Is either all or nothing
def select_balancing_strategy(dataset, iteration, number_of_workers):

    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.

    keys = range(0, len(dataset) - g_conf.NUMBER_IMAGES_SEQUENCE)

    # In the case we are using the balancing
    if g_conf.SPLIT is not None and g_conf.SPLIT is not "None":
        name, params = parse_split_configuration(g_conf.SPLIT)
        splitter_function = getattr(sys.modules[__name__], name)
        keys_splitted = splitter_function(dataset.measurements, params)

        for i in range(len(keys_splitted)):
            keys_splitted[i] = np.array(list(set(keys_splitted[i]).intersection(set(keys))))
        if params['weights'] == 'inverse':
            weights = get_inverse_freq_weights(keys_splitted, len(dataset.measurements)
                                               - g_conf.NUMBER_IMAGES_SEQUENCE)
        else:
            weights = params['weights']
        sampler = PreSplittedSampler(keys_splitted, iteration * g_conf.BATCH_SIZE, weights)
    else:
        sampler = RandomSampler(keys, iteration * g_conf.BATCH_SIZE)

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                              sampler=sampler,
                                              num_workers=number_of_workers,
                                              pin_memory=True)
    return data_loader
