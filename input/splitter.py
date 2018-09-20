from configs import g_conf
from logger import coil_logger
import numpy as np



def order_sequence(steerings, keys_sequence):
    sequence_average = []
    # print 'keys'
    #print (" NUMBER OF IMAGES IN A SEQUENCE ", g_conf.NUMBER_IMAGES_SEQUENCE)
    # print keys_sequence

    for i in keys_sequence:
        sampled_sequence = steerings[(i):(i + g_conf.NUMBER_IMAGES_SEQUENCE)]

        sequence_average.append(sum(sampled_sequence) / len(sampled_sequence))

    # sequence_average =  get_average_over_interval_stride(steerings_train,sequence_size,stride_size)

    return [i[0] for i in sorted(enumerate(sequence_average), key=lambda x: x[1])], sequence_average


def partition_keys_by_percentiles(steerings, keys, percentiles):
    iter_index = 0
    quad_pos = 0
    splited_keys = []
    # print 'len steerings'
    # print len(steerings
    quad_vec = [percentiles[0]]
    for i in range(1, len(percentiles)):
        quad_vec.append(quad_vec[-1] + percentiles[i])

    #print(quad_vec)

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


            if (eliminated_positions) > g_conf.NUMBER_IMAGES_SEQUENCE/2:
                del_pos.append(count * g_conf.SEQUENCE_STRIDE)
                break_sequence = True
                break

        if break_sequence:
            break_sequence = False
            count += 1
            continue

        count += 1

    return del_pos


""" Split the outputs keys with respect to the labels. The selected labels represents how it is going to be split """


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

    print (selected_data_vec)

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

    data = convert_measurements(data)
    keys = np.where(np.logical_or(data['angle'] != positions_dict['angle'],
                                  np.logical_and(data['angle']==positions_dict['angle'],
                                                 data['traffic_lights'] ==positions_dict['traffic_lights']
                                                 )
                                  )
                    )[0]
    return keys

def remove_angle(data, positions_dict):
    # This will remove a list of angles that you dont want
    # Usually used to get just the central camera

    data = convert_measurements(data)
    keys = np.where(np.logical_and(data['angle'] != positions_dict['angle'][0],
                                   data['angle'] != positions_dict['angle'][1]
                                   )
                    )[0]

    return keys



def remove_all(data, positions_dict):
    # This will remove a list of angles that you dont want
    # Usually used to get just the central camera

    data = convert_measurements(data)
    keys = np.where(np.logical_and(np.logical_and(data['pedestrian'] == 1,
                                                  data['traffic_lights'] == 1,
                                                  ),
                                   data['vehicle'] == 1
                                   )
                    )[0]

    return keys

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
    if data[key]['pedestrian']  < 1.0 and data[key]['pedestrian'] > 0:
        boost += positions_dict['boost'][0]

    if data[key]['pedestrian'] == 0.:
        boost += positions_dict['boost'][1]

    if data[key]['vehicle'] < 1.:
        boost +=  positions_dict['boost'][2]

    if data[key]['pedestrian'] == 1.0 and data[key]['vehicle'] == 1. and data[key]['traffic_lights'] == 1. :
        boost += positions_dict['boost'][3]

    return boost





def full_split(dataset):
    
    S = np.zeros(len(dataset.measurements))
    T = np.zeros(len(dataset.measurements))
    B = np.zeros(len(dataset.measurements))
    V = np.zeros(len(dataset.measurements))
    C = np.zeros(len(dataset.measurements))

    for i, M in tqdm(enumerate(dataset.measurements)):
        S[i] = M['steer']
        T[i] = M['throttle']
        B[i] = M['brake']
        V[i] = M['speed_module']
        C[i] = M['directions']
    
    control = [[0, 2, 5], [3], [4]]
    steering = [-1.1, -0.9, -0.8, -0.6, 0, 0.6, 0.8, 0.9, 1.1]
    throttle = [0., 0.1, 0.3, 0.5, 1.1]
    brake = [0., 0.1, 0.3, 0.5, 1.1]
    speed = [-1., 2., 4., 6., 8., 11.]
    keys = list()
    
    counter = 0
    for c in range(3):  # control
        for s in range(len(steering)-1):  # steer
            for t in range(len(throttle)-1): # throttle
                for b in range(len(brake)-1): # brake
                    for v in range(len(speed)-1): # speed
                        true_c = [False, ] * C.shape[0]
                        for vals in control[c]:
                            true_c = np.logical_or(true_c, C==vals)
                        # true_c = [m in control[c] for m in M[D[b'control']]]
                        true_s = np.logical_and(S>=steering[s], S<steering[s+1])
                        true_t = np.logical_and(T>=throttle[t], T<throttle[t+1])
                        true_b = np.logical_and(B>=brake[b], B<brake[b+1])
                        true_v = np.logical_and(V>=speed[v], V<speed[v+1])
                        k1 = np.logical_and(true_c, true_s)
                        k2 = np.logical_and(true_t, true_b)
                        k3 = np.logical_and(k1, k2)
                        k = np.logical_and(k3, true_v)
                        k = np.where(k)[0]
                        counter += 1
                 
                        print(counter, end="\r")

                        if len(k) > 0:
                            this_d = {'keys': k, 'control': c, 'steer': s, 'throttle': t, 'brake': b, 'speed': v}
                            keys.append(this_d)
    
    print('\nNumber key splits: {}'.format(len(keys)))
    return keys
