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



#TODO: Refactor this splitting strategy.

def control_steer_split(float_data, meta_data, keys):

    # TODO: WHY EVERY WHERE MAKE THIS TO BE USED ??
    steerings = float_data[np.where(meta_data[:, 0] == b'steer'), :][0][0]

    print ("steer shape", steerings.shape)

    # TODO: read meta data and turn into a coool dictionary ?
    # TODO ELIMINATE ALL NAMES CALLED LABEL OR MEASUREMENTS , MORE GENERIC FLOAT DATA AND SENSOR DATA IS BETTER
    labels = float_data[np.where(meta_data[:, 0] == b'control'), :][0][0]

    print ("labels shape ", labels.shape)
    #keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)

    splitted_labels = label_split(labels, keys, g_conf.LABELS_DIVISION)

    # Another level of splitting
    splitted_steer_labels = []

    for keys in splitted_labels:
        splitter_steer = float_split(steerings, keys, g_conf.STEERING_DIVISION)
        splitted_steer_labels.append(splitter_steer)

    coil_logger.add_message('Loading', {'KeysDivision': splitted_steer_labels})

    return splitted_steer_labels

def control_speed_split(float_data, meta_data, keys):

    # TODO: WHY EVERY WHERE MAKE THIS TO BE USED ??
    speeds = float_data[np.where(meta_data[:, 0] == b'speed_module'), :][0][0]

    print ("steer shape", speeds.shape)

    # TODO: read meta data and turn into a coool dictionary ?
    # TODO ELIMINATE ALL NAMES CALLED LABEL OR MEASUREMENTS , MORE GENERIC FLOAT DATA AND SENSOR DATA IS BETTER
    labels = float_data[np.where(meta_data[:, 0] == b'control'), :][0][0]

    print ("labels shape ", labels.shape)
    #keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)

    splitted_labels = label_split(labels, keys, g_conf.LABELS_DIVISION)

    # Another level of splitting
    splitted_steer_labels = []

    for keys in splitted_labels:
        splitter_steer = float_split(speeds, keys, g_conf.SPEED_DIVISION)
        splitted_steer_labels.append(splitter_steer)

    coil_logger.add_message('Loading', {'KeysDivision': splitted_steer_labels})

    return splitted_steer_labels


def pedestrian_speed_split(float_data, meta_data, keys):

    # TODO: WHY EVERY WHERE MAKE THIS TO BE USED ??
    speeds = float_data[np.where(meta_data[:, 0] == b'speed_module'), :][0][0]

    print ("steer shape", speeds.shape)

    # TODO: read meta data and turn into a coool dictionary ?
    # TODO ELIMINATE ALL NAMES CALLED LABEL OR MEASUREMENTS , MORE GENERIC FLOAT DATA AND SENSOR DATA IS BETTER
    labels = float_data[np.where(meta_data[:, 0] == b'pedestrian'), :][0][0].astype(np.bool) & \
             (float_data[np.where(meta_data[:, 0] == b'camera'), :][0][0] == 1)
    print ("labels shape ", labels.shape)
    #keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)

    splitted_labels = label_split(labels, keys, g_conf.PEDESTRIAN_PERCENTAGE)

    # Another level of splitting
    splitted_steer_labels = []



    for keys in splitted_labels:
        splitter_steer = float_split(speeds, keys, g_conf.SPEED_DIVISION)
        splitted_steer_labels.append(splitter_steer)

    coil_logger.add_message('Loading', {'KeysDivision': splitted_steer_labels})

    return splitted_steer_labels

def lambda_splitter(float_data, meta_data, lambda_list):
    key_list = []
    for l in lambda_list:
        keys = l(float_data, meta_data)
        key_list.append(keys)
    return key_list


def full_split(dataset):
    control = [[0, 2, 5], [3], [4]]
    steering = np.cumsum([0, 0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05])
    throttle = [0, 0.1, 0.45, 1]
    brake = [0, 0.1, 0.3, 0.5, 1]
    speed = [0, 5, 15, 40]
    keys = list()

    M = dataset.measurements
    D = {}  # meta_dict
    for i, (k, _) in enumerate(dataset.meta_data):
        if len(k) > 0:
            D[k] = i
    counter = 0
    for c in range(3):  # control
        for s in range(len(steering)-1):  # steer
            for t in range(len(throttle)-1): # throttle
                for b in range(len(brake)-1): # brake
                    for v in range(len(speed)-1): # speed
                        true_c = [True, ] * M.shape[1]
                        for vals in control[c]:
                            true_c = np.logical_and(true_c, M[D[b'control']]==vals)
                        # true_c = [m in control[c] for m in M[D[b'control']]]
                        S = M[D[b'steer']]
                        true_s = np.logical_and(S>=steering[s], S<steering[s+1])
                        T = M[D[b'throttle']]
                        true_t = np.logical_and(T>=throttle[t], T<throttle[t+1])
                        B = M[D[b'brake']]
                        true_b = np.logical_and(B>=brake[b], B<brake[b+1])
                        V = M[D[b'speed_module']]
                        true_v = np.logical_and(V>=speed[v], V<speed[v+1])
                        k1 = np.logical_and(true_c, true_s)
                        k2 = np.logical_and(true_t, true_b)
                        k3 = np.logical_and(k1, k2)
                        k = np.logical_and(k3, true_v)
                        k = np.where(k)[0]
                        if len(k) > 0:
                            this_d = {'keys': k, 'control': c, 'steer': s, 'throttle': t, 'brake': b, 'speed': v}
                            keys.append(this_d)
                        counter += 1
                        # bar.next()
                        print(counter, end="\r")
                        keys.append(k)
    keys.append({'keys': list(np.arange(M.shape[1])), 'control': np.inf, 'steer': np.inf, 'throttle': np.inf, 'brake': np.inf, 'speed': np.inf})
    print('pre-filter length: {}'.format(len(keys)))
    keys = [k for k in keys if len(k)>0]
    print('post-filter length: {}'.format(len(keys)))
    return keys
