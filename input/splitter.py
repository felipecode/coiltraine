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

    for j in range(len(selected_data)):

        keys_to_delete = select_data_sequence(labels, selected_data[j])
        # print got_keys_for_divison
        # keys_for_this_part = range(0, len(labels) - g_conf.NUMBER_IMAGES_SEQUENCE,
        #                           g_conf.SEQUENCE_STRIDE)

        keys_for_this_part = list(set(keys) - set(keys_to_delete))
        # If it is empty, kindly ask the user to change the label division
        if not keys_for_this_part:
            raise RuntimeError("No Element found of the key ", selected_data[j],
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