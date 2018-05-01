from configs import g_conf


def order_sequence(steerings, keys_sequence):
    sequence_average = []
    # print 'keys'
    # print keys_sequence

    for i in keys_sequence:
        sampled_sequence = steerings[(i):(i + g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE)]

        sequence_average.append(sum(sampled_sequence) / len(sampled_sequence))

    # sequence_average =  get_average_over_interval_stride(steerings_train,sequence_size,stride_size)

    return [i[0] for i in sorted(enumerate(sequence_average), key=lambda x: x[1])], sequence_average


def partition_keys_by_percentiles(steering_unordered, steerings, keys, percentiles):
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
            iter_index = i
            quad_pos += 1
            # TODO This could be very important log information
            print('split on ', i, 'with ', steerings[i])
            print(len(splited_keys))
            print(len(splited_keys[-1]))

    return splited_keys


def select_data_sequence(control, selected_data):
    i = 0
    break_sequence = False

    count = 0
    del_pos = []
    # print "SELECTED"
    # print selected_data
    while count * g_conf.param.MISC.SEQUENCE_STRIDE <= (
            len(control) - g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE):

        # print 'sequence starting on : ', count*self._sequence_stride
        for iter_sequence in range((count * g_conf.param.MISC.SEQUENCE_STRIDE),
                                   (count * g_conf.param.MISC.SEQUENCE_STRIDE) +
                                   g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE):

            # print ' control ', control[iter_sequence], ' selected ', selected_data
            # The position is one
            # print control.shape
            if control[iter_sequence] not in selected_data:
                # print control[j,iter_sequence]
                # print 'OUT'
                del_pos.append(count * g_conf.param.MISC.SEQUENCE_STRIDE)

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
    # TODO: here we divide keys in the same way as for float split .

    for j in range(len(selected_data)):
        keys_to_delete = select_data_sequence(labels, selected_data[j])
        # print got_keys_for_divison
        # keys_for_this_part = range(0, len(labels) - g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE,
        #                           g_conf.param.MISC.SEQUENCE_STRIDE)

        keys_for_this_part = list(set(keys) - set(keys_to_delete))

        keys_for_divison.append(keys_for_this_part)

    return keys_for_divison


def float_split(output_to_split, keys, percentiles):
    """
    Split data based on the the float value of some variable.
    Everything is splitted with respect to the percentages.

    Arguments :




    """

    # TODO: test the number of parts that the splitter keys have.

    # We use this keys to grab the steerings we want... divided into groups
    # TODO: Test the spliting based on median.
    #print ('Start keys ',keys)
    keys_ordered, average_outputs = order_sequence(output_to_split, keys)

    # we get new keys and order steering, each steering group
    sorted_outputs = [average_outputs[j] for j in keys_ordered]
    corresponding_keys =[keys[j] for j in keys_ordered]


    # We split each group...
    if len(keys_ordered) > 0:
        splitted_keys = partition_keys_by_percentiles(average_outputs, sorted_outputs,
                                                      corresponding_keys, percentiles)
    else:
        splitted_keys = []

    return splitted_keys
