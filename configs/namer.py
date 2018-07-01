

def generate_name(g_conf):
    # TODO: Make a cool name generator, maybe in another class
    """

        The name generator is currently formed by the following parts
        Dataset_name.
        THe type of network used, got directly from the class.
        The regularization
        The strategy with respect to time
        The type of output
        The preprocessing made in the data
        The type of loss function
        The parts  of data that where used.

        Take into account if the variable was not set, it set the default name, from the global conf



    Returns:
        a string containing the name


    """

    final_name_string = ""
    # Addind dataset
    final_name_string += g_conf.TRAIN_DATASET_NAME
    # Model type
    final_name_string += '_' + g_conf.MODEL_TYPE
    # Model Size
    #TODO: for now is just saying the number of convs, add a layer counting
    final_name_string += '_' + str(len(g_conf.MODEL_CONFIGURATION['perception']['conv']['kernels'])) +'conv'

    # Model Regularization
    # We start by checking if there is some kind of augmentation, and the schedule name.

    if g_conf.AUGMENTATION is not None and g_conf.AUGMENTATION != 'None':
        final_name_string += '_' + g_conf.AUGMENTATION
    else:
        # We check if there is dropout
        if sum(g_conf.MODEL_CONFIGURATION['branches']['fc']['dropouts']) > 0:
            final_name_string += '_dropout'
        else:
            final_name_string += '_none'

    # Temporal

    if g_conf.NUMBER_FRAMES_FUSION > 1 and g_conf.NUMBER_IMAGES_SEQUENCE > 1:
        final_name_string += '_lstm_fusion'
    elif g_conf.NUMBER_FRAMES_FUSION > 1:
        final_name_string += '_fusion'
    elif g_conf.NUMBER_IMAGES_SEQUENCE > 1:
        final_name_string += '_lstm'
    else:
        final_name_string += '_single'

    # THe type of output

    if 'waypoint1_angle' in set(g_conf.TARGETS):

        final_name_string += '_waypoints'
    else:
        final_name_string += '_control'

    # The pre processing ( Balance or not )
    if g_conf.BALANCE_DATA and len(g_conf.STEERING_DIVISION) > 0:
        final_name_string += '_balance'
    else:
        final_name_string += '_random'

    # The type of loss function

    final_name_string += '_'+g_conf.LOSS_FUNCTION

    # the parts of the data that were used.

    if g_conf.USE_NOISE_DATA:
        final_name_string += '_noise_'
    else:
        final_name_string += '_'

    final_name_string += g_conf.DATA_USED


    return final_name_string