
def get_network_input_size(input_dict):
    """
        Checking the input dictionary for this network module we check what should be its size
    
    :return: 
    """

    measurements_size = 0
    for _, sizes in input_dict.items():
        measurements_size += sizes

    return measurements_size