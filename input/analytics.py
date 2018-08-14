import numpy as np

# Get a bunch of data that matches certain conditions
def get_data_conditioned(conditions, data, meta_data):

    """

    conditions: The operators must be boolean.

    """

    final_mask = np.bool(np.size(data))

    for condition in conditions:
        operand1 = condition[0]
        operator = condition[1]
        operand2 = condition[2]
        # TODO for now we have the case where operand 2 is a string or when it is a float
        if isinstance(operand2, str):
            final_mask = final_mask & operator(data[np.where(meta_data[:, 0] == operand1)][0][0],
                                               data[np.where(meta_data[:, 0] == operand2)][0][0]
                                               )
        else:
            final_mask = final_mask & operator(data[np.where(meta_data[:, 0] == operand1)][0][0],
                                               0
                                               )

    return final_mask





# Condition, get all data that match certain condition

def get_conditioned_histogran(conditions, variable, number_of_bins, data):
    """

        conditions: set of conditions that your variable can be conditioned on.
        variable: the variable that you want to plot the histogram about
        number_of_bins: the number of bins that you intend to divide the variables.


        return
            the ready to plot histogram
    """

    get_data_conditioned(conditions, data)

