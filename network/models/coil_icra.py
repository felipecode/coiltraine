from logger import coil_logger
import torch.nn as nn
import torch
import importlib

from configs import g_conf
from utils.general import command_number_to_index

from .building_blocks import Conv, Residuals
from .building_blocks import Branching
from .building_blocks import FC, FCD
from .building_blocks import Join


# TODO: REFACTOR
# TODO: it is interesting the posibility to loop over many models.
# TODO: Having multiple experiments, over the same alias.

def get_layer_sequence_size(initial_shape, network_sequence):
    """
    Function to get the output size from a series of convolutions with and without residuals
    :param initial_shape:
    :param network_sequence:
    :return: the final shape
    """

    iterating_shape = initial_shape
    print(iterating_shape)
    for network in network_sequence:

        iterating_shape = network.get_conv_output(iterating_shape)
        print('iter ', iterating_shape)

    return iterating_shape


class CoILICRA(nn.Module):

    def __init__(self, params):
        # TODO: Improve the model autonaming function

        self.intermediate_layers = None
        super(CoILICRA, self).__init__()

        # TODO: Make configurable function on the config files by reading other dictionary
        number_first_layer_channels = 0


        for name, sizes in g_conf.SENSORS.items():
            """ We check to see if the sensor is in the list of sensors to be input"""
            if name in g_conf.INPUT_SENSORS:
                number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # ==============================================================================
        # -- Perception Layers -----------------------------------------------------------
        # ==============================================================================
        perception_layers = []

        # For this case we check if the perception layer has a initial convolution
        if 'conv' in params['perception']:
            perception_layers.append(Conv(params={'channels': [number_first_layer_channels] +
                                                          params['perception']['conv']['channels'],
                                            'kernels': params['perception']['conv']['kernels'],
                                            'strides': params['perception']['conv']['strides'],
                                            'padding': params['perception']['conv']['padding'],
                                               'bias': params['perception']['conv']['bias'],
                                            'dropouts': params['perception']['conv']['dropouts'],
                                            'end_layer': False}))

        if 'res' in params['perception']:
            perception_layers.append(Residuals(params={
                                            'block_type': params['perception']['res']['block_type'],
                                            'channels': params['perception']['res']['channels'],
                                            'layers': params['perception']['res']['layers'],
                                            'strides': params['perception']['res']['strides'],
                                            'end_layer': True}))
        if 'fc' in params['perception']:
            perception_layers.append(FC(params={'neurons': [get_layer_sequence_size(
                                        sensor_input_shape, perception_layers)]
                                        + params['perception']['fc']['neurons'],
                                        'dropouts': params['perception']['fc']['dropouts'],
                                        'end_layer': False}))


        print (perception_layers)
        self.perception = nn.Sequential(*perception_layers)

        number_output_neurons = params['perception']['fc']['neurons'][-1]


        # WILL NOT WORK FOR SMALL AND DEEP LAYERS
        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})



        self.join = Join(
            params={'after_process':
                         FC(params={'neurons':
                                        [params['measurements']['fc']['neurons'][-1] +
                                         number_output_neurons] +
                                        params['join']['fc']['neurons'],
                                     'dropouts': params['join']['fc']['dropouts'],
                                     'end_layer': False}),
                     'mode': 'cat'
                    }
         )

        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                  params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})


        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                         params['branches']['fc']['neurons'] +
                                                         [len(g_conf.TARGETS)],
                                               'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector) #  Here we set branching automatically

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x, a, intentions=None):


        """ ###### APPLY THE PERCEPTION MODULE """
        x = self.perception(x)
        #self.intermediate_layers = inter

        """ ###### APPLY THE MEASUREMENT MODUES """

        m = self.measurements(a)

        """ Join measurements and perception"""
        j = self.join(x, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.
        return branch_outputs + [speed_branch_output]

    def forward_branch(self, x, a, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned

        Returns:
            the wanted branch,
            the speed predicted by the model.

        """

        # TODO: take four branches, this is hardcoded
        output_vec = torch.stack(self.forward(x, a)[0:4])


        return self.extract_branch(output_vec, branch_number), self.forward(x, a)[-1]



    def extract_branch(self, output_vec, branch_number):


        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        # branch_output_vector = []
        # for i in range(len(branch_number)):
        #    branch_output_vector.append(output_vec[branch_number[i]][i])


        return output_vec[branch_number[0], branch_number[1], :]

    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })


