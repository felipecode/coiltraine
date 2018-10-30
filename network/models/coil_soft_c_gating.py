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

def get_layer_sequence_size(initial_shape, network_sequence):
    """
    Function to get the output size from a series of convolutions with and without residuals
    """

    iterating_shape = initial_shape
    print(iterating_shape)
    for network in network_sequence:

        iterating_shape = network.get_conv_output(iterating_shape)
        print('iter ', iterating_shape)

    return iterating_shape
# TODO: REFACTOR
# TODO: it is interesting the posibility to loop over many models.
# TODO: Having multiple experiments, over the same alias.
class CoILSoftCGating(nn.Module):
       # TODO: Improve the model autonaming function

    def __init__(self, params):
        self.intermediate_layers = None
        super(CoILSoftCGating, self).__init__()

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

        # For this case we check if the perception layer has a initial convolution

        self.low_perception = self.build_perception(params['low_perception'],
                                                    sensor_input_shape,
                                                    False)
        low_perception_o_shape = get_layer_sequence_size(sensor_input_shape, self.low_perception)
        self.low_perception = nn.Sequential(*self.low_perception)
        print ('low perception', low_perception_o_shape)

        ### Declare for the branched perception part used for complex cases ###
        self.mid_complex_perception = self.build_perception(params['mid_complex_perception'],
                                                            low_perception_o_shape,
                                                            False)
        mid_complex_perception_o_shape = get_layer_sequence_size(low_perception_o_shape,
                                                                 self.mid_complex_perception)
        print('mid complex perception', mid_complex_perception_o_shape)
        self.mid_complex_perception = nn.Sequential(*self.mid_complex_perception)

        ### Declare for the branched perception part used for complex cases ###
        self.mid_easy_perception = self.build_perception(params['mid_easy_perception'],
                                                         low_perception_o_shape,
                                                         False)
        mid_easy_perception_o_shape = get_layer_sequence_size(low_perception_o_shape,
                                                              self.mid_easy_perception)
        print('mid easy perception', mid_easy_perception_o_shape)

        self.mid_easy_perception = nn.Sequential(*self.mid_easy_perception)

        self.join_perceptions = Join(
            params={'after_process': None,
                    'mode': 'cat'
                    }
        )
        # TODO: we assume both parts output same shape
        high_perception_start_shape = torch.Size([
                mid_easy_perception_o_shape[0] + mid_complex_perception_o_shape[0],
                                      mid_easy_perception_o_shape[1],
                                      mid_easy_perception_o_shape[2]])


        print ('high perception', high_perception_start_shape)

        self.high_perception = self.build_perception(params['high_perception'],
                                                     high_perception_start_shape,
                                                     True)

        self.high_perception = nn.Sequential(*self.high_perception)

        # ==============================================================================
        # -- Measurement Layers -----------------------------------------------------------
        # ==============================================================================



        number_output_neurons = params['high_perception']['fc']['neurons'][-1]


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


    def build_perception(self, params, sensor_input_shape, end_layer):
        perception_layers = []
        if 'conv' in params:
            perception_layers.append(Conv(params={'channels': [sensor_input_shape[0]] +
                                                              params['conv'][
                                                                  'channels'],
                                                  'kernels': params['conv'][
                                                      'kernels'],
                                                  'strides': params['conv'][
                                                      'strides'],
                                                  'padding': params['conv'][
                                                      'padding'],
                                                  'bias': params['conv']['bias'],
                                                  'dropouts': params['conv'][
                                                      'dropouts'],
                                                  'end_layer': False}))

        if 'res' in params:
            if len(perception_layers) > 0:
                inplanes = get_layer_sequence_size(sensor_input_shape, perception_layers)[0]
            else:
                inplanes = sensor_input_shape[0]

            perception_layers.append(Residuals(params={
                'block_type': params['res']['block_type'],
                'channels': params['res']['channels'],
                'layers': params['res']['layers'],
                'strides': params['res']['strides'],
                'end_layer': end_layer},
                inplanes=inplanes))
        if 'fc' in params:
            perception_layers.append(FC(params={'neurons': [get_layer_sequence_size(
                sensor_input_shape, perception_layers)]
                                                           + params['fc']['neurons'],
                                                'dropouts': params['fc']['dropouts'],
                                                'end_layer': False}))

        return perception_layers

    def forward(self, x, a, intentions=None):

        """ ###### APPLY THE PERCEPTION MODULES """
        x = self.low_perception(x)

        ## We get the complexity indicator by using the intentions ##
        if intentions is not None:
            complexity_indicator = self.make_complexity_indicator(intentions)

            complexity_indicator = \
                torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(complexity_indicator, 1), 1), 1)

            x_complex = self.mid_complex_perception(x * (1 - complexity_indicator))

            x_easy = self.mid_easy_perception(x * complexity_indicator)
        else:
            x_complex = self.mid_complex_perception(x)

            x_easy = self.mid_easy_perception(x)

        easy_complex = self.join_perceptions(x_complex, x_easy)

        x = self.high_perception(easy_complex)

        """ ###### APPLY THE MEASUREMENT MODULES """

        m = self.measurements(a, intentions)

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


    def make_complexity_indicator(self, intention_factors):

        intention, _ = torch.min(intention_factors, 1)
        return intention


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


