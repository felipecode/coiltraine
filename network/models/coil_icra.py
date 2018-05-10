from logger import coil_logger
import torch.nn as nn
import torch

from utils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join

class CoILICRA(nn.Module):

    def __init__(self):
        # TODO: Make an auto naming function for this.

        super(CoILICRA, self).__init__()

        # TODO: AUTOMATICALLY GET THE OUTSIZES
        # TODO: Make configurable function on the config files by reading other dictionary

        self.perception = nn.Sequential(*[
                            Conv(params={'channel_sizes': [3, 32, 32, 64, 64, 128, 128, 256, 256],
                                         'kernel_sizes': [5] + [3]*7,
                                         'strides': [2, 1, 2, 1, 2, 1, 1, 1],
                                         'dropouts': [0.2]*8,
                                         'end_layer': True}),
                            FC(params={'kernel_sizes': [8192, 512, 512],
                                       'dropouts': [0.5, 0.5],
                                       'end_layer': False})]
                            )


        self.measurements = FC(params={'kernel_sizes': [1, 128, 128],
                                       'dropouts': [0.5, 0.5],
                                       'end_layer': False})



        self.join = Join(params={'after_process': FC(params={'kernel_sizes': [640, 512],
                                                             'dropouts': [0.5],
                                                             'end_layer': False}),
                                 'mode': 'cat'
                                }
                         )

        self.speed_branch = FC(params={'kernel_sizes': [512, 256, 256, 1],
                                       'dropouts': [0.5, 0.5, 0.0],
                                       'end_layer': False})

        self.branches = Branching([FC(params={'kernel_sizes': [512, 256, 256, 3],
                                               'dropouts': [0.5, 0.5, 0.0],
                                               'end_layer': True})]*4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)




    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, a):
        # get only the speeds from measurement labels
        #speed = labels[:, 10, :]

        #coil_logger.add_message('Model', {
        #    "Iteration": 765,
        #    "Output": [1.0, 12.3, 124.29]
        #}
        #                        )

        """ ###### APPLY THE PERCEPTION MODULE """
        x = self.perception(x)


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

        """
        # Convert to integer just in case .

        #print (self.forward(x, a))
        # TODO: unit test this function
        output_vec = torch.stack(self.forward(x, a)[0:4])

        branch_number = command_number_to_index(branch_number)
        if len(branch_number) > 1:
            branch_number =torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)


        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])


        #branch_output_vector = []
        #for i in range(len(branch_number)):
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


