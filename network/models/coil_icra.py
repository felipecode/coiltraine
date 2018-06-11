from logger import coil_logger
import torch.nn as nn
import torch

from utils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join

# TODO: it is interesting the posibility to loop over many models.
# TODO: Having multiple experiments, over the same alias.
class CoILICRA(nn.Module):

    def __init__(self, params):
        # TODO: Make an auto naming function for this.

        super(CoILICRA, self).__init__()

        # TODO: AUTOMATICALLY GET THE OUTSIZES
        # TODO: Make configurable function on the config files by reading other dictionary


        self.perception = nn.Sequential(*[
                            Conv(params={'channels': params['perception']['conv']['channels'],
                                         'kernels': params['perception']['conv']['kernels'],
                                         'strides': params['perception']['conv']['strides'],
                                         'dropouts': params['perception']['conv']['dropouts'],
                                         'end_layer': True}),
                            FC(params={'neurons': params['perception']['fc']['channels'],
                                       'dropouts': params['perception']['fc']['dropouts'],
                                       'end_layer': False})]
                            )


        self.measurements = FC(params={'neurons': params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})



        self.join = Join(params={'after_process': FC(params={'neurons': params['join']['fc']['neurons'],
                                                             'dropouts': params['join']['fc']['dropouts'],
                                                             'end_layer': False}),
                                 'mode': 'cat'
                                }
                         )

        self.speed_branch = FC(params={'neurons': params['speed_branch']['fc']['neurons'],
                                       'dropouts': params['speed_branch']['fc']['dropouts'],
                                       'end_layer': False})

        self.branches = Branching([FC(params={'neurons': params['branches']['fc']['neurons'],
                                               'dropouts': params['branches']['fc']['dropouts'],
                                               'end_layer': True})] # TODO: THE number of branches multiplying does not work.
                                                * params['number_of_branches']) #  Here we set branching automatically

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)




    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, a):


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


        return self.extract_branch(output_vec, branch_number)



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


