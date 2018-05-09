
from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'
                 ):

        #         input_size=, kernel_sizes= [512, 128,128], end_module=False):
        # TODO: Make an auto naming function for this.
        #  OBS, only xavier init is available
        #  OBS, only constant = 0.1 initialization of bias
        super(FC, self).__init__()


        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'kernel_sizes' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['kernel_sizes'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")


        self.layers = []


        for i in range(0, len(params['kernel_sizes']) -1):

            fc = nn.Linear(params['kernel_sizes'][i], params['kernel_sizes'][i+1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params['kernel_sizes'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))


        self.layers = nn.Sequential(*self.layers)





    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels
        # TODO: TRACK NANS OUTPUTS
        """

        # TODO: Control the frequency of postion log
        coil_logger.add_message('Model', {
            {'Perception': { "Iteration": 765,
             "Output": [1.0, 12.3, 124.29]
                          }
             }
        } )
        """
        return self.layers(x)





    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })



        # TODO: implement



