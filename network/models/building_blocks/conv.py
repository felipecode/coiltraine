
from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO:  For now the end module is a case
        # TODO: Make an auto naming function for this.

        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channels'])-1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")




        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization


        self.layers = []

        # TODO: need to log the loaded networks


        for i in range(0, len(params['channels'])-1):
            conv = nn.Conv2d(in_channels=params['channels'][i], out_channels=params['channels'][i+1],
                             kernel_size=params['kernels'][i], stride=params['strides'][i])


            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params['channels'][i+1])

            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name





    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels

        # TODO: TRACK NANS OUTPUTS
        # TODO: Maybe change the name
        # TODO: Maybe add internal logs !

        """ conv1 + batch normalization + dropout + relu """
        x = self.layers(x)

        x = x.view(-1, self.num_flat_features(x))


        return x, None  # output, intermediate


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def get_conv_output(self, shape):
        """
           By inputing the shape of the input, simulate what is the ouputsize.
        """

        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

