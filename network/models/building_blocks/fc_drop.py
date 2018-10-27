from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(FC, self).__init__()

        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")


        self.params = params
        self.layers = []


        for i in range(0, len(params['neurons']) -1):

            fc = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, ]))
            else:
                self.layers.append(nn.Sequential(*[fc, relu]))

    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, intentions=None):
        # intentions define if max dropout is to be applied
        # get only the speeds from measurement labels
        # TODO: TRACK NANS OUTPUTS
        for L, drop in zip(self.layers, self.params['dropouts']):
            x = L(x)
            if self.training:  # apply dropout
                if intentions is None:
                    intentions = torch.ones(x.shape[0], dtype=torch.float32)
                keepprob = 1. - drop
                d = (intentions * keepprob).view_as(-1, 1)
                d = d.view_as.expand_as(x)
                mask = torch.bernoulli(d)
                x = x * mask
                x = x / d

        return x

