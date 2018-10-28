from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np


class FCD(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(FCD, self).__init__()

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

        self.layers = nn.ModuleList(self.layers)


    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, intentions=None):
        # intentions define if max dropout is to be applied
        # get only the speeds from measurement labels
        # TODO: TRACK NANS OUTPUTS
        for L, drop in zip(self.layers, self.params['dropouts']):
            x = L(x)
            if self.training:  # apply dropout
                if intentions is None:
                    tensor_intentions = torch.ones(x.shape[0], dtype=torch.float32).cuda()
                else:
                    tensor_intentions, _ = torch.min(intentions, 1)
                keepprob = 1. - drop
                print (keepprob)
                print (tensor_intentions * keepprob)
                d = (tensor_intentions * keepprob)
                d = torch.clip(d, 0.1, 1)
                print(d.shape)
                d = torch.unsqueeze(d, 1)
                print(d.shape)
                mask = torch.bernoulli(d)
                print (" MASK ", mask)
                x = x * mask
                x = x / d


        return x

