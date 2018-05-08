import numpy as np
import random

import torch

from torch.utils.data.sampler import Sampler

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

class CoILSampler(Sampler):

    def __init__(self, keys):

        self.keys = keys

        self.replacement = True

    def __iter__(self):
        """

            OBS: One possible thing to be done is the possibility to have a matrix of ids
            of rank N
            OBS2: Probably we dont need weights right now


        Returns:
            Iterator to get ids for the dataset

        """
        shape_keys = np.shape(self.keys)
        # First we check how many subdivisions there are
        if len(shape_keys) == 1:

            weights = torch.tensor([1.0/float(len(self.keys))]*len(self.keys), dtype=torch.double)

            idx = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idx = idx.tolist()
            return iter([random.choice(self.keys[i]) for i in idx])
        elif len(np.shape(self.keys)) == 2:
            weights = torch.tensor([1.0 / float(len(self.keys))] * len(self.keys),
                                   dtype=torch.double)
            idx = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idx = idx.tolist()
            weights = torch.tensor([1.0 / float(len(self.keys[0]))] * len(self.keys[0]),
                                   dtype=torch.double)
            idy = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idy = idy.tolist()

            return iter([random.choice(self.keys[i][j]) for i, j in zip(idx,idy)])

        else:
            raise ValueError("Keys have invalid rank")




    def __len__(self):
        return g_conf.NUMBER_ITERATIONS