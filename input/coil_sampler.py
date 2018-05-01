import random

import torch

from torch.utils.data.sampler import Sampler

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

class CoILSampler(Sampler):

    def __init__(self, keys, weights):

        self.weights = torch.tensor(weights, dtype=torch.double)
        self.keys = keys

        self.replacement = True

    def __iter__(self):
        # Chose here
        #print(self.weights)

        # TODO: take into acount multiple divisions
        idx = torch.multinomial(self.weights, g_conf.param.MISC.DATASET_SIZE, True)
        idx = idx.tolist()
        #print (self.keys[0])
        #print (random.choice(self.keys[0]))
        #print ([self.keys[i] for i in idx])
        #print (iter(self.keys[i] for i in idx))
        #print ([random.choice(self.keys[i]) for i in idx])


        return iter([random.randint(self.keys[i]) for i in idx])


    def __len__(self):
        return g_conf.param.MISC.DATASET_SIZE