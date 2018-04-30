import os
import numpy as np
import unittest

import torch

from input.coil_sampler import CoILSampler
from input.coil_dataset import CoILDataset
import input.splitter as spliter
from torch.utils.data.sampler import BatchSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testSampler(unittest.TestCase):


    def test_fake_data(self):

        # This is actual data.
        keys = [[0,1,1,1] * 30] + [[2,3,4,5] * 120] + [[7,8,9,12] * 30]


        weight = [0.2, 0.2, 0.2]

        sampler = CoILSampler(keys, weight)

        for i in BatchSampler(sampler, 120, False):
            print(i)




    def test_real_data(self):



        dataset = CoILDataset(self.root_test_dir)

        keys = range(0, dataset.measurements)
        splited_keys = spliter.label_split(dataset.measurements[:, g_conf.param.VARIABLE_NAMES['Control']], keys,
                            g_conf.param.SPLITTER_PARAMS)

        splited_keys = spliter.float_split(dataset.measurements[:, g_conf.param.VARIABLE_NAMES['Steer']], splited_keys,
                            g_conf.param.SPLITTER_PARAMS)



        weights = [1.0/len(g_conf.param.INPUT.STEERING_DIVISION)]*len(g_conf.param.INPUT.STEERING_DIVISION)

        sampler = CoILSampler(keys, weights)

        for i in BatchSampler(sampler, 120, False):
            print(i)





