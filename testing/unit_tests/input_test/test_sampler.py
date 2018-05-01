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
        keys = [[0, 1, 1, 1] * 30] + [[2,3,4,5] * 120] + [[7, 8, 9, 12] * 30]


        weight = [0.2, 0.2, 0.2]

        sampler = CoILSampler(keys, weight)

        for i in BatchSampler(sampler, 120, False):
            print(i)




    def test_real_data(self):



        dataset = CoILDataset(self.root_test_dir)

        keys = range(0, dataset.measurements)

        splitted_labels = splitter.label_split(labels, keys, g_conf.param.INPUT.LABELS_DIVISION)

        # print (splitted_labels)
        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                  g_conf.param.INPUT.STEERING_DIVISION)

            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += steerings[key]

                avg_now = sum_now / len(splitter_steer[i])
                print(avg_now)
                # if i > 0:
                #    self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now

            splitted_steer_labels.append(splitter_steer)


        weights = [1.0/len(g_conf.param.INPUT.STEERING_DIVISION)]*len(g_conf.param.INPUT.STEERING_DIVISION)

        sampler = CoILSampler(splitted_steer_labels, weights)

        for i in BatchSampler(sampler, 120, False):
            print(i)





