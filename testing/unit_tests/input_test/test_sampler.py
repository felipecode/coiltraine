import os
import numpy as np
import unittest

import torch

from input.coil_sampler import CoILSampler
from input.coil_dataset import CoILDataset
import input.splitter as splitter
from torch.utils.data.sampler import BatchSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testSampler(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testSampler, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def test_fake_data(self):

        # This is actual data.
        keys = [[0, 1, 1, 1] * 30] + [[2,3,4,5] * 120] + [[7, 8, 9, 12] * 30]


        weight = [0.2, 0.2, 0.2]

        sampler = CoILSampler(keys)

        for i in BatchSampler(sampler, 120, False):

            pass

        keys = [
                [[1, 2, 3] * 30] + [[4, 5, 6] * 120] + [[7, 8, 9] * 30],
                [[10, 20, 30] * 30] + [[40, 50, 60] * 120] + [[70, 80, 90] * 30],
                [[100, 200, 300] * 30] + [[400, 500, 600] * 120] + [[700, 800, 900] * 30]
               ]



        weight = [0.2, 0.2, 0.2]

        sampler = CoILSampler(keys)

        for i in BatchSampler(sampler, 120, False):
            print(len(np.unique(i)))
            # We just check if there is enough variability
            self.assertGreater(len(np.unique(i)), 20)



    def test_real_data(self):



        dataset = CoILDataset(self.root_test_dir)


        steerings = dataset.measurements[0, :]
        print (dataset.meta_data)
        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == 'control'))
        labels = dataset.measurements[24, :]

        print (np.unique(labels))

        keys = range(0, len(steerings))


        splitted_labels = splitter.label_split(labels, keys, g_conf.param.INPUT.LABELS_DIVISION)

        # print (splitted_labels)
        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                  g_conf.param.INPUT.STEERING_DIVISION)

            splitted_steer_labels.append(splitter_steer)


        #weights = [1.0/len(g_conf.param.INPUT.STEERING_DIVISION)]*len(g_conf.param.INPUT.STEERING_DIVISION)

        sampler = CoILSampler(splitted_steer_labels)

        for i in BatchSampler(sampler, 120, False):
            print(i)





