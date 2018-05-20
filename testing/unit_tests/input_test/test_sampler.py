import os
import numpy as np
import unittest

import torch

from input.coil_sampler import BatchSequenceSampler
from input.coil_dataset import CoILDataset
import input.splitter as splitter

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, RandomSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testSampler(unittest.TestCase):

    #def __init__(self, *args, **kwargs):
    #    super(testSampler, self).__init__(*args, **kwargs)
    #self.root_test_dir = '/home/felipe/Datasets/CVPR02Noise/SeqTrain'

    #self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def test_fake_data(self):

        # This is actual data.
        keys = [[0, 1, 1, 1] * 30] + [[2,3,4,5] * 120] + [[7, 8, 9, 12] * 30]


        weight = [0.2, 0.2, 0.2]

        sampler = BatchSequenceSampler(keys, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)

        for i in sampler:

            pass

        keys = [
                [[1, 2, 3] * 30] + [[4, 5, 6] * 120] + [[7, 8, 9] * 30],
                [[10, 20, 30] * 30] + [[40, 50, 60] * 120] + [[70, 80, 90] * 30],
                [[100, 200, 300] * 30] + [[400, 500, 600] * 120] + [[700, 800, 900] * 30]
               ]



        weight = [0.2, 0.2, 0.2]

        sampler = BatchSequenceSampler(keys, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)

        for i in sampler:
            print(len(np.unique(i)))
            # We just check if there is enough variability
            self.assertGreater(len(np.unique(i)), 20)



    """
    def test_real_data(self):



        dataset = CoILDataset(self.root_test_dir)


        steerings = dataset.measurements[0, :]
        print (dataset.meta_data)
        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[24, :]

        print (np.unique(labels))

        keys = range(0, len(steerings))


        splitted_labels = splitter.label_split(labels, keys, g_conf.LABELS_DIVISION)

        # print (splitted_labels)
        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                  g_conf.STEERING_DIVISION)

            splitted_steer_labels.append(splitter_steer)


        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        sampler = CoILSampler(splitted_steer_labels)

        for i in BatchSampler(sampler, 120, False):
            pass
            #print(i)
            
    """


    """
    def test_real_small_data_sequence(self):



        dataset = CoILDataset('/Users/felipecode/CIL/testing/unit_tests/data/')

        g_conf.NUMBER_IMAGES_SEQUENCE = 20
        g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 120

        steerings = dataset.measurements[0, :]

        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[24, :]

        print (np.unique(labels))

        keys = range(0, len(steerings))

        splitted_steer_labels= splitter.control_steer_split(dataset.measurements, dataset.meta_data)

        print (splitted_steer_labels)
        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        sampler =  BatchSequenceSampler(splitted_steer_labels, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)


        #sampler = BatchSampler(RandomSampler(keys), 120, False)

        big_steer_vec = []

        for i in sampler:
            print (steerings[i])
            big_steer_vec += list(steerings[i])

        plt.hist(big_steer_vec, 8)
        plt.show()
    """

    def test_real_data_sequence(self):



        dataset = CoILDataset('/Users/felipecode/Datasets/ValTrainSmall/')

        g_conf.NUMBER_IMAGES_SEQUENCE = 20
        g_conf.SEQUENCE_STRIDE = 5
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 1200

        steerings = dataset.measurements[0, :]

        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[24, :]

        print (np.unique(labels))

        keys = range(0, len(steerings))

        splitted_steer_labels = splitter.control_steer_split(dataset.measurements, dataset.meta_data)

        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        sampler = BatchSequenceSampler(splitted_steer_labels, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)


        #sampler = BatchSampler(RandomSampler(keys), 120, False)

        big_steer_vec = []

        for i in sampler:
            print (i)
            #big_steer_vec += list(steerings[i])
