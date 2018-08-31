import os
import numpy as np
import unittest

import torch

from input.coil_sampler import BatchSequenceSampler, SubsetSampler, RandomSampler
from input.coil_dataset import CoILDataset
from input import Augmenter
import input.splitter as splitter
from PIL import Image
#from utils.general import plot_test_image

from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testSampler(unittest.TestCase):

    def test_regular_sampler(self):

        try:
            os.mkdir('_images')
        except:
            pass
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset('/Users/felipecode/Datasets/1HoursW1-3-6-8',
                              augmenter)

        g_conf.NUMBER_IMAGES_SEQUENCE = 1
        g_conf.SEQUENCE_STRIDE = 1
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 120000
        g_conf.BATCH_SIZE = 120

        steerings = dataset.measurements[0, :]

        # TODO: read meta data and turn into a coool dictionary ?

        labels = dataset.measurements[24, :]

        print (np.unique(labels))


        print ('position of camera', np.where(dataset.meta_data[:, 0] == b'camera'))


        camera_names = dataset.measurements[np.where(dataset.meta_data[:, 0] == b'camera'), :][0][0]
        print (" Camera names ")
        print (camera_names)

        keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)


        splitted_steer_labels = splitter.control_steer_split(dataset.measurements,
                                                             dataset.meta_data, keys)





        # one_camera_data = splitter.label_split(camera_names, keys, [[0]])
        #
        #
        #
        # splitted_steer_labels = splitter.control_steer_split(dataset.measurements,
        #                                                      dataset.meta_data, one_camera_data[0])
        #
        #
        # for split_1 in splitted_steer_labels:
        #     for split_2 in split_1:
        #         for split_3 in split_2:
        #             if split_3 not in one_camera_data[0]:
        #                 raise ValueError("not one camera")

        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        sampler = BatchSequenceSampler(splitted_steer_labels, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)




        big_steer_vec = []
        count =0

        print ("len keys", len(keys))

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_sampler=sampler,
                                                  num_workers=0,
                                                  pin_memory=True)
        dist_calc =  [0] * (len(keys)+1)

        print (len(dist_calc))

        for data in data_loader:

            sensor, float_data = data

            print (sorted(float_data[:,0]))


            count += 1

        print (dist_calc)
        #plt.hist(dist_calc,1400)

        #plt.show()


