import os
import numpy as np
import unittest

import torch

from input.coil_sampler import BatchSequenceSampler, SubsetSampler
from input.coil_dataset import CoILDataset
from input import Augmenter
import input.splitter as splitter
from PIL import Image
#from utils.general import plot_test_image

from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, RandomSampler, SubsetRandomSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testSampler(unittest.TestCase):

    #def __init__(self, *args, **kwargs):
    #    super(testSampler, self).__init__(*args, **kwargs)
    #self.root_test_dir = '/home/felipe/Datasets/CVPR02Noise/SeqTrain'

    #self.test_images_write_path = 'testing/unit_tests/_test_images_'

    """
    def test_fake_data(self):

        # This is actual data.
        keys = [[0, 1, 1, 1] * 30] + [[2,3,4,5] * 120] + [[7, 8, 9, 12] * 30]


        weight = [0.2, 0.2, 0.2]

        sampler = BatchSequenceSampler(keys, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)

        for i in sampler:

            pass

        keys = [
                [[1, 2, 3] * 30] + [[4, 5, 6] * 120] + [[7, 8, 9] * 30],
                [[10, 20, 30] * 30] + [[40, 50, 60] * 120] + [[70, 80, 90] * 30],
                [[100, 200, 300] * 30] + [[400, 500, 600] * 120] + [[700, 800, 900] * 30]
               ]



        weight = [0.2, 0.2, 0.2]

        sampler = BatchSequenceSampler(keys, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)

        for i in sampler:
            print(len(np.unique(i)))
            # We just check if there is enough variability
            self.assertGreater(len(np.unique(i)), 20)

    """

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
    """
    def test_real_data_sequence(self):



        dataset = CoILDataset('/Users/felipecode/Datasets/ValTrain')

        g_conf.NUMBER_IMAGES_SEQUENCE = 1
        g_conf.SEQUENCE_STRIDE = 1
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 1200
        g_conf.BATCH_SIZE = 120

        steerings = dataset.measurements[0, :]

        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[24, :]

        print (np.unique(labels))


        keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)

        splitted_steer_labels = splitter.control_steer_split(dataset.measurements, dataset.meta_data, keys)

        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        sampler = BatchSequenceSampler(splitted_steer_labels, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
                                      g_conf.SEQUENCE_STRIDE, False)


        #sampler = BatchSampler(RandomSampler(keys), 120, False)

        big_steer_vec = []
        count =0
        for i in sampler:
            #print(count)
            #print("len", len(i))
            count += 1
            #big_steer_vec += list(steerings[i])
    """


    def test_real_data_central_sampler(self):

        try:
            os.mkdir('_images')
        except:
            pass
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset('/home/felipe/Datasets/1HoursW1-3-6-8',
                              augmenter)

        g_conf.NUMBER_IMAGES_SEQUENCE = 1
        g_conf.SEQUENCE_STRIDE = 1
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 1200
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

        one_camera_data = splitter.label_split(camera_names, keys, [[0]])



        splitted_steer_labels = splitter.control_steer_split(dataset.measurements,
                                                             dataset.meta_data, one_camera_data[0])


        for split_1 in splitted_steer_labels:
            for split_2 in split_1:
                for split_3 in split_2:
                    if split_3 not in one_camera_data[0]:
                        raise ValueError("not one camera")

        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        #sampler = BatchSequenceSampler(splitted_steer_labels, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
        #                              g_conf.SEQUENCE_STRIDE, False)

        sampler = SubsetSampler(one_camera_data[0])

        big_steer_vec = []
        count =0

        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                                   num_workers=12, pin_memory=True)

        for data in data_loader:





            image, measurements = data

            print (image['rgb'].shape)
            for i in range(120):
                name = '_images/' + str(count) + '.png'
                image_to_save = transforms.ToPILImage()(image['rgb'][i][0].cpu())
                image_to_save.save(name)

                count += 1






            #print (list(steerings[i]))
            #big_steer_vec += list(steerings[i])

"""
    def test_real_data_central_sampler_sequence(self):

        try:
            os.mkdir('_images')
        except:
            pass
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset('/home/felipe/Datasets/1HoursW1-3-6-8',
                              augmenter)

        g_conf.NUMBER_IMAGES_SEQUENCE = 1
        g_conf.SEQUENCE_STRIDE = 1
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 1200
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

        one_camera_data = splitter.label_split(camera_names, keys, [[0]])



        splitted_steer_labels = splitter.control_steer_split(dataset.measurements,
                                                             dataset.meta_data, one_camera_data[0])


        print (splitted_steer_labels)

        #weights = [1.0/len(g_conf.STEERING_DIVISION)]*len(g_conf.STEERING_DIVISION)

        #sampler = BatchSequenceSampler(splitted_steer_labels, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
        #                              g_conf.SEQUENCE_STRIDE, False)

        sampler = SubsetSampler(one_camera_data[0])

        big_steer_vec = []
        count =0

        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                                   num_workers=12, pin_memory=True)

        for data in data_loader:





            image, measurements = data

            print (image['rgb'].shape)
            for i in range(120):
                name = '_images/' + str(count) + '.png'
                image_to_save = transforms.ToPILImage()(image['rgb'][i][0].cpu())
                image_to_save.save(name)

                count += 1






            #print (list(steerings[i]))
            #big_steer_vec += list(steerings[i])
"""