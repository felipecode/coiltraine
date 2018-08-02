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

    def test_new_sampler(self):


        pass




    def test_division_by_rank_3(self):
        return
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



    def test_pedestrian_splitted_sampler(self):

        try:
            os.mkdir('_images')
        except:
            pass
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset('/home/felipecodevilla/Datasets/25HoursW1-3-6-8',
                              augmenter)

        g_conf.NUMBER_IMAGES_SEQUENCE = 1
        g_conf.SEQUENCE_STRIDE = 1
        #g_conf.LABELS_DIVISION = [[0,2,5], [0,2,5], [0,2,5]]
        g_conf.NUMBER_ITERATIONS = 120000
        g_conf.BATCH_SIZE = 120
        g_conf.STEERING_DIVISION = []
        g_conf.PEDESTRIAN_PERCENTAGE = 50
        g_conf.SPEED_DIVISION = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05]
        steerings = dataset.measurements[0, :]

        # TODO: read meta data and turn into a coool dictionary ?

        labels = dataset.measurements[24, :]

        print (np.unique(labels))


        print ('position of camera', np.where(dataset.meta_data[:, 0] == b'camera'))
        print (dataset.meta_data)

        camera_names = dataset.measurements[np.where(dataset.meta_data[:, 0] == b'camera'), :][0][0]
        print (" Camera names ")
        print (camera_names)

        keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)


        splitted_steer_labels = splitter.pedestrian_speed_split(dataset.measurements,
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
            print ( " where ", np.where(dataset.meta_data[:, 0] == b'pedestrian'))


            print (sorted(float_data[:, np.where(dataset.meta_data[:, 0] == b'pedestrian')]))


            count += 1

        print (dist_calc)
        #plt.hist(dist_calc,1400)

        #plt.show()

    def test_regular_sampler(self):
        return
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



    def test_random_sampler(self):
        return
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

        #sampler = BatchSequenceSampler(splitted_steer_labels, 0, 120, g_conf.NUMBER_IMAGES_SEQUENCE,
        #                              g_conf.SEQUENCE_STRIDE, False)



        sampler = RandomSampler(keys, 0)

        big_steer_vec = []
        count =0

        print ("len keys", len(keys))

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  sampler=sampler,
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



    def test_real_data_central_sampler(self):
        return
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

    def test_lambda_sampler(self):
        float_data = np.random.randn(10, 3)
        meta_data = {'speed': 0, 'brake': 1, 'throttle': 2}
        keys = splitter.lambda_splitter(float_data, meta_data, [
            lambda x,y: np.where(
                np.logical_and(x[y['speed']]>0., x[y['brake']>0.])),
            lambda x,y: np.where(
                x[y['throttle']] > 0)]
        for k in keys[0]
            assert float_data[k, 0] > 0 and float_data[k, 1] > 0.
        for k in keys[1]
            assert float_data[k, 3] > 0

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
