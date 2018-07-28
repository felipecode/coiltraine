import os
import numpy as np
import unittest

from input import coil_sampler, splitter, CoILDataset, Augmenter
from configs import g_conf

class testSpliter(unittest.TestCase):

    def generate_float_data(self):


        return np.random.normal(scale=0.1, size=(1400))



    def generate_label_data(self):

        return [2]*650 + [3]*400 + [4]*400




    def test_split(self):
        return
        measurements = self.generate_float_data()
        labels = self.generate_label_data()

        keys = range(0, measurements.shape[0])
        splitted_labels = splitter.label_split(labels, keys, g_conf.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(measurements, keys,
                                                 g_conf.STEERING_DIVISION)



            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += measurements[key]


                avg_now = sum_now/len(splitter_steer[i])
                print (avg_now)
                if i > 0:
                    self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now


            splitted_steer_labels.append(splitter_steer)


            # We assert if the new key is always bigger than the previous one
    def test_split_sequence(self):
        return
        measurements = self.generate_float_data()
        labels = self.generate_label_data()

        g_conf.NUMBER_IMAGES_SEQUENCE = 20
        keys = range(0, measurements.shape[0])
        splitted_labels = splitter.label_split(labels, keys, g_conf.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(measurements, keys,
                                                 g_conf.STEERING_DIVISION)



            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += measurements[key]


                avg_now = sum_now/len(splitter_steer[i])
                print (avg_now)
                #if i > 0:
                #self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now


            splitted_steer_labels.append(splitter_steer)


            # We assert if the new key is always bigger than the previous one


    def test_split_real_data(self):
        return
        root_test_dir = '/home/felipe/Datasets/CVPR02Noise/SeqTrain'

        print ('SPLITING REAL DATA !')
        dataset = CoILDataset(root_test_dir)
        steerings = dataset.measurements[0, :]

        print (dataset.meta_data)
        print (dataset.meta_data[:, 0])
        print ( " Where is control ",np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0]  == b'control'), :]


        keys = range(0, len(steerings))

        print (labels)
        splitted_labels = splitter.label_split(labels[0][0], keys, g_conf.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                 g_conf.STEERING_DIVISION)

            print(splitter_steer)


            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += steerings[key]


                avg_now = sum_now/len(splitter_steer[i])
                #print (avg_now)
                if i > 0:
                    self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now


            splitted_steer_labels.append(splitter_steer)


            # We assert if the new key is always bigger than the previous one


    def test_split_real_data_speed(self):
        return
        root_test_dir = '/home/felipecodevilla/Datasets/02HoursW1-3-6-8'

        print ('SPLITING REAL DATA !')
        augmenter = Augmenter(g_conf.AUGMENTATION)
        dataset = CoILDataset(root_test_dir, augmenter)
        speed = dataset.measurements[10, :]

        print (dataset.meta_data)
        print (dataset.meta_data[:, 0])
        print ( " Where is control ",np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0]  == b'control'), :]


        keys = range(0, len(speed))

        print (labels)

        splitted_labels = splitter.label_split(labels[0][0], keys, g_conf.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(speed, keys,
                                                 g_conf.SPEED_DIVISION)

            print(splitter_steer)


            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += speed[key]


                avg_now = sum_now/len(splitter_steer[i])
                print (avg_now)
                if i > 0:
                    self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now


            splitted_steer_labels.append(splitter_steer)


            # We assert if the new key is always bigger than the previous one


    def test_split_label_percentage(self):

        root_test_dir = '/home/felipecodevilla/Datasets/02HoursW1-3-6-8'

        print ('SPLITING REAL DATA !')
        augmenter = Augmenter(g_conf.AUGMENTATION)
        dataset = CoILDataset(root_test_dir, augmenter)
        speed = dataset.measurements[10, :]

        print (dataset.meta_data)
        print (dataset.meta_data[:, 0])
        print ( " Where is control ",np.where(dataset.meta_data[:, 0] == b'control'))
        print (dataset.measurements[np.where(dataset.meta_data[:, 0] == b'pedestrian'), :][0][0].astype(np.bool))
        print (dataset.measurements[np.where(dataset.meta_data[:, 0] == b'camera'), :][0][0] == 1)
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0] == b'pedestrian'), :][0][0].astype(np.bool) & \
                 (dataset.measurements[np.where(dataset.meta_data[:, 0] == b'camera'), :][0][0] == 1)



        keys = range(0, len(speed))

        print (sum(labels))
        print (sum(dataset.measurements[np.where(dataset.meta_data[:, 0] == b'pedestrian'), :][0][0].astype(np.bool)))

        splitted_labels = splitter.label_split(labels, keys, 10)


        #print (splitted_labels)

        # Another level of splitting
        splitted_steer_labels = []


        for key in splitted_labels[0]:

            self.assertEqual(labels[key], 1.0)

        for key in splitted_labels[1]:

            self.assertEqual(labels[key], 0.0)


        # The first part should have all the label



    def test_split_real_data_sequence(self):

        return
        root_test_dir = 'testing/unit_tests/data'


        g_conf.NUMBER_IMAGES_SEQUENCE = 20
        g_conf.LABELS_DIVISION = [[0, 2, 5], [0, 2, 5], [0, 2, 5]]
        dataset = CoILDataset(root_test_dir)
        steerings = dataset.measurements[0, :]
        print (dataset.meta_data)
        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == 'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0] == 'control'), :]

        print ("SEQUENCE LABELS ")
        print (labels)
        keys = range(0, len(steerings) - g_conf.NUMBER_IMAGES_SEQUENCE)

        splitted_labels = splitter.label_split(labels[0][0], keys, g_conf.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                 g_conf.STEERING_DIVISION)

            print (splitter_steer)


            for i in range(0, len(splitter_steer)):
                sum_now = 0
                for key in splitter_steer[i]:
                    sum_now += steerings[key]


                avg_now = sum_now/len(splitter_steer[i])
                #print (avg_now)
                if i > 0:
                    self.assertLess(avg_previous, avg_now)

                avg_previous = avg_now


            splitted_steer_labels.append(splitter_steer)


            # We assert if the new key is always bigger than the previous one