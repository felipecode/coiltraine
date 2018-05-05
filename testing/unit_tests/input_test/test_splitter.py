import os
import numpy as np
import unittest

from input import coil_sampler, splitter, CoILDataset
from configs import g_conf

class testSpliter(unittest.TestCase):

    def generate_float_data(self):


        return np.random.normal(scale=0.1, size=(1400))



    def generate_label_data(self):

        return [2]*650 + [3]*400 + [4]*400




    def test_split(self):
        measurements = self.generate_float_data()
        labels = self.generate_label_data()

        keys = range(0, measurements.shape[0])
        splitted_labels = splitter.label_split(labels, keys, g_conf.param.INPUT.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(measurements, keys,
                                                 g_conf.param.INPUT.STEERING_DIVISION)



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
        measurements = self.generate_float_data()
        labels = self.generate_label_data()

        g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE = 20
        keys = range(0, measurements.shape[0])
        splitted_labels = splitter.label_split(labels, keys, g_conf.param.INPUT.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(measurements, keys,
                                                 g_conf.param.INPUT.STEERING_DIVISION)



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

        root_test_dir = 'testing/unit_tests/data'


        dataset = CoILDataset(root_test_dir)
        steerings = dataset.measurements[0, :]
        print (dataset.meta_data)
        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == 'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0] == 'control'), :]
        print (labels)

        print (np.unique(labels))

        keys = range(0, len(steerings))

        splitted_labels = splitter.label_split(labels[0][0], keys, g_conf.param.INPUT.LABELS_DIVISION)

        print (splitted_labels)
        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                 g_conf.param.INPUT.STEERING_DIVISION)

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



    def test_split_real_data_sequence(self):

        root_test_dir = 'testing/unit_tests/data'


        g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE = 20
        dataset = CoILDataset(root_test_dir)
        steerings = dataset.measurements[0, :]
        print (dataset.meta_data)
        # TODO: read meta data and turn into a coool dictionary ?
        print (np.where(dataset.meta_data[:, 0] == 'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0] == 'control'), :]

        print ("SEQUENCE LABELS ")
        print (labels)
        keys = range(0, len(steerings) - g_conf.param.MISC.NUMBER_IMAGES_SEQUENCE)

        splitted_labels = splitter.label_split(labels[0][0], keys, g_conf.param.INPUT.LABELS_DIVISION)

        # Another level of splitting
        splitted_steer_labels = []
        for keys in splitted_labels:
            splitter_steer = splitter.float_split(steerings, keys,
                                                 g_conf.param.INPUT.STEERING_DIVISION)

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