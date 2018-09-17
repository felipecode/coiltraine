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



    def test_splitter_brake(self):


        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_brake(dataset.measurements, {'brake': [0]})
        print (" number of keys ", len(dataset.measurements))

        print ( " Lenghts of bins brake")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)

        print ("sum ", sum_of_keys)

    def test_left_right_brake(self):


        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_brake(dataset.measurements, {'left': [], 'central': [], 'right': []})
        print (" number of keys ", len(dataset.measurements))

        print ( " Lenghts of bins brake")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)

        print ("sum ", sum_of_keys)




    def test_splitter_speed(self):

        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_speed_module(dataset.measurements, {'speed_module': [0.0666,  0.208, 0.39]})

        print ( " Lenghts of bins speed")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)
            print ("sum ", sum_of_keys)

        print ("sum ", sum_of_keys)



    def test_splitter_speed_throttle(self):

        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_speed_module_throttle(dataset.measurements, {'speed_module': [0.8], 'throttle': [0.1]})

        print ( " Lenghts of bins speed throtle")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)

        print ("sum ", sum_of_keys)
    def test_lateral_noise_longitudinal_noise(self):

        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_speed_module_throttle(dataset.measurements, {'speed_module': [0.8], 'throttle': [0.1]})

        print ( " Lenghts of bins speed throtle")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)

        print ("sum ", sum_of_keys)

    def test_splitter_pedestrian_vehicle_tl(self):
        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None)

        keys = splitter.split_pedestrian_vehicle_tl(dataset.measurements, {})
        print ( " Lenghts of bins vehicle tl")
        sum_of_keys = 0
        for key in keys:
            print (len(key))
            sum_of_keys += len(key)
            print ("sum ", sum_of_keys)

        print ("sum ", sum_of_keys)


