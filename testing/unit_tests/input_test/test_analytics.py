import os
import numpy as np
import unittest
import operator

from input import coil_sampler, splitter, CoILDataset, Augmenter, analytics
from configs import g_conf

class testSpliter(unittest.TestCase):


    def test_split_real_data_speed(self):

        root_test_dir = '/home/felipecodevilla/Datasets/02HoursW1-3-6-8'

        augmenter = Augmenter(g_conf.AUGMENTATION)
        dataset = CoILDataset(root_test_dir, augmenter)
        speed = dataset.measurements[10, :]

        print (dataset.meta_data)
        print (dataset.meta_data[:, 0])
        print ( " Where is control ",np.where(dataset.meta_data[:, 0] == b'control'))
        labels = dataset.measurements[np.where(dataset.meta_data[:, 0]  == b'control'), :]


        keys = range(0, len(speed))

        print (labels)


        conditions_vec = [(b'speed_module', operator.eq, 0)]

        mask = analytics.get_data_conditioned(conditions_vec, dataset.measurements, dataset.meta_data)


        print (" The sum is ", np.sum(dataset.measurements[np.where(dataset.meta_data[:, 0]  == b'speed_module'), mask]))




        """
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
        """
