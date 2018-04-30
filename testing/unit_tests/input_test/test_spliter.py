import os
import numpy as np
import unittest

from input import coil_sampler,splitter
from configs import g_conf

class testSpliter(unittest.TestCase):

    def generate_float_data(self):


        return np.random.normal(scale=0.1, size=(1400))



    def generate_label_data(self):


        return [2]*500 + [3]*250 + [4]*2500

    def test_split(self):
        measurements = self.generate_float_data()
        labels = self.generate_label_data()


        keys = range(0, measurements.shape[0])
        splitted_keys = splitter.label_split(labels, keys)

        splitted_keys = splitter.float_split(measurements, splitted_keys,
                                             self.param.INPUT.STEERING_DIVISION)