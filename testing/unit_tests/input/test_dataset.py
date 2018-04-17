import os
import numpy as np
import time
import unittest



from input import CILDataset


class testCILDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'

    def test___get_item__(self):

        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        dataset = CILDataset(self.root_test_dir)

        capture_time = time.time()
        for i in range(len(dataset)):

            read_data = dataset[i]

        print ("Time to load", time.time() - capture_time)
        # number of frames fused equal 1, should return a simple case with three channels in the end.
        #dataset_configuration
        #self.assertEqual()
        # number of frames fused equal 3, should return 9 frames in the end

        pass


    def test_init(self):

        # Assert for error when read on wrong place
        with self.assertRaises(ValueError):
            _ = CILDataset("Wrong place")

        #
        dataset = CILDataset(self.root_test_dir)

        print dataset.shape

        # Assert for all
        print (dataset.images)


