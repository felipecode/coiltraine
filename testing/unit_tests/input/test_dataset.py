import os
import numpy as np
import time
import unittest
import torch


from input import CoILDataset


class testCILDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'

    def test___get_item__(self):

        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        dataset = CoILDataset(self.root_test_dir)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)
        capture_time = time.time()
        for data in data_loader:

            image, labels = data
            print (image['rgb'].shape)

        print ("Time to load", time.time() - capture_time)
        # number of frames fused equal 1, should return a simple case with three channels in the end.
        #dataset_configuration
        #self.assertEqual()
        # number of frames fused equal 3, should return 9 frames in the end

        pass


    def test_init(self):

        # Assert for error when read on wrong place
        with self.assertRaises(ValueError):
            _ = CoILDataset("Wrong place")

        #
        dataset = CoILDataset(self.root_test_dir)

        print (len(dataset.sensor_data))
        print (dataset.sensor_data[0])
        # Assert for all
        #print (dataset.images)


