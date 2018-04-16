import os
import numpy as np
import unittest

from input import CILDataset

class testCILDataset(unittest.TestCase):

    def __init__(self,*args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)

        self.root_test_dir = 'test/unit_tests/data'





    def test___get_item__(self):

        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        CILDataset()

        # number of frames fused equal 1, should return a simple case with three channels in the end.
        dataset_configuration
        self.assertEqual()
        # number of frames fused equal 3, should return 9 frames in the end



    def test_init(self):

        dataset = CILDataset(self.root_test_dir)


