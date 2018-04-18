import os
import numpy as np
import unittest


import time


import torch


from input import CoILDataset

from torchvision import transforms
from .augmenter_compositions import add_test_augmenter, mul_test_augmenter


class testAugmenter(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testAugmenter, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def get_data_loader(self):

        # TODO: FIND A solution for this TO TENSOR
        dataset = CoILDataset(self.root_test_dir, transform=transforms.Compose([transforms.ToTensor()]))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        return data_loader

    def test_basic_add(self):
        """ Test for the add augmentation, Basic test we assert doing operation and doing inverse"""
        data_loader = self.get_data_loader()

        count = 0
        for data in data_loader:
            image, labels = data
            result = add_test_augmenter(count, image['rgb'])
            self.assertAlmostEqual(torch.mean(result[0][0].cpu()),
                                   torch.mean(image['rgb'][0][0]))

            count += 1

    def test_random_add(self):
        if not os.path.exists(self.test_images_write_path + 'test_basic_add'):
            os.mkdir(self.test_images_write_path + 'test_basic_add')

        data_loader = self.get_data_loader()


        count = 0
        for data in data_loader:
            image, labels = data
            result = add_random_augmenter(count, image['rgb'])
            self.assertNotAlmostEqual(torch.mean(result[0][0], 0.0))

            count += 1


    def test_basic_mul(self):
        """ Test for the add augmentation, Basic test we assert doing operation and doing inverse"""
        data_loader = self.get_data_loader()

        count = 0
        for data in data_loader:
            image, labels = data
            result = mul_test_augmenter(count, image['rgb'])
            self.assertAlmostEqual(torch.mean(result[0][0].cpu()),
                                   torch.mean(image['rgb'][0][0]))

            count += 1

    def test_random_mul(self):
        pass



# TODO: We should remind our dear user to visually check the images after a test.