import os
import numpy as np
import unittest


import time


import torch


from input import CoILDataset

from torchvision import transforms
from .augmenter_compositions import add_test_augmenter, \
    mul_test_augmenter, add_random_augmenter, mul_random_augmenter, muladd_cpu, muladd_gpu, mul_random_augmenter_color


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
            self.assertAlmostEqual(torch.mean(result[0].cpu()),
                                   torch.mean(image['rgb'][0][0]))

            count += 1

    def test_random_add(self):
        if not os.path.exists(self.test_images_write_path + 'test_basic_add'):
            os.mkdir(self.test_images_write_path + 'test_basic_add')

        data_loader = self.get_data_loader()


        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_basic_add', str(count)+'b.png'))
            result = add_random_augmenter(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_basic_add', str(count)+'.png'))
            count += 1


    def test_basic_mul(self):
        """ Test for the add augmentation, Basic test we assert doing operation and doing inverse"""
        data_loader = self.get_data_loader()

        count = 0
        for data in data_loader:
            image, labels = data
            result = mul_test_augmenter(count, image['rgb'])
            self.assertAlmostEqual(torch.mean(result[0].cpu()),
                                   torch.mean(image['rgb'][0][0]))

            count += 1



    def test_random_mul(self):
        if not os.path.exists(self.test_images_write_path + 'test_basic_mul'):
            os.mkdir(self.test_images_write_path + 'test_basic_mul')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_basic_mul', str(count)+'b.png'))
            result = mul_random_augmenter(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_basic_mul', str(count)+'.png'))
            count += 1

    def test_random_mul_color(self):
        if not os.path.exists(self.test_images_write_path + 'test_random_mul_color'):
            os.mkdir(self.test_images_write_path + 'test_random_mul_color')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_random_mul_color',
                                            str(count)+'b.png'))
            result = mul_random_augmenter_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_basic_mul_color',
                                            str(count)+'.png'))
            count += 1

    # TODO: We should remind our dear user to visually check the images after a test.

    def test_gpu_vs_cpu_speed(self):





        dataset_cpu = CoILDataset(self.root_test_dir, transform=muladd_cpu)
        data_loader_cpu = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        capture_time = time.time()
        for data in data_loader_cpu:
            image, labels = data

            result = image['rgb']

            count += 1
        print ("CPU Time =  ", time.time() - capture_time)

        data_loader = self.get_data_loader()

        count = 0
        capture_time = time.time()
        for data in data_loader:
            image, labels = data

            result = muladd_gpu(0, image['rgb'])

            count += 1

        print("Gpu Time =  ", time.time() - capture_time)