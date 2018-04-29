import os
import numpy as np
import unittest


import time


import torch


from .coil_dataset import CoILDataset

from torchvision import transforms
from .augmenter_compositions import *

class testAugmenter(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testAugmenter, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/data'
        self.test_images_write_path = 'testing/_test_images_'


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
                                   torch.mean(image['rgb'][0][0]), places=5)

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


    def test_random_add_color(self):

        if not os.path.exists(self.test_images_write_path + 'add_random_augmenter_color'):
            os.mkdir(self.test_images_write_path + 'add_random_augmenter_color')

        data_loader = self.get_data_loader()


        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'add_random_augmenter_color', str(count)+'b.png'))
            result = add_random_augmenter_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())

            image_to_save.save(os.path.join(self.test_images_write_path + 'add_random_augmenter_color', str(count)+'.png'))
            count += 1


    def test_random_add_color_cpu(self):

        if not os.path.exists(self.test_images_write_path + 'add_random_augmenter_color_cpu'):
            os.mkdir(self.test_images_write_path + 'add_random_augmenter_color_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=add_random_augmenter_color_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)


        count = 0
        for data in data_loader:
            image, labels = data

            #print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][1][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'add_random_augmenter_color_cpu', str(count)+'.png'))


            count += 1


    def test_basic_mul(self):
        """ Test for the add augmentation, Basic test we assert doing operation and doing inverse"""
        data_loader = self.get_data_loader()

        count = 0
        for data in data_loader:
            image, labels = data
            result = mul_test_augmenter(count, image['rgb'])
            self.assertAlmostEqual(torch.mean(result[0].cpu()),
                                   torch.mean(image['rgb'][0][0]),  places=5)

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

            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_random_mul_color',
                                            str(count)+'b.png'))
            result = mul_random_augmenter_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[count].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_random_mul_color',
                                            str(count)+'.png'))
            count += 1

    def test_random_mul_color_sometimes(self):
        if not os.path.exists(self.test_images_write_path + 'test_random_mul_color_sometimes'):
            os.mkdir(self.test_images_write_path + 'test_random_mul_color_sometimes')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data

            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_random_mul_color_sometimes',
                                            str(count)+'b.png'))
            result = mult_random_color_sometimes(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[count].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_random_mul_color_sometimes',
                                            str(count)+'.png'))
            count += 1

    # TODO: We should remind our dear user to visually check the images after a test.





    def test_dropout(self):
        if not os.path.exists(self.test_images_write_path + 'test_dropout'):
            os.mkdir(self.test_images_write_path + 'test_dropout')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_dropout',
                                            str(count)+'b.png'))
            result = dropout_random(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_dropout',
                                            str(count)+'.png'))
            count += 1

    def test_dropout_color(self):
        if not os.path.exists(self.test_images_write_path + 'test_dropout_color'):
            os.mkdir(self.test_images_write_path + 'test_dropout_color')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_dropout_color',
                                            str(count)+'b.png'))
            result = dropout_random_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_dropout_color',
                                            str(count)+'.png'))
            count += 1


    def test_coarse_dropout(self):
        if not os.path.exists(self.test_images_write_path + 'test_coarse_dropout'):
            os.mkdir(self.test_images_write_path + 'test_coarse_dropout')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_coarse_dropout',
                                            str(count)+'b.png'))
            result = coarse_dropout_random(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[count].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_coarse_dropout',
                                            str(count)+'.png'))
            count += 1


    def test_coarse_dropout_color(self):
        if not os.path.exists(self.test_images_write_path + 'test_coarse_dropout_color'):
            os.mkdir(self.test_images_write_path + 'test_coarse_dropout_color')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_coarse_dropout_color',
                                            str(count)+'b.png'))
            result = coarse_dropout_random_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_coarse_dropout_color',
                                            str(count)+'.png'))
            count += 1


    def test_dropout_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_dropout_cpu'):
            os.mkdir(self.test_images_write_path + 'test_dropout_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=dropout_random_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)


        count = 0
        for data in data_loader:
            image, labels = data

            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'test_dropout_cpu', str(count)+'.png'))


            count += 1


    def test_coarse_dropout_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_coarse_dropout_cpu'):
            os.mkdir(self.test_images_write_path + 'test_coarse_dropout_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=coarse_dropout_random_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)


        count = 0
        for data in data_loader:
            image, labels = data

            #print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'test_coarse_dropout_cpu', str(count)+'.png'))


            count += 1

    def test_grayscale(self):
        if not os.path.exists(self.test_images_write_path + 'test_grayscale'):
            os.mkdir(self.test_images_write_path + 'test_grayscale')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_grayscale', str(count)+'b.png'))
            result = grayscale_test(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_grayscale', str(count)+'.png'))
            count += 1

    def test_grayscale_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_grayscale_cpu'):
            os.mkdir(self.test_images_write_path + 'test_grayscale_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=gaussian_blur_test_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        for data in data_loader:
            image, labels = data

            # print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'test_grayscale_cpu',
                                            str(count) + '.png'))

            count += 1

    def test_gaussian_blur(self):
        if not os.path.exists(self.test_images_write_path + 'test_gaussian_blur'):
            os.mkdir(self.test_images_write_path + 'test_gaussian_blur')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_gaussian_blur', str(count)+'b.png'))
            result = gaussian_blur_test(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_gaussian_blur', str(count)+'.png'))
            count += 1


    def test_gaussian_blur_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_gaussian_blur_cpu'):
            os.mkdir(self.test_images_write_path + 'test_gaussian_blur_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=gaussian_blur_test_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        for data in data_loader:
            image, labels = data

            # print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'test_gaussian_blur_cpu',
                                            str(count) + '.png'))

            count += 1

    def test_contrast_normalization(self):
        if not os.path.exists(self.test_images_write_path + 'test_contrast_normalization'):
            os.mkdir(self.test_images_write_path + 'test_contrast_normalization')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_contrast_normalization', str(count)+'b.png'))
            result = contrast_normalization_test(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_contrast_normalization', str(count)+'.png'))
            count += 1

    def test_contrast_normalization_color(self):
        if not os.path.exists(self.test_images_write_path + 'test_contrast_normalization_color'):
            os.mkdir(self.test_images_write_path + 'test_contrast_normalization_color')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_contrast_normalization_color', str(count)+'b.png'))
            result = contrast_normalization_test_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_contrast_normalization_color', str(count)+'.png'))
            count += 1


    def test_contrast_normalization_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_contrast_normalization_cpu'):
            os.mkdir(self.test_images_write_path + 'test_contrast_normalization_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=contrast_normalization_test_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        for data in data_loader:
            image, labels = data

            # print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(os.path.join(self.test_images_write_path + 'test_contrast_normalization_cpu',
                                            str(count) + '.png'))

            count += 1


    def test_additive_gaussian(self):
        if not os.path.exists(self.test_images_write_path + 'test_additive_gaussian'):
            os.mkdir(self.test_images_write_path + 'test_additive_gaussian')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_additive_gaussian', str(count)+'b.png'))
            result = additive_gaussian_test(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_additive_gaussian', str(count)+'.png'))
            count += 1

    def test_additive_gaussian_color(self):
        if not os.path.exists(self.test_images_write_path + 'test_additive_gaussian_color'):
            os.mkdir(self.test_images_write_path + 'test_additive_gaussian_color')

        data_loader = self.get_data_loader()
        count = 0
        for data in data_loader:
            image, labels = data
            image_to_save = transforms.ToPILImage()(image['rgb'][0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_additive_gaussian_color', str(count)+'b.png'))
            result = additive_gaussian_test_color(count, image['rgb'])
            image_to_save = transforms.ToPILImage()(result[0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'test_additive_gaussian_color', str(count)+'.png'))
            count += 1


    def test_additive_gaussian_cpu(self):
        if not os.path.exists(self.test_images_write_path + 'test_additive_gaussian_cpu'):
            os.mkdir(self.test_images_write_path + 'test_additive_gaussian_cpu')

        dataset_cpu = CoILDataset(self.root_test_dir, transform=additive_gaussian_test_cpu)
        data_loader = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        for data in data_loader:
            image, labels = data

            # print (torch.FloatTensor(image['rgb'][0][0])/255.0)
            image_to_save = transforms.ToPILImage()(image['rgb'][count][0].byte())

            image_to_save.save(
                os.path.join(self.test_images_write_path + 'test_additive_gaussian_cpu',
                             str(count) + '.png'))

            count += 1
