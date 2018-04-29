import os
import numpy as np
import time
import unittest
import torch


from PIL import Image
from input import CoILDataset
from input import aug
from input import aug_cpu

from torchvision import transforms



#TODO: verify images, there should be something for that !!


class testCILDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'

    def test_get_item(self):
        if not os.path.exists(self.test_images_write_path + 'normal'):
            os.mkdir(self.test_images_write_path + 'normal')

        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        dataset = CoILDataset(self.root_test_dir)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)
        capture_time = time.time()
        count = 0
        for data in data_loader:

            image, labels = data
            print (image['rgb'].shape)

            image_to_save = transforms.ToPILImage()(image['rgb'][0][0])

            image_to_save.save(os.path.join(self.test_images_write_path + 'normal', str(count)+'.png'))
            count += 1

        print ("Time to load", time.time() - capture_time)
        # number of frames fused equal 1, should return a simple case with three channels in the end.
        #dataset_configuration
        #self.assertEqual()
        #TODO: Test frame fusion
        # number of frames fused equal 3, should return 9 frames in the end


#TODO Basic Augmentation tests to be removed ! !! ! ! ! ! ! ! ! ! ! ! ! !

    def test_get_item_augmented(self):
        # Function to test the augmentation

        if not os.path.exists(self.test_images_write_path + 'augmented'):
            os.mkdir(self.test_images_write_path + 'augmented')


        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        dataset = CoILDataset(self.root_test_dir, transform=aug_cpu)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)
        capture_time = time.time()
        count=0
        for data in data_loader:

            image, labels = data
            print (image['rgb'].shape)

            image_to_save = transforms.ToPILImage()(image['rgb'][0][0])

            image_to_save.save(os.path.join(self.test_images_write_path + 'augmented', str(count)+'.png'))
            count +=1

        print("Time to load AUGMENT", time.time() - capture_time)
        # number of frames fused equal 1, should return a simple case with three channels in the end.
        #dataset_configuration
        #self.assertEqual()
        #TODO: Test frame fusion
        # number of frames fused equal 3, should return 9 frames in the end


    def test_get_item_augmented_gpu(self):


        if not os.path.exists(self.test_images_write_path + 'augmented_gpu'):
            os.mkdir(self.test_images_write_path + 'augmented_gpu')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        dataset = CoILDataset(self.root_test_dir, transform=transforms.Compose([
        transforms.ToTensor()]))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)
        capture_time = time.time()
        count = 0
        for data in data_loader:
            image, labels = data
            result = aug(image['rgb'])

            image_to_save = transforms.ToPILImage()(result[0][0].cpu())
            image_to_save.save(os.path.join(self.test_images_write_path + 'augmented_gpu', str(count)+'.png'))
            count += 1


        print ("Time to load AUGMENT", time.time() - capture_time)

        # number of frames fused equal 1, should return a simple case with three channels in the end.
        #dataset_configuration


        #self.assertEqual()
        #TODO: Test frame fusion
        # number of frames fused equal 3, should return 9 frames in the end


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


