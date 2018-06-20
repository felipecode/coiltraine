import os
import numpy as np
import time
import unittest
import torch
import random
import math


from PIL import Image
from input import CoILDataset, Augmenter, BatchSequenceSampler, splitter

from configs import g_conf

from torchvision import transforms



#TODO: verify images, there should be something for that !!


class testCILDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'


    """
    def test_get_item(self):
        if not os.path.exists(self.test_images_write_path + 'normal'):
            os.mkdir(self.test_images_write_path + 'normal')

        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames

        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset(self.root_test_dir, transform=augmenter)

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)
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
    """
    def test_speed_reading(self):
        if not os.path.exists(self.test_images_write_path + 'normal_steer'):
            os.mkdir(self.test_images_write_path + 'normal_steer')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], '1HoursW1-3-6-8')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter)

        keys = range(0, len(dataset.measurements[0, :]) - g_conf.NUMBER_IMAGES_SEQUENCE)
        sampler = BatchSequenceSampler(
            splitter.control_steer_split(dataset.measurements, dataset.meta_data, keys),
            0 * g_conf.BATCH_SIZE,
            g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE
        )

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)

        count = 0
        print('len ', len(data_loader))
        max_steer = 0
        for data in data_loader:
            print(count)
            image, labels = data

            count += 1
        print("MAX STEER ", max_steer)
    """
    def test_augmented_steering_batch(self):
        if not os.path.exists(self.test_images_write_path + 'normal_steer'):
            os.mkdir(self.test_images_write_path + 'normal_steer')



        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], '1HoursW1-3-6-8')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter)

        keys = range(0, len(dataset.measurements[0, :]) - g_conf.NUMBER_IMAGES_SEQUENCE)
        sampler = BatchSequenceSampler(
                splitter.control_steer_split(dataset.measurements, dataset.meta_data, keys),
                0 * g_conf.BATCH_SIZE,
                g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE
        )


        #data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        #capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)

        count = 0
        print ('len ', len(data_loader))
        max_steer = 0
        for data in data_loader:
            print (count)
            image, labels = data


            for pos in range(0,120,2):

                image_to_save = transforms.ToPILImage()(image['rgb'][0][0])
                if math.fabs(float(labels[pos][0][0])) > max_steer:
                    max_steer = math.fabs(float(labels[pos][0][0]))


                image_to_save.save(
                    os.path.join(self.test_images_write_path + 'normal_steer',
                                 str(float(labels[pos][0][0])) + '.png'))
            count += 1
        print ("MAX STEER ", max_steer)
    """