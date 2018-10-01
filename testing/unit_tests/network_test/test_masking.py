import os

#import matplotlib.pyplot as plt
import numpy as np
import time
import unittest
import random
import math

import torch

from PIL import Image

from torchvision import transforms
from configs import g_conf


import input
import reference

from coil_core.train import select_balancing_strategy
from network import CoILModel


#TODO: verify images, there should be something for that !!


class testMasking(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testMasking, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def test_masking(self):

        if not os.path.exists(self.test_images_write_path + 'central'):
            os.mkdir(self.test_images_write_path + 'central')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'AttValidation'
        g_conf.NUMBER_OF_ITERATIONS = 10000
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'all'



        g_conf.SPLIT = [['pedestrian', []], ['vehicle', []], ['traffic_lights', []],
                        ['weights', [0.0, 0.0, 0.0, 0.0, 1.0]]]

        augmenter = input.Augmenter(None)
        dataset = input.CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) +
                                           'hours_' + g_conf.TRAIN_DATASET_NAME)



        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()


        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:


            branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())



            #image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            #b, g, r = image_to_save.split()
            #image_to_save = Image.merge("RGB", (r, g, b))

            #image_to_save.save(os.path.join(self.test_images_write_path + 'central', str(count)+'l.png'))



            # Test steerings after augmentation
            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            #print ("directions", data['directions'], " speed_module", data['speed_module'])
            count += 1

