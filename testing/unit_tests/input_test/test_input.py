import os
import numpy as np
import unittest
import time

import torch
from coil_core.train import select_balancing_strategy

from input.coil_sampler import BatchSequenceSampler, SubsetSampler, RandomSampler, PreSplittedSampler
from input.coil_dataset import CoILDataset
from input import Augmenter
import input.splitter as splitter
from PIL import Image
#from utils.general import plot_test_image

from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


from torch.utils.data import TensorDataset as dset
from configs import g_conf

class testInput(unittest.TestCase):


    def test_new_sampler(self):
        test_images_write_path = 'testing/unit_tests/_test_images_'

        if not os.path.exists(test_images_write_path):
            os.mkdir(test_images_write_path)




        #augmenter = Augmenter(g_conf.AUGMENTATION)

        root_path = '/home/felipecodevilla/Datasets/Carla100Test2'

        dataset = CoILDataset(root_path, transform=None)

        g_conf.SPLIT = [['pedestrian', []], ['vehicle', []], ['traffic_lights', []],
                        ['weights', [0.0, 0.0, 0.0, 0.0, 1]]]


        data_loader = select_balancing_strategy(dataset, 0)



        count = 0
        capture_time = time.time()
        for data in data_loader:
            controls = data['directions']

            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]

            #print (controls)
            #print (dataset.extract_inputs(data))
            #print (dataset.extract_inputs(data))

            print (data['rgb'].shape)

            image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            image_to_save.save(os.path.join(test_images_write_path,
                                            str(count)+'c.png'))
            """
            image_to_save = transforms.ToPILImage()((data['rgb'][1].cpu()*255).type(torch.ByteTensor))
            image_to_save.save(os.path.join(test_images_write_path,
                                            str(count)+'l.png'))

            image_to_save = transforms.ToPILImage()((data['rgb'][2].cpu()*255).type(torch.ByteTensor))
            image_to_save.save(os.path.join(test_images_write_path,
                                            str(count)+'r.png'))
            """

                                            

            count += 1

        print ("Time =  ", ((time.time() - capture_time)/len(dataset.sensor_data_names)))