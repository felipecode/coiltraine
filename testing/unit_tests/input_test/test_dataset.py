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


#TODO: verify images, there should be something for that !!


class testCILDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testCILDataset, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def test_pre_load_augmentation(self):
        return
        if not os.path.exists(self.test_images_write_path + 'normal_steer'):
            os.mkdir(self.test_images_write_path + 'normal_steer')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100_2')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.NUMBER_OF_HOURS = 0.2
        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter)

        keys = range(0, len(dataset.sensor_data_names))


        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)

        count = 0
        print('len ', len(data_loader))
        max_steer = 0
        for data in data_loader:




            # Test steerings after augmentation

            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])

            #print ("directions", data['directions'], " speed_module", data['speed_module'])

            print ("brake ", data['brake'].data, " throttle", data['throttle'])


            self.assertLess(data['speed_module'],1)

            count += 1

    def test_pre_load_weather_augmentation(self):
        return
        if not os.path.exists(self.test_images_write_path + 'weather_aug'):
            os.mkdir(self.test_images_write_path + 'weather_aug')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_HOURS = 25
        g_conf.WEATHERS = [1]
        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        keys = range(0, len(dataset.sensor_data_names))


        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:


            image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            image_to_save.save(os.path.join(self.test_images_write_path + 'weather_aug', str(count)+'l.png'))
            # Test steerings after augmentation

            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])

            #print ("directions", data['directions'], " speed_module", data['speed_module'])
            count += 1

    def test_pre_select_central(self):
        return
        if not os.path.exists(self.test_images_write_path + 'central'):
            os.mkdir(self.test_images_write_path + 'central')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'central'

        g_conf.SPLIT = [['speed_module', [0.0666, 0.208, 0.39]], ['weights', [1.0, 0.0, 0.0, 0.0]]]

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)



        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:
            print (data['angle'])
            print (data['speed_module'])

            #image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            #image_to_save.save(os.path.join(self.test_images_write_path + 'central', str(count)+'l.png'))
            # Test steerings after augmentation
            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            #print ("directions", data['directions'], " speed_module", data['speed_module'])
            count += 1

    def test_pre_select_no_left_traffic(self):
        return
        if not os.path.exists(self.test_images_write_path + 'central'):
            os.mkdir(self.test_images_write_path + 'central')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 10000
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'central'

        g_conf.REMOVE = [['angle', -30], ['traffic_lights', 1]]

        g_conf.SPLIT = [['speed_module', [0.0666,  0.208, 0.39]], ['weights', [1.0, 0.0, 0.0, 0.0]]]

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)



        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:
            print (data['angle'])
            print (data['speed_module'])
            for i in range(120):
                if data['angle'][i][0] == -30:



                    self.assertEqual(data['traffic_lights'][i][0], 1)

            #image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            #image_to_save.save(os.path.join(self.test_images_write_path + 'central', str(count)+'l.png'))
            # Test steerings after augmentation
            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            #print ("directions", data['directions'], " speed_module", data['speed_module'])
            count += 1

    def test_pre_select_no_dynamic(self):
        return
        if not os.path.exists(self.test_images_write_path + 'central'):
            os.mkdir(self.test_images_write_path + 'central')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 10000
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'central'

        g_conf.REMOVE = [['traffic_lights', 0]]

        g_conf.SPLIT = [['speed_module', [0.0666,  0.208, 0.39]], ['weights', [1.0, 0.0, 0.0, 0.0]]]

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)



        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:
            print (data['pedestrian'])
            print (data['vehicle'])
            print (data['traffic_lights'])
            for i in range(120):


                    self.assertEqual(data['traffic_lights'][i][0], 1)

                    self.assertEqual(data['vehicle'][i][0], 1)

                    self.assertEqual(data['pedestrian'][i][0], 1)

            #image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            #image_to_save.save(os.path.join(self.test_images_write_path + 'central', str(count)+'l.png'))
            # Test steerings after augmentation
            #print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            #print ("directions", data['directions'], " speed_module", data['speed_module'])
            count += 1
    def test_add_remove_on_preload(self):
        return

        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 10000
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'central'

        g_conf.REMOVE = [['angle', -30],['traffic_lights', 1]]

        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:

            print (data['angle'])
            print (data['traffic_lights'])
            for i in range(120):
                if data['angle'][i][0] == -30:



                    self.assertEqual(data['traffic_lights'][i][0], 1)


            """
            print (count)
            count += 1
            if count % 200 != 0:
                continue

            for i in range(120):
                image_to_save = transforms.ToPILImage()((data['rgb'][i].cpu()*255).type(torch.ByteTensor))

                b, g, r = image_to_save.split()
                image_to_save = Image.merge("RGB", (r, g, b))

                image_to_save.save(os.path.join(self.test_images_write_path + 'augmentation', str(count) + '_' + str(i) + '.png'))
            """

            # Test steerings after augmentation
            # print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            # print ("directions", data['directions'], " speed_module", data['speed_module'])

    def test_eliminating_brake(self):
        return
        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TARGETS = ['steer', 'throttle']
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 2000
        g_conf.NUMBER_OF_HOURS = 1
        g_conf.DATA_USED = 'all'


        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        throttle_vec = []
        for data in data_loader:
            print (count)
            for i in range(120):

                throttle_vec.append(data['throttle'].cpu()[0])

            count += 1
            if count > 2000:
                break

            """
            print (count)
            count += 1
            if count % 200 != 0:
                continue

            for i in range(120):
                image_to_save = transforms.ToPILImage()((data['rgb'][i].cpu()*255).type(torch.ByteTensor))

                b, g, r = image_to_save.split()
                image_to_save = Image.merge("RGB", (r, g, b))

                image_to_save.save(os.path.join(self.test_images_write_path + 'augmentation', str(count) + '_' + str(i) + '.png'))
            """
        x = range(len(throttle_vec))
        plt.plot(x, throttle_vec)
        plt.show()
        # Test steerings after augmentation
        # print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
        # print ("directions", data['directions'], " speed_module", data['speed_module'])

    def test_compare_hdf5_other(self):

        print (" COMPARE HDF5")
        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset_images = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA80TL')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TARGETS = ['steer', 'throttle', 'brake']
        g_conf.TRAIN_DATASET_NAME = 'CARLA80TL'
        g_conf.NUMBER_OF_ITERATIONS = 200000
        g_conf.NUMBER_OF_HOURS = 2
        g_conf.DATA_USED = 'all'
        print ("COMPARE HDF5")

        augmenter = input.Augmenter(None)
        dataset = input.CoILDataset(full_dataset_images, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()

        data_loader_images = iter(torch.utils.data.DataLoader(dataset,
                                                       num_workers=0,
                                                       pin_memory=True))

        #data_loader_images = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader_images))
        max_steer = 0
        count = 0
        throttle_vec = []


        full_dataset_hdf5 = os.path.join('/media/eder/Seagate Expansion Drive/data/CVPR1Noise/SeqTrain')

        dataset = reference.CoILDataset(full_dataset_hdf5, transform=augmenter)
        # capture_time = time.time()

        print("Getting dataloader")

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader_hdf5 = iter(torch.utils.data.DataLoader(dataset,
                                                       num_workers=0,
                                                       pin_memory=True))

        max_steer = 0
        count = 0
        throttle_vec = []

        for i in range(30000):

            h_images, h_labels = next(data_loader_hdf5)
            i_data = next(data_loader_images)

            print ('hdf5 ', (h_labels[0][2].data), ' ', h_labels[0][26], ' ', h_labels[0][20],
                   'images ', i_data['brake'].data, ' ', i_data['angle'], ' ', i_data['game_time'])


            """
            for i in range(120):

                throttle_vec.append(data['throttle'].cpu()[0])

            count += 1
            if count > 2000:
                break
            """
            """
            print (count)
            count += 1
            if count % 200 != 0:
                continue

            for i in range(120):
                image_to_save = transforms.ToPILImage()((data['rgb'][i].cpu()*255).type(torch.ByteTensor))

                b, g, r = image_to_save.split()
                image_to_save = Image.merge("RGB", (r, g, b))

                image_to_save.save(os.path.join(self.test_images_write_path + 'augmentation', str(count) + '_' + str(i) + '.png'))
            """
        #x = range(len(throttle_vec))
        #plt.plot(x, throttle_vec)
        #plt.show()
        # Test steerings after augmentation
        # print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
        # print ("directions", data['directions'], " speed_module", data['speed_module'])

    def test_speed(self):
        return
        print(" COMPARE HDF5")
        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset_images = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TARGETS = ['steer', 'throttle', 'brake']
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 200000
        g_conf.NUMBER_OF_HOURS = 2
        g_conf.DATA_USED = 'all'
        print("COMPARE HDF5")

        augmenter = input.Augmenter(None)
        dataset = input.CoILDataset(full_dataset_images, transform=augmenter,
                                    preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()

        data_loader_images = iter(torch.utils.data.DataLoader(dataset,
                                                              num_workers=0,
                                                              pin_memory=True))

        # data_loader_images = select_balancing_strategy(dataset, 0, 12)
        # print('len ', len(data_loader_images))
        max_steer = 0
        count = 0
        throttle_vec = []


        max_steer = 0
        count = 0
        throttle_vec = []

        for i in range(5000):
            i_data = next(data_loader_images)

            print('images ', i_data['speed_module'].data*12.0, ' ', i_data['angle'], ' ', i_data['game_time'],
                  'images_speed ', i_data['speed_module'].data * 3.6 *12.0, ' ', i_data['angle'], ' ', i_data['game_time'])




    def test_old_data(self):
        return
        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA80TL')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TARGETS = ['steer', 'throttle', 'brake']
        g_conf.TRAIN_DATASET_NAME = 'CARLA80TL'
        g_conf.NUMBER_OF_ITERATIONS = 200000
        g_conf.NUMBER_OF_HOURS = 240
        g_conf.DATA_USED = 'all'


        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        throttle_vec = []
        for data in data_loader:
            print (count)
            print (data['steer'])
            print (data['angle'])

            print (data['speed_module'])

            """
            for i in range(120):

                throttle_vec.append(data['throttle'].cpu()[0])

            count += 1
            if count > 2000:
                break
            """
            """
            print (count)
            count += 1
            if count % 200 != 0:
                continue

            for i in range(120):
                image_to_save = transforms.ToPILImage()((data['rgb'][i].cpu()*255).type(torch.ByteTensor))

                b, g, r = image_to_save.split()
                image_to_save = Image.merge("RGB", (r, g, b))

                image_to_save.save(os.path.join(self.test_images_write_path + 'augmentation', str(count) + '_' + str(i) + '.png'))
            """
        #x = range(len(throttle_vec))
        #plt.plot(x, throttle_vec)
        #plt.show()
        # Test steerings after augmentation
        # print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
        # print ("directions", data['directions'], " speed_module", data['speed_module'])


    def test_add_augmentation(self):
        return
        if not os.path.exists(self.test_images_write_path + 'augmentation'):
            os.mkdir(self.test_images_write_path + 'augmentation')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.TRAIN_DATASET_NAME = 'CARLA100'
        g_conf.NUMBER_OF_ITERATIONS = 10000
        g_conf.NUMBER_OF_HOURS = 50
        g_conf.DATA_USED = 'central'

        #g_conf.REMOVE = [['traffic_lights', 0]]

        g_conf.SPLIT = \
            [['pedestrian', []], ['vehicle', []], ['traffic_lights', []], ['weights', [0.2, 0.2, 0.2, 0.2, 0.2]],
             ['boost', [20, 20, 0, 0, 0]]]

        augmenter = Augmenter('soft_harder')
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = select_balancing_strategy(dataset, 0, 12)
        print('len ', len(data_loader))
        max_steer = 0
        count = 0
        for data in data_loader:

            """
            for i in range(120):
                self.assertEqual(data['traffic_lights'][i][0], 1)

                self.assertEqual(data['vehicle'][i][0], 1)

                self.assertEqual(data['pedestrian'][i][0], 1)
            """
            print (count)
            count += 1
            if count % 200 != 0:
                continue

            for i in range(120):
                image_to_save = transforms.ToPILImage()((data['rgb'][i].cpu()*255).type(torch.ByteTensor))

                b, g, r = image_to_save.split()
                image_to_save = Image.merge("RGB", (r, g, b))

                image_to_save.save(os.path.join(self.test_images_write_path + 'augmentation', str(count) + '_' + str(i) + '.png'))
            # Test steerings after augmentation
            # print("steer: ", data['steer'][0], "angle: ", data['angle'][0])
            # print ("directions", data['directions'], " speed_module", data['speed_module'])



    def test_steering_augmentation(self):
        return
        if not os.path.exists(self.test_images_write_path + 'normal_steer'):
            os.mkdir(self.test_images_write_path + 'normal_steer')


        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100_2')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.NUMBER_OF_HOURS = 0.2
        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter)

        keys = range(0, len(dataset.sensor_data_names))


        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)


        count = 0
        print('len ', len(data_loader))
        max_steer = 0
        for data in data_loader:

            image, labels = data
            print (count)
            print ('steer', labels[0][0])
            print ('speed', labels[0][np.where(dataset.meta_data[:, 0] == b'speed_module')])
            print ('angle', labels[0][np.where(dataset.meta_data[:, 0] == b'angle')])


            count += 1
        print("MAX STEER ", max_steer)

    def test_speed_reading(self):
        return
        if not os.path.exists(self.test_images_write_path + 'normal_steer'):
            os.mkdir(self.test_images_write_path + 'normal_steer')
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CARLA100_2')
        # This depends on the number of fused frames. A image could have
        # A certain number of fused frames
        g_conf.NUMBER_OF_HOURS = 0.2
        augmenter = Augmenter(None)
        g_conf.NUMBER_OF_ITERATIONS = 120
        dataset = CoILDataset(full_dataset, transform=augmenter)

        keys = range(0, len(dataset.sensor_data_names))

        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
        #                                          shuffle=True, num_workers=12, pin_memory=True)
        # capture_time = time.time()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)

        count = 0
        print('len ', len(data_loader))
        max_steer = 0
        start_time = time.time()
        for data in data_loader:

            print (count)

            count += 1
            if count > g_conf.NUMBER_OF_ITERATIONS:
                break

        print("Imgs /s ", (120*120)/(time.time() - start_time))
