import os
import numpy as np
import time
import unittest
import random
import math


from input import CoILDataset, Augmenter, BatchSequenceSampler, splitter

from configs import g_conf



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
