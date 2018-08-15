import os
import sys
import copy
import glob
import json
import math
import traceback

import h5py
import torch
import numpy as np

from scipy.misc import imread
from torch.utils.data import Dataset

# TODO: Warning, maybe this does not need to be included everywhere.
from logger import coil_logger
from configs import g_conf
from utils.general import sort_nicely


class NewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.sensor_data_names, self.measurements = self.pre_load_image_folders(root_dir)
        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        img_path = self.sensor_data_names[index]
        img = imread(img_path)
        if self.transform is not None:
            img = self.transform(self.batch_read_number, img)
        else:
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
        img = img/255.

        measurements = self.measurements[index]
        for k, v in measurements.items():
            v = torch.from_numpy(np.asarray([v, ]))
            measurements[k] = v.float()

        measurements['rgb'] = img.float()
        return measurements

    def pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now

        args
            the path for the dataset


        returns
         sensor data names: it is a vector with n dimensions being one for each sensor modality
         for instance, rgb only dataset will have a single vector with all the image names.
         float_data: all the wanted float data is loaded inside a vector, that is a vector
         of dictionaries.

        """

        episodes_list = glob.glob(os.path.join(path, 'episode_*'))

        sensor_data_names = []
        float_dicts = []

        for episode in episodes_list:
            print('Episode ', episode)

            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            for measurement in measurements_list:
                data_point_number = measurement.split('_')[-1].split('.')[0]

                # TODO the dataset camera name can be a parameter
                with open(measurement) as f:
                    measurement_data = json.load(f)
                # We extract the interesting subset from the measurement dict
                float_dicts.append(
                    {'steer': measurement_data['steer'],
                     'throttle':  measurement_data['throttle'],
                     'brake': measurement_data['brake'],
                     'speed_module': measurement_data['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_data['directions']}
                )

                rgb = 'CentralRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))

                # We do measurements for the left side camera
                # #TOdo the angle does not need to be hardcoded
                measurement_left = self.augment_measurement(measurement_data, -30.0)

                # We extract the interesting subset from the measurement dict
                float_dicts.append(
                    {'steer': measurement_left['steer'],
                     'throttle':  measurement_left['throttle'],
                     'brake': measurement_left['brake'],
                     'speed_module': measurement_left['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_left['directions']}
                )
                rgb = 'LeftRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))

                # We do measurements augmentation for the right side cameras

                measurement_right = self.augment_measurement(measurement_data, 30.0)
                float_dicts.append(
                    {'steer': measurement_right['steer'],
                     'throttle':  measurement_right['throttle'],
                     'brake': measurement_right['brake'],
                     'speed_module': measurement_right['playerMeasurements']['forwardSpeed'],
                     'directions': measurement_right['directions']}
                )
                rgb = 'RightRGB_' + data_point_number + '.jpg'
                sensor_data_names.append(os.path.join(episode, rgb))

        return sensor_data_names, float_dicts

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras.
        Args:
            camera_angle_batch:
            steer_batch:
            speed_batch:

        Returns:
            the augmented steering

        """

        time_use = 1.0
        car_length = 6.0
        old_steer = steer
        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))


        #print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer


    def augment_measurement(self, measurements, angle):
        """

            Augment the steering of a measurement dict


        """

        new_steer = self.augment_steering(angle, measurements['steer'],
                                          measurements['playerMeasurements']['forwardSpeed'])

        measurements['steer'] = new_steer
        return measurements


if __name__ == "__main__":
    dataset = NewDataset(root_dir='/home/eder/datasets/new_carla/')
    print(len(dataset))
    # print(dataset.sensor_data_names)
    for k, v in dataset[0].items():
        print(k, v.shape)
