import os
import glob
import traceback
import sys
import math
import copy
import json
import random
import numpy as np

import torch
import cv2

from torch.utils.data import Dataset

from logger import coil_logger

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from utils.general import sort_nicely



class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None):

        self.preload_name = preload_name
        if preload_name is not None and os.path.exists(os.path.join('_preloads', preload_name + '.npy')):
            print ( " Loading from NPY ")
            self.sensor_data_names, self.measurements  = np.load(os.path.join('_preloads', preload_name + '.npy'))
            print (self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements = self.pre_load_image_folders(root_dir)
        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):


        img_path = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME,
                                self.sensor_data_names[index])
        #print (img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transform is not None:
            img = self.transform(self.batch_read_number, img)
        else:
            img = img.transpose(2, 0, 1)

        img = img.astype(np.float)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img / 255.


        measurements = self.measurements[index].copy()
        for k, v in measurements.items():
            v = torch.from_numpy(np.asarray([v, ]))
            measurements[k] = v.float()

        # TODO: here just one image
        measurements['rgb'] = img

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
        sort_nicely(episodes_list)
        print (path)
        print (" Episodes list ")
        print (episodes_list)

        sensor_data_names = []
        float_dicts = []

        number_of_hours_pre_loaded = 0

        for episode in episodes_list:

            print('Episode ', episode)

            if not os.path.exists(os.path.join(episode, "checked")) and not os.path.exists(os.path.join(episode, "processed2")) \
                  and not os.path.exists(os.path.join(episode, "bad_episode")):
                # Episode was not checked. So we dont load it.
                print (" Not checked")
                continue


            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
                 # The number of wanted hours achieved
                 break


            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            if len (measurements_list) == 0:
                continue

            last_data_point_number = measurements_list[-1].split('_')[-1].split('.')[0]
            number_of_hours_pre_loaded += (float(last_data_point_number) / 10.0)/3600.0
            print (" Added ", ((float(last_data_point_number) / 10.0)/3600.0))


            for measurement in measurements_list[:-3]:



                data_point_number = measurement.split('_')[-1].split('.')[0]

                # TODO the dataset camera name can be a parameter
                with open(measurement) as f:
                    measurement_data = json.load(f)
                # We extract the interesting subset from the measurement dict
                if 'forwardSpeed' in  measurement_data['playerMeasurements']:
                    speed = measurement_data['playerMeasurements']['forwardSpeed']
                else:
                    speed = 0

                directions = self.augment_directions(measurement_data['directions'])


                float_dicts.append(
                    {'steer': measurement_data['steer'],
                     'throttle': measurement_data['throttle'],
                     'brake': measurement_data['brake'],
                     'speed_module': speed/g_conf.SPEED_FACTOR,
                     'directions': directions,
                     "pedestrian": measurement_data['stop_pedestrian'],
                     "traffic_lights": measurement_data['stop_traffic_lights'],
                     "vehicle": measurement_data['stop_vehicle'],
                     'angle': 0}
                )

                rgb = 'CentralRGB_' + data_point_number + '.png'
                print (os.path.join(episode.split('/')[-1], rgb))
                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))

                # We do measurements for the left side camera
                # #TOdo the angle does not need to be hardcoded
                # We convert the speed to KM/h for the augmentaiton
                measurement_left = self.augment_measurement(copy.copy(measurement_data), -30.0, 3.6*speed)

                # We extract the interesting subset from the measurement dict
                float_dicts.append(
                    {'steer': measurement_left['steer'],
                     'throttle': measurement_left['throttle'],
                     'brake': measurement_left['brake'],
                     'speed_module': speed/g_conf.SPEED_FACTOR,
                     'directions': directions,
                     "pedestrian": measurement_left['stop_pedestrian'],
                     "traffic_lights": measurement_left['stop_traffic_lights'],
                     "vehicle": measurement_left['stop_vehicle'],
                     'angle': -30.0}
                )
                rgb = 'LeftRGB_' + data_point_number + '.png'

                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))

                # We do measurements augmentation for the right side cameras

                measurement_right = self.augment_measurement(copy.copy(measurement_data), 30.0, 3.6*speed)


                float_dicts.append(
                    {'steer': measurement_right['steer'],
                     'throttle': measurement_right['throttle'],
                     'brake': measurement_right['brake'],
                     'speed_module': speed/g_conf.SPEED_FACTOR,
                     'directions': directions,
                     "pedestrian": measurement_right['stop_pedestrian'],
                     "traffic_lights": measurement_right['stop_traffic_lights'],
                     "vehicle": measurement_right['stop_vehicle'],
                     'angle': 30.0}
                )
                rgb = 'RightRGB_' + data_point_number + '.png'
                sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))


        print ( " LOADED ", number_of_hours_pre_loaded, " This hours")
        print ()
        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')

        np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts


    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions


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

        # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    def augment_measurement(self, measurements, angle, speed):
        """
            Augment the steering of a measurement dict

        """

        new_steer = self.augment_steering(angle, measurements['steer'],
                                          speed)

        measurements['steer'] = new_steer

        return measurements


    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)






if __name__ == "__main__":
    dataset = NewDataset(root_dir='')
    print(len(dataset))
    # print(dataset.sensor_data_names)
    for k, v in dataset[0].items():
        print(k, v.shape)
