import os
import glob
import traceback
import collections
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
from . import splitter

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from utils.general import sort_nicely

def parse_boost_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print ('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'get_boost'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key



    return name, conf_dict


# THIS PARSING IS A BIT HARDCODED


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print ('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key



    return name, conf_dict



def get_episode_weather(episode):

    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print (" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])

class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None):

        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        # Add no brake on the preload name if there is no brake
        if 'brake' not in g_conf.TARGETS:
            self.preload_name += '_nobrake'

        # If  not all the weathers are present we keep without anything ( WE ASSUME THAT THIS HAS LENGHT 4)
        if len(g_conf.WEATHERS) < 4:
            self.preload_name = self.preload_name + '-'.join(str(e) for e in g_conf.WEATHERS)

        if self.preload_name is not None and os.path.exists(os.path.join('_preloads', self.preload_name + '.npy')):
            print ( " Loading from NPY ")
            self.sensor_data_names, self.measurements  = np.load(os.path.join('_preloads', self.preload_name + '.npy'))
            print (self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements = self.pre_load_image_folders(root_dir)
        self.transform = transform
        self.batch_read_number = 0
        #name, self.boost_params = parse_boost_configuration(g_conf.SPLIT)
        #self.boost_function = getattr(splitter, name)

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):

        img_path = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME,
                                self.sensor_data_names[index].split('/')[-2],
                                self.sensor_data_names[index].split('/')[-1])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transform is not None:
            #if g_conf.SPLIT is not None and g_conf.SPLIT is not 'None' and 'boost' in self.boost_params:
            #    boost = self.boost_function(self.measurements, index, self.boost_params)
            #else:
            #    boost = 1
            boost = 1
            img = self.transform(self.batch_read_number * boost, img)

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
        self.batch_read_number += 100

        return measurements

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measument data is not removable is because it is part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)


    def _get_final_measurement(self, speed, measurement_data, angle, directions):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

        :return:
        """
        if angle != 0:
            measurement_augmented = self.augment_measurement(copy.copy(measurement_data), angle, 3.6 * speed)
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)


        if 'brake' not in g_conf.TARGETS:
            # A bit of repeating code, but helps for the sake of clarity



            if measurement_augmented['brake'] > 0.01:
                final_throtle = -measurement_augmented['brake']
                final_throtle_noise = -measurement_augmented['brake_noise']
            else:
                final_throtle = measurement_augmented['throttle']
                final_throtle_noise = measurement_augmented['throttle_noise']


            final_measurement = {'steer': measurement_augmented['steer'],
                             'steer_noise': measurement_augmented['steer_noise'],
                             'throttle': final_throtle,
                             'throttle_noise': final_throtle_noise,
                             'speed_module': speed/g_conf.SPEED_FACTOR,
                             'directions': directions,
                             "pedestrian": measurement_augmented['stop_pedestrian'],
                             "traffic_lights": measurement_augmented['stop_traffic_lights'],
                             "vehicle": measurement_augmented['stop_vehicle'],
                             'angle': angle}

        else:
            final_measurement = {'steer': measurement_augmented['steer'],
                             'steer_noise': measurement_augmented['steer_noise'],
                             'throttle': measurement_augmented['throttle'],
                             'throttle_noise': measurement_augmented['throttle_noise'],
                             'brake': measurement_augmented['brake'],
                             'brake_noise': measurement_augmented['brake_noise'],
                             'speed_module': speed/g_conf.SPEED_FACTOR,
                             'directions': directions,
                             "pedestrian": measurement_augmented['stop_pedestrian'],
                             "traffic_lights": measurement_augmented['stop_traffic_lights'],
                             "vehicle": measurement_augmented['stop_vehicle'],
                             'angle': angle}


        return final_measurement

    def pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now

        args
            the path for the dataset
0

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
                  and not os.path.exists(os.path.join(episode, "bad_episode")) and \
                    not g_conf.TRAIN_DATASET_NAME == 'CARLA80TL':
                # Episode was not checked. So we dont load it.
                print (" Not checked")
                continue


            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
                 # The number of wanted hours achieved
                 break



            # Get all the measuremensts from this episode

            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            if len (measurements_list) == 0:
                print ("EMPTY EPISODE")
                continue

            if get_episode_weather(episode) not in g_conf.WEATHERS:
                print("WEATHER NOT CORRECT")
                continue



            # A simple count to keep track how many measurements were added this episode.
            count_added_measurements = 0

            for measurement in measurements_list[:-3]:


                data_point_number = measurement.split('_')[-1].split('.')[0]

                # TODO the dataset camera name can be a parameter
                with open(measurement) as f:
                    measurement_data = json.load(f)

                # depending on the configuration file, we eliminated the kind of measurements that are not
                # going to be used for this experiment



                # We extract the interesting subset from the measurement dict
                if 'forwardSpeed' in  measurement_data['playerMeasurements']:
                    speed = measurement_data['playerMeasurements']['forwardSpeed']
                else:
                    speed = 0




                directions = self.augment_directions(measurement_data['directions'])

                final_measurement = self._get_final_measurement(speed, measurement_data, 0, directions)

                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'CentralRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1


                # We do measurements for the left side camera
                # #TOdo the angle does not need to be hardcoded
                # We convert the speed to KM/h for the augmentaiton

                # We extract the interesting subset from the measurement dict

                final_measurement = self._get_final_measurement(speed, measurement_data, -30.0, directions)

                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'LeftRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1

                # We do measurements augmentation for the right side cameras


                final_measurement = self._get_final_measurement(speed, measurement_data, 30.0, directions)


                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'RightRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1

            # Check how many hours were actually added

            last_data_point_number = measurements_list[-4].split('_')[-1].split('.')[0]
            print ("last and float dicts len", last_data_point_number, count_added_measurements )

            print ("ERASED ", float(last_data_point_number)*3 -  count_added_measurements)

            number_of_hours_pre_loaded += (float(count_added_measurements / 10.0)/3600.0)
            print (" Added ", ((float(count_added_measurements) / 10.0)/3600.0))
            print (" TOtal Hours (partial) ", number_of_hours_pre_loaded)



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
