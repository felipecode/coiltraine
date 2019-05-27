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

from . import splitter
from . import data_parser

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

from coilutils.general import sort_nicely

from cexp.cexp import CEXP
from cexp.env.environment import NoDataGenerated



def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def convert_scenario_name_number(measurements):

    if measurements['scenario'] == 'S0_lane_following':
        measurements['scenario'] = 0.0
    elif measurements['scenario'] == 'S1_intersection':
        measurements['scenario'] = 1.0
    elif measurements['scenario'] == 'S2_before_intersection':
        measurements['scenario'] = 2.0
    else:
        measurements['scenario'] = 3.0





def check_size(image_filename, size):
    img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    return img.shape[0] == size[1] and img.shape[1] == size[2]


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, transform=None, preload_name=None):

        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY ")
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'))
            print(self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements = self._pre_load_image_folders()

        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            img = cv2.imread(self.sensor_data_names[index], cv2.IMREAD_COLOR)
            # Apply the image transformation

            if self.transform is not None:
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

            # TODO remove hardcode
            measurements['rgb_central'] = img

            self.batch_read_number += 1
        except AttributeError:
            traceback.print_exc()
            print ("Blank IMAGE")
            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb_central'] = np.zeros(3, 88, 200)

        return measurements

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)

    def _get_final_measurement(self, speed, measurement_data, angle,
                               directions, avaliable_measurements_dict):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

        Returns
            The final measurement dict
        """
        if angle != 0:
            measurement_augmented = self.augment_measurement(copy.copy(measurement_data), angle,
                                                             3.6 * speed,
                                                 steer_name=avaliable_measurements_dict['steer'])
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)

        if 'gameTimestamp' in measurement_augmented:
            time_stamp = measurement_augmented['gameTimestamp']
        else:
            time_stamp = measurement_augmented['elapsed_seconds']

        final_measurement = {}
        # We go for every available measurement, previously tested
        # and update for the measurements vec that is used on the training.
        for measurement, name_in_dataset in avaliable_measurements_dict.items():
            # This is mapping the name of measurement in the target dataset
            final_measurement.update({measurement: measurement_augmented[name_in_dataset]})

        # Add now the measurements that actually need some kind of processing
        final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
        final_measurement.update({'directions': directions})
        final_measurement.update({'game_time': time_stamp})

        return final_measurement

    def _pre_load_image_folders(self):
        """
        We preload a dataset compleetely and pre process if necessary by using the
        C-EX interface.
        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """


        sensor_data_names = []
        jsonfile = g_conf.EXPERIENCE_FILE   # The experience file full path.
        # We check one image at least to see if matches the size expected by the network
        checked_image = False
        float_dicts = []
        env_batch = CEXP(jsonfile, params=None, iterations_to_execute=g_conf.NUMBER_ITERATIONS,
                         sequential=True)
        # Here we start the server without docker
        env_batch.start(no_server=True)  # no carla server mode.
        # count, we count the environments that are read

        # TODO add the lateral cameras for training.
        for env in env_batch:
            # it can be personalized to return different types of data.
            print("Environment Name: ", env)
            try:
                env_data = env.get_data()  # returns a basically a way to read all the data properly
            except NoDataGenerated:
                print("No data generate for episode ", env)
            else:
                for exp in env_data:
                    print("    Exp: ", exp[1])

                    for batch in exp[0]:
                        print("      Batch: ", batch[1])
                        for data_point in batch[0]:
                            for sensor in g_conf.SENSORS.keys():
                                if not checked_image:
                                    print (data_point[sensor], g_conf.SENSORS[sensor])
                                    if not check_size(data_point[sensor], g_conf.SENSORS[sensor]):
                                        raise RuntimeError("Unexpected image size for the network")
                                    checked_image = True
                                # TODO lunch meaningful exception if not found sensor name

                                sensor_data_names.append(data_point[sensor])
                            # We delete some non floatable cases
                            del data_point['measurements']['ego_actor']
                            del data_point['measurements']['opponents']
                            del data_point['measurements']['lane']
                            del data_point['measurements']['hand_brake']
                            del data_point['measurements']['reverse']
                            # Convert the scenario name to some floatable type.
                            convert_scenario_name_number(data_point['measurements'])
                            float_dicts.append(data_point['measurements'])

        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

        #episodes_list = glob.glob(os.path.join(path, 'episode_*'))
        #sort_nicely(episodes_list)
        # Do a check if the episodes list is empty
        #if len(episodes_list) == 0:
        #    raise ValueError("There are no episodes on the training dataset folder %s" % path)


    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

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

    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]


    """
        Methods to interact with the dataset attributes that are used for training.
    """

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

    def extract_intentions(self, data):
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
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

