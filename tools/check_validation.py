#!/usr/bin/env python

import glob
import re


import argparse

import os
from collections import deque
import math
import copy
import json

import numpy as np

class Control:
    steer = 0
    throttle = 0
    brake = 0
    hand_brake = 0
    reverse = 0


# Configurations for this script


sensors = {'RGB': 3, 'labels': 3, 'depth': 0}
resolution = [800, 600]

classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



def augment_steering(camera_angle, steer, speed):
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
    val = 6 * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
    steer -= pos * min(val, 0.3)
    steer += neg * min(val, 0.3)

    steer = min(1.0, max(-1.0, steer))

    # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
    return steer

def augment_measurement(measurements, angle, speed):
    """
        Augment the steering of a measurement dict

    """
    new_steer = augment_steering(angle, measurements['steer'],
                                      speed)
    measurements['steer'] = new_steer
    return measurements


def _get_final_measurement( speed, measurement_data, angle, directions):
    """
    Function to load the measurement with a certain angle and augmented direction.
    Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

    :return:
    """
    if angle != 0:
        measurement_augmented = augment_measurement(copy.copy(measurement_data), angle, 3.6 * speed)
    else:
        # We have to copy since it reference a file.
        measurement_augmented = copy.copy(measurement_data)


    if 'gameTimestamp' in measurement_augmented:
        time_stamp = measurement_augmented['gameTimestamp']
    else:
        time_stamp =measurement_augmented['game_time']

    final_measurement = {'steer': measurement_augmented['steer'],
                     'steer_noise': measurement_augmented['steer_noise'],
                     'throttle': measurement_augmented['throttle'],
                     'throttle_noise': measurement_augmented['throttle_noise'],
                     'brake': measurement_augmented['brake'],
                     'brake_noise': measurement_augmented['brake_noise'],
                     'speed_module': speed/12.0,
                     'directions': directions,
                     "pedestrian": measurement_augmented['stop_pedestrian'],
                     "traffic_lights": measurement_augmented['stop_traffic_lights'],
                     "vehicle": measurement_augmented['stop_vehicle'],
                     "game_time": time_stamp,
                     'angle': angle}




    return final_measurement


def get_error(meas1, meas2):

    print (math.fabs(meas1 - meas2))
    return math.fabs(meas1 - meas2)


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    parser.add_argument(
        '--episodes',
        nargs='+',
        dest='episodes',
        type=str,
        default ='all'
    )

    parser.add_argument(
        '-ps',
        '--pred-file',
        dest='pred_file',
        type=str,
    )

    args = parser.parse_args()
    path = args.path

    # By setting episodes as all, it means that all episodes should be visualized
    if args.episodes == 'all':
        episodes_list = glob.glob(os.path.join(path, 'episode_*'))
    else:
        episodes_list = args.episodes



    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    step_size = 1
    # initial_positions =[20,25,48,68,79,105,108,120,130]
    # positions_to_test = []
    # for i in initial_positions:
    #  positions_to_test += range(i-1,i+2)



    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful

    ts = []

    total_number_of_seconds = 0
    total_number_of_checked_seconds = 0
    total_number_of_bad_seconds = 0

    ground_truth = np.loadtxt(args.pred_file, delimiter=",", skiprows=0, usecols=([0]))

    count = 0
    error = 0
    for episode in episodes_list:
        print ('Episode ', episode)
        if 'episode' not in episode:
            episode = 'episode_' + episode

        # Take all the measurements from a list
        measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
        sort_nicely(measurements_list)


        for measurement in measurements_list:

            with open(measurement) as f:
                measurement_data = json.load(f)

            # We extract the interesting subset from the measurement dict
            if 'forwardSpeed' in measurement_data['playerMeasurements']:
                speed = measurement_data['playerMeasurements']['forwardSpeed']
            else:
                speed = 0

            directions = measurement_data['directions']

            # Make central
            # Get line from predictions
            central_measurements = _get_final_measurement(speed, measurement_data, -30.0,
                                                            directions)


            error += get_error(ground_truth[count], central_measurements['steer'])
            count += 1


            # Left measurements
            left_measurements = _get_final_measurement(speed, measurement_data, -30.0,
                                                            directions)
            error += get_error(ground_truth[count], central_measurements['steer'])
            count += 1

            # Right measurements
            right_measurements = _get_final_measurement(speed, measurement_data, 0.0,
                                                            directions)
            error += get_error(ground_truth[count], central_measurements['steer'])
            count +=1
            #

