#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')

import argparse
import numpy as np
import h5py
import pygame


import argparse

import math

import os
import scipy
from collections import deque
from skimage.transform import resize

sys.path.append('drive_interfaces')


def sldist(c1, c2): return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)



class Control:
    steer = 0
    throttle = 0
    brake = 0
    hand_brake = 0
    reverse = 0


# Configurations for this script


sensors = {'RGB': 3, 'labels': 3, 'depth': 0}
resolution = [200, 88]
camera_id_position = 25
direction_position = 24
speed_position = 10
pos_x_position = 8
pos_y_position = 9
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


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
    old_steer = steer
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

def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def join_classes_for(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    # print compressed_labels_image.shape
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            compressed_labels_image[i, j, 0] = join_dic[labels_image[i, j, 0]]

    return compressed_labels_image


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    parser.add_argument('-ot', '--out_path', default=".")

    args = parser.parse_args()
    path = args.path
    out_path = args.out_path

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    step_size = 1
    # initial_positions =[20,25,48,68,79,105,108,120,130]
    # positions_to_test = []
    # for i in initial_positions:
    #  positions_to_test += range(i-1,i+2)

    positions_to_test = range(0, len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])+1)

    image_queue = deque()

    actions_queue = deque()

    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful

    with open(os.path.join(out_path, 'predictions.csv'), 'w') as camera_label_file:


        ts = []
        images = [np.zeros([resolution[1], resolution[0], 3])] * sensors['RGB']
        labels = [np.zeros([resolution[1], resolution[0], 1])] * sensors['labels']
        depths = [np.zeros([resolution[1], resolution[0], 3])] * sensors['depth']
        actions = [Control()] * sensors['RGB']
        actions_noise = [Control()] * sensors['RGB']

        first_time = True
        end_of_episodes = []
        count = 0
        for h_num in positions_to_test:

            print (" SEQUENCE NUMBER ", h_num)
            try:
                data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r")
            except Exception as e:
                print (e)
                continue

            for i in range(0, 200):
                steer = data['targets'][i][0]
                camera_label = data['targets'][i][26]
                speed = data['targets'][i][10]
                steer = augment_steering(camera_label, steer, speed)

                camera_label_file.write(str(steer) + ',' +
                                        str(data['targets'][i][1]) + ',' +
                                        str(data['targets'][i][2]) + '\n')

