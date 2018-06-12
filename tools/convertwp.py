#!/usr/bin/env python
import sys

sys.path.append('utils')
sys.path.append('configuration')

import glob
import argparse
import numpy as np
import h5py
import math
import time
import scipy
import os

sys.path.append('drive_interfaces')


class Control:
    steer = 0
    gas = 0
    brake = 0
    hand_brake = 0
    reverse = 0


def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / dist, dist


def get_angle(vec_dst, vec_src):
    angle = math.atan2(vec_dst[1], vec_dst[0]) - math.atan2(vec_src[1], vec_src[0])
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


import random

###Specify maps, directory, number of h-files
# ***** main loop *****
if __name__ == "__main__":

    # Concatenate all files
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")
    parser.add_argument('-ot', '--out_path', default="")

    args = parser.parse_args()
    path_in = args.path
    out_path_dir = args.out_path

    if not os.path.exists(path_in):
        os.makedirs(path_in)
    if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)
    files = [os.path.join(path_in, f) for f in glob.glob1(path_in, "data_*.h5")]

    # We will append 4 entries for mag, angle of 2 waypoints
    num_new_entries = 4
    num_cameras = 3
    num_channels = 1
    addWPs = True  # end of file or replace old
    h5_last = 18700
    exception_list = []

    pos_x_ind = 8
    pos_y_ind = 9
    ori_x_ind = 21
    ori_y_ind = 22
    camera_ind = 25
    camera_angle_ind = 26
    wp1_x_ind = 27
    wp1_y_ind = 28
    wp2_x_ind = 29
    wp2_y_ind = 30

    sequence_num = range(0, h5_last + 1)

    for h_num in sequence_num:

        if (not (h_num in exception_list)):
            print (" SEQUENCE NUMBER ", h_num)
            data = h5py.File(path_in + 'data_' + str(h_num).zfill(5) + '.h5', "r")
            if (addWPs):  # Add waypoints or just update them
                num_data_entry = data['targets'][0].shape[0]
            else:
                num_data_entry = data['targets'][0].shape[0] - num_new_entries

            new_data = h5py.File(out_path_dir + 'data_' + str(h_num).zfill(5) + '.h5', "w")
            rgb = new_data.create_dataset('rgb', (200, 88, 200, 3), dtype=np.uint8)
            labels = new_data.create_dataset('labels', (200, 88, 200, num_channels), dtype=np.uint8)
            depth = new_data.create_dataset('depth', (200, 88, 200, 3), dtype=np.uint8)
            targets = new_data.create_dataset('targets', (200, num_data_entry + num_new_entries),
                                              'f')

            for i in range(0, 200):
                rgb[i] = data['rgb'][i]
                labels[i] = data['labels'][i]
                depth[i] = data['depth'][i]

                target_array = np.zeros(num_data_entry + num_new_entries)
                if (addWPs):
                    target_array[:-num_new_entries] = data['targets'][i]
                else:
                    target_array = data['targets'][i]

                loc_x_player = target_array[pos_x_ind]
                loc_y_player = target_array[pos_y_ind]
                ori_x_player = target_array[ori_x_ind]
                ori_y_player = target_array[ori_y_ind]
                wp1_x_player = target_array[wp1_x_ind]
                wp1_y_player = target_array[wp1_y_ind]
                wp2_x_player = target_array[wp2_x_ind]
                wp2_y_player = target_array[wp2_y_ind]

                wp1_vector, wp1_mag = get_vec_dist(wp1_x_player, wp1_y_player, loc_x_player,
                                                   loc_y_player)
                if wp1_mag > 0:
                    wp1_angle = get_angle(wp1_vector, [ori_x_player, ori_y_player]) - target_array[
                        camera_angle_ind] * math.pi / 180
                else:
                    wp1_angle = 0

                wp2_vector, wp2_mag = get_vec_dist(wp2_x_player, wp2_y_player, loc_x_player,
                                                   loc_y_player)
                if wp2_mag > 0:
                    wp2_angle = get_angle(wp2_vector, [ori_x_player, ori_y_player]) - target_array[
                        camera_angle_ind] * math.pi / 180
                else:
                    wp2_angle = 0

                # print(target_array[camera_ind], target_array[camera_angle_ind], wp1_angle, wp2_angle)

                target_array[num_data_entry] = wp1_angle
                target_array[num_data_entry + 1] = wp1_mag
                target_array[num_data_entry + 2] = wp2_angle
                target_array[num_data_entry + 3] = wp2_mag

                new_data['targets'][i] = target_array
        else:
            print (" SEQUENCE NUMBER ", h_num, " was skipped")
