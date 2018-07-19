#!/usr/bin/env python

import argparse
import numpy as np
import h5py
from PIL import Image
import os

import random
import glob
import time


sensors = {'RGB': 3, 'labels': 3, 'depth': 3}
resolution = [200, 88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def plot_test_image(image, name):

    image_to_plot = Image.fromarray(image)
    image_to_plot.save(name)


MIN_PEDESTRIAN_PIXELS = 350

def is_there_a_pedestrian(image):

    number_pedestrian_pixels = len(np.argwhere(image == 0))

    return number_pedestrian_pixels > MIN_PEDESTRIAN_PIXELS






# gta_surface = get_gta_map_surface()


# THis script has the following objectives

# * Convert the semantic segmentation format
# * Add random augmentation to the direction
# * Point critical points passible of mistakes


def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


# ***** main loop *****

def add_pedestrian_label(data):
    num_data_entry = data['targets'][0].shape[0]

    # You copy all the targets to an array
    target_dataset = np.zeros((200, num_data_entry + 1))
    for i in range(0, 200):
        target_array = np.zeros(num_data_entry + 1)

        target_array[:-1] = data['targets'][i]

        target_dataset[i, :] = target_array

    del data['targets']
    targets = data.create_dataset('targets', (200, num_data_entry + 1,), dtype='f')

    for i in range(200):

        target_dataset[i, num_data_entry] = float(is_there_a_pedestrian(data['labels'][i]))

        data['targets'][i] = target_dataset[i]


    return data







# Code to add a label saying if there is a pedestrian

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    args = parser.parse_args()
    path = args.path



    count = 0


    files = glob.glob(os.path.join(path, 'data_*.h5'))
    for f in files:

        try:
            data = h5py.File(f, "r+")
        except Exception as e:
            continue


        data = add_pedestrian_label(data)


        # redata = h5py.File('/media/adas/012B
