#!/usr/bin/env python

import argparse
import numpy as np
import h5py

import os

import random


from PIL import Image


resolution = [200, 88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    args = parser.parse_args()
    path = args.path

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []

    positions_to_test = range(0, len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))




    for h_num in positions_to_test:


        try:
            data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r+")
        except Exception as e:
            continue

        if not os.path.exists(os.path.join('_hdf5_images', 'seq' + str(h_num))):
            os.makedirs(os.path.join('_hdf5_images', 'seq' + str(h_num)))


        for i in range(200):
            image_to_save = Image.fromarray(np.array(data['rgb'][i].astype(np.uint8)))
            image_to_save.save(
                os.path.join('_hdf5_images', 'seq' + str(h_num),str(i)+'.jpg'))
            # redata = h5py.File('/media/adas/012B









