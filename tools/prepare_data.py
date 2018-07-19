#!/usr/bin/env python

import argparse
import numpy as np
import h5py

import os

import random

from add_metadata import fill_metadata
from add_pedestrian_label import add_pedestrian_label



import glob



sensors = {'RGB': 3, 'labels': 3, 'depth': 3}
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




def is_hdf5_prepared(filename):
    """
        We add this checking to verify if the hdf5 file has all the necessary metadata needed for performing,
        our trainings.
        # TODO: I dont know the scope but maybe this can change depending on the system. BUt i want to keep this for
        CARLA

    """

    data = h5py.File(filename, "r+")

    # Check if the number of metadata is correct, the current number is 28

    print (len(data['targets'][0]))

    if len(data['metadata_targets']) < 32:
        return False
    if len(data['targets'][0]) < 32:
        return False





    return True

    # Check if the target data has the same size as the metadata




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



    files = glob.glob(os.path.join(path, 'data_*.h5'))
    for f in files:

        try:
            data = h5py.File(f, "r+")
        except Exception as e:
            continue

        data = add_pedestrian_label(data)
        data = fill_metadata(data)

        data.close()

        if not is_hdf5_prepared(f):
            raise ValueError("Not working ")



