#!/usr/bin/env python

import argparse
import numpy as np
import h5py

import os

import random



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


    #if not os.path.exists(path_clean):
    #    os.mkdir(path_clean)

    for h_num in positions_to_test:


        try:
            data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r+")
        except Exception as e:
            continue

        # redata = h5py.File('/media/adas/012B

        #new_data_images = new_data.create_dataset('rgb', (200, 88, 200, 3), dtype=np.uint8)
        #new_data_labels = new_data.create_dataset('labels', (200, 88, 200, 1), dtype=np.uint8)
        # new_data_depth= new_data.create_dataset('depth', (200,88,200,3),dtype=np.uint8)
        #targets = new_data.create_dataset('targets', (200, data['targets'][0].shape[0]), 'f')

        dt = h5py.special_dtype(vlen=str)  # PY3

        metadata = data.create_dataset('metadata_targets', (31, 2, ), dtype=dt)

        metadata[0,0], metadata[0,1] = 'steer', 'float'
        metadata[1,0], metadata[1,1] = 'throttle', 'float'
        metadata[2,0], metadata[2,1] = 'brake','float'
        metadata[3,0], metadata[3,1] = 'hand_brake','bool'
        metadata[4,0], metadata[4,1] = 'reverse_gear','bool'
        metadata[5,0], metadata[5,1] = 'steer_noise','float'
        metadata[6,0], metadata[6,1] = 'gas_noise','float'
        metadata[7,0], metadata[7,1] = 'brake_noise','float'
        metadata[8,0], metadata[8,1] = 'x_position','float'
        metadata[9,0], metadata[9,1] = 'y_position','float'
        metadata[10,0], metadata[10,1] = 'speed_module','float'
        metadata[11,0], metadata[11,1] = 'collision_other', 'float'
        metadata[12,0], metadata[12,1] = 'collision_pedestrian', 'float'
        metadata[13,0], metadata[13,1] = 'collision_vehicles', 'float'
        metadata[14,0], metadata[14,1] = 'opposite_lane_intersection', 'float'
        metadata[15,0], metadata[15,1] = 'sidewalk_intersection', 'float'
        metadata[16,0], metadata[16,1] = 'acceleration_x', 'float'
        metadata[17,0], metadata[17,1] = 'acceleration_y', 'float'
        metadata[18,0], metadata[18,1] = 'acceleration_z', 'float'
        metadata[19,0], metadata[19,1] = 'plataform_time', 'float'
        metadata[20,0], metadata[20,1] = 'game_time', 'float'
        metadata[21,0], metadata[21,1] = 'orientation_x', 'float'
        metadata[22,0], metadata[22,1] = 'orientation_y', 'float'
        metadata[23, 0], metadata[23, 1] = 'orientation_z', 'float'
        metadata[24, 0], metadata[24, 1] = 'control', 'int'
        metadata[25, 0], metadata[25, 1] = 'camera', 'int'
        metadata[26, 0], metadata[26, 1] = 'angle', 'float'
        metadata[27, 0], metadata[27, 1] = 'waypoint1_x', 'float'
        metadata[28, 0], metadata[28, 1] = 'waypoint1_y', 'float'
        metadata[29, 0], metadata[29, 1] = 'waypoint2_x', 'float'
        metadata[30, 0], metadata[30, 1] = 'waypoint2_y', 'float'






