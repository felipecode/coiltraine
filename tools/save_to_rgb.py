#!/usr/bin/env python

import argparse
import numpy as np
import h5py
import glob

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




def write_csv_header(filename):

    csv_outfile = open(filename, 'w')

    csv_outfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
                      "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                      % ('steer', 'throttle', 'brake', 'hand_brake', 'reverse_gear', 'steer_noise',
                         'gas_noise', 'brake_noise', 'x_position', 'y_position', 'speed_module',
                         'collision_other', 'collision_pedestrian', 'collision_vehicles',
                         'opposite_lane_intersection', 'sidewalk_intersection', 'acceleration_x',
                         'acceleration_y', 'acceleration_z', 'plataform_time', 'game_time',
                         'orientation_x', 'orientation_y', 'orientation_z', 'control', 'camera',
                         'angle', 'waypoint1_x', 'waypoint1_y', 'waypoint2_x', 'waypoint2_y',
                         'waypoint1_angle', 'waypoint1_mag', 'waypoint2_angle', 'waypoint2_mag'))

    csv_outfile.close()


def write_csv_data(filename, float_data):


    csv_outfile = open(filename, 'a+')

    first_time = True
    for value in float_data:

        if first_time:
            csv_outfile.write("%f" % (value))
        else:
            csv_outfile.write(",%f" % (value))
        first_time = False



    csv_outfile.write("\n")






# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")
    parser.add_argument('-ot', '--out_path', default="")

    args = parser.parse_args()
    path = args.path
    out_path = args.out_path

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []

    positions_to_test = range(0, len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))

    files = glob.glob(os.path.join(path, 'data_*.h5'))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.exists(os.path.join(out_path, 'rgb')):
        os.mkdir(os.path.join(out_path, 'rgb'))

    if not os.path.exists(os.path.join(out_path, 'labels')):
        os.mkdir(os.path.join(out_path, 'labels'))

    if not os.path.exists(os.path.join(out_path, 'depth')):
        os.mkdir(os.path.join(out_path, 'depth'))

    number_of_files_per_folder = 20

    count_files = 0
    count_images = 0

    image_folder_name = 0

    for f in files:

        try:
            data = h5py.File(f, "r+")
        except Exception as e:
            continue

        if count_files % number_of_files_per_folder == 0:
            image_folder_name = count_files * 200

            if not os.path.exists(os.path.join(out_path, 'rgb', str(image_folder_name))):
                os.mkdir(os.path.join(out_path, 'rgb', str(image_folder_name)))
            if not os.path.exists(os.path.join(out_path, 'labels', str(image_folder_name))):
                os.mkdir(os.path.join(out_path, 'labels', str(image_folder_name)))
            if not os.path.exists(os.path.join(out_path, 'depth', str(image_folder_name))):
                os.mkdir(os.path.join(out_path, 'depth', str(image_folder_name)))

            write_csv_header(os.path.join(out_path, 'float_data_'+str(image_folder_name)+'.csv'))



            print ("Reset at file namer", image_folder_name )
            count_images = 0


        count_files += 1

        print ("File name ", f)


        for i in range(200):

            image_to_save = Image.fromarray(np.array(data['rgb'][i].astype(np.uint8)))
            image_to_save.save(
                os.path.join(out_path, 'rgb', str(image_folder_name)
                             , str(count_images)+'.jpg'))
            image_to_save = Image.fromarray(np.array(data['labels'][i].astype(np.uint8))[:,:,0])

            image_to_save.save(
                os.path.join(out_path, 'labels', str(image_folder_name)
                             , str(count_images)+'.png'))

            depth_array = np.array(data['depth'][i].astype(np.uint8))
            depth_array = depth_array[:, :, ::-1]
            image_to_save = Image.fromarray(depth_array)
            image_to_save.save(
                os.path.join(out_path, 'depth', str(image_folder_name)
                             , str(count_images)+'.png'))

            write_csv_data(os.path.join(out_path, 'float_data_'+str(image_folder_name)+'.csv'),
                           data['targets'][i])

            count_images += 1


        print (" image count ", count_images)




        # new_data_depth = new_data.create_dataset('depth', (200, 88, 200, 3), dtype=np.uint8)



