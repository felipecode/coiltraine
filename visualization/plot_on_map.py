#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import csv
import datetime
import math
import os
import abc
import logging


from skimage.transform import rescale
from carla.planner import map
from PIL import Image
import numpy as np

def plot_test_image(image, name):

    image_to_plot = Image.fromarray(image.astype("uint8"))

    image_to_plot.save(name)



def sldist(c1, c2): return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)




def plot_on_map(map_image, position, color, size):

    for i in range(0, size):
        map_image[int(position[1]), int(position[0])] = color
        map_image[int(position[1]) + i, int(position[0])] = color
        map_image[int(position[1]), int(position[0]) + i] = color
        map_image[int(position[1]) - i, int(position[0])] = color
        map_image[int(position[1]), int(position[0]) - i] = color
        map_image[int(position[1]) + i, int(position[0]) + i] = color
        map_image[int(position[1]) - i, int(position[0]) - i] = color
        map_image[int(position[1]) + i, int(position[0]) - i] = color
        map_image[int(position[1]) - i, int(position[0]) + i] = color




def split_episodes(meas_file):

    """

    Args:
        meas_file: the file containing the measurements.

    Returns:
        a matrix where each vector is a vector of points from the episodes.
        a vector with the travelled distance on each episode

    """
    f = open(meas_file, "rU")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-2]
    f.close()


    details_matrix = np.loadtxt(open(meas_file, "rb"), delimiter=",", skiprows=1)

    #
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    #
    episode_positions_matrix = []
    positions_vector = []
    travelled_distances = []
    travel_this_episode = 0
    previous_start_point = details_matrix[0, header_details.index('start_point')]
    previous_end_point = details_matrix[0, header_details.index('end_point')]
    for i in range(1, len(details_matrix)):
        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]

        start_point = details_matrix[i, header_details.index('start_point')]
        end_point = details_matrix[i, header_details.index('end_point')]

        positions_vector.append(point)
        if previous_start_point != start_point and end_point != previous_end_point:


            travelled_distances.append(travel_this_episode)
            travel_this_episode = 0

            episode_positions_matrix.append(positions_vector)
            positions_vector = []


        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point

        previous_start_point = start_point
        previous_end_point = end_point

    return episode_positions_matrix, travelled_distances





def plot_episodes_tracks(exp_batch, experiment, checkpoint, city_name, exp_suite, meas_file):

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(city_name, 0.164, 50)


    episodes_positions, travelled_distances = split_episodes(meas_file)

    root_folder = "../_logs"
    paths_dir = os.path.join(root_folder, exp_batch, experiment,
                             'drive_' + exp_suite + '_' + city_name + '_paths')

    if not os.path.exists(paths_dir):
        os.mkdir(paths_dir)

    if not os.path.exists(os.path.join(paths_dir, str(checkpoint))):
        os.mkdir(os.path.join(paths_dir, str(checkpoint)))

    # For each position vec in all episodes
    count = 0  # To count the number
    for episode_vec in episodes_positions:

        map_image = Image.open(os.path.join(image_location, city_name + '.png'))
        map_image.load()
        map_image = np.asarray(map_image, dtype="int32")

        travel_this_episode = 0
        previous_pos = episode_vec[0]
        for point in episode_vec[1:]:

            travel_this_episode += sldist(point, previous_pos)
            previous_pos = point
            value = travel_this_episode / travelled_distances[count]

            color_palate_inst = [0 + (value * x) for x in [255, 0, 0]]
            color_palate_inst.append(255)

            point.append(0.0)

            plot_on_map(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 4)


        count += 1

        map_image = rescale(map_image.astype('float'), 1.0 / 4.0)
        plot_test_image(map_image, os.path.join(paths_dir, str(checkpoint), str(count) + '.png'))







if __name__ == '__main__':


    city_name = 'Town01'



    plot_episodes_tracks('eccv',
                         'experiment_11',
                         '200000',
                         city_name,
                         'ECCVTrainingSuite',
                         '../_benchmarks_results/eccv_experiment_11_200000_drive_control_output_auto_ECCVTrainingSuite_Town01/measurements.csv',
                         )

