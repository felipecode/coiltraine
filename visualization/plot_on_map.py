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


def circle(map_image, position, colour=[255, 0, 0, 255], radius=6):

    y0, x0 = position
    f = 1 - radius
    ddf_x = 1
    ddf_y = -2 * radius
    x = 0
    y = radius
    plot_point(map_image,x0, y0 + radius, colour)
    plot_point(map_image,x0, y0 - radius, colour)
    plot_point(map_image,x0 + radius, y0, colour)
    plot_point(map_image,x0 - radius, y0, colour)

    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        plot_point(map_image,x0 + x, y0 + y, colour)
        plot_point(map_image,x0 - x, y0 + y, colour)
        plot_point(map_image,x0 + x, y0 - y, colour)
        plot_point(map_image,x0 - x, y0 - y, colour)
        plot_point(map_image,x0 + y, y0 + x, colour)
        plot_point(map_image,x0 - y, y0 + x, colour)
        plot_point(map_image,x0 + y, y0 - x, colour)
        plot_point(map_image,x0 - y, y0 - x, colour)




def filled_circle(map_image, position, colour=[255, 0, 0, 255], radius=6):

    for i in range(0, radius):
        circle(map_image, position,colour=colour, radius=i)



def plot_point(map_image, x, y, colour):

    if (0 < x < map_image.shape[1]) and (0 < x < map_image.shape[0]):
        map_image[x, y] = colour



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

    print (header_details)


    details_matrix = np.loadtxt(open(meas_file, "rb"), delimiter=",", skiprows=1)

    #
    #print (details_matrix)
    previous_pos = [details_matrix[0, header_details.index('pos_')],
                 details_matrix[0, header_details.index('pos_y')]]

    #

    episode_positions_matrix = []
    positions_vector = []
    travelled_distances = []
    travel_this_episode = 0
    previous_start_point = details_matrix[0, header_details.index('start_point')]
    previous_end_point = details_matrix[0, header_details.index('end_point')]
    for i in range(1, len(details_matrix)):
        point = [details_matrix[i, header_details.index('pos_')],
                 details_matrix[i, header_details.index('pos_y')]]

        start_point = details_matrix[i, header_details.index('start_point')]
        end_point = details_matrix[i, header_details.index('end_point')]

        positions_vector.append(point)
        #print (start_point, end_point)
        if previous_start_point != start_point and end_point != previous_end_point:

            travelled_distances.append(travel_this_episode)
            travel_this_episode = 0
            positions_vector.pop()
            episode_positions_matrix.append(positions_vector)
            positions_vector = []


        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point

        previous_start_point = start_point
        previous_end_point = end_point

    return episode_positions_matrix, travelled_distances


def get_start_end_points(summary):

    f = open(summary, "rU")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-2]
    f.close()

    # TODO: implement





def plot_episodes_tracks(exp_batch, experiment, checkpoint, city_name, exp_suite, meas_file):

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(city_name, 0.164, 50)


    episodes_positions, travelled_distances = split_episodes(meas_file)

    root_folder = "_logs"
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

            plot_on_map(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 8)


        count += 1

        map_image = rescale(map_image.astype('float'), 1.0 / 4.0)
        plot_test_image(map_image, os.path.join(paths_dir, str(checkpoint), str(count) + '.png'))




def plot_episodes_tracks_sameimage(exp_batch, experiment, checkpoint,
                                   city_name, exp_suite, meas_file, color_palete, episode_list):

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(city_name, 0.164, 50)


    episodes_positions, travelled_distances = split_episodes(meas_file)

    root_folder = "_logs"
    paths_dir = os.path.join(root_folder, exp_batch, experiment,
                             'drive_' + exp_suite + '_' + city_name + '_paths')

    if not os.path.exists(paths_dir):
        os.mkdir(paths_dir)

    if not os.path.exists(os.path.join(paths_dir, str(checkpoint))):
        os.mkdir(os.path.join(paths_dir, str(checkpoint)))



    #for j in range(0, 25 - len(color_palete)):

    # For each position vec in all episodes
    count = 0  # To count the number
    map_image = Image.open(os.path.join(image_location, city_name + '.png'))
    map_image.load()
    map_image = np.asarray(map_image, dtype="int32")

    for i in episode_list:

        episode_vec = episodes_positions[i]

        travel_this_episode = 0
        previous_pos = episode_vec[0]
        color = color_palete[count]
        for point in episode_vec[1:]:

            travel_this_episode += sldist(point, previous_pos)
            previous_pos = point
            value = travel_this_episode / travelled_distances[count]
            color_palate_inst = [0 + (value * x) for x in color]
            color_palate_inst.append(255)
            #print (point)
            point.append(0.2)

            filled_circle(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 8)


        count += 1



    plot_test_image(map_image, os.path.join(paths_dir, str(checkpoint), 'episodes.png'))




if __name__ == '__main__':


    episode_list = [4,5,6,7]

    # 4,1,5, 23

    city_name = 'Town02'

    color_palete = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0]

    ]

    plot_episodes_tracks_sameimage('eccv_debug',
                         'experiment_24',
                         '200000',
                         city_name,
                         'ECCVGeneralizationSuite',
                         '_benchmarks_results/eccv_debug_experiment_24_200000_drive_control_output_auto_ECCVGeneralizationSuite_Town02/measurements.csv',
                         color_palete,
                         episode_list
                         )


    color_palete = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0]
    ]

    plot_episodes_tracks_sameimage('eccv_debug',
                         'experiment_64',
                         '200000',
                         city_name,
                         'ECCVGeneralizationSuite',
                         '_benchmarks_results/eccv_debug_experiment_64_200000_drive_control_output_auto_ECCVGeneralizationSuite_Town02/measurements.csv',
                         color_palete,
                         episode_list
                         )
