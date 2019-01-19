#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import math
import os


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
        plot_point(map_image, x0 + x, y0 + y, colour)
        plot_point(map_image, x0 - x, y0 + y, colour)
        plot_point(map_image, x0 + x, y0 - y, colour)
        plot_point(map_image,x0 - x, y0 - y, colour)
        plot_point(map_image,x0 + y, y0 + x, colour)
        plot_point(map_image,x0 - y, y0 + x, colour)
        plot_point(map_image,x0 + y, y0 - x, colour)
        plot_point(map_image,x0 - y, y0 - x, colour)




def filled_circle(map_image, position, colour=[255, 0, 0, 255], radius=6):

    for i in range(0, radius):
        circle(map_image, position, colour=colour, radius=i)



def plot_point(map_image, x, y, colour):

    if (x <map_image.shape[1]  and x > 0) and (y <map_image.shape[0]  and y > 0):
        map_image[x, y] = colour




def plot_on_map(map_image, position, color, size):
    def plot_square(map_image, position, color, size):
        for i in range(0, size):
            for j in range(0, size):
                map_image[int(position[1]) + i, int(position[0]) + j] = color

    for i in range(size):
        plot_square(map_image, position, color, i)


def split_episodes(meas_file, exp_id=-1):

    """
        The idea is to split the positions assumed by the ego vehicle on every episode.
    Args:
        meas_file: the file containing the measurements.
        exp_id: the expid used

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


    exp_id_vec = details_matrix[:, header_details.index('exp_id')]
    #
    if exp_id != -1:
        exp_id_vec = np.where(exp_id_vec == exp_id)
        print(" Exp id ", len(details_matrix[exp_id_vec[0], :]))
        print(exp_id_vec[0][0:50])
        details_matrix = details_matrix[exp_id_vec[0], :]

    episode_positions_matrix = []
    positions_vector = []
    travelled_distances = []
    travel_this_episode = 0

    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]
    previous_start_point = details_matrix[0, header_details.index('start_point')]
    previous_end_point = details_matrix[0, header_details.index('end_point')]
    previous_repetition = details_matrix[0, header_details.index('rep')]
    for i in range(1, len(details_matrix)):

        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]

        start_point = details_matrix[i, header_details.index('start_point')]
        end_point = details_matrix[i, header_details.index('end_point')]
        repetition = details_matrix[i, header_details.index('rep')]

        positions_vector.append(point)
        if (previous_start_point != start_point and end_point != previous_end_point) or \
                repetition != previous_repetition:

            travelled_distances.append(travel_this_episode)
            travel_this_episode = 0
            positions_vector.pop()
            episode_positions_matrix.append(positions_vector)
            print ("episode ", len(episode_positions_matrix))
            print ("len ", len(positions_vector))
            positions_vector = []

            travel_this_episode += sldist(point, previous_pos)
        previous_pos = point

        previous_start_point = start_point
        previous_end_point = end_point
        previous_repetition = repetition

    return episode_positions_matrix, travelled_distances




def get_causes_of_end(summary_file, exp_id=-1):
    """
        The dot that finalizes the printing is codified differently depending on the
        cause ( pedestrian, vehicle, timeout, other)

    """
    f = open(summary_file, "rU")
    header_summary = f.readline()

    header_summary = header_summary.split(',')
    header_summary[-1] = header_summary[-1][:-2]
    f.close()

    summary_matrix = np.loadtxt(open(summary_file, "rb"), delimiter=",", skiprows=1)

    success = summary_matrix[:, header_summary.index('result')]
    end_pedestrian = summary_matrix[:, header_summary.index('end_pedestrian_collision')]
    end_vehicle = summary_matrix[:, header_summary.index('end_vehicle_collision')]
    end_other = summary_matrix[:, header_summary.index('end_other_collision')]


    exp_id_vec = summary_matrix[:, header_summary.index('exp_id')]

    print ("end peds ", end_pedestrian)
    print ("success ", success)
    all_ends = np.concatenate((np.expand_dims(success, axis=1),
                               np.expand_dims(end_pedestrian, axis=1),
                               np.expand_dims(end_vehicle, axis=1),
                               np.expand_dims(end_other, axis=1)),
                              axis=1)

    print ( "All ends shape")
    print ("all_ends ", all_ends.shape)
    print ("Which won")
    print (np.where(all_ends == 1))
    no_timeout_pos, end_cause = np.where(all_ends == 1)
    final_end_cause = np.zeros((len(success)))
    final_end_cause[no_timeout_pos] = end_cause + 1

    if exp_id != -1:
        exp_id_vec = np.where(exp_id_vec == exp_id)
        print(" Exp id ", len(final_end_cause[exp_id_vec[0]]))
        print(exp_id_vec)
        final_end_cause = final_end_cause[exp_id_vec[0]]

    return final_end_cause



def plot_episodes_tracks(exp_batch, experiment, checkpoint, town_name, exp_suite, exp_id=-1):

    # We build the measurement file used for the benchmarks.
    meas_file = os.path.join('_benchmarks_results',
                             exp_batch + '_' + experiment + '_'
                             + str(checkpoint) + '_drive_control_output_'
                             + exp_suite + '_' + town_name,
                             'measurements.csv')
    # We build the summary file used for the benchmarks.
    summary_file = os.path.join('_benchmarks_results',
                                exp_batch + '_' + experiment + '_'
                                + str(checkpoint) + '_drive_control_output_'
                                + exp_suite + '_' + town_name,
                                'summary.csv')


    """
    meas_file = os.path.join('_benchmarks_results',
                                exp_batch + '_'
                                + exp_suite + '_' + town_name,
                             'measurements.csv')
    # We build the summary file used for the benchmarks.
    summary_file = os.path.join('_benchmarks_results',
                                exp_batch + '_'
                                + exp_suite + '_' + town_name,
                             'summary.csv')
    """

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(town_name, 0.164, 50)

    # Split the measurements for each of the episodes
    episodes_positions, travelled_distances = split_episodes(meas_file, exp_id)

    # Get causes of end
    end_cause = get_causes_of_end(summary_file, exp_id)

    print ("End casues ", len(end_cause))
    print (end_cause)

    # Prepare the folder where the results are going to be written
    root_folder = "_logs"
    paths_dir = os.path.join(root_folder, exp_batch, experiment,
                             'drive_' + exp_suite + '_' + town_name + '_paths')

    # Create the paths just in case they don't exist.
    if not os.path.exists(paths_dir):
        os.makedirs(paths_dir)

    if not os.path.exists(os.path.join(paths_dir, str(checkpoint))):
        os.mkdir(os.path.join(paths_dir, str(checkpoint)))

    # For each position vec in all episodes
    count = 0  # To count the number

    # Color pallet for the causes of episodes to end
    end_color_palete = [
        [255, 0, 0, 255],  # Red for timeout
        [0, 255, 0, 255],  # Green for success
        [0, 0, 255, 255],  # Blue for End pedestrian
        [255, 255, 0, 255],  # Yellow for end car
        [255, 0, 255, 255],  # Magenta for end other

    ]
    print("Number of episodes ", len(episodes_positions))
    #episodes_positions = episodes_positions[::-1]

    # We instance an image that is going to have all the final position plots
    map_image_dots = Image.open(os.path.join(image_location, town_name + '.png'))
    map_image_dots.load()
    map_image_dots = np.asarray(map_image_dots, dtype="int32")

    # Cluttered
    #positions = list(range(149, 225, 1)) + list(range(376, 449, 1))
    #print (positions)
    for i in range(len(episodes_positions)):
        episode_vec = episodes_positions[i]
        map_image = Image.open(os.path.join(image_location, town_name + '.png'))
        map_image.load()
        map_image = np.asarray(map_image, dtype="int32")

        # This is for plotting the path driven by the car.
        for point in episode_vec[1:]:

            point[1] = point[1] - 3
            point[0] = point[0] - 2
            color_palate_inst = [255, 0, 0, 255]
            point.append(0.0)

            plot_on_map(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 8)

        # Plot the end point on the path map
        plot_on_map(map_image, carla_map.convert_to_pixel(point),
                    end_color_palete[int(end_cause[i])], 16)
        # Plot the end point on the map just showing the dots
        plot_on_map(map_image_dots, carla_map.convert_to_pixel(point),
                    end_color_palete[int(end_cause[i])], 16)

        count += 1
        map_image = rescale(map_image.astype('float'), 1.0 / 4.0)
        plot_test_image(map_image, os.path.join(paths_dir, str(checkpoint), str(i) + '.png'))

    map_image_dots = rescale(map_image_dots.astype('float'), 1.0 / 4.0)
    plot_test_image(map_image_dots, os.path.join(paths_dir, str(checkpoint), 'all_dots.png'))


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


    plot_episodes_tracks('cvprfinal_valstop_seed1', 'res34-100-lowdropout-imnet-nospeed',
                          500000, 'Town02', 'Long3NewWeatherTown', 0)

