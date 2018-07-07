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


from carla.planner import map
import numpy as np



def sldist(c1, c2): return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def plot_episode(city_name, file_name, number_of_episodes, color_palate):


    # This can be hardcoded since the map file name is always the same
    image_location = (map.__file__)

    self.map_image = Image.open(city_map_file)
    self.map_image.load()
    self.map_image = np.asarray(self.map_image, dtype="int32")

    carla_map = map.CarlaMap(city_name, 0.1643, 50.0)



    f = open(file_name, "rb")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-2]
    f.close()

    details_matrix = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)


    #
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    count_episodes = 0
    episode_palete = [0]
    travelled_distance =[]
    travel_this_episode =0
    for i in range(len(details_matrix)):
        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]

        if sldist(point,previous_pos) > 500.0:
            count_episodes +=1
            travelled_distance.append(travel_this_episode)
            travel_this_episode=0
            episode_palete.append(i)
            previous_pos = point
        if count_episodes == number_of_episodes:
            break

        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point
        #print point

    count_episodes = 1
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    for i in range(0, episode_palete[-1]):

        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]


        if sldist(point,previous_pos) > 500.0: # DUMB BUT WHATEVER
            count_episodes +=1
            travel_this_episode =0
            previous_pos= point
        if count_episodes == number_of_episodes+1:
            break

        travel_this_episode += sldist(point, previous_pos)

        previous_pos = point



        #print travel_this_episode,travelled_distance[count_episodes]

        value = travel_this_episode/travelled_distance[count_episodes-1]
        #print '     ',value

        color_palate_inst = [0+(value*x) for x in color_palate[count_episodes-1][:-1]]
        color_palate_inst.append(255)


        carla_map.plot_on_map(point,12,color_palate_inst)

    carla_map.save_image('')


if __name__ == '__main__':


    city_name = 'Town01'
    name_to_save = 'map saved'


    color_palate =[
        [255,0,0,255],
        [0,255,0,255],
        [0,0,255,255],
        [255,128,0,255],
        [0,255,255,255],
        [255,0,255,255]
    ]




    plot_episode(city_name, '/Users/felipecode/chauffeur/500000_25_nor_saug_single_ctrl_bal_regr_allauto_databench_Town01/details_w1.',6,color_palate)

