#!/usr/bin/env python
import sys

import argparse
import numpy as np
import h5py


import argparse
from PIL import Image
import matplotlib.pyplot as plt
import math

import time
import os
from collections import deque
import seaborn as sns

sns.set(color_codes=True)


from screen_manager import ScreenManager


class Control:
    steer = 0
    throttle = 0
    brake = 0
    hand_brake = 0
    reverse = 0


# Configurations for this script


sensors = {'RGB': 3, 'labels': 3, 'depth': 0}
resolution = [200, 88]
camera_id_position = 25
direction_position = 24
speed_position = 10
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def join_classes_for(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    # print compressed_labels_image.shape
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            compressed_labels_image[i, j, 0] = join_dic[labels_image[i, j, 0]]

    return compressed_labels_image


def figure_plot(steer_pred1, steer_pred2, steer_gt, iteration):
    fig, ax = plt.subplots(figsize=(16, 7))

    time_vec = range(0, len(steer_pred1))

    time_vec = [float(x) / 10.0 + 200 for x in time_vec]

    ax.plot(time_vec, steer_pred1, 'g', label='Model 2')
    ax.plot(time_vec, steer_pred2, 'r', label='Model 1')
    ax.plot(time_vec, steer_gt, 'b', label='Ground Truth')

    ax.set_ylim([-0.6, 0.8])
    ax.set_xlim([200, 650])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(24)

    ax.legend(loc='upper center', ncol=3)
    # plt.title('Steering Angle Time Series')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Steering Value (radians)')

    fig.savefig('footage_offline/plot' + str(iteration) + '.png', orientation='landscape',
                bbox_inches='tight')

    plt.close(fig)


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    args = parser.parse_args()
    path = args.path

    steer_pred1 = np.loadtxt(
        '/home/felipe/CIL/_logs/eccv/experiment_1/validation_Town02W14_csv/500000.csv',
        delimiter=",", skiprows=0, usecols=([0]))
    steer_pred2 = np.loadtxt(
        '/home/felipe/CIL/_logs/eccv/experiment_11/validation_Town02W14_csv/16000.csv',
        delimiter=",", skiprows=0, usecols=([0]))
    steer_gt = np.loadtxt(
        '/home/felipe/Datasets/Town02W14/ground_truth.csv',
        delimiter=",", skiprows=0, usecols=([0]))

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    step_size = 1
    # initial_positions =[20,25,48,68,79,105,108,120,130]
    # positions_to_test = []
    # for i in initial_positions:
    #  positions_to_test += range(i-1,i+2)

    positions_to_test = range(0, len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
    #positions_to_test = range(10 * 3, 33 * 3)

    screen = ScreenManager()

    image_queue = deque()

    actions_queue = deque()

    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful

    screen.start_screen([resolution[0], resolution[1]], [1, 1], 2)
    ts = []
    images = [np.zeros([resolution[1], resolution[0], 3])] * sensors['RGB']
    labels = [np.zeros([resolution[1], resolution[0], 1])] * sensors['labels']
    depths = [np.zeros([resolution[1], resolution[0], 3])] * sensors['depth']

    steer_gt_order = [0] * 3
    steer_pred1_order = [0] * 3
    steer_pred2_order = [0] * 3

    steer_pred1_vec = []
    steer_pred2_vec = []
    steer_gt_vec = []

    actions = [Control()] * sensors['RGB']
    actions_noise = [Control()] * sensors['RGB']

    for h_num in positions_to_test:

        print(" SEQUENCE NUMBER ", h_num)
        try:
            data = h5py.File(path + 'data_' + str(h_num).zfill(5) + '.h5', "r")
        except Exception as e:
            print (e)
            continue

        for i in range(0, 197, sensors['RGB'] * step_size):

            speed = math.fabs(data['targets'][i + 2][speed_position])

            for j in range(sensors['RGB']):
                capture_time = time.time()
                images[int(data['targets'][i + j][camera_id_position])] = np.array(
                    data['rgb'][i + j]).astype(np.uint8)
                steer_gt_order[int(data['targets'][i + j][camera_id_position])] = steer_gt[
                    (h_num * 200) + i + j]
                steer_pred1_order[int(data['targets'][i + j][camera_id_position])] = steer_pred1[
                    (h_num * 200) + i + j]
                steer_pred2_order[int(data['targets'][i + j][camera_id_position])] = steer_pred2[
                    (h_num * 200) + i + j]

                # print ' Read RGB time ',time.time() - capture_time
                # depths[int(data['targets'][i +j][25])] = np.array(data['depth'][i+j]).astype(np.uint8)
                action = Control()
                angle = data['targets'][i + j][26]

                #########Augmentation!!!!
                # time_use =  1.0
                # car_lenght = 6.0
                # targets[count][i] -=min(4*(math.atan((angle*car_lenght)/(time_use*float_data[speed_pos,i]+0.05)))/3.1415,0.2)

                action.steer = data['targets'][i + j][0]
                # print 'action', action.steer
                steering_pred.append(action.steer)
                action.throttle = data['targets'][i + j][1]
                action.brake = data['targets'][i + j][2]
                time_use = 1.0
                car_lenght = 6.0
                extra_factor = 4.0
                threshold = 0.3
                if angle > 0.0:
                    angle = math.radians(math.fabs(angle))
                    action.steer -= min(
                        extra_factor * (
                        math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.6)
                else:
                    angle = math.radians(math.fabs(angle))
                    action.steer += min(
                        extra_factor * (
                        math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.6)

                    # print 'Angle : ',angle,'Steer : ',action.steer

                actions[int(data['targets'][i + j][camera_id_position])] = action

                action_noise = Control()
                action_noise.steer = data['targets'][i + j][0]
                action_noise.throttle = data['targets'][i + j][1]
                action_noise.brake = data['targets'][i + j][2]

                actions_noise[int(data['targets'][i + j][camera_id_position])] = action_noise

            for j in range(sensors['labels']):
                capture_time = time.time()
                labels[(int(data['targets'][i + j][camera_id_position]))] = np.array(
                    data['labels'][i + j]).astype(
                    np.uint8)
                # print ' Read Label time ',time.time() - capture_time
            for j in range(sensors['depth']):
                depths[int(data['targets'][i + j][camera_id_position])] = np.array(
                    data['depth'][i + j]).astype(
                    np.uint8)

            direction = data['targets'][i][direction_position]
            # print direction

            speed = data['targets'][i + 2][speed_position]


            steer_pred1_vec.append(steer_pred1_order[1] * 1.22)
            steer_pred2_vec.append(steer_pred2_order[1] * 1.22)
            steer_gt_vec.append(steer_gt_order[1] * 1.22)



            #    print actions[j].steer

            screen.plot3camrcnoise(images[1], steer_pred1_order[1], steer_pred2_order[1],
                                   steer_gt_order[1], [0, 0])



            #figure_plot(steer_pred1_vec, steer_pred2_vec, steer_gt_vec, count)
            count += 1
            # for j in range(sensors['depth']):
            #  #print j

            #  screen.plot_camera(depths[j] ,[j,2])

            # pygame.display.flip()
            # time.sleep(0.05)

    # save_gta_surface(gta_surface)
