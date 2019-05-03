#!/usr/bin/env python
import json
import numpy as np
import glob
import re
import scipy

import argparse
import math

import time
import os
from collections import deque
import scipy.ndimage
from cexp.cexp import CEXP




class Control:
    steer = 0
    throttle = 0
    brake = 0
    hand_brake = 0
    reverse = 0


# Configurations for this script


sensors = {'RGB': 3, 'labels': 3, 'depth': 0}
resolution = [800, 600]

""" Position to cut the image before reshapping """
""" This is used to cut out the sky (Kind of useless for learning) """
IMAGE_CUT = [90, 485]





def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))



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
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



def reshape_images(image_type, image_name, in_path, out_path):

    """
    Function for reshaping all the images of an episode and save it again on the

    Params:
        image_type: The type of images that is going to be reshaped.

    """

    if image_type == 'labels':
        interp_type = 'nearest'
    else:
        interp_type = 'bicubic'


    image = scipy.ndimage.imread(os.path.join(in_path, image_name))

    if image_name.shape[0] == 600:
        center = image[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
        center = scipy.misc.imresize(center, (88, 200), interp=interp_type)
        scipy.misc.imsave(os.path.join(out_path, image_name), center)

    else:
        raise ValueError("Unexpected Size")


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    parser.add_argument(
        '--dataset',
        help=' the json configuration file name',
        default='all'
    )
    """ You should pass this extra arguments if you want to delete the semantic segmenation labels"""
    parser.add_argument(
        '-ds', '--delete-semantic-segmentation',
        dest='delete_semantic_segmentation',
        action='store_true',
        help='Flag to tell the system to NOT erase the semantic segmentation labels from the dataset'
    )
    """ You should pass this extra arguments if you want to delete the depth labels"""
    parser.add_argument(
        '-dd', '--delete-depth',
        dest='delete_depth',
        action='store_true',
        help='Flag to tell the system to NOT erase the semantic segmentation labels from the dataset'
    )
    parser.add_argument(
        '-c', '--copy-path',
        help=' Set the path when coppying to another folder insteaed of the same'
    )

    args = parser.parse_args()
    path = args.path


    data_configuration_name = 'coil_training_dataset'
    print ( data_configuration_name)
    print ('dataset_configurations.' + (data_configuration_name) )
    settings_module = __import__('dataset_configurations.' + (data_configuration_name),
                                 fromlist=['dataset_configurations'] )

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []
    step_size = 1
    image_queue = deque()

    actions_queue = deque()

    params = {'save_dataset': False,
              'docker_name': 'carlalatest:latest',
              'gpu': 0,
              'batch_size': 1,
              'remove_wrong_data': False,
              'non_rendering_mode': False,
              'carla_recording': False
              }

    # Start a screen to show everything. The way we work is that we do IMAGES x Sensor.
    # But maybe a more arbitrary configuration may be useful

    ts = []

    env_batch = CEXP(args.dataset, params, 100000, 1, sequential=True)
    # Here some docker was set
    env_batch.start(no_server=True)  # no carla server mode.
    # count, we count the environments that are read
    for env in env_batch:
        steer_vec = []
        throttle_vec = []
        brake_vec = []
        # it can be personalized to return different types of data.
        print("Environment Name: ", env)
        try:
            env_data = env.get_data()  # returns a basically a way to read all the data properly
        except NoDataGenerated:
            print("No data generate for episode ", env)

        else:

            #TODO CHECK FOR ALREADY PRE PROCESSED
            count_exp = 0
            for exp in env_data:
                print("    Exp: ", count_exp)
                count_batch = 0
                for batch in exp:
                    print("      Batch: ", count_batch)
                    for data_point in batch:
                        # assume standard name
                        for key in data_point.keys():
                            print (data_point[key])
                            print
                            if args.delete_depth and 'depth' in key:
                                print ("***Depth***")
                                purge(episode, "*depth*")

                            if args.delete_semantic_segmentation and 'labels' in key:
                                print ("***Purging SemanticSeg***")
                                purge(episode, "*labels*")
                        #Process in the orign or copy ??

                        reshape_images("rgb", episode, data_point_number)
                        if not args.delete_depth:
                            reshape_images("labels", episode, data_point_number)

                        if not args.delete_depth:
                            reshape_images("depth", episode, data_point_number)


                    count_batch += 1
                count_exp += 1

    """
    for episode in episodes_list[args.start_episode:]:
        print ('Episode ', episode)
        if 'episode' not in episode:
            episode = 'episode_' + episode

        if os.path.exists(os.path.join(episode, "checked")) or os.path.exists(
                os.path.join(episode, "processed2")) \
                or os.path.exists(os.path.join(episode, "bad_episode")):
            # Episode was not checked. So we dont load it.
            print(" This episode was already checked ")
            continue
        # Take all the measurements from a list
        try:
            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)
            print (" Purging other data")
            print ("Lidar")
            purge(episode, "Lidar*")

            print (episode)
            if args.delete_depth:
                print ("***Depth***")
                purge(episode, "CentralDepth*")
                purge(episode, "LeftDepth*")
                purge(episode, "RightDepth*")

            if args.delete_semantic_segmentation:
                print ("***Purging SemanticSeg***")
                purge(episode, "CentralSemanticSeg*")
                purge(episode, "LeftSemanticSeg*")
                purge(episode, "RightSemanticSeg*")

            bad_episode = False
            if len(measurements_list) <= 1:
                print (" Episode is empty")
                purge(episode, '.')
                bad_episode = True
                continue

            for measurement in measurements_list[:-3]:

                data_point_number = measurement.split('_')[-1].split('.')[0]

                with open(measurement) as f:
                    measurement_data = json.load(f)

                reshape_images("RGB", episode, data_point_number)
                if not args.delete_depth:
                    reshape_images("SemanticSeg", episode, data_point_number)

                if not args.delete_depth:
                    reshape_images("Depth", episode, data_point_number)

                if 'forwardSpeed' in  measurement_data['playerMeasurements']:
                    speed = measurement_data['playerMeasurements']['forwardSpeed']
                else:
                    speed = 0

                float_dicts = {'steer': measurement_data['steer'],
                               'throttle': measurement_data['throttle'],
                               'brake': measurement_data['brake'],
                               'speed_module': speed,
                               'directions': measurement_data['directions'],
                               "pedestrian": measurement_data['stop_pedestrian'],
                               "traffic_lights": measurement_data['stop_traffic_lights'],
                               "vehicle": measurement_data['stop_vehicle'],
                               'angle': -30.0}


            for measurement in measurements_list[-3:]:
                data_point_number = measurement.split('_')[-1].split('.')[0]
                purge(episode, "CentralRGB_" + data_point_number + '.png')
                purge(episode, "LeftRGB_" + data_point_number + '.png')
                purge(episode, "RightRGB_" + data_point_number + '.png')
                purge(episode, "CentralSemanticSeg_" + data_point_number + '.png')
                purge(episode, "LeftSemanticSeg_" + data_point_number + '.png')
                purge(episode, "RightSemanticSeg_" + data_point_number + '.png')
                purge(episode, "CentralDepth_" + data_point_number + '.png')
                purge(episode, "LeftDepth_" + data_point_number + '.png')
                purge(episode, "RightDepth_" + data_point_number + '.png')
                os.remove(measurement)

            if not bad_episode:
                done_file = open(os.path.join(episode, "processed2"), 'w')
                done_file.close()

        except:
            import traceback
            traceback.print_exc()
            print (" Error on processing")
            done_file = open(os.path.join(episode, "bad"), 'w')
            done_file.close()

            continue




        # The last one we delete
        data_point_number = measurements_list[-1].split('_')[-1].split('.')[0]
        print (data_point_number)
        purge(episode, "CentralRGB_" + data_point_number + '.png')
        purge(episode, "LeftRGB_" + data_point_number + '.png')
        purge(episode, "RightRGB_" + data_point_number + '.png')



if ss_center.shape[0] == 600:
    ss_center = rgb_center[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
    ss_center = scipy.misc.imresize(rgb_center, (88, 200))
    scipy.misc.imsave(os.path.join(episode, rgb_center_name), rgb_center)


"""