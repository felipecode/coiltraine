import sys
import os
import time
import socket

import re
import math
import numpy as np
import copy
import random

#from sklearn import preprocessing

import scipy
#from Queue import Queue


#from carla.autopilot.autopilot import Autopilot
#from carla.autopilot.pilotconfiguration import ConfigAutopilot

from carla.agent import Agent
from PIL import Image


#TODO: The network is defined and toguether there is as forward pass operation to be used for testing, depending on the configuration

from network import CoILModel
from configs import g_conf
from logger import coil_logger


try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')


"""
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def restore_session(sess, saver, models_path, checkpoint_number):
    ckpt = 0
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    ckpt = tf.train.get_checkpoint_state(models_path)
    if checkpoint_number != None:
        ckpt.model_checkpoint_path = os.path.join(models_path,
                                                  'model.ckpt-' + str(checkpoint_number))
    if ckpt:
        print 'Restoring from ', ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        ckpt = 0

    return ckpt
"""

""" Initializing Session as variables that control the session """


def convert_to_car_coord(goal_x, goal_y, pos_x, pos_y, car_heading_x, car_heading_y):
    start_to_goal = (goal_x - pos_x, goal_y - pos_y)

    car_goal_x = -(-start_to_goal[0] * car_heading_y + start_to_goal[1] * car_heading_x)
    car_goal_y = start_to_goal[0] * car_heading_x + start_to_goal[1] * car_heading_y

    return [car_goal_x, car_goal_y]

#TODO , this should drastically change.

class CoILAgent(Agent):

    def __init__(self, checkpoint):



        #experiment_name='None', driver_conf=None, memory_fraction=0.18,
        #image_cut=[115, 510]):

        # use_planner=False,graph_file=None,map_file=None,augment_left_right=False,image_cut = [170,518]):

        Agent.__init__(self)
        # This should likely come from global
        #config_gpu = tf.ConfigProto()
        #config_gpu.gpu_options.visible_device_list = '0'

        #config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        #self._sess = tf.Session(config=config_gpu)

        # THIS DOES NOT WORK FOR FUSED PLUS LSTM
        #if self._config.number_frames_sequenced > self._config.number_frames_fused:
        #    self._config_train.batch_size = self._config.number_frames_sequenced
        #else:
        #    self._config_train.batch_size = self._config.number_frames_fused

        #self._train_manager = load_system(self._config_train)
        #self._config.train_segmentation = False
        self.model = Model(g_conf.param.NETWORK.MODEL_DEFINITION)
        self.model.load_network(checkpoint)

        #self._sess.run(tf.global_variables_initializer())

        #self._control_function = getattr(machine_output_functions,
        #                                 self._train_manager._config.control_mode)
        # More elegant way to merge with autopilot
        #self._agent = Autopilot(ConfigAutopilot(driver_conf.city_name))

        #self._image_cut = driver_conf.image_cut
        #self._auto_pilot = driver_conf.use_planner

        #self._recording = False
        #self._start_time = 0


    def run_step(self, measurements, sensor_data, directions, target):


        # pos = (rewards.player_x,rewards.player_y,22)
        # ori =(rewards.ori_x,rewards.ori_y,rewards.ori_z)
        # pos,point = self.planner.get_defined_point(pos,ori,(target[0],target[1],22),(1.0,0.02,-0.001),self._select_goal)
        # direction = convert_to_car_coord(point[0],point[1],pos[0],pos[1],ori[0],ori[1])
        # image_filename_format = '_images/episode_{:0>3d}/{:s}/image_{:0>5d}.png'
        sensors = []

        #control_agent = self._agent.run_step(measurements, None, target)


        for name in g_conf.param.SENSORS.keys():
            if name == 'rgb':
                sensors.append(sensor_data['RGB'].data)
            elif name == 'labels':
                sensors.append(sensor_data['Labels'].data)

        control = self.compute_action(sensors, measurements.player_measurements.forward_speed,
                                      directions)

        # if self._auto_pilot:
        #    control.steer = control_agent.steer
        # TODO: adapt the client side agent for the new version.
        #control.throttle = control_agent.throttle
        #control.brake = control_agent.brake

        # TODO: maybe change to a more meaningfull message ??


        return control

    def compute_action(self, sensors, speed, direction):

        capture_time = time.time()


        sensor_pack = []

        for i in range(len(sensors)):

            sensor = sensors[i]
            if g_conf.param.SENSORS.keys()[i] == 'rgb':

                sensor = sensor[self._image_cut[0]:self._image_cut[1], :]
                sensor = scipy.misc.imresize(sensor, [self._config.sensors_size[i][0],
                                                      self._config.sensors_size[i][1]])


            elif g_conf.param.SENSORS.keys()[i] == 'labels':

                sensor = sensor[self._image_cut[0]:self._image_cut[1], :]

                sensor = scipy.misc.imresize(sensor, [self._config.sensors_size[i][0],
                                                      self._config.sensors_size[i][1]],
                                             interp='nearest')

                sensor = join_classes(sensor) * int(255 / (number_of_seg_classes - 1))

                sensor = sensor[:, :, np.newaxis]

            sensor_pack.append(sensor)

        if len(sensor_pack) > 1:

            image_input = np.concatenate((sensor_pack[0], sensor_pack[1]), axis=2)

        else:
            image_input = sensor_pack[0]

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        # TODO: This will of course depend on the model , if it is based on sequences there are
        # TODO: different requirements
        tensor = self.model(image_input)

        """
        if brake < 0.2:
            brake = 0.0

        if acc > brake:
            brake = 0.0
        else:
            acc = acc * 2
        if speed > 35.0 and brake == 0.0:
            acc = 0.0
        """
        control = carla_protocol.Control()
        control.steer = 0.0
        control.throttle = 0.6
        control.brake = 0.0
        # print brake



        control.hand_brake = 0
        control.reverse = 0

        return control  # ,machine_output_functions.get_intermediate_rep(image_input,speed,self._config,self._sess,self._train_manager)


    """
    def compute_perception_activations(self, sensor, speed):

        sensor = sensor[self._image_cut[0]:self._image_cut[1], :, :]

        sensor = scipy.misc.imresize(sensor, [self._config.network_input_size[0],
                                              self._config.network_input_size[1]])

        image_input = sensor.astype(np.float32)

        # print future_image

        # print "2"
        image_input = np.multiply(image_input, 1.0 / 255.0)

        vbp_image = machine_output_functions.vbp(image_input, speed, self._config, self._sess,
                                                 self._train_manager)

        min_max_scaler = preprocessing.MinMaxScaler()
        vbp_image = min_max_scaler.fit_transform(np.squeeze(vbp_image))

        # print vbp_image
        # print vbp_image
        # print grayscale_colormap(np.squeeze(vbp_image),'jet')

        vbp_image_3 = np.copy(image_input)
        vbp_image_3[:, :, 0] = vbp_image
        vbp_image_3[:, :, 1] = vbp_image
        vbp_image_3[:, :, 2] = vbp_image
        # print vbp_image

        return 0.4 * grayscale_colormap(np.squeeze(vbp_image), 'inferno') + 0.6 * image_input

    def get_waypoints(self):

        wp1, wp2 = self._agent.get_active_wps()
        return [wp1, wp2]
    """