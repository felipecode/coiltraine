import sys
import os
import time
import socket

import re
import math
import numpy as np
import copy
import random

from sklearn import preprocessing

import scipy
from Queue import Queue



from carla.benchmarks.agent import Agent
from PIL import Image


#TODO: The network is defined and toguether there is as forward pass operation to be used for testing, depending on the configuration
import drive_interfaces.machine_output_functions as machine_output_functions

from carla.autopilot.autopilot import Autopilot

from carla.autopilot.pilotconfiguration import ConfigAutopilot

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

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
        self._sess = tf.Session(config=config_gpu)

        # THIS DOES NOT WORK FOR FUSED PLUS LSTM
        if self._config.number_frames_sequenced > self._config.number_frames_fused:
            self._config_train.batch_size = self._config.number_frames_sequenced
        else:
            self._config_train.batch_size = self._config.number_frames_fused

        #self._train_manager = load_system(self._config_train)
        #self._config.train_segmentation = False

        model.load_network(checkpoint)
        #self._sess.run(tf.global_variables_initializer())

        self._control_function = getattr(machine_output_functions,
                                         self._train_manager._config.control_mode)
        # More elegant way to merge with autopilot
        #self._agent = Autopilot(ConfigAutopilot(driver_conf.city_name))

        self._image_cut = driver_conf.image_cut
        self._auto_pilot = driver_conf.use_planner

        self._recording = False
        self._start_time = 0

    def _load_model(self, checkpoint_number=None):
        """
        if self._config_train.restore_seg_test:
          if  self._config.segmentation_model != None:
            exclude = ['global_step']

            variables_to_restore = slim.get_variables(scope="ENet_Small")

            saver = tf.train.Saver(variables_to_restore,max_to_keep=0)

            seg_ckpt = restore_session(self._sess,saver,self._config.segmentation_model)


          variables_to_restore = list(set(tf.global_variables()) - set(slim.get_variables(scope="ENet_Small")))

        else:
          variables_to_restore = tf.global_variables()
        """
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore)
        cpkt = restore_session(self._sess, saver, self._config.models_path, checkpoint_number)



    def compute_goal(self, pos, ori):  # Return the goal selected
        pos, point = self.planner.get_defined_point(pos, ori, (
            self.positions[self._target][0], self.positions[self._target][1], 22),
                                                    (1.0, 0.02, -0.001),
                                                    1 + self._select_goal)
        return convert_to_car_coord(point[0], point[1], pos[0], pos[1], ori[0], ori[1])

    def compute_direction(self, pos,
                          ori):  # This should have maybe some global position... GPS stuff

        if self._train_manager._config.control_mode == 'goal':
            return self.compute_goal(pos, ori)

        elif self.use_planner:

            command, made_turn, completed = self.planner.get_next_command(pos, ori, (
                self.positions[self._target].location.x, self.positions[self._target].location.y,
                22), (1.0, 0.02, -0.001))
            return command

        else:
            # BUtton 3 has priority
            if 'Control' not in set(self._config.inputs_names):
                return None

            button_vec = self._get_direction_buttons()
            if sum(button_vec) == 0:  # Nothing
                return 2
            elif button_vec[0] == True:  # Left
                return 3
            elif button_vec[1] == True:  # RIght
                return 4
            else:
                return 5

    def get_recording(self):

        return False

    def get_reset(self):
        return False

    def get_all_turns(self, data, target):
        rewards = data[0]
        sensor = data[2][0]
        speed = rewards.speed
        return self.planner.get_all_commands((rewards.player_x, rewards.player_y, 22),
                                             (rewards.ori_x, rewards.ori_y, rewards.ori_z), \
                                             (target[0], target[1], 22), (1.0, 0.02, -0.001))

    def run_step(self, measurements, sensor_data, target):

        direction = self._planner.get_next_command(
            (measurements.player_measurements.transform.location.x,
             measurements.player_measurements.transform.location.y, 22), \
            (measurements.player_measurements.transform.orientation.x,
             measurements.player_measurements.transform.orientation.y,
             measurements.player_measurements.transform.orientation.z), \
            (target.location.x, target.location.y, 22),
            (target.orientation.x, target.orientation.y, 0.0012))
        # pos = (rewards.player_x,rewards.player_y,22)
        # ori =(rewards.ori_x,rewards.ori_y,rewards.ori_z)
        # pos,point = self.planner.get_defined_point(pos,ori,(target[0],target[1],22),(1.0,0.02,-0.001),self._select_goal)
        # direction = convert_to_car_coord(point[0],point[1],pos[0],pos[1],ori[0],ori[1])
        # image_filename_format = '_images/episode_{:0>3d}/{:s}/image_{:0>5d}.png'
        sensors = []

        control_agent = self._agent.run_step(measurements, None, target)

        for name in self._config.sensor_names:
            if name == 'rgb':
                sensors.append(sensor_data['RGB'].data)
            elif name == 'labels':
                sensors.append(sensor_data['Labels'].data)

        control = self.compute_action(sensors, measurements.player_measurements.forward_speed,
                                      direction)

        # if self._auto_pilot:
        #    control.steer = control_agent.steer

        control.throttle = control_agent.throttle
        control.brake = control_agent.brake

        return control

    def compute_action(self, sensors, speed, direction=None):

        capture_time = time.time()

        if direction == None:
            direction = self.compute_direction((0, 0, 0), (0, 0, 0))

        sensor_pack = []

        for i in range(len(sensors)):

            sensor = sensors[i]
            if self._config.sensor_names[i] == 'rgb':

                sensor = sensor[self._image_cut[0]:self._image_cut[1], :]
                sensor = scipy.misc.imresize(sensor, [self._config.sensors_size[i][0],
                                                      self._config.sensors_size[i][1]])


            elif self._config.sensor_names[i] == 'labels':

                sensor = sensor[self._image_cut[0]:self._image_cut[1], :]

                sensor = scipy.misc.imresize(sensor, [self._config.sensors_size[i][0],
                                                      self._config.sensors_size[i][1]],
                                             interp='nearest')

                sensor = join_classes(sensor) * int(255 / (number_of_seg_classes - 1))

                sensor = sensor[:, :, np.newaxis]

            sensor_pack.append(sensor)

        if len(sensor_pack) > 1:

            print sensor_pack[0].shape

            print sensor_pack[1].shape
            image_input = np.concatenate((sensor_pack[0], sensor_pack[1]), axis=2)

        else:
            image_input = sensor_pack[0]

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._config,
                                                   self._sess,
                                                   self._train_manager)

        print steer, acc, brake

        if brake < 0.2:
            brake = 0.0

        if acc > brake:
            brake = 0.0
        else:
            acc = acc * 2
        if speed > 35.0 and brake == 0.0:
            acc = 0.0

        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake
        # print brake

        control.hand_brake = 0
        control.reverse = 0

        return control  # ,machine_output_functions.get_intermediate_rep(image_input,speed,self._config,self._sess,self._train_manager)

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
