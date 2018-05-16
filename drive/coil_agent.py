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


#from carla.autopilot.autopilot import Autopilot
#from carla.autopilot.pilotconfiguration import ConfigAutopilot

from carla.agent import Agent
from PIL import Image


#TODO: The network is defined and toguether there is as forward pass operation to be used for testing, depending on the configuration

from network import CoILModel
from configs import g_conf
from logger import coil_logger
from torchvision import transforms
import imgauggpu as iag
import torch


try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}

# TODO: implement this as a torch operation.
def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


"""


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



class CoILAgent(Agent):

    def __init__(self, checkpoint):

        Agent.__init__(self)

        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.model = CoILModel(g_conf.MODEL_NAME)

        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.cuda()




    def run_step(self, measurements, sensor_data, directions, target):



        #control_agent = self._agent.run_step(measurements, None, target)

        speed = torch.cuda.FloatTensor([measurements.player_measurements.forward_speed]).unsqueeze(0)
        print ("Speed shape ", speed)
        directions_tensor = torch.cuda.LongTensor([directions])
        model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), speed,
                                                  directions_tensor)

        print (model_outputs)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0],
                                         measurements.player_measurements.forward_speed)

        #control = self.compute_action(,
        #                              ,
        #                              directions)
        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        # if self._auto_pilot:
        #    control.steer = control_agent.steer
        # TODO: adapt the client side agent for the new version. ( PROBLEM )
        #control.throttle = control_agent.throttle
        #control.brake = control_agent.brake

        # TODO: maybe change to a more meaningfull message ??
        return control


    def _process_sensors(self, sensors):


        iteration = 0
        for name, size in g_conf.SENSORS.items():

            sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]
            if  sensors[name].type == 'SemanticSegmentation':


                # TODO: the camera name has to be sincronized with what is in the experiment...
                sensor = join_classes(sensor)

                sensor = sensor[:, :, np.newaxis]

                image_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((size[1], size[2]), interpolation=Image.NEAREST),
                                   iag.ToGPU(), iag.Multiply((1 / (number_of_seg_classes - 1)))])
            else:

                image_transform = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((size[1], size[2])),
                                   transforms.ToTensor(), transforms.Normalize((0, 0 ,0), (255, 255, 255)),
                                   iag.ToGPU()])


            sensor = np.swapaxes(sensor, 0, 1)

            sensor = np.flip(sensor.transpose((2, 0, 1)), axis=0)

            if iteration == 0:
                image_input = image_transform(sensor)
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1

        image_input = image_input.unsqueeze(0)


        return image_input


    def _process_model_outputs(self,outputs, speed):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0
        else:
            throttle = throttle * 2
        if speed > 35.0 and brake == 0.0:
            throttle = 0.0


        return steer, throttle, brake


    """

    def compute_action(self, sensors, speed, direction):

        capture_time = time.time()


        sensor_pack = []



        for i in range(len(sensors)):

            sensor = sensors[i]

            sensor = sensor[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], :]

            if g_conf.param.SENSORS.keys()[i] == 'rgb':

                sensor = scipy.misc.imresize(sensor, [self._config.sensors_size[i][0],
                                                      self._config.sensors_size[i][1]])


            elif g_conf.param.SENSORS.keys()[i] == 'labels':

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
        

        image_input = sensors[0]

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        # TODO: This will of course depend on the model , if it is based on sequences there are
        # TODO: different requirements
        #tensor = self.model(image_input)
        outputs = self.model.forward_branch(image_input, speed, direction)



        return control  # ,machine_output_functions.get_intermediate_rep(image_input,speed,self._config,self._sess,self._train_manager)

    """
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