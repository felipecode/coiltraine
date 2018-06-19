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

from utils.general import plot_test_image


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

# TODO: implement this as a torch operation. maybe...
def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image



class CoILAgent(Agent):

    def __init__(self, checkpoint):

        Agent.__init__(self)

        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)

        # TODO: just  trick, remove this leatter then I learn how to suppress stdout
        self.first_iter = True

        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.cuda()




    def run_step(self, measurements, sensor_data, directions, target):



        #control_agent = self._agent.run_step(measurements, None, target)

        norm_speed = measurements.player_measurements.forward_speed/g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)

        directions_tensor = torch.cuda.LongTensor([directions])
        model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)


        steer, throttle, brake = self._process_model_outputs(model_outputs[0])


        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        # if self._auto_pilot:
        #    control.steer = control_agent.steer
        # TODO: adapt the client side agent for the new version. ( PROBLEM )
        #control.throttle = control_agent.throttle
        #control.brake = control_agent.brake
        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False
        return control


    def _process_sensors(self, sensors):


        iteration = 0
        for name, size in g_conf.SENSORS.items():
            print ("Sensors")
            print (sensors)
            sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]


            if  sensors[name].type == 'SemanticSegmentation':
                # For now we have just for RGB images and semantic segmentation.

                # TODO: the camera name has to be sincronized with what is in the experiment...
                sensor = join_classes(sensor)
                sensor = sensor[:, :, np.newaxis]

                sensor = scipy.misc.imresize(sensor, (size[1], size[2]), interp='nearest')
                sensor = sensor * (1 / (number_of_seg_classes - 1))

                sensor = torch.from_numpy(sensor).type(torch.FloatTensor).cuda()

                # OBS: Is using image transform better ?

            else:

                sensor = scipy.misc.imresize(sensor, (size[1], size[2]))

                plot_test_image(sensor, "image.png")
                sensor = np.swapaxes(sensor, 0, 1)

                sensor = np.transpose(sensor, (2, 0, 1))


                sensor = torch.from_numpy(sensor).type(torch.FloatTensor).cuda()





            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1

        image_input = image_input.unsqueeze(0)

        print (image_input.shape)

        return image_input


    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0
        #else:
        #    throttle = throttle * 2
        #if speed > 35.0 and brake == 0.0:
        #    throttle = 0.0


        return steer, throttle, brake
