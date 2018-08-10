import numpy as np
import copy
import random
import gc
import os
from collections import deque
import scipy

import json
from google.protobuf.json_format import MessageToDict
from torchvision import transforms

from carla.agent import Agent, CommandFollower

# TODO: The network is defined and toguether there is as forward pass operation to be used for testing, depending on the configuration

from network import CoILModel
from configs import g_conf
from logger import coil_logger

from utils.general import plot_test_image



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

    def __init__(self, checkpoint, town_name, record_collisions):

        Agent.__init__(self)

        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)



        # TODO: just  trick, remove this leatter then I learn how to suppress stdout
        self.first_iter = True

        self.model.load_state_dict(checkpoint['state_dict'])
        print("loaded state", checkpoint)

        self.model.cuda()

        self.model.eval()

        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE:
            self.control_agent = CommandFollower(town_name)

        self._recording_collisions = record_collisions
        if self._recording_collisions:
            # The parameters used for the case we want to detect collisions

            self._collision_other_thresh = 400

            self._collision_vehicles_thresh = 400

            self._collision_pedestrians_thresh = 300

            self._previous_pedestrian_collision = 0

            self._previous_vehicle_collision = 0

            self._previous_other_collision = 0

            self._image_queue = deque([])

            self._measurements_queue = deque([])
            self._collision_time = -1
            self._count_collisions = 0
            self._writting_path_collisions = os.path.join('_logs', g_conf.EXPERIMENT_BATCH_NAME,
                                                          g_conf.EXPERIMENT_NAME, g_conf.PROCESS_NAME + '_' +
                                                          'collisions')

            if not os.path.exists(self._writting_path_collisions):
                os.mkdir(self._writting_path_collisions)





    def _add_image_and_record(self, sensor_data, player_measurements, game_timestamp):
        # The clip size for recording is 3 seconds before and 2 seconds after.

        before_collision_clip_size = 3
        after_collision_clip_size = 4
        clip_size = before_collision_clip_size + after_collision_clip_size # in seconds

        def _add_clip_to_disk(clip, meas_clip, writting_path, collision_number):
            # Add it to disk
            # The path for the collision images
            collision_path = os.path.join(writting_path, 'collision_' + str(collision_number))
            if not os.path.exists(collision_path):
                os.mkdir(collision_path)

            count = 0
            for image in clip:
                image.save_to_disk(os.path.join(collision_path, 'img_' + str(count) + '.png'))


                with open(os.path.join(collision_path, 'measurements' + str(count) + '.json'), 'w') as fo:
                    jsonObj = MessageToDict(meas_clip[count])
                    fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

                count += 1



        def _test_for_collision(player_measurements, previous_vehicle_collision,
                                previous_pedestrian_collision, previous_other_collision,
                                thresh_vehicle, thresh_pedestrian, thresh_other):

            if (player_measurements.collision_vehicles - previous_vehicle_collision) > thresh_vehicle:
                return True
            if (player_measurements.collision_pedestrians - previous_pedestrian_collision) > thresh_pedestrian:
                return True
            if (player_measurements.collision_other - previous_other_collision) > thresh_other:
                return True

            return False



        #print ("Pedestrian Collision", player_measurements.collision_pedestrians)

        print ("Instant Pedestrian Collision",
               (player_measurements.collision_pedestrians - self._previous_pedestrian_collision))


        #TODO We will hardcode the sensor that is going to be used as RGB to record the collisions
        if len(self._image_queue) < clip_size*10:

            self._image_queue.append(sensor_data['rgb'])
            self._measurements_queue.append(player_measurements)
        else:
            self._image_queue.popleft()
            self._measurements_queue.popleft()
            self._image_queue.append(sensor_data['rgb'])
            self._measurements_queue.append(player_measurements)


        #print ('images on clip', len(self._image_queue))

        # If it collided, prepare to save things. after after_collision_clipe_size seconds

        # print ( ' TEST FOR COLLISION ', _test_for_collision(player_measurements, self._previous_vehicle_collision,
        #                       self._previous_pedestrian_collision, self._previous_other_collision,
        #                       self._collision_vehicles_thresh,
        #                       self._collision_pedestrians_thresh, self._collision_other_thresh))



        if _test_for_collision(player_measurements, self._previous_vehicle_collision,
                               self._previous_pedestrian_collision, self._previous_other_collision,
                               self._collision_vehicles_thresh,
                               self._collision_pedestrians_thresh, self._collision_other_thresh)\
                and self._collision_time == -1:
            # This use of col time helps it to make sure that you dont overlap collisions
            #print (" COLLLLIDED")
            self._collision_time = game_timestamp/100.0





        if self._collision_time > 0 and ((game_timestamp/100.0) - self._collision_time) > \
                                         after_collision_clip_size:
            # This use of col time helps it to make sure that you dont overlap collisions
            self._collision_time = -1
            _add_clip_to_disk(self._image_queue, self._measurements_queue,
                              self._writting_path_collisions, self._count_collisions)
            self._count_collisions += 1

        self._previous_pedestrian_collision = player_measurements.collision_pedestrians

        self._previous_vehicle_collision = player_measurements.collision_vehicles

        self._previous_other_collision = player_measurements.collision_other





    def run_step(self, measurements, sensor_data, directions, target):

        # control_agent = self._agent.run_step(measurements, None, target)

        norm_speed = (measurements.player_measurements.forward_speed * 3.6) / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)

        directions_tensor = torch.cuda.LongTensor([directions])

        if self._recording_collisions:
            self._add_image_and_record(sensor_data, measurements.player_measurements,
                                       measurements.game_timestamp)


        model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)

        # TODO: for now this is hard coded, but on the first week on TRI i will adapt
        if 'waypoint1_angle' in g_conf.TARGETS:
            steer, throttle, brake = self._process_model_outputs_wp(model_outputs[0])
        else:
            steer, throttle, brake = self._process_model_outputs(model_outputs[0])

        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if g_conf.USE_FULL_ORACLE:
            control.steer, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        # TODO: adapt the client side agent for the new version. ( PROBLEM )

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False


        print ('Steer', control.steer)
        return control





    def _process_sensors(self, sensors):

        iteration = 0
        for name, size in g_conf.SENSORS.items():

            sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]

            if sensors[name].type == 'SemanticSegmentation':
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

                sensor = np.swapaxes(sensor, 0, 1)

                sensor = np.transpose(sensor, (2, 1, 0))

                sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1

        image_input = image_input.unsqueeze(0)

        print(image_input.shape)

        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0
        # else:
        #    throttle = throttle * 2
        # if speed > 35.0 and brake == 0.0:
        #    throttle = 0.0

        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.8 * (wpa1 + wpa2)/0.5

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        # else:
        #    throttle = throttle * 2
        # if speed > 35.0 and brake == 0.0:
        #    throttle = 0.0

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):



        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake
