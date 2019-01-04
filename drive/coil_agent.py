import numpy as np
import scipy

import torch
try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from carla.agent import Agent, CommandFollower

from network import CoILModel
from configs import g_conf
from logger import coil_logger

# Parameters for using semantic segmentation as input.
number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}


def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


class CoILAgent(Agent):


    def __init__(self, checkpoint, town_name):


        Agent.__init__(self)

        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self.first_iter = True

        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.cuda()
        self.model.eval()

        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE:
            self.control_agent = CommandFollower(town_name)


    def run_step(self, measurements, sensor_data, directions, target):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: All the float measurements from CARLA ( Just speed is used)
            sensor_data: All the sensor data used on this benchmark
            directions: The directions, high level commands
            target: Final objective.

        Returns:

        """

        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)

        if 'brake' in g_conf.TARGETS:
            steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        else:
            steer, throttle, brake = self._process_model_outputs_no_brake(model_outputs[0])

        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if g_conf.USE_FULL_ORACLE:
            control.steer, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False

        print("speed ", measurements.player_measurements.forward_speed)
        print('Steer', control.steer, 'Gas', control.throttle, 'Brake', control.brake)
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


        return steer, throttle, brake

    def _process_model_outputs_no_brake(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle_brake = outputs[0], outputs[1]

        if throttle_brake >= 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            brake = -throttle_brake
            throttle = 0.0



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

        steer = 0.7 * wpa2

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
