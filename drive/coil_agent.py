import numpy as np
import scipy

import torch
from agent import Agent, CommandFollower
from carla import VehicleControl

from network import CoILModel
from configs import g_conf
from logger import coil_logger

# Parameters for using semantic segmentation as input.
number_of_seg_classes = 4
classes_join = {0: 1, 1: 1, 2: 1, 3: 1, 5: 1, 12: 1, 9: 1, 11: 1, 4: 1, 10: 0, 8: 2, 6: 2, 7: 3}


def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.items():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


class CoILAgent(Agent):

    def __init__(self, town_name, checkpoint):
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self.model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self.first_iter = True

        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.cuda()
        self.model.eval()

        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE:
            self.control_agent = CommandFollower(town_name, None)

    def run_step(self, vehicle, sensor_data, direction, timestamp):
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
        our_car_transform = vehicle.geo.transform
        our_car_velocity = vehicle.geo.velocity

        norm_speed = vehicle.geo.forward_speed / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([direction])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)

        if 'brake' in g_conf.TARGETS:
            steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        else:
            steer, throttle, brake = self._process_model_outputs_no_brake(model_outputs[0])

        control = VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                vehicle, sensor_data, direction, timestamp)

        if g_conf.USE_FULL_ORACLE:
            control.steer, control.throttle, control.brake = self._get_oracle_prediction(
                vehicle, sensor_data, direction, timestamp)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False

        #print("speed ", vehicle.geo.forward_speed)
        print('COILAGENT Steer', control.steer, 'Gas', control.throttle, 'Brake', control.brake)

        state = {
            'control': control
        }
        return state





    def _process_sensors(self, sensors):

        translate_collect_system = {'rgb': 'rgb',
                                    'labels_front': 'semantic',
                                    'labels_left': 'left_augmentation_semantic' ,
                                    'labels_right': 'right_augmentation_semantic'
                                    }
        iteration = 0
        for o_name, size in g_conf.SENSORS.items():
            name = translate_collect_system[o_name]
            raw_data = np.array(sensors[name].raw_data)
            raw_data = np.reshape(raw_data, (600, 800, 4))
            raw_data = raw_data[:, :, 0] # remove transparency channel
            sensor = raw_data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]

            if 'semantic' in name:
                # For now we have just for RGB images and semantic segmentation.

                # TODO: the camera name has to be sincronized with what is in the experiment...
                sensor = join_classes(sensor)
                sensor = sensor[:, :, 0]
                #sensor = sensor[:, :, np.newaxis]

                sensor = scipy.misc.imresize(sensor, (size[1], size[2]), interp='nearest')
                sensor = sensor * (1 / (number_of_seg_classes - 1))

                sensor = torch.from_numpy(sensor).type(torch.FloatTensor).cuda()
                sensor = sensor.unsqueeze(0)
                #print('sensor.shape', sensor.shape)

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
        #print('image_input.shape', image_input.shape)



        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0].item(), outputs[1].item(), outputs[2].item()
        brake = brake
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
        steer, throttle_brake = outputs[0].item(), outputs[1].item()

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

    def _get_oracle_prediction(self, vehicle, sensor_data, direction, timestamp):



        # For the oracle, the current version of sensor data is not really relevant.
        state = self.control_agent.run_step(vehicle, sensor_data, direction, timestamp)

        return state['control'].steer, state['control'].throttle, state['control'].brak