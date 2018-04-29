# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function


from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suite.experiment_suite import ExperimentSuite



class CoRL2017(ExperimentSuite):

    @property
    def train_weathers(self):
        return [1, 3, 6, 8]
    @property
    def test_weathers(self):
        return [4, 14]

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """


        return [[[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
                [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
                [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
                [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
                [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]]


    def _poses_town02(self):


        return [[[19, 66], [79, 14], [19, 57], [23, 1],
                [53, 76], [42, 13], [31, 71], [33, 5],
                [54, 30], [10, 61], [66, 3], [27, 12],
                [79, 19], [2, 29], [16, 14], [5, 57],
                [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                [51, 81], [77, 68], [56, 65], [43, 54]]]



    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.


        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0, 0, 0, 20]
            pedestrians_tasks = [0, 0, 0, 50]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0, 0, 0, 15]
            pedestrians_tasks = [0, 0, 0, 50]

        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector


