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
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite



class EccvGeneralization(ExperimentSuite):

    def __init__(self):
        super(EccvGeneralization, self).__init__('Town02')

    @property
    def train_weathers(self):
        return []
    @property
    def test_weathers(self):
        return [14]


    def _poses(self):


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

        camera = Camera('rgb')
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)


        poses_tasks = self._poses()
        vehicles_tasks = [15]
        pedestrians_tasks = [50]

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


