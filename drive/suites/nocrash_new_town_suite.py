# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function


from carla08.driving_benchmark.experiment import Experiment
from carla08.sensor import Camera
from carla08.settings import CarlaSettings
from carla08.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite



class NocrashNewTown(ExperimentSuite):

    def __init__(self):
        super(NocrashNewTown, self).__init__('Town02')

    @property
    def train_weathers(self):
        return [1, 3, 6, 8]

    @property
    def test_weathers(self):
        return []

    @property
    def collision_as_failure(self):
        return True

    def calculate_time_out(self, path_distance):
        """
        Function to return the timeout ,in milliseconds,
        that is calculated based on distance to goal.
        This is the same timeout as used on the CoRL paper.
        """
        return ((path_distance / 1000.0) / 5.0) * 3600.0 + 20.0

    def _poses(self):

        def _poses_navigation():
            return [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
                    [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
                    [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
                    [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
                    [70, 73], [46, 67], [34, 77], [61, 49], [21, 12]]


        return [_poses_navigation(),
                _poses_navigation(),
                _poses_navigation()]



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
        vehicles_tasks = [0, 15, 70]
        pedestrians_tasks = [0, 50, 150]

        task_names = ['empty', 'normal', 'cluttered']

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
                conditions.set(DisableTwoWheeledVehicles=True)
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    TaskName=task_names[iteration],
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector


