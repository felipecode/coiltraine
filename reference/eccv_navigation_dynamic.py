from __future__ import print_function

from carla.driving_benchmark.experiment_sets import _build_experiments
from carla.sensor import Camera






def build_eccv_navigation_dynamic():
    def _poses_town01():
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_navigation():
            return [[36, 40], [39, 35]]
            #return [[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
            #        [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
            #        [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
            #        [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
            #        [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]

        return [_poses_navigation()]

    def _poses_town02():


        def _poses_navigation():
            return [[38, 34], [4, 2]]
            #return [[19, 66], [79, 14], [19, 57], [23, 1],
            #        [53, 76], [42, 13], [31, 71], [33, 5],
            #        [54, 30], [10, 61], [66, 3], [27, 12],
            #        [79, 19], [2, 29], [16, 14], [5, 57],
            #        [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
            #        [51, 81], [77, 68], [56, 65], [43, 54]]

        return [_poses_navigation()]

    # We check the town, based on that we define the town related parameters
    # The size of the vector is related to the number of tasks, inside each
    # task there is also multiple poses ( start end, positions )

    exp_set_dict = {
        'Name': 'eccv_navigation_dynamic',
        'Town01': {'poses': _poses_town01(),
                   'vehicles': [20],
                   'pedestrians': [50],
                   'weathers_train': [1],
                   'weathers_validation': []

                   },
        'Town02': {'poses': _poses_town02(),
                   'vehicles': [15],
                   'pedestrians': [50],
                   'weathers_train': [],
                   'weathers_validation': [14]

                   }
    }

    # We set the camera
    # This single RGB camera is used on every experiment
    camera = Camera('rgb')
    camera.set(FOV=100)
    camera.set_image_size(800, 600)
    camera.set_position(2.0, 0.0, 1.4)
    camera.set_rotation(-15.0, 0, 0)
    sensor_set = [camera]

    return _build_experiments(exp_set_dict, sensor_set), exp_set_dict

