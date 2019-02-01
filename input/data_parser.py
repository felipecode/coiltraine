import glob
import os
import json
import numpy as np
"""
Module used to check attributes existent on data before incorporating them
to the coil dataset
"""


def orientation_vector(measurement_data):
    pitch = np.deg2rad(measurement_data['rotation_pitch'])
    yaw = np.deg2rad(measurement_data['rotation_yaw'])
    orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    return orientation


def forward_speed(measurement_data):
    vel_np = np.array([measurement_data['velocity_x'], measurement_data['velocity_y'],
                       measurement_data['velocity_z']])
    speed = np.dot(vel_np, orientation_vector(measurement_data))

    return speed


def get_speed(measurement_data):
    """ Extract the proper speed from the measurement data dict """

    # If the forward speed is not on the dataset it is because speed is zero.
    if 'playerMeasurements' in measurement_data and \
            'forwardSpeed' in measurement_data['playerMeasurements']:
        return measurement_data['playerMeasurements']['forwardSpeed']
    elif 'velocity_x' in measurement_data:  # We have a 0.9.X data here
        return forward_speed(measurement_data)
    else: # There is no speed key, probably speed is zero.
        return 0


def check_available_measurements(episode):

    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
    # Open a sample measurement
    with open(measurements_list[0]) as f:
        measurement_data = json.load(f)

    available_measurements = {}
    for meas_name in measurement_data.keys():

        # Add steer
        if 'steer' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'steer': meas_name})

        # Add Throttle
        if 'throttle' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'throttle': meas_name})

        # Add brake
        if 'brake' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'brake': meas_name})

        # add game time
    """
    'steer': measurement_augmented['steer'],
    'steer_noise': measurement_augmented['steer_noise'],
    'throttle': measurement_augmented['throttle'],
    'throttle_noise': measurement_augmented['throttle_noise'],
    'brake': measurement_augmented['brake'],
    'brake_noise': measurement_augmented['brake_noise'],
    'speed_module': speed / g_conf.SPEED_FACTOR,
    'directions': directions,
    "pedestrian": measurement_augmented['stop_pedestrian'],
    "traffic_lights": measurement_augmented['stop_traffic_lights'],
    "vehicle": measurement_augmented['stop_vehicle'],
    "game_time": time_stamp,
    'angle': angle}
    """

    return available_measurements

