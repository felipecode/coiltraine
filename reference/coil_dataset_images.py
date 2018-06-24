import os
import glob
import h5py
import traceback
import sys
import math



import gc
import numpy as np

from torch.utils.data import Dataset
import torch

from logger import coil_logger

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None):  # The transformation object.
        """
        Function to encapsulate the dataset

        Arguments:
            root_dir (string): Directory with all the hdfiles from the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.sensor_data, self.measurements, self.meta_data = self.pre_load_hdf5_files(root_dir)
        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        # This is seems to be the entire dataset size

        return self.measurements.shape[1]

    def __getitem__(self, used_ids):
        """
        Function to get the items from a dataset

        Arguments
            us
        """
        # We test here directly and include the other images here.
        batch_sensors = {}

        # Number of positions

        try:
            number_of_position = len(used_ids)
        except:
            number_of_position = 1
            used_ids = [used_ids]

        # Initialization of the numpy arrays
        for sensor_name, sensor_size in g_conf.SENSORS.items():
            sensor_data = np.zeros(
                (number_of_position, sensor_size[0], sensor_size[1],
                 sensor_size[2] * g_conf.NUMBER_FRAMES_FUSION),
                dtype='float32'
            )


            batch_sensors.update({sensor_name: sensor_data})

        for sensor_name, sensor_size in g_conf.SENSORS.items():
            count = 0
            for chosen_key in used_ids:

                for i in range(g_conf.NUMBER_FRAMES_FUSION):
                    chosen_key = chosen_key + i * 3


                    """
                    for es, ee, x in self.sensor_data[count]:

                        if chosen_key >= es and chosen_key < ee:


                            pos_inside = chosen_key - es
                            sensor_image = np.array(x[pos_inside, :, :, :])
                    """


                    """ We found the part of the data to open """

                    pos_inside = chosen_key - (chosen_key // 200)*200
                    # TODO: converting to images. The two goes out.
                    sensor_image = self.sensor_data[count][chosen_key // 200][2][pos_inside]


                    if self.transform is not None:
                        sensor_image = self.transform(self.batch_read_number, sensor_image)
                    else:

                        sensor_image = np.swapaxes(sensor_image, 0, 2)
                        sensor_image = np.swapaxes(sensor_image, 1, 2)
                    # Do not forget the final normalization step
                    batch_sensors[sensor_name][count, (i * 3):((i + 1) * 3), :, :
                    ] = sensor_image/255.0

                    del sensor_image



                count += 1

        #TODO: if experiments change name there should be an error

        if g_conf.AUGMENT_LATERAL_STEERINGS > 0:

            camera_angle = self.measurements[np.where(self.meta_data[:, 0] == b'angle'), used_ids][0][0]
            speed = self.measurements[np.where(self.meta_data[:, 0] == b'speed_module'), used_ids][0][0]
            steer = self.measurements[np.where(self.meta_data[:, 0] == b'steer'), used_ids][0][0]

            self.measurements[np.where(self.meta_data[:, 0] == b'steer'), used_ids] =\
                self.augment_steering(camera_angle, steer, speed)
            #print ( 'camera angle', camera_angle,
            #        'new_steer' , self.measurements[np.where(self.meta_data[:, 0] == b'steer'), used_ids],
            #       'old_steer', steer)

        self.measurements[np.where(self.meta_data[:, 0] == b'speed_module'), used_ids] /= g_conf.SPEED_FACTOR




        self.batch_read_number += 1
        # TODO: IMPORTANT !!!
        # TODO: ADD GROUND TRUTH CONTROL IN SOME META CONFIGURATION FOR THE DATASET
        # TODO: SO if the data read and manipulate is outside some range, it should report error
        return batch_sensors, self.measurements[:, used_ids]

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras.
        Args:
            camera_angle_batch:
            steer_batch:
            speed_batch:

        Returns:
            the augmented steering

        """

        time_use = 1.0
        car_length = 6.0
        old_steer = steer
        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))


        #print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer


    # file_names, image_dataset_names, dataset_names
    def pre_load_hdf5_files(self, path_for_files):
        """
        Function to load all hdfiles from a certain folder
        TODO: Add partially loading of the data
        Returns
            TODO: IMPROVE
            A list with the read sensor data ( h5py)

            All the measurement data

        """

        # Take the names of all measurements from the dataset
        meas_names = list(g_conf.MEASUREMENTS.keys())
        # take the names for all sensors
        sensors_names = list(g_conf.SENSORS.keys())

        # From the determined path take all the possible file names.
        # TODO: Add more flexibility for the file base names ??

        float_data_files = glob.glob1(os.path.join(path_for_files), '*.csv')

        # Concatenate all the sensor names and measurements names
        # TODO: This structure is very ugly.
        meas_data_cat = [list([]) for _ in range(len(meas_names))]
        sensors_data_cat = [list([]) for _ in range(len(sensors_names))]

        # We open one dataset to get the metadata for targets
        # that is important to be able to reference variables in a more legible way

        metadata_targets = np.array(read_metadata_from_csv(measurement_files))

        # Forcing the metadata to be bytes
        """
        if not isinstance(metadata_targets[0][0], bytes):
            metadata_targets = np.array(
                [[some_meta_data[0].encode('utf-8'), some_meta_data[1].encode('utf-8')]
                 for some_meta_data in metadata_targets])
        """
        lastidx = 0
        count = 0
        # TODO: More logs to be added ??
        coil_logger.add_message('Loading', {'FilesLoaded': folder_file_names,
                                            'NumberOfImages': len(folder_file_names)})


        # Read sensor images locations
        for i in range(len(sensors_names)):
            # Put all the folders with images inside the vector
            folder_names = glob.glob1(os.path.join(path_for_files, sensors_names[i]))

            # Get the len of each folder , keep in mind that just the last one will be bad
            number_of_images_folder = []
            for folder in folder_names:
                number_of_images_folder.append(
                    len(glob.glob1(os.path.join(path_for_files, sensors_names[i],folder), '.jpg')))

            # Check if they are numbers
            for number_folder in folder_names:
                try:
                    float(number_folder.split('/')[-1])
                except:
                    raise ValueError(" A folder inside a sensor dataset is not a number")

            for i in range(len(number_of_images_folder)):

                sensors_data_cat[i].append( (number_of_images_folder[i], ))



        for m in measurement_files:



            dset_to_append = dataset[meas_names[i]]
            meas_data_cat[i].append(dset_to_append[:])





        for file_name in folder_file_names:

            for f in :


            try:
                dataset = h5py.File(file_name, "r")

                    x = dataset[sensors_names[i]]
                    old_shape = x.shape[0]
                    #  Concatenate all the datasets for a given sensor.
                    .append((lastidx, lastidx + x.shape[0], x))

                for i in range(len(meas_names)):


                lastidx += old_shape
                dataset.flush()
                count += 1

            except IOError:

                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print("failed to open", file_name)


        # For the number of datasets names that are going to be used for measurements cat all.
        for i in range(len(meas_names)):
            meas_data_cat[i] = np.concatenate(meas_data_cat[i], axis=0)
            meas_data_cat[i] = meas_data_cat[i].transpose((1, 0))

        return sensors_data_cat, meas_data_cat[0], metadata_targets

    # TODO: MAKE AN "EXTRACT" method used by both of the functions above.

    # TODO: Turn into a static property

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]

    def extract_targets(self, float_data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(float_data[:, np.where(self.meta_data[:, 0] == target_name.encode())
                                             [0][0], :])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, float_data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(float_data[:, np.where(self.meta_data[:, 0] == input_name.encode())
                                            [0][0], :])

        return torch.cat(inputs_vec, 1)
