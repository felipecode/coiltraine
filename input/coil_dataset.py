import os
import glob
import h5py
import traceback
import sys
import numpy as np


from torch.utils.data import Dataset


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

                    for es, ee, x in self.sensor_data[count]:

                        if chosen_key >= es and chosen_key < ee:
                            """ We found the part of the data to open """

                            pos_inside = chosen_key - es
                            sensor_image = np.array(x[pos_inside, :, :, :])

                            try:
                                sensor_image = self.transform(sensor_image)
                            except:
                                sensor_image = self.transform(0, sensor_image)


                            batch_sensors[sensor_name][count, (i * 3):((i + 1) * 3), :, :
                            ] = sensor_image



                count += 1


        # TODO: iteration is wrong
        coil_logger.add_message('Reading', {'Iteration': 25, 'ReadKeys': used_ids})
        # TODO: add tensorboard image adding
        # TODO: Do we need to limit the number of iterations the images are saved ??
        # TODO: ADD GROUND TRUTH CONTROL IN SOME META CONFIGURATION FOR THE DATASET
        # TODO: SO if the data read and manipulate is outside some range, it should report error
        return batch_sensors, self.measurements[:, used_ids]

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
        folder_file_names = [os.path.join(path_for_files, f)
                      for f in glob.glob1(path_for_files, "data_*.h5")]


        # Concatenate all the sensor names and measurements names
        # TODO: This structure is very ugly.
        meas_data_cat = [list([]) for _ in range(len(meas_names))]
        sensors_data_cat = [list([]) for _ in range(len(sensors_names))]



        # We open one dataset to get the metadata for targets
        # that is important to be able to reference variables in a more legible way
        dataset = h5py.File(folder_file_names[0], "r")
        metadata_targets = np.array(dataset['metadata_'+ meas_names[0]])

        lastidx = 0
        count = 0
        # TODO: More logs to be added ??
        coil_logger.add_message('Loading', {'FilesLoaded': folder_file_names})

        for file_name in folder_file_names:
            try:
                dataset = h5py.File(file_name, "r")

                for i in range(len(sensors_names)):
                    x = dataset[sensors_names[i]]
                    old_shape = x.shape[0]

                    #  Concatenate all the datasets for a given sensor.

                    sensors_data_cat[i].append((lastidx, lastidx + x.shape[0], x))

                for i in range(len(meas_names)):
                    dset_to_append = dataset[meas_names[i]]
                    meas_data_cat[i].append(dset_to_append[:])

                lastidx += old_shape
                dataset.flush()
                count += 1

            except IOError:

                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print("failed to open", file_name )


        # For the number of datasets names that are going to be used for measurements cat all.
        for i in range(len(meas_names)):
            meas_data_cat[i] = np.concatenate(meas_data_cat[i], axis=0)
            meas_data_cat[i] = meas_data_cat[i].transpose((1, 0))

        return sensors_data_cat, meas_data_cat[0], metadata_targets



