import os
import glob
import h5py
import traceback
import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader


# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

class CILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None):# The transformation object.
        """
        Function to encapsulate the dataset

        Arguments:
            root_dir (string): Directory with all the hdfiles from the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images = self.pre_load_hdf5_files(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, used_ids):


        # We test here directly and include the other images here.

        for s in range(len(self.images)):
            count = 0

            for chosen_key in used_ids:

                count_seq = 0
                first_enter = True

                for i in range(g_conf.MISC.NUMBER_FRAMES_FUSION):
                    chosen_key = chosen_key + i * 3

                    for es, ee, x in self.images[s]:

                        if chosen_key >= es and chosen_key < ee:
                            """ We found the part of the data to open """
                            # print x[]
                            first_enter = False

                            pos_inside = chosen_key - es

                            # print 'el i'
                            # print chosen_key
                            # print pos_inside
                            # print x[chosen_key - es - 1 + 1:chosen_key - es + 1,:,:,:].shape

                            sensors_batch[s][count, :, :,
                            (i * 3):((i + 1) * 3)] = np.array(x[pos_inside, :, :, :])

                            # print sensors_batch[s][count].shape
                            # if not self._perform_sequential:
                            # img = Image.fromarray(sensors_batch[s][count])
                            # img.save('test' + str(self._current_position_on_dataset +count) + '_0_.jpg')
                count += 1
        return SensorBatch

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
        meas_names = g_conf.param.INPUT.DATASET.MEASUREMENTS_DATASET_NAMES
        # take the names for all sensors
        sensors_names = g_conf.param.INPUT.DATASET.SENSOR_DATASET_NAMES

        # From the determined path take all the possible file names.
        # TODO: Add more flexibility for the file base names ??
        folder_file_names = [os.path.join(path_for_files, f)
                      for f in glob.glob1(path_for_files, "data_*.h5")]


        # Concatenate all the sensor names and measurements names
        # TODO: This structure is very ugly.
        meas_data_cat = [list([]) for _ in range(len(meas_names))]
        sensors_data_cat = [list([]) for _ in range(len(sensors_names))]


        lastidx = 0
        count = 0
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

        return sensors_data_cat, meas_data_cat



