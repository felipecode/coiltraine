import unittest
import os

from configs import g_conf, set_type_of_process, merge_with_yaml
from input import CoILDataset
from coilutils.general import create_log_folder, create_exp_path
"""
The idea for this test is to check if the system is able to open all the available datasets.
After you add anything the changes the input you should test if the datasets (The sample ones)
are possible to be open.

For now it is testing:
    CoILTrain ( The sample dataset)

"""


class TestLoadData(unittest.TestCase):

    def get_datasets(self):
        """
            Function to download the datasets to perform the testing
        Return
        """

    def test_basic_data(self):
        # the town2-town01 data, try to load.
        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'coil_icra'
        create_log_folder('sample')
        create_exp_path('sample', 'coil_icra')
        merge_with_yaml('configs/sample/coil_icra.yaml')

        set_type_of_process('train')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CoILTrain')

        dataset = CoILDataset(full_dataset, transform=None,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                               + 'hours_' + g_conf.TRAIN_DATASET_NAME)

    def test_town3_data(self):
        # the town3 data has different names and does not have pedestrians of vehicle stop
        # indications
        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'resnet34imnet'
        create_log_folder('town03')
        create_exp_path('town03', 'resnet34imnet')
        merge_with_yaml('configs/town03/resnet34imnet.yaml')

        set_type_of_process('train')

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CoILTrainTown03')

        dataset = CoILDataset(full_dataset, transform=None,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                               + 'hours_' + g_conf.TRAIN_DATASET_NAME)



