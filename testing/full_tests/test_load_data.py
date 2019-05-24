import unittest
import torch

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
    def test_new_data(self):

        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'resnet34imnet'
        create_log_folder('new_baseline')
        create_exp_path('new_baseline', 'resnet34imnet')
        merge_with_yaml('configs/new_baseline/resnet34imnet.yaml')
        set_type_of_process('train')

        dataset = CoILDataset(transform=None,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                               + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  pin_memory=True)

        for data in data_loader:

            print (data)



