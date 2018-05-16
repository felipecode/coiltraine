import os
import numpy as np
import unittest
from configs import g_conf, merge_with_yaml, set_type_of_process
from input import CoILDataset, CoILSampler
from torchvision import transforms
import torch

class testConfigs(unittest.TestCase):

    def test_config_integrity(self):


        pass


    def test_merge_yaml_line_globaldict(self):


        g_conf.NAME = 'experiment_1'
        merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')

        print (g_conf)
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # augmenter_cpu = iag.AugmenterCPU(g_conf.AUGMENTATION_SUITE_CPU)

        dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        data_loader = torch.utils.data.DataLoader(dataset , batch_size=120,
                                                  shuffle=False, num_workers=12, pin_memory=True)
        # By instanciating the augmenter we get a callable that augment images and transform them
        for data in data_loader:


            a,b =data








