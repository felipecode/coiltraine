import os
import unittest

from input.coil_dataset import CoILDataset
from input import Augmenter



from configs import g_conf, merge_with_yaml, set_type_of_process

class testDataset(unittest.TestCase):


    def test_load_diff_parts(self):

        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'CoILTrain')

        g_conf.EXPERIMENT_NAME = 'coil_icra'
        merge_with_yaml('configs/sample/coil_icra.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(None)

        dataset1 = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                               + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.1)

        dataset2 = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                           + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.2)

        dataset3 = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                           + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.3)

        dataset4 = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                           + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.4)

        dataset5 = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                           + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.5)



        # The datasets should be all different


        print (dataset1.sensor_data_names)

        print(dataset2.sensor_data_names)

        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS)
                                           + 'hours_' + g_conf.TRAIN_DATASET_NAME,
                              start_part=0.1)

        # This one should be exactly the same as 1

        print ("a")