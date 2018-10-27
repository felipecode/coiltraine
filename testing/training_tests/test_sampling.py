import os
import numpy as np
import unittest


from input.coil_sampler import BatchSequenceSampler, SubsetSampler, RandomSampler
from input.coil_dataset import CoILDataset
from input import Augmenter
import input.splitter as splitter
from PIL import Image
# from utils.general import plot_test_image

from configs import set_type_of_process, merge_with_yaml
import torch
from torchvision import transforms
from utils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint

from coil_core.train import get_inverse_freq_weights

from  coil_core.train import select_balancing_strategy, parse_split_configuration, select_data
from configs import g_conf

def create_log_folder(exp_batch_name):
    """
        Only the train creates the path. The validation should wait for the training anyway,
        so there is no need to create any path for the logs. That avoids race conditions.
    Returns:

    """
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))


def create_exp_path(exp_batch_name, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(os.path.join(root_path, exp_batch_name, experiment_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, experiment_name))


class testValidation(unittest.TestCase):
    # def __init__(self, *args, **kwargs):
    #    super(testSampler, self).__init__(*args, **kwargs)
    # self.root_test_dir = '/home/felipe/Datasets/CVPR02Noise/SeqTrain'

    # self.test_images_write_path = 'testing/unit_tests/_test_images_'

    def __init__(self, *args, **kwargs):
        super(testValidation, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/unit_tests/data'
        self.test_images_write_path = 'testing/unit_tests/_test_images_'

    def test_core_validation(self):

        if not os.path.exists(self.test_images_write_path + 'weather_aug'):
            os.mkdir(self.test_images_write_path + 'weather_aug')

        dataset_name = 'CARLA100'

        exp_batch  = 'manual_balance50'
        exp_alias = 'experiment_1'
        full_dataset = os.path.join('/home/eder/data', dataset_name)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        create_log_folder('manual_balance50')
        create_exp_path(exp_batch, exp_alias)


        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
        set_type_of_process('drive', dataset_name)
        augmenter = Augmenter(None)

        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)

        g_conf.immutable(False)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file


        #g_conf.SPLIT = [['lateral_noise', []], ['longitudinal_noise', []], ['weights', [0.0, 0.0, 1.0]]]
        data_loader = select_balancing_strategy(dataset, 0, 12)



        #name, params = parse_split_configuration(g_conf.SPLIT)
        #splitter_function = getattr(splitter, name)

        #keys = splitter_function(dataset.measurements, params)
        #print (" The keys are ", keys)
        count = 0

        for data in data_loader:
            #print (data)



            image_to_save = transforms.ToPILImage()((data['rgb'][0].cpu()*255).type(torch.ByteTensor))
            image_to_save.save(os.path.join(self.test_images_write_path + 'weather_aug', str(count)+'l.png'))

            """
            for i in range(120):
                print (data[i]['steer'], data[i]['steer_noise'])
                self.assertEqual(data[i]['steer'], data[i]['steer_noise'])
                self.assertEqual(data[i]['throttle'], data[i]['throttle_noise'])
                self.assertEqual(data[i]['brake'], data[i]['brake_noise'])
            """



