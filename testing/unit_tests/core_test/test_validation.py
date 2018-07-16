import os
import numpy as np
import unittest

import torch

from input.coil_sampler import BatchSequenceSampler, SubsetSampler, RandomSampler
from input.coil_dataset import CoILDataset
from input import Augmenter
import input.splitter as splitter
from PIL import Image
# from utils.general import plot_test_image

from configs import set_type_of_process, merge_with_yaml

from torchvision import transforms
from utils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from network import CoILModel
from torch.utils.data import TensorDataset as dset
from configs import g_conf


class testValidation(unittest.TestCase):
    # def __init__(self, *args, **kwargs):
    #    super(testSampler, self).__init__(*args, **kwargs)
    # self.root_test_dir = '/home/felipe/Datasets/CVPR02Noise/SeqTrain'

    # self.test_images_write_path = 'testing/unit_tests/_test_images_'


    def test_core_validation(self):

        dataset_name = 'DataVerySmall'

        exp_batch  = 'eccv_debug'
        exp_alias = 'experiment_1'
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
        set_type_of_process('validation', dataset_name)

        augmenter = Augmenter(None)

        dataset = CoILDataset(full_dataset, transform=augmenter)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)


        # TODO: here there is clearly a posibility to make a cool "conditioning" system.
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)



        model.cuda()

        print (dataset.meta_data)
        best_loss = 1000
        best_error = 1000
        best_loss_iter = 0
        best_error_iter = 0


        checkpoint_avg_loss_vec = []


        for latest in [ 2000, 4000, 8000, 16000, 32000, 64000, 100000, 200000, 300000, 400000, 500000]:


            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                    , 'checkpoints', str(latest) + '.pth'))
            checkpoint_iteration = checkpoint['iteration']
            print ("Validation loaded ", checkpoint_iteration)

            model.load_state_dict(checkpoint['state_dict'])

            model.eval()
            accumulated_loss = 0
            accumulated_error = 0
            iteration_on_checkpoint = 0
            for data in data_loader:

                input_data, float_data = data
                control_position = np.where(dataset.meta_data[:, 0] == b'control')[0][0]
                speed_position = np.where(dataset.meta_data[:, 0] == b'speed_module')[0][0]



                print ("image ", input_data['rgb'].shape)
                print (float_data)

                output = model.forward_branch(torch.squeeze(input_data['rgb']).cuda(),
                                              float_data[:, speed_position, :].cuda(),
                                              float_data[:, control_position, :].cuda())




                # TODO: Change this a functional standard using the loss functions.

                loss = torch.mean((output - dataset.extract_targets(float_data).cuda())**2).data.tolist()
                mean_error = torch.mean(torch.abs(output - dataset.extract_targets(float_data).cuda())).data.tolist()
                #print ("Loss", loss)
                #print ("output", output[0])
                accumulated_error += mean_error
                accumulated_loss += loss
                error = torch.abs(output - dataset.extract_targets(float_data).cuda())


                # Log a random position

                iteration_on_checkpoint += 1

            checkpoint_average_loss = accumulated_loss/(len(data_loader))

            checkpoint_avg_loss_vec.append(checkpoint_average_loss)

        count = 0
        for latest in [2000, 4000, 8000, 16000, 32000, 64000, 100000, 200000, 300000, 400000, 500000]:

            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                                 , 'checkpoints', str(latest) + '.pth'))
            checkpoint_iteration = checkpoint['iteration']
            print ("Validation loaded ", checkpoint_iteration)

            model.load_state_dict(checkpoint['state_dict'])

            model.eval()
            accumulated_loss = 0
            accumulated_error = 0
            iteration_on_checkpoint = 0
            for data in data_loader:
                input_data, float_data = data
                control_position = np.where(dataset.meta_data[:, 0] == b'control')[0][0]
                speed_position = np.where(dataset.meta_data[:, 0] == b'speed_module')[0][0]

                print ("image ", input_data['rgb'].shape)
                print (float_data)

                output = model.forward_branch(torch.squeeze(input_data['rgb']).cuda(),
                                              float_data[:, speed_position, :].cuda(),
                                              float_data[:, control_position, :].cuda())

                # TODO: Change this a functional standard using the loss functions.

                loss = torch.mean(
                    (output - dataset.extract_targets(float_data).cuda()) ** 2).data.tolist()
                mean_error = torch.mean(
                    torch.abs(output - dataset.extract_targets(float_data).cuda())).data.tolist()
                # print ("Loss", loss)
                # print ("output", output[0])
                accumulated_error += mean_error
                accumulated_loss += loss
                error = torch.abs(output - dataset.extract_targets(float_data).cuda())

                # Log a random position

                iteration_on_checkpoint += 1

            checkpoint_average_loss = accumulated_loss / (len(data_loader))

            self.assertEqual(checkpoint_avg_loss_vec[count], checkpoint_average_loss)
            count += 1





