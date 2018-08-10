import os
import traceback
import time
import sys
import json
import multiprocessing

import numpy as np
import torch
import traceback
import torch.optim as optim
import random

from torchvision import transforms
# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, Augmenter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint
from .configs import MODEL_TYPE, MODEL_CONFIGURATION


def execute_validation(checkpoint, output_file, gpu):
    p = multiprocessing.Process(target=execute,
                                args=(checkpoint, output_file, gpu))
    p.start()
    # execute(checkpoint, output_file, gpu)
    return p


def execute(checkpoint, output_file, gpu):
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        #Define the dataset. This structure is has the __get_item__ redefined in a way
        #that you can access the HDFILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], 'Town02W14Noise')
        augmenter = Augmenter(None)
        dataset = CoILDataset(full_dataset, transform=augmenter)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)
        model = CoILModel(MODEL_TYPE, MODEL_CONFIGURATION)
        ckpt = torch.load(checkpoint)
        checkpoint_iteration = ckpt['iteration']
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda()

        model.eval()
        accumulated_loss = []
        accumulated_error = []
        iteration_on_checkpoint = 0
        for data in data_loader:
            input_data, float_data = data
            control_position = np.where(dataset.meta_data[:, 0] == b'control')[0][0]
            speed_position = np.where(dataset.meta_data[:, 0] == b'speed_module')[0][0]
            output = model.forward_branch(torch.squeeze(input_data['rgb']).cuda(),
                                          float_data[:, speed_position, :].cuda(),
                                          float_data[:, control_position, :].cuda())
            loss = torch.mean((output - dataset.extract_targets(float_data).cuda())**2).data.tolist()
            mean_error = torch.mean(torch.abs(output - dataset.extract_targets(float_data).cuda())).data.tolist()
            accumulated_error.append(mean_error)
            accumulated_loss.append(loss)
            error = torch.abs(output - dataset.extract_targets(float_data).cuda())
            iteration_on_checkpoint += 1

        checkpoint_average_loss = np.percentile(accumulated_loss, 85)  # accumulated_loss/(len(data_loader))
        checkpoint_average_error = np.percentile(accumulated_error, 85)  # accumulated_error/(len(data_loader))

        with open(output_file, 'w') as ofile:
            json.dump({'avg_loss': checkpoint_average_loss,
                       'avg_error': checkpoint_average_error},
                       ofile, indent=4, sort_keys=True)


    except KeyboardInterrupt:
        print('Error', 'Message: Killed By User')

    except:
        traceback.print_exc()

        print('Error', 'Message: Something Happened')
