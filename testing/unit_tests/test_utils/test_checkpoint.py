import os
import numpy as np
import unittest
import shutil
import torch

from input import CoILDataset
from logger import coil_logger
from logger import monitorer
from logger import readJSONlog
from configs import g_conf, GlobalConfig
from utils.checkpoint_schedule import  *




class testCheckpoint(unittest.TestCase):


    def __init__(self, *args, **kwargs):

        super(testCheckpoint, self).__init__(*args, **kwargs)



    def test_simulate_save_and_read(self):

        exp_batch = 'eccv'
        exp_alias = 'experiment_1'

        checkpoint = get_latest_saved_checkpoint(exp_batch, exp_alias)

        for iteration in range(0, g_conf.param.MISC.NUMBER_ITERATIONS/2):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join(exp_batch, exp_alias,
                                               'configurations', str(iteration) + '.pth'))

        if is_next_checkpoint_ready(exp_batch, exp_alias):
            latest = get_next_checkpoint(exp_batch, exp_alias)

        print (latest)
        #self.assertEqual(latest == )


        for iteration in range(g_conf.param.MISC.NUMBER_ITERATIONS/2,
                               g_conf.param.MISC.NUMBER_ITERATIONS):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join(exp_batch, exp_alias,
                                               'configurations', str(iteration) + '.pth'))


        if is_next_checkpoint_ready(exp_batch, exp_alias):
            latest = get_next_checkpoint(exp_batch, exp_alias)

        print (latest)