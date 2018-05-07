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
        if os.path.exists('_logs/test/test_checkpoint'):
            shutil.rmtree('_logs/test/test_checkpoint')


    def test_simulate_save_and_read(self):
        g_conf = GlobalConfig()
        g_conf.param.NAME = 'test_checkpoint'
        # TODO: this merge is weird.
        g_conf.merge_with_yaml('test/test_checkpoint.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('validation')
        exp_batch = 'test'
        exp_alias = 'test_checkpoint'

        checkpoint = get_latest_saved_checkpoint('test', g_conf.param.NAME)

        self.assertEqual(checkpoint, None)

        for iteration in range(0, int(g_conf.param.MISC.NUMBER_ITERATIONS/2)):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }


                torch.save(state, os.path.join('_logs', exp_batch, exp_alias,
                                               'checkpoints', str(iteration) + '.pth'))




        for validation in g_conf.param.MISC.TEST_SCHEDULE:

            if is_next_checkpoint_ready(exp_batch, exp_alias, g_conf.param.PROCESS_NAME,
                                        g_conf.param.MISC.TEST_SCHEDULE):

                latest = get_next_checkpoint(exp_batch, exp_alias, g_conf.param.PROCESS_NAME,
                                             g_conf.param.MISC.TEST_SCHEDULE)

                # Create the checkpoint file
                coil_logger.add_scalar()


                print (latest)
        #self.assertEqual(latest == )


        for iteration in range(int(g_conf.param.MISC.NUMBER_ITERATIONS/2),
                               g_conf.param.MISC.NUMBER_ITERATIONS):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias,
                                               'checkpoints', str(iteration) + '.pth'))

        if is_next_checkpoint_ready(exp_batch, exp_alias, g_conf.param.PROCESS_NAME,
                                    g_conf.param.MISC.TEST_SCHEDULE):
            latest = get_next_checkpoint(exp_batch, exp_alias, g_conf.param.PROCESS_NAME,
                                         g_conf.param.MISC.TEST_SCHEDULE)

