import os
import numpy as np
import unittest
import shutil
import torch

from logger import coil_logger
from logger import monitorer
from logger import readJSONlog
from configs import g_conf, merge_with_yaml, set_type_of_process
from utils.checkpoint_schedule import  *




class testCheckpoint(unittest.TestCase):


    def __init__(self, *args, **kwargs):

        super(testCheckpoint, self).__init__(*args, **kwargs)
        if os.path.exists('_logs/test/test_checkpoint'):
            shutil.rmtree('_logs/test/test_checkpoint')


    def test_simulate_save_and_read(self):
        g_conf.immutable(False)
        # TODO: this merge is weird.
        merge_with_yaml('test/test_checkpoint.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('validation')
        exp_batch = 'test'
        exp_alias = 'test_checkpoint'



        checkpoint = get_latest_saved_checkpoint()

        self.assertEqual(checkpoint, None)

        for iteration in range(0, int(g_conf.NUMBER_ITERATIONS/2)):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }


                torch.save(state, os.path.join('_logs', exp_batch, exp_alias,
                                               'checkpoints', str(iteration) + '.pth'))




        for validation in g_conf.TEST_SCHEDULE[0:int(len(g_conf.TEST_SCHEDULE)/2)]:

            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

                latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
                # Create the checkpoint file
                coil_logger.write_on_csv(latest, [0.1, 0.2, 0.0])



                print (latest)
        self.assertEqual(latest, 800)


        for iteration in range(int(g_conf.NUMBER_ITERATIONS/2),
                               g_conf.NUMBER_ITERATIONS):

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias,
                                               'checkpoints', str(iteration) + '.pth'))

        if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):
            latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
            coil_logger.write_on_csv(latest, [0.1, 0.2, 0.0])


        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):

            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

                latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
                # Create the checkpoint file
                coil_logger.write_on_csv(latest, [0.1, 0.2, 0.0])
