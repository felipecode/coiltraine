import os
import numpy as np
import unittest


from input import CoILDataset
from logger import coil_logger
from logger import readJSONlog
from logger.coil_logger import recover_loss_window
from configs import g_conf, merge_with_yaml, set_type_of_process

#import logging
#logging.getLogger('tensorflow').disabled = True

class testLogger(unittest.TestCase):

    def test_recover_loss_window(self):

        g_conf.EXPERIMENT_NAME = 'res34-50-lowdropout'
        merge_with_yaml('configs/cvprfinal_valstop_seed1/res34-50-lowdropout.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')

        print (recover_loss_window('TrainValidation', None))