import os
import numpy as np
import unittest
from configs import g_conf, merge_with_yaml, set_type_of_process


class testConfigs(unittest.TestCase):

    def test_config_integrity(self):


        pass


    def test_merge_yaml_line_globaldict(self):


        g_conf.NAME = 'experiment_1'
        merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')


