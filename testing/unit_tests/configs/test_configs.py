import os
import numpy as np
import unittest
from configs import g_cfg
from configs import merge_g_cfg_from_file
from configs import merge_g_cfg_from_list



class testConfigs(unittest.TestCase):

    def test_config_integrity(self):

        #


    def test_merge_command_line_globaldict(self):

        # Test if the merged configuration works
        merge_g_cfg_from_list(parameter_list)


        # Have the result dict

        self.assertDictEqual()


    def test_merge_yaml_line_globaldict(self):

        # Test if the merged configuration works
        merge_g_cfg_from_file(yaml_file)

        # Check if the new config file is different, or if the changed configs worked

        self.assertDictEqual()