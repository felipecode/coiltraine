import os
import numpy as np
import unittest
from configs import g_conf, merge_with_yaml, set_type_of_process
from torchvision import transforms
from logger.printer import print_folder_process_names
import torch

class testName(unittest.TestCase):




    def test_name_generation(self):


        g_conf.NAME = 'experiment_1'
        merge_with_yaml('configs/test_exps/experiment_1.yaml')

        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        set_type_of_process('train')




    def test_name_generation_printing(self):


        print_folder_process_names('eccv')




