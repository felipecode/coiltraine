import os
import numpy as np
import unittest


from input import CoILDataset
from logger import coil_logger
from logger import readJSONlog
from configs import g_conf



class testLogger(unittest.TestCase):




    def test_global_logger_train(self):
        # TODO: THERE WILL BE A NAME GENERATOR
        g_conf.param.NAME = 'experiment_1'
        g_conf.merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')


        coil_logger.add_message('Loading',{
                                    "Keys_Division":[1,123,1,1,2,12,3,12,31,2,1,1]
                                })

        coil_logger.add_message('Loading',{
                                    "Models_loaded": ' VUALA ',
                                    "Checkpoint": "988765"
                                })


        for i in range(0, 10):

            coil_logger.add_message('Reading', {
                                        "Iteration": i,
                                        "ReadKeys": [1,123,5,1,34,1,23]

                                    })
            coil_logger.add_message('Network', {
                                        "Iteration": i,
                                        "Output-": ["output"]
                                    })






    def test_global_logger_drive(self):
        """
        # TODO: THERE WILL BE A NAME GENERATOR
        g_conf.param.NAME = 'experiment_1'
        g_conf.merge_with_yaml('configs/eccv/experiment_1.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')


        coil_logger.add_message('Loading',{
                                    "Keys_Division": [1, 123, 1, 1, 2, 1, 2, 3, 12, 31, 2, 1, 1]
                                })

        coil_logger.add_message('Loading',{
                                    "Models_loaded": ' VUALA ',
                                    "Checkpoint": "988765"
                                })


        for i in range(0,10):

            coil_logger.add_message()
        """
        pass






    def test_log_monitor(self):



        # This function should make a log and them read and report the status.

        # I will simulate an entire execution pass of the system

        pass
    def test_writing_data_message(self):

        pass