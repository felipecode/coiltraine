import os
import numpy as np
import unittest
import shutil

from input import CoILDataset
from logger import coil_logger
from logger import monitorer
from logger import readJSONlog
from configs import g_conf, GlobalConfig



class testMonitorer(unittest.TestCase):


    def __init__(self, *args, **kwargs):

        super(testMonitorer, self).__init__(*args, **kwargs)

        # Recreate some folder
        if os.path.exists('_logs/monitor_test'):
            shutil.rmtree('_logs/monitor_test')




    def test_check_status_not_existent(self):

        # Check if status could be check for unexistent experiments
        g_conf = GlobalConfig()
        status = monitorer.get_status('monitor_test', 'experiment_25',
                                      g_conf.param.PROCESS_NAME)
        self.assertEqual(status[0], "Does Not Exist")



    def test_check_status_to_run(self):

        # Check for an experiment that exists in the config files but has not been started
        g_conf = GlobalConfig()
        g_conf.param.NAME = 'experiment_to_run'

        status = monitorer.get_status('monitor_test', 'experiment_to_run',
                                      g_conf.param.PROCESS_NAME)
        self.assertEqual(status[0], "Not Started")


    def test_check_status_running_loading(self):
        g_conf = GlobalConfig()
        g_conf.param.NAME = 'experiment_running_loading'
        # TODO: this merge is weird.
        g_conf.merge_with_yaml('configs/monitor_test/experiment_running_loading.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')




        coil_logger.add_message('Loading',{
                                    "Keys_Division": [1,123,1,1,2,12,3,12,31,2,1,1]
                                })

        coil_logger.add_message('Loading',{
                                    "Models_loaded": ' VUALA ',
                                    "Checkpoint": "988765"
                                })


        # TODO: Check how the alias will work.
        status = monitorer.get_status('monitor_test', 'experiment_running_loading',
                                      g_conf.param.PROCESS_NAME)

        self.assertEqual(status[0], "Loading")





    def test_check_status_running_iter(self):
        g_conf = GlobalConfig()
        g_conf.param.NAME = 'experiment_running_iter'
        # TODO: this merge is weird.
        g_conf.merge_with_yaml('configs/monitor_test/experiment_running_iter.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')

        coil_logger.add_message('Loading', {
            "Keys_Division": [1, 123, 1, 1, 2, 12, 3, 12, 31, 2, 1, 1]
        })

        coil_logger.add_message('Loading', {
            "Models_loaded": ' VUALA ',
            "Checkpoint": "988765"
        })



        for i in range(0, 10):

            coil_logger.add_message('Reading', {
                                        "Iteration": i,
                                        "ReadKeys": [1, 123, 5, 1, 34, 1, 23]

                                    })
            coil_logger.add_message('Model', {
                                        "Iteration": i,
                                        "Output": ["output"]
                                    })


        # TODO: Check how the alias will work.
        status = monitorer.get_status('monitor_test', 'experiment_running_iter',
                                      g_conf.param.PROCESS_NAME)

        self.assertEqual(status[0], "Iterating")




    def test_check_status_error(self):
        g_conf = GlobalConfig()
        # TODO: THe error ? How do nicely merge with the other parts ??
        g_conf.param.NAME = 'experiment_running_error'
        # TODO: this merge is weird.
        g_conf.merge_with_yaml('configs/monitor_test/experiment_running_error.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')

        coil_logger.add_message('Loading', {
            "Keys_Division": [1, 123, 1, 1, 2, 12, 3, 12, 31, 2, 1, 1]
        })

        coil_logger.add_message('Loading', {
            "Models_loaded": ' VUALA ',
            "Checkpoint": "988765"
        })



        for i in range(0, 10):

            coil_logger.add_message('Reading', {
                                        "Iteration": i,
                                        "ReadKeys": [1, 123, 5, 1, 34, 1, 23]

                                    })
            coil_logger.add_message('Model', {
                                        "Iteration": i,
                                        "Output": ["output"]
                                    })

        coil_logger.add_message('Error', {
                    "Iteration": 10,
                    "Message": " Some data integrity problems ! "

                })

        # TODO: Check how the alias will work.


        status = monitorer.get_status('monitor_test', 'experiment_running_error',
                                      g_conf.param.PROCESS_NAME)

        self.assertEqual(status[0], "Error")

    def test_check_status_finished(self):
        g_conf = GlobalConfig()
        # TODO: THe error ? How do nicely merge with the other parts ??
        g_conf.param.NAME = 'experiment_finished'
        # TODO: this merge is weird.
        g_conf.merge_with_yaml('configs/monitor_test/experiment_finished.yaml')

        g_conf.param.MISC.NUMBER_OF_ITERATIONS = 20
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.set_type_of_process('train')

        # We set the number of iterations as

        coil_logger.add_message('Loading', {
            "Keys_Division": [1, 123, 1, 1, 2, 12, 3, 12, 31, 2, 1, 1]
        })

        coil_logger.add_message('Loading', {
            "Models_loaded": ' VUALA ',
            "Checkpoint": "988765"
        })



        for i in range(0, 21):

            coil_logger.add_message('Reading', {
                                        "Iteration": i,
                                        "ReadKeys": [1, 123, 5, 1, 34, 1, 23]

                                    })
            coil_logger.add_message('Model', {
                                        "Iteration": i,
                                        "Output": ["output"]
                                    })


        # TODO: Check how the alias will work.

        status = monitorer.get_status('monitor_test', 'experiment_finished',
                                      g_conf.param.PROCESS_NAME)

        self.assertEqual(status[0], "Finished")