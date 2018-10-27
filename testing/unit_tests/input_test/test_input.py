import os
import unittest
import time

from coil_core.train import select_balancing_strategy

from input.coil_dataset import CoILDataset


from configs import g_conf

class testInput(unittest.TestCase):

    def test_new_sampler(self):


        """
        Tests a balancing strategy and checks if by only getting images with 30 degrees
        the sampler actually just gets images with 30 degrees.
        Returns:

        """

        test_images_write_path = 'testing/unit_tests/_test_images_'

        if not os.path.exists(test_images_write_path):
            os.mkdir(test_images_write_path)





        root_path = '/home/felipecodevilla/Datasets/CARLA100'

        dataset = CoILDataset(root_path, transform=None, preload_name=str(g_conf.NUMBER_OF_HOURS)
                                                                      + 'hours_CARLA100')

        g_conf.SPLIT = [['left', []], ['central', []], ['right', []], ['weights', [0., 0., 1.]]]

        data_loader = select_balancing_strategy(dataset, 0)



        count = 0
        capture_time = time.time()
        for data in data_loader:
            controls = data['directions']


            for i in range( len(data['angle'])):

                self.assertEqual(data['angle'][i], 30)


            count += 1
