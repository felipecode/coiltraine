import os
import unittest
import torch
import numpy as np
import random

from logger.coil_logger import recover_loss_window
from configs import g_conf, merge_with_yaml, set_type_of_process

from input import Augmenter, CoILDataset, RandomSampler
from network import CoILModel

class testDeterminism(unittest.TestCase):

    """
    def __init__(self):

        self.init_seeds = []
        self.init_seeds = []
        self.init_seeds = []
    """
    def get_results_based_on_seeds(self, init_seed, dropout_seed, sampling_seed, magical_seed):

        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'res34'
        merge_with_yaml('configs/sample/res34.yaml')
        # JUST A TRICK TO CONTAIN THE CURRENT LIMITATIONS
        g_conf.INIT_SEED = init_seed
        g_conf.DROPOUT_SEED = dropout_seed
        g_conf.MAGICAL_SEED = magical_seed
        g_conf.SAMPLING_SEED = sampling_seed

        set_type_of_process('train')

        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        if g_conf.DROPOUT_SEED is not None:
           torch.manual_seed(g_conf.DROPOUT_SEED)
           torch.cuda.manual_seed_all(g_conf.DROPOUT_SEED)



        model.zero_grad()

        input = torch.randn(1, 256)
        weights = model.speed_branch.layers[1][0].weight
        _, zero_pos = np.where(model.speed_branch.layers[1](input).data == 0)
        keys = range(0, 5000 - g_conf.NUMBER_IMAGES_SEQUENCE)
        sampler = RandomSampler(keys, 0 * g_conf.BATCH_SIZE)
        iterator = sampler.__iter__()
        samples = []
        for i in range(20):
            torch.manual_seed(i)
            torch.cuda.manual_seed_all(i)
            samples.append(next(iterator))

        return zero_pos, weights, samples

    def test_determinism(self):
        """
             ONLY DROPOUT
        """

        torch.backends.cudnn.deterministic = True

        # Test 1 Get zero positions
        zero_pos1, w1, samples1 = self.get_results_based_on_seeds(333, 32, 123, 666)
        zero_pos2, w2, samples2 = self.get_results_based_on_seeds(333, 42, 123, 666)

        try:
            np.testing.assert_equal(zero_pos1, zero_pos2)
            dropout_res = True
        except AssertionError as err:
            dropout_res = False
            print(err)
        self.assertFalse(dropout_res)

        try:
            np.testing.assert_almost_equal(w1.data.numpy(), w2.data.numpy())
            init_res = True
        except AssertionError as err:
            init_res = False
            print(err)

        self.assertTrue(init_res)

        try:
            np.testing.assert_equal(samples1, samples2)
            samples_res = True
        except AssertionError as err:
            samples_res = False
            print(err)

        self.assertTrue(samples_res)

        """
            ONLY CHANGING INIT
        """
        # Test 1 Get zero positions
        zero_pos1, w1, samples1 = self.get_results_based_on_seeds(222, 42, 123, 666)
        zero_pos2, w2, samples2 = self.get_results_based_on_seeds(333, 42, 123, 666)

        print(zero_pos1)
        print(zero_pos2)

        try:
            np.testing.assert_equal(zero_pos1, zero_pos2)
            dropout_res = True
        except AssertionError as err:
            dropout_res = False
            print(err)
        self.assertFalse(dropout_res)

        try:
            np.testing.assert_almost_equal(w1.data.numpy(), w2.data.numpy())
            init_res = True
        except AssertionError as err:
            init_res = False
            print(err)

        self.assertFalse(init_res)

        try:
            np.testing.assert_equal(samples1, samples2)
            samples_res = True
        except AssertionError as err:
            samples_res = False
            print(err)

        self.assertTrue(samples_res)

        """
            ONLY CHANGING SAMPLING
        """
        # Test 1 Get zero positions
        zero_pos1, w1, samples1 = self.get_results_based_on_seeds(333, 42, 123, 666)
        zero_pos2, w2, samples2 = self.get_results_based_on_seeds(333, 42, 456, 666)

        try:
            np.testing.assert_equal(zero_pos1, zero_pos2)
            dropout_res = True
        except AssertionError as err:
            dropout_res = False
            print(err)
        self.assertTrue(dropout_res)

        try:
            np.testing.assert_almost_equal(w1.data.numpy(), w2.data.numpy())
            init_res = True
        except AssertionError as err:
            init_res = False
            print(err)

        self.assertTrue(init_res)

        try:
            np.testing.assert_equal(samples1, samples2)
            samples_res = True
        except AssertionError as err:
            samples_res = False
            print(err)

        self.assertFalse(samples_res)