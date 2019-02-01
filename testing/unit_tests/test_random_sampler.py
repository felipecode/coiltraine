import unittest
import os

from configs import g_conf, set_type_of_process, merge_with_yaml
from input import RandomSequenceSampler
from utils.general import create_log_folder, create_exp_path
"""
The idea of this test is to check if sampling is able to sample random sequencial images
inside a batch

"""


class TestRandomSampler(unittest.TestCase):

    def test_init(self):
        keys = range(0, 10000)
        executed_iterations = 0
        stride = 2
        sequence_size = 1
        batch_size = 120
        # The initialization needs to have an stride and a sequence size.
        # The stride determines if t
        random_sampler = RandomSequenceSampler(keys, executed_iterations,
                                               batch_size, stride, sequence_size)

    def test_correct(self):
        # TODO this is not working
        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'coil_icra'
        create_log_folder('sample')
        create_exp_path('sample', 'coil_icra')
        merge_with_yaml('configs/sample/coil_icra.yaml')

        set_type_of_process('train')

        keys = range(0, 10000)
        stride = 2
        sequence_size = 12

        executed_iterations = 100
        batch_size = 120
        random_sampler = RandomSequenceSampler(keys, executed_iterations*batch_size,
                                               batch_size, stride, sequence_size)

        # The len should be equal to 120 and it should not have so much sequences

        iteration = 100
        for sample in random_sampler:
            iteration += 1

        self.assertEqual(iteration, g_conf.NUMBER_ITERATIONS)


    def test_sampling_sequences(self):
        # Thest if the sequences are sampled following a correct order and correct id jumps
        g_conf.immutable(False)
        g_conf.EXPERIMENT_NAME = 'coil_icra'
        create_log_folder('sample')
        create_exp_path('sample', 'coil_icra')
        merge_with_yaml('configs/sample/coil_icra.yaml')

        set_type_of_process('train')

        keys = range(0, 10000)
        executed_iterations = 0
        stride = 2
        sequence_size = 1
        batch_size = 120
        random_sampler = RandomSequenceSampler(keys, executed_iterations,
                                               batch_size, stride, sequence_size)

        # The len should be equal to 120 and it should not have so much sequences

        for sampled_ids in random_sampler:
            count_truths = 0
            self.assertTrue(len(sampled_ids) == 120)
            previous_id = sampled_ids[0]
            for id in sampled_ids[1:]:
                if id - previous_id == stride:
                    count_truths += 1
            self.assertLess(count_truths, 10)

        # You set the stride and the iterations
        executed_iterations = 0
        stride = 10
        sequence_size = 2
        random_sampler = RandomSequenceSampler(keys, executed_iterations,
                                               batch_size, stride, sequence_size)
        for sampled_ids in random_sampler:
            self.assertTrue(len(sampled_ids) == 120)
            for id in range(0, len(sampled_ids), sequence_size):
                self.assertTrue(sampled_ids[id+1] - sampled_ids[id] == stride)


        # this parameters should raise a value error
        executed_iterations = 0
        stride = 10
        sequence_size = 13

        self.assertRaises(ValueError, RandomSequenceSampler, keys, executed_iterations,
                         batch_size, stride, sequence_size)

