import numpy as np
import random

import torch

from torch.utils.data.sampler import Sampler

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf


# TODO: When putting sequences, the steering continuity and integrity needs to be verified
def get_rank(input_array):

    rank = 0
    while True:
        try:
            length = len(input_array)
            input_array = input_array[0]
            rank += 1
        except:
            return rank



class PreSplittedSampler(Sampler):
    """ Sample on a list of keys that was previously splitted

    """


    def __init__(self, keys):

        self.keys = keys

        self.replacement = True

    def __iter__(self):
        """

            OBS: One possible thing to be done is the possibility to have a matrix of ids
            of rank N
            OBS2: Probably we dont need weights right now


        Returns:
            Iterator to get ids for the dataset

        """
        rank_keys = get_rank(self.keys)


        # First we check how many subdivisions there are
        if rank_keys == 2:

            weights = torch.tensor([1.0/float(len(self.keys))]*len(self.keys), dtype=torch.double)

            idx = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idx = idx.tolist()
            return iter([random.choice(self.keys[i]) for i in idx])

        elif rank_keys == 3:
            weights = torch.tensor([1.0 / float(len(self.keys))] * len(self.keys),
                                   dtype=torch.double)
            idx = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idx = idx.tolist()
            weights = torch.tensor([1.0 / float(len(self.keys[0]))] * len(self.keys[0]),
                                   dtype=torch.double)
            idy = torch.multinomial(weights, g_conf.NUMBER_ITERATIONS, True)
            idy = idy.tolist()


            return iter([random.choice(self.keys[i][j]) for i, j in zip(idx,idy)])

        else:
            raise ValueError("Keys have invalid rank")


    def __len__(self):
        return g_conf.NUMBER_ITERATIONS * g_conf.BATCH_SIZE


class BatchSequenceSampler(object):
    r"""Wraps another sampler to yield a mini-batch of indices taking a certain sequence size

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, keys, batch_size, sequence_size, sequence_stride, drop_last=True):
        sampler = PreSplittedSampler(keys)

        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(sequence_size, int) or isinstance(sequence_size, bool) or \
                sequence_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(sequence_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sequence_stride = sequence_stride

    def __iter__(self):


        batch = []
        for idx in self.sampler:
            for seq in range(0, self.sequence_size * self.sequence_stride, self.sequence_stride):
                batch.append(int(idx)+seq)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


        #if len(batch) > 0 and not self.drop_last:
        #    yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


