import torch as th
import torch.utils.data
import numpy as np


class sampler_t:
    def __init__(self, batch_size, x, y, train=True):
        """
        Data loader in batches for supervised problems.

        Args:
            batch_size (int): size of the batches.
            x (th.Tensor): networks's input dataset on the format [N, C, H, W].
            y (th.Tensor): annotation of the input dataset with size N.
                           N is the size of the whole dataset.
            train (bool): True, if the batch samples must be randomly selected.
                          False, otherwise.
        """
        self.n = x.size(0)
        self.x, self.y = x, y
        self.b = batch_size

        # Initializes tensor to receive the batch indexes
        # [0, ..., batch_size - 1]
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.train_index = 0
        self.sidx = 0

    def __next__(self):
        """
        Selects samples equal to the batch size.

        Returns:
            x (th.Tensor): network's input data on the format [batch, C, H, W].
            y (th.Tensor): annotation of the input data with size batch.
        """
        if self.train:
            self.idx.random_(0, self.n)
        else:
            s = self.sidx
            e = min(s + self.b, self.n)

            self.idx = th.arange(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

        x = th.index_select(self.x, 0, self.idx)
        y = th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self


def cifar10(opt):
    """
    Defines the data loader for CIFAR-10.

    Args:
        opt (dict): Dictionary with data parameters.
                    b = batch size.
                    eval_b = larger batch for evaluation.

    Returns:
        train (sampler_t): Loader for the training set.
        train_eval (sampler_t): Loader for the complexity measures.
        val (sampler_t): Loader for the validation set.
    """
    loc = './proc/'
    d1 = np.load(loc + 'cifar10-train.npz')
    d2 = np.load(loc + 'cifar10-test.npz')

    # Loads all the train dataset (50000)
    # and the samples are accessed randomly (train=True)
    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                      th.from_numpy(d1['labels']))

    # Loads all the train dataset (50000)
    # which is accessed in order. All samples are reached
    train_eval = sampler_t(opt['eval_b'], th.from_numpy(d1['data']),
                           th.from_numpy(d1['labels']), train=False)

    # Loads all the validation set (10000)
    # which is accessed in order. All samples are reached
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                    th.from_numpy(d2['labels']), train=False)

    return train, train_eval, val
