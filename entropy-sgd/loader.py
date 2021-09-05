import torch as th
import math
import torch.utils.data
import numpy as np

from torchvision import datasets


class sampler_t:
    def __init__(self, batch_size, x, y, train=True):
        self.n = x.size(0)
        self.x, self.y = x, y
        self.b = batch_size
        self.idx = th.arange(0, self.b).long()
        self.train = train
        self.train_index = 0
        self.sidx = 0
        self.total_loops = int(math.floor(self.n/self.b))

    def __next__(self):
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


def mnist(opt):
    d1, d2 = datasets.MNIST('../proc', download=True, train=True), \
            datasets.MNIST('../proc', train=False)

    train = sampler_t(opt['b'],
                      d1.train_data.view(-1, 1, 28, 28).float(),
                      d1.train_labels)
    val = sampler_t(opt['b'],
                    d2.test_data.view(-1, 1, 28, 28).float(),
                    d2.test_labels, train=False)
    return train, val, val


def cifar10(opt):
    loc = './proc/'
    d1 = np.load(loc+'cifar10-train.npz')
    d2 = np.load(loc+'cifar10-test.npz')

    # Loads all the train dataset and the samples are
    # acessed randomly
    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                      th.from_numpy(d1['labels']))

    # Loads all the train dataset which is acessed in order
    # All samples are reached
    train_eval = sampler_t(opt['eval_b'], th.from_numpy(d1['data']),
                           th.from_numpy(d1['labels']), train=False)
    # Loads all the validation set which is acessed in order
    # All samples are reached
    val = sampler_t(opt['eval_b'], th.from_numpy(d2['data']),
                    th.from_numpy(d2['labels']), train=False)

    return train, train_eval, val
