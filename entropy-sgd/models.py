import torch.nn as nn
from torch import Tensor

from experiment_config import DatasetType


class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


# TODO: Remove MNIST code
class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.5

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784, c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c, c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c, 10))

        s = '[%s] Num parameters: %d' % (self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


class mnistconv(nn.Module):
    def __init__(self, opt):
        super(mnistconv, self).__init__()
        self.name = 'mnistconv'
        opt['d'] = 0.5

        def convbn(ci, co, ksz, psz, p):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz, stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1, 20, 5, 3, opt['d']),
            convbn(20, 50, 5, 2, opt['d']),
            View(50 * 2 * 2),
            nn.Linear(50 * 2 * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500, 10))

        s = '[%s] Num parameters: %d' % (self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


class allcnn(nn.Module):
    def __init__(self, opt={'d': 0.5}, c1=96, c2=192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        opt['d'] = 0.5

        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(0.2),            # (N, 03, 32, 32)
            convbn(3, c1, 3, 1, 1),     # (N, 96, 32, 32)
            convbn(c1, c1, 3, 1, 1),    # (N, 96, 32, 32)
            convbn(c1, c1, 3, 2, 1),    # (N, 96, 16, 16)
            nn.Dropout(opt['d']),       # (N, 96, 16, 16)
            convbn(c1, c2, 3, 1, 1),    # (N, 192, 16, 16)
            convbn(c2, c2, 3, 1, 1),    # (N, 192, 16, 16)
            convbn(c2, c2, 3, 2, 1),    # (N, 192, 08, 08)
            nn.Dropout(opt['d']),       # (N, 192, 08, 08)
            convbn(c2, c2, 3, 1, 1),    # (N, 192, 08, 08)
            convbn(c2, c2, 3, 1, 1),    # (N, 192, 08, 08)
            convbn(c2, 10, 1, 1),       # (N, 10, 08, 08)
            nn.AvgPool2d(8),            # (N, 10, 01, 01)
            View(10))

        s = '[%s] Num parameters: %d' % (self.name,
                                         num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


# Currently being used for typing, we may remove it
# TODO: Remove it from typing
class ExperimentBaseModel(nn.Module):
    def __init__(self, dataset_type: DatasetType):
        super().__init__()
        self.dataset_type = dataset_type

    def forward(self, x) -> Tensor:
        raise NotImplementedError
