import torch as th
import torch.nn as nn
import math

from torch.autograd import Variable
from torch import Tensor

from experiment_config import DatasetType


class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class mnistfc(nn.Module):
    def __init__(self, opt):
        super(mnistfc, self).__init__()
        self.name = 'mnsitfc'

        c = 1024
        opt['d'] = 0.5

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


class mnistconv(nn.Module):
    def __init__(self, opt):
        super(mnistconv, self).__init__()
        self.name = 'mnistconv'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


class allcnn(nn.Module):
    def __init__(self, opt = {'d':0.5}, c1=96, c2= 192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        opt['d'] = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        s = '[%s] Num parameters: %d'%(self.name, num_parameters(self.m))
        print(s)

    def forward(self, x):
        return self.m(x)


class ExperimentBaseModel(nn.Module):
  def __init__(self, dataset_type: DatasetType):
    super().__init__()
    self.dataset_type = dataset_type

  def forward(self, x) -> Tensor:
    raise NotImplementedError


class NiNBlock(nn.Module):
  def __init__(self, inplanes: int, planes: int) -> None:
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn3 = nn.BatchNorm2d(planes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    return x


class NiN(ExperimentBaseModel):
  def __init__(self, depth: int, width: int, base_width: int, dataset_type: DatasetType) -> None:
    super().__init__(dataset_type)

    self.base_width = base_width

    blocks = []
    blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width*width))
    for _ in range(depth-1):
      blocks.append(NiNBlock(self.base_width*width,self.base_width*width))
    self.blocks = nn.Sequential(*blocks)

    self.conv = nn.Conv2d(self.base_width*width, self.dataset_type.K, kernel_size=1, stride=1)
    self.bn = nn.BatchNorm2d(self.dataset_type.K)
    self.relu = nn.ReLU(inplace=True)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x):
    x = self.blocks(x)

    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.avgpool(x)
    return x.squeeze()
