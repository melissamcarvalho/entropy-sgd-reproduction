import torch.nn as nn


class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class allcnn(nn.Module):
    def __init__(self, opt, c1=96, c2=192):
        super(allcnn, self).__init__()
        self.name = 'allcnn'
        self.p = opt['dropout']

        def convbn(ci, co, ksz, s=1, pz=0):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.BatchNorm2d(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            nn.Dropout(self.p),            # (N, 03, 32, 32)
            convbn(3, c1, 3, 1, 1),     # (N, 96, 32, 32)
            convbn(c1, c1, 3, 1, 1),    # (N, 96, 32, 32)
            convbn(c1, c1, 3, 2, 1),    # (N, 96, 16, 16)
            nn.Dropout(0.5),       # (N, 96, 16, 16)
            convbn(c1, c2, 3, 1, 1),    # (N, 192, 16, 16)
            convbn(c2, c2, 3, 1, 1),    # (N, 192, 16, 16)
            convbn(c2, c2, 3, 2, 1),    # (N, 192, 08, 08)
            nn.Dropout(0.5),       # (N, 192, 08, 08)
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
