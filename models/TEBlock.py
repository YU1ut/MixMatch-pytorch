import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TEBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(TEBlock, self).__init__()
        padding = int((kernel_size-1)/2)#same size output
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.LeakyReLU(negative_slope=0.1,inplace=True))

    def forward(self, x):
        return self.layers(x)
class TEBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(TEBlock1, self).__init__()
        padding = 0
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.LeakyReLU(negative_slope=0.1,inplace=True))

    def forward(self, x):
        return self.layers(x)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)


