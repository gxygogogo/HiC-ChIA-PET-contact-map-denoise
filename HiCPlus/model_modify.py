import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import gzip
import sys
import torch.optim as optim

conv2d1_filters_numbers = 16
conv2d1_filters_size = 5
conv2d2_filters_numbers = 32
conv2d2_filters_size = 3
conv2d3_filters_numbers = 64
conv2d3_filters_size = 3
conv2d4_filters_numbers = 32
conv2d4_filters_size = 3
conv2d5_filters_numbers = 16
conv2d5_filters_size = 3
conv2d6_filters_numbers = 1
conv2d6_filters_size = 3

class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size, padding=2)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size, padding=1)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, conv2d3_filters_numbers, conv2d3_filters_size, padding=1)
        self.conv4 = nn.Conv2d(conv2d3_filters_numbers, conv2d4_filters_numbers, conv2d4_filters_size, padding=1)
        self.conv5 = nn.Conv2d(conv2d4_filters_numbers, conv2d5_filters_numbers, conv2d5_filters_size, padding=1)
        self.conv6 = nn.Conv2d(conv2d5_filters_numbers, conv2d6_filters_numbers, conv2d6_filters_size, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)  # 最后一层不使用激活函数
        return x