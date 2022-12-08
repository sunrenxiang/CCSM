import numpy as np
import torch
from torch import nn
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ori = x
        out1 = self.conv2(self.relu(self.conv1(self.avg(x))))
        out2 = self.conv2(self.relu(self.conv1(self.max(x))))
        out = self.sigmoid(out1 + out2)
        out = ori * out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size, padding=3, dilation=3, bias=False)
        self.conv4 = nn.Conv2d(2, 1, kernel_size, padding=4, dilation=4, bias=False)
        self.conv = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()


class AttentionBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()

        self.channel = ChannelAttention(channel=channel,reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        ori = x
        out = x * self.channel(x)
        out = out * self.spatial(out)
        return out + ori

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    att = AttentionBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=att(input)
    print(output.shape)
