from torch import nn
import torch
import torch.nn.functional as F


class ConfBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_conv1 = nn.Conv2d(64, 32, kernel_size=3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(32, 16, kernel_size=3, dilation=2, padding=2)
        self.d_conv3 = nn.Conv2d(16, 1, kernel_size=3, dilation=5, padding=5)
        
        self.p_conv1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.p_conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x1 = self.d_conv1(x)
        x2 = self.d_conv2(x1)
        x3 = self.d_conv3(x2)
        
        t1 = self.p_conv1(x1)
        t2 = self.p_conv2(x2)
        
        x = torch.cat([x3, t1], dim=1)
        x = torch.cat([x, t2], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.bn(x)
        x = self.sigmoid(x)

        return x
