import torch
import torch.nn as nn

class squeeze_excitation(nn.Module):
    def __init__(self, in_c, reduction_ratio):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c//4, in_c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.layers(y).view(batch_size, channel_size, 1, 1)
        
        return x*y.expand_as(x)

class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid(x)

class MBConv1(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_c),
            nn.BatchNorm2d(in_c),
            swish(),
            squeeze_excitation(in_c, reduction_ratio=4),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.layers(x)

class MBConv6(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, 6*in_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(6*in_c),
            swish(),
            nn.Conv2d(6*in_c, 6*in_c, kernel_size=kernel_size, stride=stride, groups=6*in_c, padding=padding),
            nn.BatchNorm2d(6*in_c),
            swish(),
            squeeze_excitation(6*in_c, reduction_ratio=4),
            nn.Conv2d(6*in_c, out_c, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.layers(x)

class efficientnetB0(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_szie=3, padding=1),
            nn.BatchNorm2d(32),
            swish(),
            MBConv1(32, 16, kernel_size=3, stride=1),
            MBConv6(16, 24, kernel_size=3, stride=1),
            MBConv6(24, 24, kernel_size=3, stride=1),
            MBConv6(24, 40, kernel_size=3, stride=1),
            MBConv6(40, 40, kernel_size=3, stride=1),
            MBConv6(40, 80, kernel_size=3, stride=1),
            MBConv6(80, 80, kernel_size=3, stride=1),
            MBConv6(80, 80, kernel_size=3, stride=1),
            MBConv6(80, 112, kernel_size=3, stride=1),
            MBConv6(112, 112, kernel_size=3, stride=1),
            MBConv6(112, 112, kernel_size=3, stride=1),
            MBConv6(112, 192, kernel_size=3, stride=1),
            MBConv6(192, 192, kernel_size=3, stride=1),
            MBConv6(192, 192, kernel_size=3, stride=1),
            MBConv6(192, 192, kernel_size=3, stride=1),
            MBConv6(192, 320, kernel_size=3, stride=1),
            nn.Conv2d(320, 1280, kernel_szie=3, padding=1),
            nn.BatchNorm2d(1280),
            swish(),
            nn.AvgPool2d()
            
        )
