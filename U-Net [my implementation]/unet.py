# Import the packages

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set device to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == 'cuda':
    print("Cuda available: " + torch.cuda.get_device_name())
else:
    print("Cuda unavailable.")

# Defining the basic conv block of U-Net
class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=0, act=True):
        super().__init__()

        self.conv2d = nn.Sequential(
                nn.Conv2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_c)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        
        if act==True:
            x = self.relu(x)

        return x


# Define U-Net

class unet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block1 = nn.Sequential(
            Conv2D(in_c=1, out_c=64, kernel_size=3),
            Conv2D(in_c=64, out_c=64, kernel_size=3),
            Conv2D(in_c=64, out_c=64, kernel_size=3),
        )

        self.block2 = nn.Sequential(
            Conv2D(in_c=64, out_c=128, kernel_size=3),
            Conv2D(in_c=128, out_c=128, kernel_size=3),
            Conv2D(in_c=128, out_c=128, kernel_size=3),
        )

        self.block3 = nn.Sequential(
            Conv2D(in_c=128, out_c=256, kernel_size=3),
            Conv2D(in_c=256, out_c=256, kernel_size=3),
            Conv2D(in_c=256, out_c=256, kernel_size=3),
        )

        self.block4 = nn.Sequential(
            Conv2D(in_c=256, out_c=512, kernel_size=3),
            Conv2D(in_c=512, out_c=512, kernel_size=3),
            Conv2D(in_c=512, out_c=512, kernel_size=3),
        )

        self.block5 = nn.Sequential(
            Conv2D(in_c=512, out_c=1024, kernel_size=3),
            Conv2D(in_c=1024, out_c=1024, kernel_size=3),
            Conv2D(in_c=1024, out_c=1024, kernel_size=3),
        )


    def forward(self, x):
        x1 = self.block1(x)
        x1p = nn.MaxPool2d(x1)
        
        x2 = self.block1(x1p)
        x2p = nn.MaxPool2d(x2)
        
        x3 = self.block1(x2p)
        x3p = nn.MaxPool2d(x3)
        
        x4 = self.block1(x3p)
        x4p = nn.MaxPool2d(x4)
        
        x5 = self.block1(x4)
        x5p = nn.MaxPool2d(x5)

        return x5p, [x1, x2, x3, x4, x5]
        
