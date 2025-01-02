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

# Defining the downsampling and upsampling conv block of U-Net
class downConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=0, stride=1, act=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride,),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride,),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=0, stride=1, act=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=kernel_size, padding=padding, stride=stride,),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // 2, out_c, kernel_size=kernel_size, padding=padding, stride=stride,),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Define U-Net

class unet_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            downConv(in_c=1, out_c=64, kernel_size=3, padding=0, stride=1),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(2),
            downConv(in_c=64, out_c=128, kernel_size=3, padding=0, stride=1),
        )

        self.block3 = nn.Sequential(
            nn.MaxPool2d(2),
            downConv(in_c=128, out_c=256, kernel_size=3, padding=0, stride=1),
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(2),
            downConv(in_c=256, out_c=512, kernel_size=3, padding=0, stride=1),
        )

        self.block5 = nn.Sequential(
            nn.MaxPool2d(2),
            downConv(in_c=512, out_c=1024, kernel_size=3, padding=0, stride=1),
        )


    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        return [x1, x2, x3, x4, x5]


class unet_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2,)
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2,)
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,)
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2,)
        
        self.block1 = upConv(in_c=1024, out_c=256, kernel_size=3, padding=0, stride=1)
        self.block2 = upConv(in_c=512, out_c=128, kernel_size=3, padding=0, stride=1)
        self.block3 = upConv(in_c=256, out_c=64, kernel_size=3, padding=0, stride=1)
        self.block4 = upConv(in_c=128, out_c=64, kernel_size=3, padding=0, stride=1)
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1,),
            nn.Sigmoid(2)
        )

    def forward(self, x_set):
        x1_1, x1_2, x1_3, x1_4, x1_5 = x_set
        
        x2_1 = self.upconv1(x1_5)
        x2_1 = torch.cat([x2_1, x1_4], axis=1)
        x2_1 = nn.block1(x2_1)
        
        x2_2 = self.upconv1(x1_4)
        x2_2 = torch.cat([x2_2, x1_3], axis=1)
        x2_2 = nn.block1(x2_2)

        x2_3 = self.upconv1(x1_3)
        x2_3 = torch.cat([x2_3, x1_2], axis=1)
        x2_3 = nn.block1(x2_3)

        x2_4 = self.upconv1(x1_2)
        x2_4 = torch.cat([x2_4, x1_1], axis=1)
        x2_4 = nn.block1(x2_4)

        x = self.seg_head(x2_4)
        
        return x


class u_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.part1 = unet_encoder()
        self.part2 = unet_decoder()

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x1)

        return x
