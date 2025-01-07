import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
import torchvision.datasets as datasets


class conv3x3(nn.Module):
    def __init__(self, in_c, out_c, stride=1, padding=1, groups=1, dilation=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c, out_c,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()

        self.expansion = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("Groups not equal to 1 or base width not 64")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in Basic Block")

        self.conv1 = conv3x3(in_c, out_c, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = bn2(out)

        if self.downsample != None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, base_width=64, dilation=1):
        super().__init__()

        self.expansion = 4
        
        if norm_layer != None:
            norm_layer = nn.BatchNorm2d

        width = int(out_c * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_c, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(width, width, )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_c * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)
