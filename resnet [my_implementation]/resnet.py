# import libraries
import torch
import torch.nn as nn
from collections import OrderedDict


# Define the Block
class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=False):
        super().__init__()

        if downsample != False:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            stride = 2
        
        else:
            self.downsample = None
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            i = self.downsample(x)
            out += i

        return out


# Define resnet34
class resnet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )

        self.layer_count = [3, 4, 6, 3]
        self.in_c = [64, 64, 128, 256]
        self.out_c = [64, 128, 256, 512]

        self.layer_1 = self._make_layers(0)
        self.layer_2 = self._make_layers(1)
        self.layer_3 = self._make_layers(2)
        self.layer_4 = self._make_layers(3)
        
    def _make_layers(self, c):
        layers = OrderedDict()
        downsample = False
        
        for i in range(self.layer_count[c]):
            layers[f'sub_layer_{i}'] = (Block(self.in_c[c], self.out_c[c], downsample=downsample))
            downsample = False

        downsample = True

        return nn.Sequential(layers)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
