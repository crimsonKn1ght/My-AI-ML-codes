# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the Block
class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        if downsample != False:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        
        else:
            self.downsample = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            x = self.downsample(x)
        
        out += x

        return out


# Define resnet34
class resnet34(nn.Module):
    def __init__(self, out_class):
        super().__init__()

        self.out_class = out_class
        
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self.out_class, bias=True)
        
        self.layer_count = [3, 4, 6, 3]
        self.channels = [64, 64, 128, 256, 512]

        self.layer_1 = self._make_layers(0)
        self.layer_2 = self._make_layers(1, True)
        self.layer_3 = self._make_layers(2, True)
        self.layer_4 = self._make_layers(3, True)
        
    def _make_layers(self, c, first=False):
        layers = OrderedDict()
        
        for i in range(self.layer_count[c]):
            if first==True:
                x = c
                downsample = True
                stride=2
            else:
                x = c+1
                downsample = False
                stride=1
            layers[f'sub_layer_{i}'] = (Block(self.channels[x], self.channels[c+1], stride=stride, downsample=downsample))
            first = False

        return nn.Sequential(layers)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)       

        return x
