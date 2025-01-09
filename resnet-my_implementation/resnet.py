# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Check for CUDA (GPU)
device = torch.device('cuda')
if device == 'cuda':
    print(torch.cuda.get_device_name())
else:
    print('CUDA not available')

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

        self.layer = self._make_layers()
        
    def _make_layers(self):
        layers = []
        downsample = False
        
        for i in range(len(self.layer_count)):
            for j in range(self.layer_count[i]):
                layers.append(Block(self.in_c[i], self.out_c[i]), downsample=downsample)
                downsample = False

            downsample = True

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer(x)

        return x
