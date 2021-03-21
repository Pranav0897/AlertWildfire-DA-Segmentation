from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class nConvReLU(torch.nn.Module):
    """(convolution => [BN] => ReLU) * n"""

    def __init__(self, in_channels, out_channels, n_convrelu_layers = 1):
        super().__init__()
        layers = OrderedDict()
        layers['conv0'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         layers['bn0'] = nn.BatchNorm2d(out_channels)
        layers['relu0'] = nn.LeakyReLU(inplace=True)
        
        for i in range(1,n_convrelu_layers):
            layers['conv%d'%i] = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#             layers['bn%d'%i] = nn.BatchNorm2d(out_channels)
            layers['relu%d'%i] = nn.LeakyReLU(inplace=True)

        self.n_conv_relu = nn.Sequential(layers)

    def forward(self, x):
        return self.n_conv_relu(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_convrelu_layers = 1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                         nConvReLU(in_channels, out_channels, n_convrelu_layers))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(torch.nn.Module):
    """Upscaling then conv relu"""

    def __init__(self, in_channels, out_channels, bilinear=True, n_convrelu_layers = 1):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upconv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nConvReLU(in_channels, out_channels, n_convrelu_layers))
        else:
            self.upconv = nn.Sequential(nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2),
                                       nConvReLU(in_channels, out_channels, n_convrelu_layers))
    def forward(self, x):
        return self.upconv(x)        

class concat(torch.nn.Module):
    """Concatenator wrapper"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x
    
class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return torch.sigmoid(self.conv(x))
