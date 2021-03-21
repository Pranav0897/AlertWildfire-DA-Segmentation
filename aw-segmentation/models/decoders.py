import torch
import torch.nn as nn
import torch.functional as tf


def conv(in_chs, out_chs, kernel_size=3, stride=1, dilation=1, bias=True, use_relu=True, use_bn=False):
  layers = []
  layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=bias))
  if use_bn:
    layers.append(nn.BatchNorm2d(out_chs))
  if use_relu:
    layers.append(nn.LeakyReLU(0.1, inplace=True))
  
  return nn.Sequential(*layers)


class DenseDecoder(nn.Module):
    def __init__(self, args, conv_chs=[128, 128, 64, 32, 32, 1], skips=None, use_bn=True):
        super(DenseDecoder, self).__init__()

        self.args = args

        layers = []
        for (lc, ln) in zip(conv_chs[:-1], conv_chs[1:]):
            layers.append(conv(lc, ln, use_bn=use_bn))
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convs(x)
        out = torch.sigmoid(out)

        return out