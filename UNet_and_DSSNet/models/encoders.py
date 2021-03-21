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


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride, downsample, pad, dilation, use_bn=True):
      super(BasicBlock, self).__init__()

      self.conv1 = conv(in_ch, out_ch, 3, stride, dilation, use_bn=use_bn)
      self.conv2 = conv(out_ch, out_ch, 3, 1, dilation, use_relu=False, use_bn=use_bn)

      self.downsample = downsample
      self.stride = stride

    def forward(self, x):
      out = self.conv1(x)
      out = self.conv2(out)

      if self.downsample is not None:
          x = self.downsample(x)

      out += x

      return out


class ResNetEncoder(nn.Module):
    def __init__(self, args, in_chs, conv_chs=None, use_bn=False):
        super(ResNetEncoder, self).__init__()

        self.args = args

        if conv_chs is None:
          self.conv_chs = [32, 32, 64, 128, 128]
        else:
          self.conv_chs = conv_chs

        self.in_chs = self.conv_chs[0]

        self.conv1 = nn.Sequential(
          conv(in_chs, self.in_chs, 3, 2, 1, 1, use_bn=use_bn),
          conv(self.in_chs, self.in_chs, 3, 1, 1, 1, use_bn=use_bn),
          conv(self.in_chs, self.in_chs, 3, 1, 1, 1, use_bn=use_bn))

        self.res_layers = nn.ModuleList()
        for conv_ch in self.conv_chs[1:]:
            self.res_layers.append(self._make_layer(BasicBlock, conv_ch, 3, 2, 1, 1, use_bn=use_bn))

        # if args.use_ppm:
        #   self.ppm = PyramidPoolingModule(args=args, encoder_planes=[32, 32, 64, 128, 128], ppm_last_conv_planes=128, ppm_inter_conv_planes=128) 
        # else:
        #   self.ppm = None

    def _make_layer(self, block, chs, blocks, stride, pad, dilation, use_bn=True):
      downsample = None
      if stride != 1 or self.in_chs != chs * block.expansion:
        if use_bn:
          downsample = nn.Sequential(
            nn.Conv2d(self.in_chs, chs * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(chs * block.expansion))
        else:
          downsample = nn.Conv2d(self.in_chs, chs * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.in_chs, chs, stride, downsample, pad, dilation, use_bn=use_bn))
        self.in_chs = chs * block.expansion
        for _ in range(1, blocks):
          layers.append(block(self.in_chs, chs, 1, None, pad, dilation, use_bn=use_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
      outs = [x]

      outs.append(self.conv1(x))

      for res_layer in self.res_layers:
        outs.append(res_layer(outs[-1]))
    #   if self.args.use_ppm:
    #     outs.append(self.ppm(outs[-1]))

      return outs[::-1]