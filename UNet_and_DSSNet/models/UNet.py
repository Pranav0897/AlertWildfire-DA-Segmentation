import torch
import torch.nn as nn
import torch.functional as tf
from models.encoders import ResNetEncoder
from models.decoders import DenseDecoder


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        self.args = args

        self.encoder = ResNetEncoder(args, 3, use_bn=args.use_bn)
        self.decoder = DenseDecoder(args, conv_chs=[128, 128, 64, 32, 32, 1], use_bn=args.use_bn)

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out[0])
        return dec_out
