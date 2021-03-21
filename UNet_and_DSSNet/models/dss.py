import torch
import torch.nn as nn
from .common import nConvReLU, Down, UpConv, OutConv, concat


class subnet1(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(subnet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

#         self.inc = nConvReLU(n_channels, 64, 2)
#         self.down1 = Down(in_channels=64, out_channels=128, n_convrelu_layers=2)
#         self.down2 = Down(in_channels=128, out_channels=256, n_convrelu_layers=3)
#         self.down3 = Down(in_channels=256, out_channels=512, n_convrelu_layers=3)
#         self.down4 = Down(in_channels=512, out_channels=512, n_convrelu_layers=3)

        self.conv1 = nConvReLU(n_channels, 32, 2)
        self.down1 = Down(in_channels=32, out_channels=64, n_convrelu_layers=2)
        self.down2 = Down(in_channels=64, out_channels=128, n_convrelu_layers=3)
        self.down3 = Down(in_channels=128, out_channels=256, n_convrelu_layers=3)
        self.down4 = Down(in_channels=256, out_channels=256, n_convrelu_layers=2)

        factor = 2 if bilinear else 1
        
#         self.up1 = UpConv(in_channels=512, out_channels=512 //factor, bilinear=bilinear)
#         self.concat1 = concat()
#         self.up2 = UpConv(in_channels=512 + (512//factor), out_channels=512 //factor, bilinear=bilinear)
#         self.concat2 = concat()
#         self.conv3 = nn.Sequential(nConvReLU(256 + (512//factor), 512, 1),
#                                    nConvReLU(512, 256, 2))
#         self.up3 = UpConv(in_channels=256, out_channels=128, bilinear=bilinear, n_convrelu_layers=2)
#         self.up4 = UpConv(in_channels=128, out_channels=64, bilinear=bilinear, n_convrelu_layers=2)
#         self.outc = OutConv(64,n_classes)

        self.up1 = UpConv(in_channels=256, out_channels=256 //factor, bilinear=bilinear)
        self.concat1 = concat()
        self.up2 = UpConv(in_channels=256 + (256//factor), out_channels=256 //factor, bilinear=bilinear)
        self.concat2 = concat()
        self.conv3 = nn.Sequential(nConvReLU(128 + (256//factor), 256, 1),
                                   nConvReLU(256, 128, 2))
        self.up3 = UpConv(in_channels=128, out_channels=64, bilinear=bilinear, n_convrelu_layers=2)
        self.up4 = UpConv(in_channels=64, out_channels=32, bilinear=bilinear, n_convrelu_layers=2)
        self.outc = OutConv(32,n_classes)

    def forward(self, x):

        x = self.conv1(x)
        xd1 = self.down1(x)
        xd2 = self.down2(xd1)
        xd3 = self.down3(xd2)
        xd4 = self.down4(xd3)

        x1 = self.up1(xd4)
        xc1 = self.concat1(x1, xd3)
        x2 = self.up2(xc1)
        xc2 = self.concat2(x2, xd2)
        x = self.conv3(xc2)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)

        # x = self.conv1(x) #64
        # x = self.down1(x) #128
        # x3 = self.down2(x)#256
        # x4 = self.down3(x3) #512
        # x5 = self.down4(x4) #512
        # x6 = self.up1(x5) #512 //factor
        # x = self.concat1(x6, x4) # 512 + 512//factor
        # x = self.up2(x) # 512//factor
        # x = self.concat2(x, x3) # 256 + 512//factor
        # x = self.conv3(x)
        # x = self.up3(x)
        # x = self.up4(x)
        # x = self.outc(x)

        return x
    
    
class subnet2(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(subnet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nConvReLU(n_channels, 64, 2)
        self.down1 = Down(in_channels=64, out_channels=128, n_convrelu_layers=2)
        self.down2 = Down(in_channels=128, out_channels=256, n_convrelu_layers=3)
        factor = 2 if bilinear else 1
        self.up1 = UpConv(in_channels=256, out_channels=256 // factor, bilinear=bilinear)
        self.up2 = UpConv(in_channels=128 + (256 //factor), out_channels=256 // factor, bilinear=bilinear)
        self.conv3 = nn.Sequential(nConvReLU((256//factor + 64), 128, 1), nConvReLU(128, 64, 1))
        self.concat1 = concat()
        self.concat2 = concat()
        self.outc = OutConv(64,n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) # c = 128
        x3 = self.down2(x2)
        x4 = self.up1(x3)
        x = self.concat1(x4, x2)
        # x = torch.cat([x4, x2], dim=1)
        x = self.up2(x)
        x = self.concat2(x, x1)
        # x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.outc(x)
        return x
    

class DSS_Net(torch.nn.Module):
    def __init__(self, args, n_channels, n_classes, bilinear=True):
        super(DSS_Net, self).__init__()

        self.args = args

        self.net1 = subnet1(n_channels, n_classes, bilinear)
#         self.net2 = subnet2(n_channels, n_classes, bilinear)
        self.outc = OutConv(n_classes, n_classes)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

    def forward(self, x):
        x1 = self.net1(x)
#         x2 = self.net2(x)
        
#         x = torch.add(x1, x2)
        x = self.outc(x1)
    
        return x
