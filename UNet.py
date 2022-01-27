"""
    By:     Huang Siyuan
    Date:   2022/1/27
"""
from Blocks import *
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # downward
        self.dConv1 = ConvBlock(3,64)
        self.dSamp1 = DownSampling(64)
        self.dConv2 = ConvBlock(64,128)
        self.dSamp2 = DownSampling(128)
        self.dConv3 = ConvBlock(128,256)
        self.dSamp3 = DownSampling(256)
        self.dConv4 = ConvBlock(256,512)
        self.dSamp4 = DownSampling(512)
        self.dConv5 = ConvBlock(512,1024)
        
        # upward
        self.uSamp1 = UpSamling(1024)
        self.uConv1 = ConvBlock(1024, 512)
        self.uSamp2 = UpSamling(512)
        self.uConv2 = ConvBlock(512, 256)
        self.uSamp3 = UpSamling(256)
        self.uConv3 = ConvBlock(256, 128)
        self.uSamp4 = UpSamling(128)
        self.uConv4 = ConvBlock(128, 64)
        
        # out
        self.outConv = nn.Conv2d(64, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        # downward
        feature1 = self.dConv1(input)
        feature2 = self.dConv2(self.dSamp1(feature1))
        feature3 = self.dConv3(self.dSamp2(feature2))
        feature4 = self.dConv4(self.dSamp3(feature3))
        feature5 = self.dConv5(self.dSamp4(feature4))
        
        # upward
        up1 = self.uConv1(self.uSamp1(feature5, feature4))
        up2 = self.uConv2(self.uSamp2(up1, feature3))
        up3 = self.uConv3(self.uSamp3(up2, feature2))
        up4 = self.uConv4(self.uSamp4(up3, feature1))
        
        return self.sigmoid(self.outConv(up4))
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 240, 240)
    net = UNet()
    print(net(x).shape)
