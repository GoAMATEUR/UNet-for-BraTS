import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import VGG

class UpSampBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.2):
        super(UpSampBlock, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(),
        )
        self.reduceLayers = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, 1, 1)
        )
           
    def forward(self, input, feature): # feature a*a*b, input: a/2 * a/2 * 2b
        interpolated = F.interpolate(input, scale_factor=2, mode='nearest') # a*a*2b
        output = self.reduceLayers(interpolated)#a*a*b
        output = torch.cat((output, feature), dim = 1) # a*a*2b
        return self.convLayers(output)#a*a*b
    
class UNet(nn.Module):
    def __init__(self, backbone_type: str, d_rate=0.3):
        super(UNet, self).__init__()
        