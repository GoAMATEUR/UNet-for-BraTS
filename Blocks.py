"""
    By:     Huang Siyuan
    Date:   2022/1/27
"""
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

dropout_rate = 0.3

class ConvBlock(nn.Module):
    """ 
    Convolution Block

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU(),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate),
            nn.LeakyReLU()
        )
        
    def forward(self, input):
        return self.layers(input)
    
class DownSampling(nn.Module):
    """
    Down Sampling
    Using 3*3 Conv instead of 2*2 max pooling. 
    
    Args:
        channels (int): number of input/output channels
    """
    def __init__(self, channel):
        super(DownSampling, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode= "reflect", bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        
    def forward(self, input):
        return self.layers(input)
    
class UpSamling(nn.Module):
    """
        Interpolation
    """
    def __init__(self, channels) -> None:
        super(UpSamling, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1, 1)
        ) # Feature intrgration
        
    def forward(self, input, feature_map):
        interpolated = F.interpolate(input, scale_factor=2, mode='nearest')
        out = self.layers(interpolated)
        return torch.cat((out, feature_map), dim = 1)
    