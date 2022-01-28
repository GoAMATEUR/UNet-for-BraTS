"""
    By:     hsy
    Date:   2022/1/28
    TODO: more self-implemented loss func
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        
    def forward(self, input, target):
        pass