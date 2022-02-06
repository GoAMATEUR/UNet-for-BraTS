"""
    By:     hsy
    Update: 2022/2/4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MetricsTracker(object):
    def __init__(self, tag=None):
        self.iter_IoUs = []
        self.iter_Dices = []
        self.iter_Accs = []
        self.epoch = 0
        self.tag = tag
    
    def update(self, input: torch.Tensor, target: torch.Tensor, dice):
        self.iter_IoUs.append(self.IoU(input, target))
        self.iter_Dices.append(self.Acc(input, target))
    
    def get_metrics(self):
        iou = np.mean(self.iter_IoUs)
        dice = np.mean(self.iter_Dices)
        acc = np.mean(self.iter_Accs)
        return acc, iou, dice
        
    def save_logs(self):
        return 
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_Dices = []
        self.iter_Accs = []
        self.iter_IoUs = []
      
    def IoU(self, input: torch.Tensor, target: torch.Tensor)->float:
        return

    def Acc(self, input: torch.Tensor, target: torch.Tensor)->float:
        return

class BinaryDiceLoss(nn.Module):
    """Simple implementation of Binary Dice Loss
    
    Args:
        nn ([type]): [description]
    """
    def __init__(self, smooth=1e-5, beta=1, size_mean=True) -> None:
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.size_mean = size_mean
        self.beta = beta
        
    def forward(self, input, target):
        """
        

        Args:
            input ([type]): [1, 1, 240, 240]
            target ([type]): [1, 1, 240, 240]
        """
        
        N = target.size()[0]
        smooth = self.smooth

        input_flat = input.view(N, -1)
        targets_flat = target.view(N, -1)
        intersection = input_flat * targets_flat 
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

        
        
        
 



class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
if __name__ == "__main__":
    loss_dice = BinaryDiceLoss()
    target = torch.tensor([[[[1, 0],
                            [0, 1]]],
                           [[[1, 0],
                            [0, 1]]]])
    input = torch.tensor([[[[1, 1],
                            [0, 1]]],
                            [[[1, 1],
                            [0, 1]]]])
