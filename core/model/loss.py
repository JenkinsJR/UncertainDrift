import torch
from torch import nn


class MSE(nn.Module):
    def __init__(self, exclude_mask):
        super().__init__()
        self._mask = ~exclude_mask

    def forward(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        return torch.square(pred - true).mean()
    
    
class MAE(nn.Module):
    def __init__(self, exclude_mask):
        super().__init__()
        self._mask = ~exclude_mask

    def forward(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        return torch.abs(pred - true).mean()
    

class ResidualLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self._loss_fn = loss_fn
    
    def forward(self, pred, true, inp):
        inp = inp[:, -1]

        return self._loss_fn(pred, (true-inp))
