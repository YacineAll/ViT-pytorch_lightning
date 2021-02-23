import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Allows us to release to slightly lower the loss target values from 1 to, say, 0.9.
    And naturally, we increase the target value of 0 for the others slightly as such. 
    This idea is called label smoothing.
    """
    def __init__(self, num_class:int, smoothing=0.0, dim=-1):
        """
        Args:
            classes (int): number of classes.
            smoothing (float, optional): La valeur smooth de combien veut relacher. Defaults to 0.0.
            dim (int, optional): dim . Defaults to -1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_class
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))