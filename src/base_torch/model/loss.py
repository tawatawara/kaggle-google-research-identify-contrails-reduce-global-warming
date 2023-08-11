# -*- coding: utf-8 -*- #
"""metric functions."""
import typing as tp

import numpy as np
import torch
from torch import nn


class FocalBCEWithLogitsLoss(nn.Module):
    r"""
    Focal Binary Cross Entropy Loss
    
    loss_{i}(p, t) = - \alpha' + (1 - p'_i) ** \gamma * ln(p'_i),

    where
        \alpha' = { \alpha (t_i = 1)
                  { 1 - \alpha (t_i = 0)
         p'_i   = { 1 - p_i (t_i = 1)
                = ( p_i (t_i = 0)
    """

    def __init__(self, alpha: float=0.25,  gamma: float=2.0, reduction: str="mean"):
        """Initialize."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.a = alpha
        self.g = gamma
        self.reduce = {
            "sum": torch.sum,
            "mean": torch.mean,
        }[reduction]

    def forward(self, y, t):
        """Forward."""
        bce_loss = self.bce(y, t)  # shape: (bs, 1)
        proba = torch.sigmoid(y)   # shape: (bs, 1)
        alpha = (1 - t) + (2 * t - 1) * self.a
        p = t + (1 - 2 * t) * proba
        loss_by_example = alpha * p ** self.g * bce_loss
        loss = self.reduce(loss_by_example)

        return loss


class LabelSmoothingFocalCrossEntropy(nn.Module):
    """"""

    def __init__(
        self,
        smooth_alpha: float=0,
        gamma       : float=0,
    ):
        """"""
        super().__init__()
        self.smooth_alpha = smooth_alpha
        self.gamma = gamma
    
    def forward(self, y, t):
        """"""
        log_p = nn.functional.log_softmax(y, dim=1)
        p     = torch.exp(log_p)
        t = (1 - self.smooth_alpha) * t + self.smooth_alpha / y.shape[-1]
        loss = - t * log_p
        loss = (1. - p) ** self.gamma * loss  # shape: (B, C)
        loss = torch.sum(loss, axis=1)        # shape: (B,)
        loss = torch.mean(loss)               # shape: (1,)
        return loss
