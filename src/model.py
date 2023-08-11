# -*- coding: utf-8 -*- #
""""""
import typing as tp

import numpy as np
import pytorch_pfn_extras as ppe
import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from torch import nn

Array = tp.Union[np.ndarray, torch.Tensor]
Batch = tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]]
ModelOut = tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor], torch.Tensor]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def global_dice_coefficient(
        X: Array, Y: Array, eps=1e-7,
) -> tp.Union[torch.Tensor, float]:
    """"""
    intersection = (X * Y).sum()
    union = X.sum() + Y.sum()

    if isinstance(union, torch.Tensor):
        union = torch.clamp(union, min=eps)
    else:
        union = np.clip(union, a_min=eps, a_max=1e+10)
    
    return 2 * intersection / union


class GlobalDiceLossWithLogits(nn.Module):
    """"""
    def __init__(self) -> None:
        """"""
        super(GlobalDiceLossWithLogits, self).__init__()

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.sigmoid()
        else:
            y = sigmoid(y)
        
        return 1 - global_dice_coefficient(y, t)
    

class GlobalDiceLossWithLogitsForPPE(nn.Module):

    def __init__(self, report_name: str):
        """"""
        super(GlobalDiceLossWithLogitsForPPE, self).__init__()
        self.report_name = report_name
        self.intersection = 0.
        self.union = 0.

    def forward(self, batch: Batch, model_output: ModelOut, is_last_batch: bool):
        X = model_output.sigmoid()
        Y = batch["target"]

        self.intersection += (X * Y).sum().item()
        self.union += (X.sum() + Y.sum()).item()

        if is_last_batch:
            final_metric = 1 - 2 * self.intersection / max(self.union, 1e-7)
            ppe.reporting.report({self.report_name: final_metric})
            self.intersection = 0.
            self.union = 0.


class GlobalDiceCoefficientWithLogitsForPPE(nn.Module):

    def __init__(self, report_name: str, threshold: float=0.5):
        """"""
        super(GlobalDiceCoefficientWithLogitsForPPE, self).__init__()
        self.report_name = report_name
        self.threshold = threshold
        self.intersection = 0.
        self.union = 0.

    def forward(self, batch: Batch, model_output: ModelOut, is_last_batch: bool):
        X = (model_output.sigmoid() >= self.threshold).float()
        Y = batch["target"]

        self.intersection += (X * Y).sum().item()
        self.union += (X.sum() + Y.sum()).item()

        if is_last_batch:
            final_metric = 2 * self.intersection / max(self.union, 1e-7)
            ppe.reporting.report({self.report_name: final_metric})
            self.intersection = 0.
            self.union = 0.


class PercentileGlobalDiceCoefficientWithLogitsForPPE(nn.Module):
    
    def __init__(self, report_name: str, percent: float):
        """"""
        super().__init__()
        self.report_name = report_name
        self.percent = percent
        self.prob_list = []
        self.mask_list = []

    def forward(self, batch: Batch, model_output: ModelOut, is_last_batch: bool):
        y = model_output.sigmoid()
        t = batch["target"]

        self.prob_list.append(y.numpy())
        self.mask_list.append(t.numpy())

        if is_last_batch:
            prob = np.concatenate(self.prob_list, axis=0)
            threshold = np.quantile(prob, self.percent)
            pred_mask = (prob > threshold).astype(int)
            mask = np.concatenate(self.mask_list, axis=0)
            final_metric = global_dice_coefficient(pred_mask, mask)
            ppe.reporting.report({self.report_name: final_metric})
            self.prob_list = []
            self.mask_list = []


class BCEandDiceLossWithLogits(nn.Module):

    def __init__(self, alpha: float=0.5, smooth_factor: float=None):
        """"""
        super(BCEandDiceLossWithLogits, self).__init__()
        self.alpha = alpha
        self.bce = SoftBCEWithLogitsLoss(smooth_factor=smooth_factor)
        self.dice = DiceLoss(mode="binary", smooth=1.0)

    def forward(self, y, t):
        bce_loss = self.bce(y, t)
        dice_loss = self.dice(y, t)

        return (1 - self.alpha) * bce_loss + self.alpha * dice_loss
