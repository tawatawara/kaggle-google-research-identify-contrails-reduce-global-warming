# -*- coding-utf8 -*- #
import timm
from torch import device, nn
from torch.cuda.amp import GradScaler

from .loss import FocalBCEWithLogitsLoss, LabelSmoothingFocalCrossEntropy
from .metric import RMSE, ROCAUC, Accuracy
from .model_basic import MLP, Conv1dBNActiv, Conv2dBNActiv, get_activation
from .model_image import ABResNetD, BasicImageModel, TimmBase

CONFIG_TYPES = {
    # # model
    "timm_create_model": timm.create_model,
    "BasicImageModel": BasicImageModel,
    "ABResNetD": ABResNetD,
    "MLP": MLP,
    "Conv1dBNActiv": Conv1dBNActiv,
    "Conv2dBNActiv": Conv2dBNActiv,

    # # loss
    "BCEWithLogitsLoss"     : nn.BCEWithLogitsLoss,
    "FocalBCEWithLogitsLoss": FocalBCEWithLogitsLoss,
    "CrossEntropyLoss"      : nn.CrossEntropyLoss,
    "LabelSmoothingFocalCrossEntropy": LabelSmoothingFocalCrossEntropy,
    "SoftMarginLoss"        : nn.SoftMarginLoss,
    "MAELoss"       : nn.L1Loss,
    "L1Loss"        : nn.L1Loss,
    "MSELoss"       : nn.MSELoss,
    "SmoothL1Loss"  : nn.SmoothL1Loss, 
    "HuberLoss"     : nn.SmoothL1Loss, 
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "KLDivLoss"     : nn.KLDivLoss,

    # # metric
    "Accuracy": Accuracy,
    "ROCAUC": ROCAUC,
    "RMSE": RMSE,

    # # other
    "torch_device": device,
    "GradScaler"  : GradScaler,
    "ModelEmaV2"  : timm.utils.ModelEmaV2,
}
