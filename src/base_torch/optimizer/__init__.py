# -*- coding: utf-8 -*- #
from torch import optim
from .lr_scheduler import CosineAnnealingWarmupRestarts
from .optimizer import SAM

CONFIG_TYPES = {
    # # Optimizer
    "SGD"  : optim.SGD  ,
    "Adam" : optim.Adam ,
    "AdamW": optim.AdamW,
    "SAM"  : SAM        ,
    "_class_SGD"  : lambda: optim.SGD  ,
    "_class_Adam" : lambda: optim.Adam ,
    "_class_AdamW": lambda: optim.AdamW,

    # # Scheduler
    "MultiStepLR"      : optim.lr_scheduler.MultiStepLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR"       : optim.lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts"  : optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "CosineAnnealingWarmupRestarts": CosineAnnealingWarmupRestarts, 
}