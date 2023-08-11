# -*- coding: utf-8 -*- #

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import (ContrastiveImageLabelLazyDataset,
                      ContrastiveImagePathLabelLazyDataset,
                      ImageLabelLazyDataset, ImagePathLabelLazyDataset,
                      PathLabelLazyDataset)
from .transform import ImageOnlyTransform, RandomErase


def worker_init_fn(worker_id):                     
    """Init function for DataLoader"""        
    np.random.seed(np.random.get_state()[1][0] + worker_id)

CONFIG_TYPES = {
    # # Dataset
    "ImagePathLabelLazyDataset": ImagePathLabelLazyDataset,
    "ImageLabelLazyDataset": ImageLabelLazyDataset,
    "ContrastiveImagePathLabelLazyDataset": ContrastiveImagePathLabelLazyDataset,
    "ContrastiveImageLabelLazyDataset": ContrastiveImageLabelLazyDataset,

    # # DataLoader
    "DataLoader": DataLoader,
    "WeightedRandomSampler": WeightedRandomSampler,
    "worker_init_fn": lambda: worker_init_fn,

    # # Data Augmentation
    "Compose": A.Compose,
    "OneOf"  : A.OneOf,

    "PadIfNeeded"    : A.PadIfNeeded,
    "Resize"         : A.Resize,
    "SmallestMaxSize": A.SmallestMaxSize,
    
    "HorizontalFlip"   : A.HorizontalFlip,
    "VerticalFlip"     : A.VerticalFlip,
    "RandomRotate90"   : A.RandomRotate90,
    "ShiftScaleRotate" : A.ShiftScaleRotate,
    "OpticalDistortion": A.OpticalDistortion,
    "GridDistortion"   : A.GridDistortion,
    "ElasticTransform" : A.ElasticTransform,
    "PiecewiseAffine"  : A.PiecewiseAffine,

    "CenterCrop"       : A.CenterCrop, 
    "RandomCrop"       : A.RandomCrop,
    "RandomResizedCrop": A.RandomResizedCrop,

    "Cutout"           : A.Cutout,
    "CoarseDropout"    : A.CoarseDropout,
    "PixelDropout"     : A.PixelDropout,
    "RandomErase"      : RandomErase,
    "RandomGridShuffle": A.RandomGridShuffle,

    "GaussNoise"  : A.GaussNoise,
    "GaussianBlur": A.GaussianBlur,
    
    "CLAHE"         : A.CLAHE,
    "ColorJitter"   : A.ColorJitter,
    "HueSaturationValue"      : A.HueSaturationValue,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "RandomContrast"          : A.RandomContrast,
    "RandomBrightness"        : A.RandomBrightness,
    "RandomGamma"             : A.RandomGamma,

    "ChannelShuffle": A.ChannelShuffle,
    "ToGray"        : A.ToGray,
    
    "Normalize" : A.Normalize,
    "ToTensorV2": ToTensorV2,
}
