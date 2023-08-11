"""Global Settings"""
from pathlib import Path

import segmentation_models_pytorch as smp

from base_torch import (PPE_TYPES, TORCH_DATA_TYPES, TORCH_MODEL_TYPES,
                        TORCH_OPTIM_TYPES)
from data import GRICRGWLazyDataset
from model import (BCEandDiceLossWithLogits,
                   GlobalDiceCoefficientWithLogitsForPPE,
                   GlobalDiceLossWithLogits, GlobalDiceLossWithLogitsForPPE,
                   PercentileGlobalDiceCoefficientWithLogitsForPPE)

ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "google-research-identify-contrails-reduce-global-warming"
PROC = INPUT / "processed_data"

RANDOM_SEED = 1086

N_TRAIN = 20529
N_VALID = 1856

CONFIG_TYPES = dict()
CONFIG_TYPES.update(TORCH_MODEL_TYPES)
CONFIG_TYPES.update(TORCH_DATA_TYPES)
CONFIG_TYPES.update(TORCH_OPTIM_TYPES)
CONFIG_TYPES.update(PPE_TYPES)

# Add newly defined dataset, models, loss, etc. in data.py and model.py
TYPES_ADHOC = dict(
    GRICRGWLazyDataset=GRICRGWLazyDataset,

    Unet=smp.Unet,
    UnetPlusPlus=smp.UnetPlusPlus,
    DeepLabV3Plus=smp.DeepLabV3Plus,

    GlobalDiceLossWithLogits=GlobalDiceLossWithLogits,
    BCEandDiceLossWithLogits=BCEandDiceLossWithLogits,
    GlobalDiceCoefficientWithLogitsForPPE=GlobalDiceCoefficientWithLogitsForPPE,
    PercentileGlobalDiceCoefficientWithLogitsForPPE=PercentileGlobalDiceCoefficientWithLogitsForPPE,
    GlobalDiceLossWithLogitsForPPE=GlobalDiceLossWithLogitsForPPE,
)
CONFIG_TYPES.update(TYPES_ADHOC)
