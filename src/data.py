# -*- coding: utf-8 -*- #
""""""
import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class GRICRGWLazyDataset(Dataset):
    """"""

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        mask_paths: tp.Sequence[FilePath],
        transform: A.Compose,
    ):
        """Initialize"""
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        """Return num of cadence snippets"""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return preprocessed input and label for given index."""
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        img = np.load(img_path)
        mask = np.load(mask_path)
        img, mask = self._apply_transform(img, mask)

        return {"data": img, "target": mask}

    def _apply_transform(self, img: np.ndarray, mask: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        return img, mask.float()

    def lazy_init(self, **kwargs):
        """Reset Members"""
        for k, v in kwargs.items():
            assert hasattr(self, k), f"not have a member `{k}`" 
            setattr(self, k, v)
