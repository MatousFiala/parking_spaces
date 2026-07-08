"""Torch Dataset over the chip directory + albumentations pipelines."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import geopandas as gpd
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from parking.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE = 255


def train_transform() -> A.Compose:
    return A.Compose(
        [
            # D4 group: aerial imagery has no canonical orientation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            # robustness across ortofoto vintages; rotation border -> ignore
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), fill=0, fill_mask=IGNORE, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_range=(70, 95), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_transform() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class ChipDataset(Dataset):
    def __init__(self, cfg: Config, split: str):
        chips_dir = Path(cfg.paths.chips_dir)
        index = gpd.read_file(chips_dir / "chips_index.gpkg")
        self.ids = index.loc[index.split == split, "chip_id"].tolist()
        if not self.ids:
            raise RuntimeError(f"no chips for split={split!r} in {chips_dir}")
        self.img_dir = chips_dir / "images"
        self.mask_dir = chips_dir / "masks"
        self.transform = train_transform() if split == "train" else val_transform()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chip_id = self.ids[i]
        img = np.asarray(Image.open(self.img_dir / f"{chip_id}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(self.mask_dir / f"{chip_id}.png"))
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()
