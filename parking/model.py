"""Model and loss factories, config-driven."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from parking.config import Config
from parking.dataset import IGNORE

ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
}


def build_model(cfg: Config) -> nn.Module:
    arch = ARCHS[cfg.train.arch]
    return arch(
        encoder_name=cfg.train.encoder,
        encoder_weights=cfg.train.encoder_weights or None,
        in_channels=3,
        classes=1,
    )


class DiceBce(nn.Module):
    """0..1-weighted Dice + label-smoothed BCE, both skipping ignore pixels."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True, ignore_index=IGNORE)
        self.bce = smp.losses.SoftBCEWithLogitsLoss(
            ignore_index=IGNORE, smooth_factor=cfg.train.bce_smooth
        )
        self.w = cfg.train.dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits (N,1,H,W), target (N,H,W) in {0,1,IGNORE}
        return self.w * self.dice(logits, target) + (1 - self.w) * self.bce(
            logits.squeeze(1), target.float()
        )


def build_loss(cfg: Config) -> nn.Module:
    return DiceBce(cfg)
