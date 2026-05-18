"""
Model definition for ROP detection.

Architecture:
    Image branch:    EfficientNet-B0 pre-trained on ImageNet, feature head
                     replaced with an Identity so we get a 1280-d feature
                     vector per image.
    Metadata branch: gestational age in days and birth weight in grams,
                     passed through a small MLP. A binary "metadata missing"
                     flag is also fed in so the model can ignore the
                     numerical values when they are unknown.
    Fusion:          concatenate the two branches and project to a single
                     logit (probability of ROP after sigmoid).

The "tabular" inputs are normalized using statistics computed at training
time and stored alongside the model in models/normalization.json so that
inference uses identical preprocessing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _build_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (backbone, feature_dim) for EfficientNet-B0."""
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
    net = models.efficientnet_b0(weights=weights)
    feature_dim = net.classifier[1].in_features  # 1280
    net.classifier = nn.Identity()  # produce features directly
    return net, feature_dim


class RopModel(nn.Module):
    """Multi-input ROP classifier."""

    META_DIM = 3  # [ga_days_norm, birth_weight_norm, metadata_missing_flag]

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_backbone(pretrained=pretrained)
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.META_DIM, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feat = self.forward_features(image)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        logit = self.head(fused).squeeze(1)
        return logit


# ---------------------------------------------------------------------------
# Metadata normalization helpers
# ---------------------------------------------------------------------------
def normalize_meta(
    ga_days: float | None,
    birth_weight_g: float | None,
    stats: dict,
) -> list[float]:
    """Return [ga_norm, bw_norm, missing_flag].

    If both values are missing we fill them with zero (the mean after
    normalization) and set the missing flag to 1.0 so the model knows to
    rely on the image alone.
    """
    missing = 0.0
    if ga_days is None or ga_days != ga_days:  # NaN check
        ga_norm = 0.0
        missing = 1.0
    else:
        ga_norm = (float(ga_days) - stats["ga_mean"]) / max(stats["ga_std"], 1e-6)
    if birth_weight_g is None or birth_weight_g != birth_weight_g:
        bw_norm = 0.0
        missing = 1.0
    else:
        bw_norm = (float(birth_weight_g) - stats["bw_mean"]) / max(stats["bw_std"], 1e-6)
    return [ga_norm, bw_norm, missing]
