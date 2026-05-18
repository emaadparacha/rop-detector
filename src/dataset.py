"""
Dataset and dataloader helpers for ROP training.

A Pandas DataFrame (loaded from dataset/labels.csv) drives everything so
augmentation, metadata lookup, and label assignment stay in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model import normalize_meta


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RopDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        project_root: Path,
        meta_stats: dict,
        transform: transforms.Compose,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.project_root = project_root
        self.meta_stats = meta_stats
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.project_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        ga = row["gestational_age_days"]
        bw = row["birth_weight_g"]
        meta_vec = normalize_meta(ga, bw, self.meta_stats)
        meta = torch.tensor(meta_vec, dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, meta, label


def compute_meta_stats(train_df: pd.DataFrame) -> dict:
    """Compute mean/std of gestational age and birth weight on train split."""
    ga = train_df["gestational_age_days"].dropna()
    bw = train_df["birth_weight_g"].dropna()
    return {
        "ga_mean": float(ga.mean()) if len(ga) else 0.0,
        "ga_std": float(ga.std()) if len(ga) > 1 else 1.0,
        "bw_mean": float(bw.mean()) if len(bw) else 0.0,
        "bw_std": float(bw.std()) if len(bw) > 1 else 1.0,
    }
