"""
Train the ROP binary classifier.

Usage examples:
    python src/train.py
    python src/train.py --epochs 25 --batch-size 16
    python src/train.py --epochs 1 --max-train-batches 5     # smoke test

The script logs progress per epoch, saves the best model (highest validation
AUROC) to models/best_model.pt, and writes:
    models/metrics.json          training history and best metrics
    models/normalization.json    metadata normalization stats and config
    models/test_report.txt       final evaluation on the test split

To choose a device pass --device {cpu, cuda, mps}. The script auto-picks
the best available device when --device auto (the default) is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from dataset import (  # noqa: E402
    RopDataset,
    build_eval_transform,
    build_train_transform,
    compute_meta_stats,
)
from model import RopModel  # noqa: E402

PROJECT_ROOT = SRC_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"


def pick_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "mps":
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    df: pd.DataFrame,
    project_root: Path,
    stats: dict,
    transform,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 2,
) -> DataLoader:
    dataset = RopDataset(df, project_root, stats, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df["label"].value_counts().to_dict()
    weights = df["label"].map(lambda y: 1.0 / counts[y]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(df), replacement=True)


def evaluate(
    model: RopModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(device)
            meta = meta.to(device)
            logits = model(images, meta)
            all_logits.extend(logits.detach().cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    preds = (probs >= 0.5).astype(int)
    labels_arr = np.array(all_labels, dtype=int)
    try:
        auroc = float(roc_auc_score(labels_arr, probs))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "auroc": auroc,
        "confusion_matrix": confusion_matrix(labels_arr, preds).tolist(),
        "probs": probs.tolist(),
        "labels": labels_arr.tolist(),
        "preds": preds.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ROP classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=3,
        help="Number of warmup epochs where only the head and meta MLP train.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="If > 0, cap the number of train batches per epoch (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="If > 0, cap the number of validation batches per epoch.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    labels_csv = DATASET_DIR / "labels.csv"
    if not labels_csv.exists():
        print(
            "ERROR: dataset/labels.csv not found. Run `python src/prepare_data.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(labels_csv)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("Class balance (train):")
    print(train_df["label_name"].value_counts())

    meta_stats = compute_meta_stats(train_df)
    print(f"Meta stats: {meta_stats}")

    train_loader = make_loader(
        train_df,
        PROJECT_ROOT,
        meta_stats,
        build_train_transform(),
        batch_size=args.batch_size,
        shuffle=True,
        sampler=build_balanced_sampler(train_df),
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_df,
        PROJECT_ROOT,
        meta_stats,
        build_eval_transform(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_df,
        PROJECT_ROOT,
        meta_stats,
        build_eval_transform(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = RopModel(pretrained=True).to(device)
    pos_weight = torch.tensor(
        [len(train_df[train_df["label"] == 0]) / max(len(train_df[train_df["label"] == 1]), 1)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head_params = list(model.meta_mlp.parameters()) + list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())

    history: list[dict] = []
    best_val_auroc = -1.0
    best_state: dict | None = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                head_params, lr=args.lr_head, weight_decay=args.weight_decay
            )
            phase = "warmup"
        else:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": args.lr_head},
                    {"params": backbone_params, "lr": args.lr_backbone},
                ],
                weight_decay=args.weight_decay,
            )
            phase = "finetune"

        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_idx, (images, meta, labels) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            images = images.to(device)
            meta = meta.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            seen += images.size(0)
        train_loss = epoch_loss / max(seen, 1)

        # Validation
        if args.max_val_batches:
            val_subset_idx = list(range(min(args.max_val_batches * args.batch_size, len(val_df))))
            partial_val_loader = make_loader(
                val_df.iloc[val_subset_idx],
                PROJECT_ROOT,
                meta_stats,
                build_eval_transform(),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            val_metrics = evaluate(model, partial_val_loader, device)
        else:
            val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{args.epochs}  phase={phase}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  "
            f"val_auroc={val_metrics['auroc']:.3f}  "
            f"elapsed={elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
            }
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.pt"
    torch.save(best_state, model_path)
    print(f"\nSaved best model to {model_path}  (val AUROC = {best_val_auroc:.3f})")

    # Save normalization stats / config so inference uses the same numbers.
    normalization = {
        "ga_mean": meta_stats["ga_mean"],
        "ga_std": meta_stats["ga_std"],
        "bw_mean": meta_stats["bw_mean"],
        "bw_std": meta_stats["bw_std"],
        "image_size": 224,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    }
    (MODEL_DIR / "normalization.json").write_text(json.dumps(normalization, indent=2))

    # Final test evaluation using the best weights.
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    test_report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=["not_rop", "rop"],
        digits=3,
    )
    print("\nTest set report")
    print("---------------")
    print(test_report)
    print(f"Test AUROC: {test_metrics['auroc']:.3f}")
    print(f"Confusion matrix [tn fp / fn tp]: {test_metrics['confusion_matrix']}")

    (MODEL_DIR / "test_report.txt").write_text(
        f"Test AUROC: {test_metrics['auroc']:.3f}\n\n"
        f"{test_report}\n"
        f"Confusion matrix [tn fp / fn tp]: {test_metrics['confusion_matrix']}\n"
    )

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_val_auroc": best_val_auroc,
                "test_accuracy": test_metrics["accuracy"],
                "test_auroc": test_metrics["auroc"],
                "test_confusion_matrix": test_metrics["confusion_matrix"],
            },
            indent=2,
        )
    )
    print(f"\nMetrics written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
