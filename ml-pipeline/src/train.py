"""Unified training script for Cityscapes semantic segmentation.

Supports SegFormer and DeepLabV3 models with mixed-precision training,
polynomial LR scheduling, and mIoU-based checkpointing.

Usage (CLI)::

    python -m src.train \\
        --model segformer \\
        --images-root /path/to/leftImg8bit \\
        --labels-root /path/to/gtFine \\
        --epochs 50 \\
        --batch-size 4 \\
        --output-dir ./runs/segformer \\
        --mixed-precision

Designed to run seamlessly on **Google Colab** (auto-detects CUDA / MPS / CPU,
shows tqdm progress bars, logs GPU memory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import (
    CITYSCAPES_CLASSES,
    IGNORE_INDEX,
    NUM_CLASSES,
    CityscapesSegDataset,
)
from .models import create_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Using Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[Device] Using CPU")
    return dev


def _print_gpu_memory() -> None:
    """Print current GPU memory usage (CUDA only)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU memory — allocated: {alloc:.1f} MB, peak: {peak:.1f} MB")


# ---------------------------------------------------------------------------
# Confusion-matrix based mIoU
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute confusion matrix from predictions and ground-truth labels.

    Args:
        preds: ``[N]`` or ``[B, H, W]`` predicted class IDs.
        labels: Same shape as ``preds``, ground-truth class IDs.
        num_classes: Total number of valid classes.
        ignore_index: Label value to ignore.

    Returns:
        Confusion matrix ``[num_classes, num_classes]`` as int64 numpy array.
    """
    preds_flat = preds.reshape(-1).cpu().numpy()
    labels_flat = labels.reshape(-1).cpu().numpy()

    # Mask out ignore
    valid = labels_flat != ignore_index
    preds_flat = preds_flat[valid]
    labels_flat = labels_flat[valid]

    # Clip predictions to valid range (safety)
    preds_flat = np.clip(preds_flat, 0, num_classes - 1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels_flat, preds_flat), 1)
    return cm


def compute_metrics_from_cm(
    cm: np.ndarray,
) -> Dict[str, Any]:
    """Derive mIoU, per-class IoU, and pixel accuracy from a confusion matrix.

    Args:
        cm: ``[C, C]`` confusion matrix.

    Returns:
        Dictionary with ``miou``, ``pixel_accuracy``, ``per_class_iou`` keys.
    """
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

    # Avoid division by zero for classes not present
    valid = union > 0
    iou = np.zeros_like(intersection, dtype=np.float64)
    iou[valid] = intersection[valid] / union[valid]

    miou = float(iou[valid].mean()) if valid.any() else 0.0
    pixel_acc = float(intersection.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    per_class = {}
    for i, name in enumerate(CITYSCAPES_CLASSES):
        per_class[name] = float(iou[i])

    return {
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": per_class,
    }


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    epoch: int,
) -> Dict[str, float]:
    """Run a single training epoch.

    Returns:
        Dict with ``train_loss`` (mean).
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            logits = model(pixel_values)

            # Ensure logits match label spatial dims
            if logits.shape[2:] != labels.shape[1:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(n_batches, 1)
    return {"train_loss": avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> Dict[str, Any]:
    """Run validation and compute mIoU + pixel accuracy.

    Returns:
        Dict with ``val_loss``, ``miou``, ``pixel_accuracy``, ``per_class_iou``.
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            logits = model(pixel_values)

            if logits.shape[2:] != labels.shape[1:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        cm += compute_confusion_matrix(preds, labels, NUM_CLASSES, IGNORE_INDEX)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(n_batches, 1)
    metrics = compute_metrics_from_cm(cm)
    metrics["val_loss"] = avg_loss
    return metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Full training pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    device = _get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets & loaders --------------------------------------------------
    print("\n=== Loading datasets ===")
    train_ds = CityscapesSegDataset(
        images_root=args.images_root,
        labels_root=args.labels_root,
        split="train",
    )
    val_ds = CityscapesSegDataset(
        images_root=args.images_root,
        labels_root=args.labels_root,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ---------------------------------------------------------------
    print(f"\n=== Creating model: {args.model} ===")
    model = create_model(args.model, num_classes=NUM_CLASSES)
    model = model.to(device)
    print(f"  Parameters: {model.get_param_count():,}")
    print(f"  Size: {model.get_model_size_mb():.1f} MB")

    # ---- Optimiser, scheduler, criterion ------------------------------------
    lr = args.lr
    if lr is None:
        lr = 6e-5 if args.model == "segformer" else 1e-4

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Mixed precision
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed-precision training: ENABLED")
    else:
        print("  Mixed-precision training: DISABLED")

    # ---- Resume from checkpoint ----------------------------------------------
    start_epoch = 0
    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
                print(f"  Resuming from epoch {start_epoch}")
        else:
            # Plain state dict
            model.load_state_dict(ckpt)

    # ---- Training log --------------------------------------------------------
    training_log: Dict[str, Any] = {
        "model": args.model,
        "config": {
            "lr": lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "mixed_precision": use_amp,
            "weight_decay": 0.01,
            "scheduler": "PolynomialLR(power=0.9)",
            "loss": "CrossEntropyLoss(ignore_index=255)",
        },
        "epochs": [],
        "best_miou": 0.0,
    }

    best_miou = 0.0
    best_ckpt_path = output_dir / "best_model.pth"

    # ---- Epoch loop ----------------------------------------------------------
    print(f"\n=== Training for {args.epochs} epochs ===\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp, epoch)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "miou": val_metrics["miou"],
            "pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        training_log["epochs"].append(epoch_record)

        # Print summary
        print(
            f"Epoch {epoch:3d}/{args.epochs} │ "
            f"train_loss={train_metrics['train_loss']:.4f} │ "
            f"val_loss={val_metrics['val_loss']:.4f} │ "
            f"mIoU={val_metrics['miou']:.4f} │ "
            f"pixAcc={val_metrics['pixel_accuracy']:.4f} │ "
            f"lr={optimizer.param_groups[0]['lr']:.2e} │ "
            f"time={elapsed:.0f}s"
        )
        _print_gpu_memory()

        # Save best
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            training_log["best_miou"] = best_miou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "miou": best_miou,
                },
                best_ckpt_path,
            )
            print(f"  ✓ New best mIoU: {best_miou:.4f} — saved → {best_ckpt_path}")

        # Save training log after every epoch
        log_path = output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

    # ---- Final summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Best mIoU:      {best_miou:.4f}")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Training log:   {log_path}")
    _print_gpu_memory()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on Cityscapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["segformer", "deeplabv3"],
        required=True,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults: 6e-5 segformer, 1e-4 deeplabv3).",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="Path to leftImg8bit/ directory.",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        required=True,
        help="Path to gtFine/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/default",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed-precision (FP16) training on CUDA.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data-loading workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
