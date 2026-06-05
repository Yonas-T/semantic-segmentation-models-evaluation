"""Evaluation script for Cityscapes semantic segmentation models.

Runs both SegFormer and DeepLabV3 on the Cityscapes validation set and
produces a comprehensive ``metrics.json`` with per-model mIoU, per-class
IoU, pixel accuracy, latency statistics, FPS, and resource usage.

Usage::

    python evaluate_local.py \\
        --images-root /path/to/leftImg8bit \\
        --labels-root /path/to/gtFine \\
        --segformer-checkpoint runs/segformer/best_model.pth \\
        --deeplabv3-checkpoint runs/deeplabv3/best_model.pth \\
        --output metrics.json

Designed for Google Colab — auto-detects CUDA, shows tqdm progress bars,
and handles single-GPU memory reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import (
    CITYSCAPES_CLASSES,
    IGNORE_INDEX,
    NUM_CLASSES,
    CityscapesSegDataset,
    get_val_transforms,
)
from src.models import DeepLabV3Model, SegFormerModel, create_model


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device(requested: Optional[str] = None) -> torch.device:
    """Resolve the compute device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple MPS")
        return torch.device("mps")
    print("[Device] CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Confusion matrix + metrics
# ---------------------------------------------------------------------------

def _compute_confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Accumulate a confusion matrix (see train.py for detailed docs)."""
    p = preds.reshape(-1).cpu().numpy()
    g = labels.reshape(-1).cpu().numpy()
    valid = g != IGNORE_INDEX
    p, g = p[valid], g[valid]
    p = np.clip(p, 0, num_classes - 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (g, p), 1)
    return cm


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """Derive mIoU, per-class IoU, pixel accuracy from confusion matrix."""
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    valid = union > 0
    iou = np.zeros_like(intersection, dtype=np.float64)
    iou[valid] = intersection[valid] / union[valid]

    miou = float(iou[valid].mean()) if valid.any() else 0.0
    pixel_acc = float(intersection.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    per_class: Dict[str, float] = {}
    for i, name in enumerate(CITYSCAPES_CLASSES):
        per_class[name] = round(float(iou[i]), 4)

    return {
        "miou": round(miou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "per_class_iou": per_class,
    }


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------

def _benchmark_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...] = (1, 3, 512, 1024),
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """Measure inference latency statistics.

    Args:
        model: Model in eval mode on ``device``.
        device: Compute device.
        input_shape: Shape of dummy input tensor.
        warmup: Number of warmup runs (not measured).
        iterations: Number of timed runs.

    Returns:
        Dict with ``mean``, ``p50``, ``p95`` latency in milliseconds.
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(latencies)
    return {
        "mean": round(float(arr.mean()), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_name: str,
    display_name: str,
    loader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    num_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full evaluation for a single model.

    Args:
        model: Segmentation model.
        model_name: Short key (``segformer`` / ``deeplabv3``).
        display_name: Human-readable name for the report.
        loader: Validation DataLoader.
        device: Compute device.
        checkpoint_path: Path to the checkpoint file (for reporting file size).
        num_images: Optional cap on the number of images to evaluate.

    Returns:
        Metrics dictionary ready for JSON serialisation.
    """
    model.to(device)
    model.eval()

    # Reset peak memory tracker
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    count = 0

    pbar = tqdm(loader, desc=f"Evaluating {display_name}", leave=True)
    for batch in pbar:
        if num_images is not None and count >= num_images:
            break

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(pixel_values)
        if logits.shape[2:] != labels.shape[1:]:
            logits = F.interpolate(
                logits, size=labels.shape[1:], mode="bilinear", align_corners=False
            )

        preds = logits.argmax(dim=1)
        cm += _compute_confusion_matrix(preds, labels, NUM_CLASSES)
        count += pixel_values.size(0)

    # Metrics
    metrics = _metrics_from_cm(cm)

    # Latency
    print(f"  Benchmarking latency for {display_name} …")
    latency = _benchmark_latency(model, device)
    fps = round(1000.0 / latency["mean"], 1) if latency["mean"] > 0 else 0.0

    # Model size from checkpoint file
    model_size_mb = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 1)
    else:
        model_size_mb = round(model.get_model_size_mb(), 1)

    # Peak GPU memory
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1
        )

    return {
        "name": display_name,
        "mIoU": metrics["miou"],
        "pixelAccuracy": metrics["pixel_accuracy"],
        "perClassIoU": metrics["per_class_iou"],
        "latencyMs": latency,
        "fps": fps,
        "modelSizeMB": model_size_mb,
        "paramCount": model.get_param_count(),
        "peakMemoryMB": peak_memory_mb,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print a comparison table to the console."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"{'Metric':<25} ", end="")
    for key, info in results.items():
        print(f"{'│ ' + info['name']:<22}", end="")
    print()
    print(sep)

    rows = [
        ("mIoU", "mIoU", ".4f"),
        ("Pixel Accuracy", "pixelAccuracy", ".4f"),
        ("Latency mean (ms)", None, None),
        ("Latency p50 (ms)", None, None),
        ("Latency p95 (ms)", None, None),
        ("FPS", "fps", ".1f"),
        ("Model Size (MB)", "modelSizeMB", ".1f"),
        ("Parameters", "paramCount", ","),
        ("Peak GPU Mem (MB)", "peakMemoryMB", ".1f"),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<23} ", end="")
        for model_key, info in results.items():
            if key is not None:
                val = info.get(key, "N/A")
                if fmt and isinstance(val, (int, float)):
                    print(f"│ {val:{fmt}:<20}", end="")
                else:
                    print(f"│ {val!s:<20}", end="")
            elif "Latency mean" in label:
                val = info.get("latencyMs", {}).get("mean", "N/A")
                print(f"│ {val:<20}", end="")
            elif "Latency p50" in label:
                val = info.get("latencyMs", {}).get("p50", "N/A")
                print(f"│ {val:<20}", end="")
            elif "Latency p95" in label:
                val = info.get("latencyMs", {}).get("p95", "N/A")
                print(f"│ {val:<20}", end="")
        print()

    # Per-class IoU
    print(sep)
    print("  Per-class IoU:")
    for cls_name in CITYSCAPES_CLASSES:
        print(f"    {cls_name:<21} ", end="")
        for model_key, info in results.items():
            val = info.get("perClassIoU", {}).get(cls_name, 0.0)
            print(f"│ {val:.4f}              ", end="")
        print()
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Run evaluation for all available models."""
    device = _get_device(args.device)

    # ---- Validation DataLoader -----------------------------------------------
    print("\n=== Loading validation dataset ===")
    val_ds = CityscapesSegDataset(
        images_root=args.images_root,
        labels_root=args.labels_root,
        split="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # batch=1 for fair latency measurement
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    results: Dict[str, Dict[str, Any]] = {}

    # ---- SegFormer -----------------------------------------------------------
    if args.segformer_checkpoint:
        print("\n=== Evaluating SegFormer ===")
        seg_model = SegFormerModel(num_classes=NUM_CLASSES)
        ckpt = torch.load(
            args.segformer_checkpoint, map_location=device, weights_only=True
        )
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            seg_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            seg_model.load_state_dict(ckpt, strict=False)

        results["segformer"] = evaluate_model(
            model=seg_model,
            model_name="segformer",
            display_name="SegFormer-B2",
            loader=val_loader,
            device=device,
            checkpoint_path=args.segformer_checkpoint,
            num_images=args.num_images,
        )
    else:
        print("[INFO] No SegFormer checkpoint provided — skipping.")

    # ---- DeepLabV3 -----------------------------------------------------------
    if args.deeplabv3_checkpoint:
        print("\n=== Evaluating DeepLabV3 ===")
        dl_model = DeepLabV3Model(num_classes=NUM_CLASSES, pretrained=False)
        ckpt = torch.load(
            args.deeplabv3_checkpoint, map_location=device, weights_only=True
        )
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            dl_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            dl_model.load_state_dict(ckpt, strict=False)

        results["deeplabv3"] = evaluate_model(
            model=dl_model,
            model_name="deeplabv3",
            display_name="DeepLabV3+",
            loader=val_loader,
            device=device,
            checkpoint_path=args.deeplabv3_checkpoint,
            num_images=args.num_images,
        )
    else:
        print("[INFO] No DeepLabV3 checkpoint provided — skipping.")

    if not results:
        print("[ERROR] No checkpoints provided. Nothing to evaluate.")
        sys.exit(1)

    # ---- Metadata ------------------------------------------------------------
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    eval_images = args.num_images if args.num_images else len(val_ds)

    output_data = {
        "models": results,
        "metadata": {
            "dataset": "Cityscapes",
            "numClasses": NUM_CLASSES,
            "evalResolution": "512x1024",
            "evalImages": eval_images,
            "gpuName": gpu_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # ---- Save JSON -----------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Saved] Metrics → {output_path}")

    # ---- Print table ---------------------------------------------------------
    _print_comparison_table(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models on Cityscapes validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--segformer-checkpoint",
        type=str,
        default=None,
        help="Path to SegFormer .pth checkpoint.",
    )
    parser.add_argument(
        "--deeplabv3-checkpoint",
        type=str,
        default=None,
        help="Path to DeepLabV3 .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Output path for metrics JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit evaluation to this many images (for quick testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
