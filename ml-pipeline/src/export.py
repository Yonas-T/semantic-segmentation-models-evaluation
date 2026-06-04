"""ONNX export script for Cityscapes segmentation models.

Exports a trained SegFormer or DeepLabV3 model to ONNX format with
dynamic axes, validates the exported model, and compares outputs against
the PyTorch model.

Usage::

    python -m src.export \\
        --model segformer \\
        --checkpoint runs/segformer/best_model.pth \\
        --output exports/segformer.onnx \\
        --opset 17

    python -m src.export \\
        --model deeplabv3 \\
        --checkpoint runs/deeplabv3/best_model.pth \\
        --output exports/deeplabv3.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .dataset import NUM_CLASSES
from .models import DeepLabV3Model, SegFormerModel, create_model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_height: int = 512,
    input_width: int = 1024,
    opset_version: int = 17,
    model_name: str = "segformer",
) -> None:
    """Export a PyTorch segmentation model to ONNX.

    Args:
        model: The PyTorch model (must be in eval mode on CPU).
        output_path: Destination path for the ``.onnx`` file.
        input_height: Spatial height of the dummy input.
        input_width: Spatial width of the dummy input.
        opset_version: ONNX opset version.
        model_name: ``'segformer'`` or ``'deeplabv3'`` (affects input naming).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, input_height, input_width)

    # Choose input/output names
    input_name = "pixel_values" if model_name == "segformer" else "images"
    output_name = "logits"

    dynamic_axes = {
        input_name: {0: "batch_size", 2: "height", 3: "width"},
        output_name: {0: "batch_size", 2: "out_height", 3: "out_width"},
    }

    print(f"[Export] Exporting {model_name} → {output_path}")
    print(f"  Input shape : {list(dummy_input.shape)}")
    print(f"  Opset       : {opset_version}")

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX file   : {file_size_mb:.1f} MB")


def validate_onnx(
    onnx_path: str,
    pytorch_model: nn.Module,
    input_height: int = 512,
    input_width: int = 1024,
    model_name: str = "segformer",
    tolerance: float = 1e-4,
) -> bool:
    """Validate an ONNX model against its PyTorch source.

    Steps:
        1. Load and check the ONNX model graph.
        2. Create an ONNX Runtime session.
        3. Run inference on a random input.
        4. Compare outputs against PyTorch with the given tolerance.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        pytorch_model: The original PyTorch model (eval mode, CPU).
        input_height: Spatial height of the test input.
        input_width: Spatial width of the test input.
        model_name: ``'segformer'`` or ``'deeplabv3'``.
        tolerance: Maximum allowed absolute difference.

    Returns:
        ``True`` if validation passes.
    """
    import onnx
    import onnxruntime as ort

    # 1. Structural check
    print("\n[Validate] Checking ONNX model structure …")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is structurally valid.")

    # Print input/output info
    for inp in onnx_model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input : {inp.name} → {shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → {shape}")

    # 2. ONNX Runtime inference
    print("\n[Validate] Running ONNX Runtime inference …")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_np = np.random.randn(1, 3, input_height, input_width).astype(np.float32)

    ort_outputs = session.run(None, {input_name: dummy_np})
    ort_logits = ort_outputs[0]
    print(f"  ONNX output shape: {ort_logits.shape}")

    # 3. PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_logits = pytorch_model(torch.from_numpy(dummy_np)).numpy()
    print(f"  PyTorch output shape: {pt_logits.shape}")

    # 4. Compare
    max_diff = float(np.abs(ort_logits - pt_logits).max())
    mean_diff = float(np.abs(ort_logits - pt_logits).mean())
    print(f"  Max  absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    if max_diff < tolerance:
        print(f"  ✓ Outputs match within tolerance ({tolerance}).")
        return True
    else:
        print(f"  ✗ Max diff {max_diff:.6e} exceeds tolerance {tolerance}.")
        print("    This may be acceptable for complex models — check manually.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run the full export + validation pipeline."""
    device = torch.device("cpu")  # ONNX export always on CPU

    # ---- Build model ---------------------------------------------------------
    print(f"\n=== Building {args.model} model ===")
    model = create_model(args.model, num_classes=NUM_CLASSES)

    # ---- Load checkpoint -----------------------------------------------------
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    print(f"  Parameters : {model.get_param_count():,}")
    print(f"  Model size : {model.get_model_size_mb():.1f} MB")

    # ---- Export --------------------------------------------------------------
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_height=args.input_height,
        input_width=args.input_width,
        opset_version=args.opset,
        model_name=args.model,
    )

    # ---- Validate ------------------------------------------------------------
    # Use a relaxed tolerance for SegFormer (transformer models tend to have
    # slightly larger numerical differences due to layer-norm / softmax).
    tol = 1e-3 if args.model == "segformer" else 1e-4
    validate_onnx(
        onnx_path=args.output,
        pytorch_model=model,
        input_height=args.input_height,
        input_width=args.input_width,
        model_name=args.model,
        tolerance=tol,
    )

    print("\n=== Export complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a trained segmentation model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["segformer", "deeplabv3"],
        required=True,
        help="Model architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint. If omitted, exports with pretrained weights.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/model.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=512,
        help="Input spatial height.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=1024,
        help="Input spatial width.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
