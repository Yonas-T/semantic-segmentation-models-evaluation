"""Model wrappers for Cityscapes semantic segmentation.

Provides:
- **SegFormerModel**: wraps HuggingFace ``SegformerForSemanticSegmentation``.
- **DeepLabV3Model / DeepLabV3Wrapper**: wraps torchvision ``deeplabv3_resnet101``.
- **create_model**: factory function.

Both wrappers expose a unified ``forward`` → logits interface plus convenience
``predict``, ``get_param_count``, and ``get_model_size_mb`` methods.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .dataset import CITYSCAPES_CLASSES, IMAGENET_MEAN, IMAGENET_STD, id2label, label2id

# ---------------------------------------------------------------------------
# SegFormer
# ---------------------------------------------------------------------------


class SegFormerModel(nn.Module):
    """Wrapper around HuggingFace SegFormer for Cityscapes segmentation.

    Args:
        num_classes: Number of semantic classes (default 19 for Cityscapes).
        pretrained_name: HuggingFace model hub identifier. The default
            checkpoint is already fine-tuned on Cityscapes at 1024×1024.
    """

    def __init__(
        self,
        num_classes: int = 19,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_name = pretrained_name

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.processor = SegformerImageProcessor.from_pretrained(pretrained_name)

    # -- persistence helpers ---------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to a ``.pth`` file.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[SegFormer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str, device: str = "cpu") -> None:
        """Load model weights from a ``.pth`` file.

        Args:
            path: Path to the checkpoint file.
            device: Device to map the weights to.
        """
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"[SegFormer] Loaded checkpoint ← {path}")

    # -- forward ---------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits upsampled to input spatial resolution.

        Args:
            pixel_values: ``[B, 3, H, W]`` normalised float tensor.

        Returns:
            Logits tensor of shape ``[B, num_classes, H, W]``.
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, num_classes, h, w] — lower res
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    # -- inference helper ------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 1024),
    ) -> np.ndarray:
        """Run full preprocessing → inference → argmax on a single PIL image.

        Args:
            image: RGB PIL Image.
            target_size: ``(height, width)`` to resize the image before inference.

        Returns:
            Predicted class-ID map as a uint8 numpy array ``[H, W]``.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # Resize
        image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)

        # Preprocess using the HF processor
        inputs = self.processor(images=image_resized, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        logits = self.forward(pixel_values)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    # -- utilities -------------------------------------------------------------

    def get_param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Estimate model size in megabytes from parameter storage."""
        param_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        )
        return (param_bytes + buffer_bytes) / (1024 * 1024)


# ---------------------------------------------------------------------------
# DeepLabV3
# ---------------------------------------------------------------------------


class DeepLabV3Wrapper(nn.Module):
    """Thin wrapper that returns only the ``'out'`` logits tensor.

    ``torchvision.models.segmentation.deeplabv3_resnet101`` returns an
    ``OrderedDict`` with keys ``'out'`` and optionally ``'aux'``.  This
    wrapper extracts the primary output so that ``forward`` returns a
    plain tensor — convenient for training loops and ONNX export.

    Args:
        backbone: A torchvision DeepLabV3 model instance.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, 3, H, W]`` normalised float tensor.

        Returns:
            Logits tensor ``[B, num_classes, H, W]``.
        """
        out: OrderedDict = self.backbone(x)
        return out["out"]


class DeepLabV3Model(nn.Module):
    """Wrapper around torchvision DeepLabV3-ResNet101 for Cityscapes.

    Loads COCO-pretrained weights and replaces the classification head
    with a new ``DeepLabHead`` for the target number of classes.

    Args:
        num_classes: Number of semantic classes (default 19).
        pretrained: Whether to load COCO-pretrained backbone weights.
    """

    def __init__(self, num_classes: int = 19, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained DeepLabV3 with ResNet-101 backbone
        weights = "DEFAULT" if pretrained else None
        backbone = deeplabv3_resnet101(weights=weights)

        # Replace the classifier head for num_classes
        backbone.classifier = DeepLabHead(2048, num_classes)

        # Replace aux classifier as well if it exists
        if backbone.aux_classifier is not None:
            backbone.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=1),
            )

        self.model = DeepLabV3Wrapper(backbone)

        # Standard ImageNet preprocessing for inference
        self._preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # -- persistence helpers ---------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to a ``.pth`` file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[DeepLabV3] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str, device: str = "cpu") -> None:
        """Load model weights from a ``.pth`` file."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"[DeepLabV3] Loaded checkpoint ← {path}")

    # -- forward ---------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            images: ``[B, 3, H, W]`` normalised float tensor.

        Returns:
            Logits tensor ``[B, num_classes, H, W]``.
        """
        return self.model(images)

    # -- inference helper ------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 1024),
    ) -> np.ndarray:
        """Run full preprocessing → inference → argmax on a single PIL image.

        Args:
            image: RGB PIL Image.
            target_size: ``(height, width)`` to resize before inference.

        Returns:
            Predicted class-ID map as uint8 numpy array ``[H, W]``.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        tensor = self._preprocess(image_resized).unsqueeze(0).to(device)
        logits = self.forward(tensor)

        # Upsample to target_size if needed (DeepLabV3 preserves spatial dims)
        if logits.shape[2:] != (target_size[0], target_size[1]):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    # -- utilities -------------------------------------------------------------

    def get_param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Estimate model size in megabytes from parameter storage."""
        param_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        )
        return (param_bytes + buffer_bytes) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(model_name: str, num_classes: int = 19) -> nn.Module:
    """Instantiate a segmentation model by name.

    Args:
        model_name: One of ``'segformer'`` or ``'deeplabv3'``.
        num_classes: Number of output semantic classes.

    Returns:
        A ``nn.Module`` model instance.

    Raises:
        ValueError: If ``model_name`` is not recognised.
    """
    name = model_name.lower().strip()
    if name == "segformer":
        return SegFormerModel(num_classes=num_classes)
    elif name == "deeplabv3":
        return DeepLabV3Model(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: segformer, deeplabv3"
        )
