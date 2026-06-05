"""Cityscapes Semantic Segmentation Dataset.

PyTorch Dataset implementation for the Cityscapes dataset with full label
mapping from raw labelIds (0-33) to the standard 19-class trainIds.

Directory layout expected:
    images_root/{split}/{city_name}/{city}_{seq}_{frame}_leftImg8bit.png
    labels_root/{split}/{city_name}/{city}_{seq}_{frame}_gtFine_labelIds.png

NOTE: The user must supply the leftImg8bit images directory. Download from
https://www.cityscapes-dataset.com/ (requires registration). The labels
(gtFine) are already available locally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Cityscapes 19-class metadata
# ---------------------------------------------------------------------------

CITYSCAPES_CLASSES: List[str] = [
    "road",          # trainId 0
    "sidewalk",      # trainId 1
    "building",      # trainId 2
    "wall",          # trainId 3
    "fence",         # trainId 4
    "pole",          # trainId 5
    "traffic light",  # trainId 6
    "traffic sign",  # trainId 7
    "vegetation",    # trainId 8
    "terrain",       # trainId 9
    "sky",           # trainId 10
    "person",        # trainId 11
    "rider",         # trainId 12
    "car",           # trainId 13
    "truck",         # trainId 14
    "bus",           # trainId 15
    "train",         # trainId 16
    "motorcycle",    # trainId 17
    "bicycle",       # trainId 18
]

NUM_CLASSES: int = 19
IGNORE_INDEX: int = 255

id2label: Dict[int, str] = {i: name for i, name in enumerate(CITYSCAPES_CLASSES)}
label2id: Dict[str, int] = {name: i for i, name in enumerate(CITYSCAPES_CLASSES)}

# Full mapping from raw Cityscapes labelId → trainId.
# Any labelId not listed maps to 255 (ignore).
_LABEL_ID_TO_TRAIN_ID: Dict[int, int] = {
    0: 255,    # unlabeled
    1: 255,    # ego vehicle
    2: 255,    # rectification border
    3: 255,    # out of roi
    4: 255,    # static
    5: 255,    # dynamic
    6: 255,    # ground
    7: 0,      # road
    8: 1,      # sidewalk
    9: 255,    # parking
    10: 255,   # rail track
    11: 2,     # building
    12: 3,     # wall
    13: 4,     # fence
    14: 255,   # guard rail
    15: 255,   # bridge
    16: 255,   # tunnel
    17: 5,     # pole
    18: 255,   # polegroup
    19: 6,     # traffic light
    20: 7,     # traffic sign
    21: 8,     # vegetation
    22: 9,     # terrain
    23: 10,    # sky
    24: 11,    # person
    25: 12,    # rider
    26: 13,    # car
    27: 14,    # truck
    28: 15,    # bus
    29: 255,   # caravan
    30: 255,   # trailer
    31: 16,    # train
    32: 17,    # motorcycle
    33: 18,    # bicycle
    -1: 255,   # license plate (sometimes present)
}

# Build a numpy lookup table for fast vectorised mapping (0-255 → trainId).
_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _raw_id, _train_id in _LABEL_ID_TO_TRAIN_ID.items():
    if 0 <= _raw_id < 256:
        _LUT[_raw_id] = _train_id


def map_label_ids_to_train_ids(label: np.ndarray) -> np.ndarray:
    """Convert raw Cityscapes labelId mask to 19-class trainId mask.

    Args:
        label: uint8 numpy array with raw Cityscapes label IDs (0-33).

    Returns:
        uint8 numpy array with train IDs (0-18) and 255 for ignored classes.
    """
    return _LUT[label]


# ---------------------------------------------------------------------------
# Cityscapes 19-class colour palette (RGB)
# ---------------------------------------------------------------------------

def get_cityscapes_colormap() -> np.ndarray:
    """Return the official Cityscapes 19-class colour palette.

    Returns:
        np.ndarray of shape (19, 3) with uint8 RGB values.
    """
    return np.array([
        [128, 64, 128],    # road
        [244, 35, 232],    # sidewalk
        [70, 70, 70],      # building
        [102, 102, 156],   # wall
        [190, 153, 153],   # fence
        [153, 153, 153],   # pole
        [250, 170, 30],    # traffic light
        [220, 220, 0],     # traffic sign
        [107, 142, 35],    # vegetation
        [152, 251, 152],   # terrain
        [70, 130, 180],    # sky
        [220, 20, 60],     # person
        [255, 0, 0],       # rider
        [0, 0, 142],       # car
        [0, 0, 70],        # truck
        [0, 60, 100],      # bus
        [0, 80, 100],      # train
        [0, 0, 230],       # motorcycle
        [119, 11, 32],     # bicycle
    ], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a trainId segmentation mask to an RGB colour image.

    Args:
        mask: 2-D uint8 array with trainId values (0-18, 255=ignore).

    Returns:
        3-D uint8 RGB array of the same spatial dimensions.
    """
    palette = get_cityscapes_colormap()
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        colour[mask == class_id] = palette[class_id]
    return colour


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(target_size: Tuple[int, int] = (512, 1024)) -> A.Compose:
    """Build albumentations augmentation pipeline for training.

    The pipeline first resizes to ``target_size`` (H, W), then applies
    random crop, flip, and colour jitter before normalising.

    Args:
        target_size: (height, width) to resize images to before cropping.

    Returns:
        An ``albumentations.Compose`` transform.
    """
    h, w = target_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=1),  # INTER_LINEAR
        A.RandomCrop(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(target_size: Tuple[int, int] = (512, 1024)) -> A.Compose:
    """Build albumentations transform pipeline for validation / inference.

    Args:
        target_size: (height, width) to resize images to.

    Returns:
        An ``albumentations.Compose`` transform.
    """
    h, w = target_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CityscapesSegDataset(Dataset):
    """PyTorch Dataset for Cityscapes semantic segmentation.

    Scans all city sub-directories under ``images_root/{split}/`` and
    ``labels_root/{split}/`` and pairs corresponding image and label files.

    Args:
        images_root: Path to ``leftImg8bit/`` directory (contains train/val/test
            subdirectories, each with per-city folders).
        labels_root: Path to ``gtFine/`` directory (same structure).
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        transform: Optional albumentations transform. If *None*, the
            default train or val transform is used based on the split.
        target_size: ``(height, width)`` for the default transforms.
    """

    def __init__(
        self,
        images_root: str,
        labels_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (512, 1024),
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.split = split
        self.target_size = target_size

        # Resolve default transforms -----------------------------------------------
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(target_size)
        else:
            self.transform = get_val_transforms(target_size)

        # Scan directory tree -------------------------------------------------------
        self.image_paths: List[str] = []
        self.label_paths: List[str] = []

        split_img_dir = self.images_root / split
        split_lbl_dir = self.labels_root / split

        if not split_img_dir.exists():
            raise FileNotFoundError(
                f"Images split directory not found: {split_img_dir}\n"
                f"Download leftImg8bit from https://www.cityscapes-dataset.com/ "
                f"and place it so that {split_img_dir} exists."
            )
        if not split_lbl_dir.exists():
            raise FileNotFoundError(
                f"Labels split directory not found: {split_lbl_dir}"
            )

        # Iterate over city directories
        for city_dir in sorted(split_img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city_name = city_dir.name
            lbl_city_dir = split_lbl_dir / city_name

            if not lbl_city_dir.exists():
                print(f"[WARN] Label city dir missing: {lbl_city_dir}, skipping.")
                continue

            for img_file in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Derive matching label filename
                stem = img_file.name.replace("_leftImg8bit.png", "")
                lbl_file = lbl_city_dir / f"{stem}_gtFine_labelIds.png"
                if not lbl_file.exists():
                    print(f"[WARN] Label not found for {img_file.name}, skipping.")
                    continue
                self.image_paths.append(str(img_file))
                self.label_paths.append(str(lbl_file))

        print(
            f"[CityscapesSegDataset] split={split}, "
            f"found {len(self.image_paths)} image-label pairs."
        )

    # ---- public helpers --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample.

        Returns:
            Dictionary with keys:
                - ``pixel_values``: float32 tensor [3, H, W] (normalised).
                - ``labels``: int64 tensor [H, W] with trainIds (0-18, 255=ignore).
        """
        # Load image as RGB numpy array
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        # Load label as raw uint8 labelIds
        label = np.array(Image.open(self.label_paths[idx]))

        # Map raw labelIds → trainIds
        label = map_label_ids_to_train_ids(label)

        # Apply augmentation (albumentations works on numpy arrays)
        transformed = self.transform(image=image, mask=label)
        pixel_values: torch.Tensor = transformed["image"]  # [3, H, W] float32
        labels:  torch.Tensor = transformed["mask"].long()  # [H, W]

        return {"pixel_values": pixel_values, "labels": labels}

    def get_sample_info(self, idx: int) -> Dict[str, str]:
        """Return metadata about a sample (useful for debugging).

        Args:
            idx: Sample index.

        Returns:
            Dict with ``image_path`` and ``label_path``.
        """
        return {
            "image_path": self.image_paths[idx],
            "label_path": self.label_paths[idx],
        }
