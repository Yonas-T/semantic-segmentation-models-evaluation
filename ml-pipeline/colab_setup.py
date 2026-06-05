"""Colab Setup Script — Downloads Cityscapes leftImg8bit directly to Colab.

Run this cell at the top of your Colab notebook to set up the dataset
and install dependencies. Requires Cityscapes credentials.

Usage in Colab:
    !python colab_setup.py --username YOUR_EMAIL --password YOUR_PASSWORD

Alternatively, set environment variables:
    import os
    os.environ['CITYSCAPES_USERNAME'] = 'your_email'
    os.environ['CITYSCAPES_PASSWORD'] = 'your_password'
    !python colab_setup.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITYSCAPES_BASE_URL = "https://www.cityscapes-dataset.com"
LOGIN_URL = f"{CITYSCAPES_BASE_URL}/login/"
DOWNLOAD_URL = f"{CITYSCAPES_BASE_URL}/file-handling/?packageID=3"  # leftImg8bit_trainvaltest.zip

DATA_DIR = Path("/content/cityscapes")
IMAGES_ZIP = DATA_DIR / "leftImg8bit_trainvaltest.zip"
LABELS_ZIP = DATA_DIR / "gtFine_trainvaltest.zip"

# Final extracted paths
IMAGES_ROOT = DATA_DIR / "leftImg8bit"
LABELS_ROOT = DATA_DIR / "gtFine"


def install_dependencies() -> None:
    """Install Python dependencies for the ML pipeline."""
    print("=" * 60)
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)

    req_path = Path(__file__).resolve().parent / "requirements.txt"
    if req_path.exists():
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)
        ])
        print("✓ Dependencies installed from requirements.txt")
    else:
        # Fallback: install key packages directly
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "torchvision", "transformers", "albumentations",
            "tqdm", "Pillow", "numpy", "scipy", "onnx", "onnxruntime",
            "optimum[onnxruntime]",
        ])
        print("✓ Dependencies installed (fallback)")


def download_cityscapes(username: str, password: str) -> None:
    """Download Cityscapes leftImg8bit using authenticated session.

    Uses requests with session cookies to authenticate against the
    Cityscapes download portal and fetch the leftImg8bit zip.

    Args:
        username: Cityscapes account email.
        password: Cityscapes account password.
    """
    import requests

    print("\n" + "=" * 60)
    print("DOWNLOADING CITYSCAPES leftImg8bit")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if IMAGES_ROOT.exists() and any(IMAGES_ROOT.rglob("*.png")):
        print(f"✓ leftImg8bit already extracted at {IMAGES_ROOT}")
        return

    if IMAGES_ZIP.exists():
        print(f"✓ ZIP already downloaded: {IMAGES_ZIP}")
    else:
        print(f"  Logging in as {username} ...")
        session = requests.Session()

        # Get CSRF token
        login_page = session.get(LOGIN_URL)
        login_page.raise_for_status()

        # Extract CSRF token from cookies or form
        csrf_token = session.cookies.get("csrftoken", "")

        # POST login
        login_data = {
            "username": username,
            "password": password,
            "csrfmiddlewaretoken": csrf_token,
            "submit": "Login",
        }
        headers = {
            "Referer": LOGIN_URL,
        }
        resp = session.post(LOGIN_URL, data=login_data, headers=headers)
        resp.raise_for_status()

        if "login" in resp.url.lower() and "logout" not in resp.text.lower():
            print("✗ Login failed. Check your credentials.")
            print("  Register at: https://www.cityscapes-dataset.com/register/")
            sys.exit(1)

        print("  ✓ Logged in successfully")

        # Download leftImg8bit
        print(f"  Downloading leftImg8bit_trainvaltest.zip (~11GB) ...")
        print(f"  This may take 15-30 minutes on Colab.")

        dl_resp = session.get(DOWNLOAD_URL, stream=True)
        dl_resp.raise_for_status()

        total_size = int(dl_resp.headers.get("content-length", 0))
        downloaded = 0

        with open(IMAGES_ZIP, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="", flush=True)

        print(f"\n  ✓ Downloaded: {IMAGES_ZIP} ({downloaded // (1024*1024)} MB)")

    # Extract
    print(f"  Extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"  ✓ Extracted leftImg8bit")


def upload_gtfine_from_drive() -> None:
    """Mount Google Drive and copy/link gtFine if available.

    This is an alternative to downloading gtFine — the user already has
    the gtFine labels locally and can upload them to Google Drive.
    """
    print("\n" + "=" * 60)
    print("SETTING UP gtFine LABELS")
    print("=" * 60)

    if LABELS_ROOT.exists() and any(LABELS_ROOT.rglob("*.png")):
        print(f"✓ gtFine already available at {LABELS_ROOT}")
        return

    # Try Google Drive
    drive_path = Path("/content/drive/MyDrive/cityscapes/gtFine")
    if drive_path.exists():
        print(f"  Found gtFine in Google Drive: {drive_path}")
        os.symlink(str(drive_path), str(LABELS_ROOT))
        print(f"  ✓ Symlinked to {LABELS_ROOT}")
        return

    # Try downloading via Cityscapes API (gtFine is only 241MB)
    print("  gtFine not found. Options:")
    print("  1. Upload gtFine_trainvaltest.zip to Google Drive at:")
    print("     /MyDrive/cityscapes/gtFine_trainvaltest.zip")
    print("  2. Or run: !unzip /path/to/gtFine_trainvaltest.zip -d /content/cityscapes/")
    print("  3. Or use the download function with your Cityscapes credentials")


def verify_dataset() -> None:
    """Verify the dataset is properly set up."""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET")
    print("=" * 60)

    for split in ["train", "val"]:
        img_dir = IMAGES_ROOT / split
        lbl_dir = LABELS_ROOT / split

        if img_dir.exists():
            n_images = sum(1 for _ in img_dir.rglob("*_leftImg8bit.png"))
            print(f"  {split} images: {n_images}")
        else:
            print(f"  ✗ {split} images directory missing: {img_dir}")

        if lbl_dir.exists():
            n_labels = sum(1 for _ in lbl_dir.rglob("*_gtFine_labelIds.png"))
            print(f"  {split} labels: {n_labels}")
        else:
            print(f"  ✗ {split} labels directory missing: {lbl_dir}")

    print(f"\n  IMAGES_ROOT = '{IMAGES_ROOT}'")
    print(f"  LABELS_ROOT = '{LABELS_ROOT}'")
    print("\n  Use these paths with train.py:")
    print(f"    --images-root {IMAGES_ROOT}")
    print(f"    --labels-root {LABELS_ROOT}")


def print_training_commands() -> None:
    """Print ready-to-use training commands for Colab."""
    print("\n" + "=" * 60)
    print("READY! Use these commands to train:")
    print("=" * 60)
    print(f"""
# Train SegFormer-B2 (recommended: ~2h on T4 GPU for 50 epochs)
!cd /content/cityscapes-benchmark/ml-pipeline && python -m src.train \\
    --model segformer \\
    --images-root {IMAGES_ROOT} \\
    --labels-root {LABELS_ROOT} \\
    --epochs 50 \\
    --batch-size 4 \\
    --lr 6e-5 \\
    --mixed-precision \\
    --output-dir /content/runs/segformer

# Train DeepLabV3 (recommended: ~3h on T4 GPU for 50 epochs)
!cd /content/cityscapes-benchmark/ml-pipeline && python -m src.train \\
    --model deeplabv3 \\
    --images-root {IMAGES_ROOT} \\
    --labels-root {LABELS_ROOT} \\
    --epochs 50 \\
    --batch-size 4 \\
    --lr 1e-4 \\
    --mixed-precision \\
    --output-dir /content/runs/deeplabv3

# Export to ONNX
!cd /content/cityscapes-benchmark/ml-pipeline && python -m src.export \\
    --model segformer \\
    --checkpoint /content/runs/segformer/best_model.pth \\
    --output /content/exports/segformer_b2.onnx

!cd /content/cityscapes-benchmark/ml-pipeline && python -m src.export \\
    --model deeplabv3 \\
    --checkpoint /content/runs/deeplabv3/best_model.pth \\
    --output /content/exports/deeplabv3_resnet101.onnx

# Evaluate both models
!cd /content/cityscapes-benchmark/ml-pipeline && python evaluate_local.py \\
    --images-root {IMAGES_ROOT} \\
    --labels-root {LABELS_ROOT} \\
    --segformer-checkpoint /content/runs/segformer/best_model.pth \\
    --deeplabv3-checkpoint /content/runs/deeplabv3/best_model.pth \\
    --output /content/metrics.json
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up Cityscapes dataset and dependencies in Google Colab."
    )
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("CITYSCAPES_USERNAME", ""),
        help="Cityscapes account email (or set CITYSCAPES_USERNAME env var).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("CITYSCAPES_PASSWORD", ""),
        help="Cityscapes account password (or set CITYSCAPES_PASSWORD env var).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading leftImg8bit (if already available).",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip installing Python dependencies.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_deps:
        install_dependencies()

    if not args.skip_download:
        if not args.username or not args.password:
            print("\n⚠ Cityscapes credentials not provided.")
            print("  Set --username and --password, or environment variables:")
            print("    export CITYSCAPES_USERNAME='your_email'")
            print("    export CITYSCAPES_PASSWORD='your_password'")
            print("\n  Skipping download. Set up the dataset manually.")
        else:
            download_cityscapes(args.username, args.password)

    upload_gtfine_from_drive()
    verify_dataset()
    print_training_commands()
