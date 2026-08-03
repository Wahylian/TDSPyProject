"""Write a train/val/test split manifest for the deepdetect-2025 dataset.

Scans the real/ and fake/ image subfolders and writes one CSV with columns
photo_name, photo_path (relative, forward slashes), label (0 real, 1 fake), and
split. Rows are shuffled with a seeded RNG and partitioned 70/15/15, so a fixed
seed reproduces the manifest. extract_features.py consumes it downstream.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd

# Image paths are written relative to the repo root (this script lives in src/)
# so the manifest stays portable.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset folder (holding real/ and fake/) as laid out by download_dataset.py.
DEFAULT_DATA_DIR = (
    PROJECT_DIR / "datasets" / "ayushmandatta1" / "deepdetect-2025" / "versions" / "1"
)

DEFAULT_OUTPUT_CSV = PROJECT_DIR / "datasets" / "dataset_split.csv"

# Class subfolder to integer label.
LABEL_BY_FOLDER: Dict[str, int] = {"real": 0, "fake": 1}

# The dataset mixes .jpg and .png images.
IMAGE_PATTERNS = ("*.jpg", "*.png")

# Per-partition fractions; test takes the remainder so they sum to the dataset size.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

COLUMNS = ["photo_name", "photo_path", "label", "split"]


def _scan_images(data_dir: Path) -> pd.DataFrame:
    """Collect every image under real/ and fake/ into a (name, path, label) frame."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    rows = []
    for folder, label in LABEL_BY_FOLDER.items():
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class subfolder is missing: {class_dir}")

        # Sort for a deterministic scan; the seeded shuffle sets the final order.
        image_paths = [p for pattern in IMAGE_PATTERNS for p in class_dir.glob(pattern)]
        for image_path in sorted(image_paths):
            rows.append(
                {
                    "photo_name": image_path.name,
                    # Relative, forward-slashed path for cross-OS portability.
                    "photo_path": os.path.relpath(image_path, PROJECT_DIR).replace(os.sep, "/"),
                    "label": label,
                }
            )

    return pd.DataFrame(rows, columns=["photo_name", "photo_path", "label"])


def _assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seeded-shuffle the rows and add a train/val/test split column."""
    # Seeded shuffle: random yet reproducible for a fixed seed.
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(TRAIN_FRACTION * n)
    n_val = int(VAL_FRACTION * n)

    # Assign by position; test absorbs the remainder so every row is covered once.
    split_labels = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )
    shuffled["split"] = split_labels
    return shuffled


def create_split(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    seed: int = 42,
) -> pd.DataFrame:
    """Scan the dataset, build a 70/15/15 split, write and return the manifest."""
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    images = _scan_images(data_dir)
    manifest = _assign_splits(images, seed)[COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

    return manifest


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the manifest generator."""
    parser = argparse.ArgumentParser(
        description="Generate a 70/15/15 train/val/test split manifest CSV "
        "for the deepfake-vs-real image dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory holding the real/ and fake/ subfolders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write the split manifest CSV to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the reproducible shuffle (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = create_split(data_dir=args.data_dir, output_csv=args.output_csv, seed=args.seed)

    # Report a short summary so a manual run confirms what was written.
    counts = df["split"].value_counts().to_dict()
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(f"  splits: train={counts.get('train', 0)}, "
          f"val={counts.get('val', 0)}, test={counts.get('test', 0)}")
    label_counts = df["label"].value_counts().to_dict()
    print(f"  labels: real(0)={label_counts.get(0, 0)}, "
          f"fake(1)={label_counts.get(1, 0)}")
