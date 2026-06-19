"""
Generate a train/val/test split manifest for the deepfake-vs-real image dataset.

Scans the on-disk dataset directory — which contains a ``Real`` and a
``Deepfake`` subfolder of ``.jpg`` images — and writes a single CSV manifest
describing every image with these columns:

    photo_name  the image filename (e.g. ``111 (1).jpg``)
    photo_path  the image path relative to the project root, using forward slashes
    label       integer class label: 0 for Real, 1 for Deepfake
    split       the assigned partition: "train", "val", or "test"

The images are shuffled with a seeded RNG (default seed 42) and partitioned
70% / 15% / 15% into train / val / test, so re-running with the same seed
reproduces an identical manifest. The manifest is consumed downstream by
``extract_features.py`` to stream ``(image, label)`` pairs per split.

Typical usage::

    python create_split.py                 # use defaults (seed 42)
    python create_split.py --seed 7        # different shuffle

Requirements:
    pip install pandas
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd

# Directory of this script. Image paths are written relative to it so the
# manifest stays valid regardless of the caller's working directory.
PROJECT_DIR = Path(__file__).resolve().parent

# Default location of the image dataset (the folder holding Real/ and Deepfake/).
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets" / "deepfake-vs-real-20k" / "Deep-vs-Real"

# Default output path for the generated split manifest.
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "datasets" / "dataset_split.csv"

# Map each class subfolder name to its integer label.
LABEL_BY_FOLDER: Dict[str, int] = {"Real": 0, "Deepfake": 1}

# Fraction of images assigned to each partition. The test split takes the
# remainder so the three fractions always sum to exactly the dataset size.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Output column order of the manifest CSV.
COLUMNS = ["photo_name", "photo_path", "label", "split"]


def _scan_images(data_dir: Path) -> pd.DataFrame:
    """Collect every ``.jpg`` image under the Real/ and Deepfake/ subfolders.

    Args:
        data_dir: Directory containing the ``Real`` and ``Deepfake`` subfolders.

    Returns:
        A DataFrame with one row per image and the columns ``photo_name``,
        ``photo_path`` (relative to the project root, forward slashes) and
        ``label`` (0 for Real, 1 for Deepfake).

    Raises:
        FileNotFoundError: If ``data_dir`` or one of its class subfolders is
            missing.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    rows = []
    for folder, label in LABEL_BY_FOLDER.items():
        class_dir = data_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class subfolder is missing: {class_dir}")

        # Sort the filenames so the scan order is deterministic; the final row
        # order is decided by the seeded shuffle in _assign_splits, not here.
        for image_path in sorted(class_dir.glob("*.jpg")):
            rows.append(
                {
                    "photo_name": image_path.name,
                    # Store the path relative to the project root with forward
                    # slashes so the manifest is portable across machines/OSes.
                    "photo_path": os.path.relpath(image_path, PROJECT_DIR).replace(os.sep, "/"),
                    "label": label,
                }
            )

    return pd.DataFrame(rows, columns=["photo_name", "photo_path", "label"])


def _assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle the rows and label each with a train/val/test partition.

    Args:
        df: The scanned image DataFrame (photo_name, photo_path, label).
        seed: Seed for the shuffle, making the partition reproducible.

    Returns:
        A new DataFrame, shuffled and carrying an added ``split`` column, with
        sizes ``int(0.70 * n)`` / ``int(0.15 * n)`` / remainder for
        train / val / test respectively.
    """
    # Shuffle every row with a seeded RNG so the partition is random yet
    # reproducible for a fixed seed.
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(TRAIN_FRACTION * n)
    n_val = int(VAL_FRACTION * n)

    # Assign partitions by position in the shuffled order; "test" absorbs the
    # remainder so every row is covered exactly once.
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
    """Scan the dataset, build a 70/15/15 split, and write the manifest CSV.

    Args:
        data_dir: Directory containing the ``Real`` and ``Deepfake`` subfolders.
        output_csv: Path the manifest CSV is written to. Parent directories are
            created if needed.
        seed: Seed for the reproducible shuffle. Defaults to 42.

    Returns:
        The manifest DataFrame that was written, with columns
        ``photo_name``, ``photo_path``, ``label`` and ``split``.

    Raises:
        FileNotFoundError: If ``data_dir`` or a class subfolder is missing.
    """
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
        help="Directory holding the Real/ and Deepfake/ subfolders.",
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
          f"deepfake(1)={label_counts.get(1, 0)}")
