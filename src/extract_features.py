"""Stream (image, label) pairs for a dataset split from the manifest CSV.

Filters the create_split.py manifest to the requested split and yields decoded
BGR images one at a time. Only the (path, label) rows are held in memory, so a
large split is never fully loaded. Relative paths resolve against this script's
directory, so streaming is independent of the caller's working directory.
"""

from __future__ import annotations

import csv
import os
import random
import warnings
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

# Valid values of the manifest's "split" column.
VALID_SPLITS = frozenset({"train", "val", "test"})

# Relative image paths resolve against the repo root (this script lives in
# src/), not the caller's cwd.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Default manifest location, matching create_split.py's default output.
DEFAULT_CSV = PROJECT_DIR / "datasets" / "dataset_split.csv"

# Manifest column names, matching create_split.py.
PATH_COLUMN = "photo_path"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"

# A single manifest entry: an image path paired with its integer label.
Entry = Tuple[str, int]


def _validate_split(split: str) -> None:
    """Raise ``ValueError`` unless *split* is one of the recognized partitions."""
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {sorted(VALID_SPLITS)}, got '{split}'"
        )


def _load_entries(split: str, csv_path: str) -> List[Entry]:
    """Load the (photo_path, label) entries for one split from the manifest."""
    _validate_split(split)

    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Manifest CSV not found: {csv_path}. Run create_split.py first."
        )

    entries: List[Entry] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Fail loudly on a missing column rather than silently yielding nothing.
        for column in (PATH_COLUMN, LABEL_COLUMN, SPLIT_COLUMN):
            if column not in fieldnames:
                raise ValueError(
                    f"Manifest {csv_path} is missing required column '{column}'. "
                    f"Found columns: {fieldnames}"
                )

        for row in reader:
            if row[SPLIT_COLUMN] != split:
                continue
            path = row[PATH_COLUMN].strip()
            # No path means no image to stream.
            if not path:
                continue
            label = int(row[LABEL_COLUMN])
            entries.append((path, label))

    return entries


def _load_image(path: str) -> Optional[np.ndarray]:
    """Decode one image to a uint8 BGR array, or None (with a warning) if unreadable.

    Relative paths resolve against the project directory; a bad file is skipped
    rather than aborting the stream.
    """
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
    image = cv2.imread(resolved, cv2.IMREAD_COLOR)
    if image is None:
        warnings.warn(f"Skipping unreadable image {path}")
        return None
    return image


def get_feature_stream(
    split: str,
    csv_path: str = DEFAULT_CSV,
    random_seed: Optional[int] = 42,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Yield (image, label) pairs for every readable image in a split.

    The (path, label) rows are loaded and shuffled up front, then images decode
    one at a time in that order. Seed the shuffle with random_seed for a
    reproducible ordering, or None for a fresh one. Images are BGR uint8; labels
    are 0 (real) / 1 (deepfake).
    """
    entries = _load_entries(split, csv_path)

    # Shuffle the cheap (path, label) rows so the stream isn't in file/class order.
    # A local Random keeps it reproducible without touching global RNG state.
    random.Random(random_seed).shuffle(entries)

    for path, label in entries:
        image = _load_image(path)
        if image is not None:
            yield image, label


if __name__ == "__main__":
    import sys

    split_name = sys.argv[1] if len(sys.argv) > 1 else "train"
    manifest = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV

    print(f"Streaming '{split_name}' images from {manifest}...")
    count = 0
    for img, lbl in get_feature_stream(split_name, csv_path=manifest):
        count += 1
        print(f"  [{count}] shape={img.shape}, dtype={img.dtype}, label={lbl}")
        if count >= 3:
            print("  (stopping after 3 images)")
            break

    if count == 0:
        print("No images yielded.")
    else:
        print(f"Done. Yielded {count} image(s).")
