"""
Stream ``(image, label)`` pairs for a dataset split from the manifest CSV.

Reads the manifest produced by ``create_split.py`` (columns ``photo_name``,
``photo_path``, ``label``, ``split``), filters it to the requested split, loads
each image from local storage, decodes it to a BGR NumPy array (OpenCV
convention, matching the ``preprocessing`` package), and yields
``(image, label)`` pairs one at a time. Only the lightweight ``(path, label)``
rows are held in memory; the heavy decoded images are streamed one by one, so a
large split never has to be loaded all at once.

Image paths recorded relative to the project root are resolved against this
script's directory, so streaming works regardless of the caller's working
directory.

Typical usage::

    for image, label in get_feature_stream("train"):
        features = pipeline.process(image)
        # train downstream on (features, label)

Requirements:
    pip install numpy opencv-python
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

# The three partition names the manifest's "split" column may contain.
VALID_SPLITS = frozenset({"train", "val", "test"})

# Directory of this script. Image paths recorded relative to the project root
# (as written by create_split.py) are resolved against it, so loading does not
# depend on the caller's current working directory.
PROJECT_DIR = Path(__file__).resolve().parent

# Default manifest location, matching create_split.py's default output.
DEFAULT_CSV = PROJECT_DIR / "datasets" / "dataset_split.csv"

# Manifest column names. These must match the columns written by create_split.py.
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
    """Load the ``(photo_path, label)`` entries for one split from the manifest.

    Reads the manifest CSV, keeps only the rows whose ``split`` column matches
    the requested split, and pairs each image path with its integer label.

    Args:
        split: Partition to load. Must be ``"train"``, ``"val"`` or ``"test"``.
        csv_path: Path to the manifest CSV produced by ``create_split.py``.

    Returns:
        A list of ``(photo_path, label)`` tuples for the requested split.

    Raises:
        ValueError: If ``split`` is unknown or the CSV lacks a required column.
        FileNotFoundError: If the manifest CSV does not exist.
    """
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

        # Fail loudly if the manifest is missing any column we depend on, rather
        # than silently yielding nothing.
        for column in (PATH_COLUMN, LABEL_COLUMN, SPLIT_COLUMN):
            if column not in fieldnames:
                raise ValueError(
                    f"Manifest {csv_path} is missing required column '{column}'. "
                    f"Found columns: {fieldnames}"
                )

        for row in reader:
            # Keep only rows belonging to the requested split.
            if row[SPLIT_COLUMN] != split:
                continue
            path = row[PATH_COLUMN].strip()
            # Skip rows with no path; there is no image to stream for them.
            if not path:
                continue
            label = int(row[LABEL_COLUMN])
            entries.append((path, label))

    return entries


def _load_image(path: str) -> Optional[np.ndarray]:
    """Read a single image from local storage and return it as a BGR NumPy array.

    Relative paths are resolved against the project directory. An image that is
    missing or cannot be decoded yields ``None`` (after a warning) so one bad
    file does not abort the stream.

    Args:
        path: Image path, absolute or relative to the project root.

    Returns:
        The decoded image as a ``uint8`` BGR array of shape
        ``(height, width, 3)``, or ``None`` if it could not be read.
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
    """Yield ``(image, label)`` pairs for every readable image in a split.

    The ``(path, label)`` rows for the split are loaded from the manifest and
    shuffled up front (so the stream is not biased by file/class order), then
    images are read and decoded one at a time in that shuffled order. Only the
    cheap path/label strings are held in memory; decoded images stay streamed.

    Args:
        split: Partition to load. Must be ``"train"``, ``"val"`` or ``"test"``.
        csv_path: Path to the manifest CSV. Defaults to
            ``datasets/dataset_split.csv``.
        random_seed: Seed for the shuffle. Pass an int to reproduce a specific
            ordering (e.g. in tests); pass ``None`` for a fresh ordering each run.

    Yields:
        ``(image, label)`` where ``image`` is the decoded picture in BGR format
        (OpenCV convention) with shape ``(height, width, 3)`` and dtype
        ``uint8``, and ``label`` is the integer class label (0 for Real, 1 for
        Deepfake) from the same manifest row.

    Raises:
        ValueError: If ``split`` is not a recognized partition name.
        FileNotFoundError: If the manifest CSV is missing.

    Example:
        >>> for image, label in get_feature_stream("train"):
        ...     print(image.shape, label)
    """
    entries = _load_entries(split, csv_path)

    # Shuffle the lightweight (path, label) rows so the stream is randomized
    # rather than following dataset/file order. A local Random instance keeps the
    # shuffle reproducible (when seeded) without touching global RNG state.
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
