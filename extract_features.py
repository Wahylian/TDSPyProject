"""
Stream image arrays from URL metadata for memory-efficient ML pipelines.

Loads image URLs from split metadata files under ``base_dir``, downloads each
image, decodes it to a BGR NumPy array (OpenCV convention, matching
``image_preprocessing.py``), and yields arrays one at a time.

Typical usage::

    for image in get_feature_stream("train", base_dir="./datasets"):
        features = pipeline.process(image)

Metadata lookup (first match wins)::

    {base_dir}/{split}.json
    {base_dir}/{split}.csv
    {base_dir}/split.json          (keyed by split name)
    {base_dir}/split.csv           (filtered by data_split / split column)
    {base_dir}/split_dataset.csv   (filtered by data_split / split column)

Requirements:
    pip install requests pillow numpy opencv-python
"""

import csv
import json
import os
import warnings
from io import BytesIO
from typing import Generator, List

import cv2
import numpy as np
import requests
from PIL import Image

VALID_SPLITS = frozenset({"train", "test", "val"})
REQUEST_TIMEOUT = 10
URL_COLUMNS = ("image_url", "url")
SPLIT_COLUMNS = ("data_split", "split")


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {sorted(VALID_SPLITS)}, got '{split}'"
        )


def _urls_from_json(path: str, split: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(url) for url in data]

    if isinstance(data, dict):
        if split in data:
            return [str(url) for url in data[split]]
        if "urls" in data:
            return [str(url) for url in data["urls"]]

    raise ValueError(f"Unrecognized JSON structure in {path}")


def _url_column(fieldnames: List[str]) -> str:
    for column in URL_COLUMNS:
        if column in fieldnames:
            return column
    raise ValueError(
        f"CSV must contain one of {URL_COLUMNS}, got columns: {fieldnames}"
    )


def _split_column(fieldnames: List[str]) -> str | None:
    for column in SPLIT_COLUMNS:
        if column in fieldnames:
            return column
    return None


def _urls_from_csv(path: str, split: str, filter_by_split: bool) -> List[str]:
    urls: List[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty: {path}")

        url_col = _url_column(reader.fieldnames)
        split_col = _split_column(reader.fieldnames) if filter_by_split else None

        if filter_by_split and split_col is None:
            raise ValueError(
                f"CSV {path} must contain a split column "
                f"({SPLIT_COLUMNS}) when not using per-split files"
            )

        for row in reader:
            if filter_by_split and row.get(split_col) != split:
                continue
            url = row.get(url_col, "").strip()
            if url:
                urls.append(url)

    return urls


def _load_image_urls(split: str, base_dir: str) -> List[str]:
    """Resolve and load image URLs for the requested split from metadata files."""
    _validate_split(split)

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")

    candidates = [
        (os.path.join(base_dir, f"{split}.json"), "json", False),
        (os.path.join(base_dir, f"{split}.csv"), "csv", False),
        (os.path.join(base_dir, "split.json"), "json", True),
        (os.path.join(base_dir, "split.csv"), "csv", True),
        (os.path.join(base_dir, "split_dataset.csv"), "csv", True),
    ]

    for path, fmt, filter_by_split in candidates:
        if not os.path.isfile(path):
            continue

        if fmt == "json":
            return _urls_from_json(path, split)
        return _urls_from_csv(path, split, filter_by_split=filter_by_split)

    raise FileNotFoundError(
        f"No metadata file found for split '{split}' in {base_dir}. "
        f"Expected one of: {split}.json, {split}.csv, split.json, "
        f"split.csv, or split_dataset.csv"
    )


def _download_image(url: str) -> np.ndarray | None:
    """Download a single image and return it as a BGR NumPy array."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (requests.RequestException, OSError, ValueError) as exc:
        warnings.warn(f"Skipping broken URL {url}: {exc}")
        return None


def get_feature_stream(
    split: str,
    base_dir: str = "./data",
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR image arrays for every downloadable URL in a dataset split.

    Args:
        split: Dataset partition to load. Must be ``"train"``, ``"test"``, or ``"val"``.
        base_dir: Directory containing split metadata files. Defaults to ``"./data"``.

    Yields:
        np.ndarray: Decoded image in BGR format (OpenCV convention) with shape
        ``(height, width, 3)`` and dtype ``uint8``.

    Raises:
        ValueError: If ``split`` is not a recognized partition name.
        FileNotFoundError: If ``base_dir`` or the required metadata file is missing.

    Example:
        >>> for image in get_feature_stream("train", base_dir="./datasets"):
        ...     print(image.shape)
    """
    urls = _load_image_urls(split, base_dir)

    for url in urls:
        image = _download_image(url)
        if image is not None:
            yield image


if __name__ == "__main__":
    import sys

    split_name = sys.argv[1] if len(sys.argv) > 1 else "train"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "./datasets"

    print(f"Streaming '{split_name}' images from {data_dir}...")
    count = 0
    for img in get_feature_stream(split_name, base_dir=data_dir):
        count += 1
        print(f"  [{count}] shape={img.shape}, dtype={img.dtype}")
        if count >= 3:
            print("  (stopping after 3 images)")
            break

    if count == 0:
        print("No images yielded.")
    else:
        print(f"Done. Yielded {count} image(s).")
