"""
Pre-download every dataset image to local storage.

Run this after ``downloaddataset.py`` has fetched ``FINAL_DATASET.csv``. It reads
that CSV, downloads each image referenced by ``image_url`` into a local
``photos`` directory (named by the row's ``image_id``), and writes a new dataset
mapping ``datasets/photos_dataset.csv`` pairing each image's local path with its
``label_numeric``.

Downloads run concurrently across a thread pool, since each download is
independent and spends almost all its time waiting on the network. The number of
parallel workers is tunable with ``--workers`` (lower it if the image host starts
rate-limiting).

An image that is already present on disk is reused rather than re-downloaded, so
the script is safe to re-run. An image that cannot be downloaded keeps its
datapoint: its path is left empty rather than dropped, so the mapping has one row
per row of ``FINAL_DATASET.csv``.

Usage:
    python downloaddataset.py
    python download_photos.py [--workers N]
"""
import argparse
import concurrent.futures
import glob
import os
import threading

import pandas as pd
import requests

REQUEST_TIMEOUT = 10
PHOTOS_DIRNAME = "photos"
MAPPING_FILENAME = "photos_dataset.csv"
DEFAULT_WORKERS = 16

# One requests.Session per worker thread, so HTTP connections are pooled and kept
# alive without sharing a single Session across threads.
_thread_local = threading.local()


def _session() -> requests.Session:
    """Return this thread's reusable HTTP session, creating it on first use."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _find_final_dataset(script_dir: str) -> str:
    """Locate ``FINAL_DATASET.csv`` anywhere under the ``datasets`` directory."""
    pattern = os.path.join(script_dir, "datasets", "**", "FINAL_DATASET.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            "FINAL_DATASET.csv not found — run downloaddataset.py first."
        )
    return matches[0]


def _download_image(url: str, dest_path: str) -> bool:
    """Download ``url`` to ``dest_path``. Return True on success, False otherwise."""
    try:
        response = _session().get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return False
    with open(dest_path, "wb") as f:
        f.write(response.content)
    return True


def _fetch_one(image_id, url: str, photos_dir: str) -> str | None:
    """Ensure the image for one row is on disk; return its relative path or None.

    Reuses an already-downloaded file, otherwise attempts the download. Returns
    the project-relative path on success, or ``None`` if the image could not be
    obtained.
    """
    filename = f"{image_id}.jpg"
    dest_path = os.path.join(photos_dir, filename)
    relative_path = os.path.join(PHOTOS_DIRNAME, filename)
    if os.path.isfile(dest_path) or _download_image(url, dest_path):
        return relative_path
    return None


def download_photos(workers: int = DEFAULT_WORKERS) -> None:
    """Download all dataset images locally and write the path/label mapping.

    Creates the ``photos`` directory, downloads each image from its
    ``image_url`` concurrently (skipping ones already on disk), and saves
    ``datasets/photos_dataset.csv`` with columns ``image_path`` and
    ``label_numeric``. Rows whose image could not be downloaded are kept with an
    empty ``image_path``.

    Args:
        workers: Number of parallel download threads (default: 16).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = _find_final_dataset(script_dir)
    dataset_df = pd.read_csv(dataset_path)
    total = len(dataset_df)
    print(f"Loaded {total} rows from {dataset_path}")

    photos_dir = os.path.join(script_dir, PHOTOS_DIRNAME)
    os.makedirs(photos_dir, exist_ok=True)

    image_ids = dataset_df["image_id"].tolist()
    urls = dataset_df["image_url"].tolist()
    labels = dataset_df["label_numeric"].tolist()

    # Download concurrently. Each result is stored back into its row's slot, so
    # the mapping stays aligned with FINAL_DATASET.csv regardless of the order in
    # which downloads finish.
    paths: list[str | None] = [None] * total
    available = 0
    print(f"Downloading with {workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_fetch_one, image_ids[i], urls[i], photos_dir): i
            for i in range(total)
        }
        for done, future in enumerate(
            concurrent.futures.as_completed(future_to_index), start=1
        ):
            index = future_to_index[future]
            path = future.result()
            paths[index] = path
            if path is not None:
                available += 1
            if done % 50 == 0:
                print(f"  processed {done}/{total} images")

    records = [
        {"image_path": paths[i], "label_numeric": labels[i]} for i in range(total)
    ]
    mapping_path = os.path.join(script_dir, "datasets", MAPPING_FILENAME)
    pd.DataFrame(records, columns=["image_path", "label_numeric"]).to_csv(
        mapping_path, index=False
    )
    print(f"Downloaded/available {available}/{total} images in {photos_dir}")
    print(f"Mapping written to {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download dataset images locally")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel download threads (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    download_photos(workers=args.workers)
