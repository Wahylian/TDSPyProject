import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_ID = "ayushmandatta1/deepdetect-2025"


def restructure_to_real_fake(dataset_root: Path) -> None:
    """Flatten the downloaded dataset so it holds only 'real' and 'fake' folders.

    The dataset ships as ddata/{train,test}/{real,fake}. This merges the train
    and test images of each label into a single <root>/<label> folder. Safe to
    re-run: if 'ddata' is already gone, the dataset is left as-is.
    """
    src = dataset_root / "ddata"
    if not src.exists():
        print("Dataset already restructured; skipping.")
        return

    for label in ("real", "fake"):
        dest = dataset_root / label
        # train is the larger split: rename it into place (instant), then merge
        # the test images into it.
        shutil.move(str(src / "train" / label), str(dest))
        for img in (src / "test" / label).iterdir():
            target = dest / img.name
            if target.exists():
                raise FileExistsError(f"Name collision, refusing to overwrite: {target}")
            shutil.move(str(img), str(target))

    shutil.rmtree(src)
    print(f"Restructured into: {dataset_root / 'real'} and {dataset_root / 'fake'}")


def main() -> None:
    """Download the dataset into the project's 'datasets' folder and restructure it.

    Isolating the network call here (rather than at module level) means importing
    this module has no side effects — the download only runs when the script is
    executed directly.
    """
    # Point KAGGLEHUB_CACHE at the repo root (this script lives in src/) so the
    # download lands under the project's 'datasets' folder. Must be set before
    # kagglehub is imported.
    os.environ["KAGGLEHUB_CACHE"] = str(Path(__file__).resolve().parent.parent)
    import kagglehub

    try:
        path = kagglehub.dataset_download(DATASET_ID)
    except Exception as exc:
        logger.error(
            "Failed to download dataset '%s': %s. Check your network connection "
            "and Kaggle credentials (see kagglehub authentication docs).",
            DATASET_ID,
            exc,
        )
        raise

    restructure_to_real_fake(Path(path))
    print("Path to dataset files:", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()