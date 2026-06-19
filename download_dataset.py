import os
import shutil
from pathlib import Path

# Make kagglehub download into the project's 'datasets' folder instead of the
# default ~/.cache/kagglehub location. kagglehub appends its own
# 'datasets/<owner>/<name>/versions/<n>' tree under KAGGLEHUB_CACHE, so pointing
# the cache at the project root lands the files inside the existing 'datasets' folder.
# Must be set before kagglehub is imported.
os.environ["KAGGLEHUB_CACHE"] = str(Path(__file__).parent)

import kagglehub


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


# Download latest version (directly into the project's 'datasets' folder)
path = kagglehub.dataset_download("ayushmandatta1/deepdetect-2025")

restructure_to_real_fake(Path(path))

print("Path to dataset files:", path)