import shutil
from pathlib import Path

import kagglehub

# Download latest version (goes to kagglehub's cache)
path = kagglehub.dataset_download("prithivsakthiur/deepfake-vs-real-20k")

# Copy the dataset into the project's 'datasets' folder
dest = Path(__file__).parent / "datasets" / "deepfake-vs-real-20k"
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Path to dataset files:", dest)
