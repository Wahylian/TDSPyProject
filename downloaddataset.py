import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = os.path.dirname(os.path.abspath(__file__))

path = kagglehub.dataset_download("chuneeb/deepfake-detection-dataset-2026")

print("Path to dataset files:", path)