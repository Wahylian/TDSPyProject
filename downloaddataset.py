import kagglehub

# Download latest version
path = kagglehub.dataset_download("chuneeb/deepfake-detection-dataset-2026")

print("Path to dataset files:", path)