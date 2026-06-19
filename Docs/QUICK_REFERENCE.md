# Image Preprocessing Pipeline - Quick Reference

## One-Liner Installation
```bash
pip install opencv-python scikit-image
```

## Quickstart (3 lines)
```python
from preprocessing import ImagePipeline
pipeline = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (64, 64)}), ('normalize', {'method': 'minmax'}), ('vectorize', {})])
features = pipeline.process(image)  # (4096,) float32 vector
```

## 3 Ways to Compose

### 1. Pipeline Class
```python
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])
features = pipeline.process(image)
```

### 2. Functional Composition
```python
from functools import partial
from preprocessing import compose
pipeline = compose(
    vectorize_image,
    partial(normalize_image, method='minmax'),
    partial(resize_image, target_size=(64, 64)),
    to_grayscale
)
features = pipeline(image)
```

### 3. Decorator
```python
from preprocessing import pipeline_decorator, to_grayscale, resize_image, vectorize_image

@pipeline_decorator(
    (to_grayscale, {}),
    (lambda x: resize_image(x, (64, 64)), {}),
    (vectorize_image, {})
)
def extract_features(image):
    return image

features = extract_features(image)
```

## Common Patterns

### SVM Training Pipeline
```python
from preprocessing import ImagePipeline
from sklearn.svm import SVC

# Preprocess
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('denoise', {'method': 'bilateral'}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])

# Extract and train
features = np.array([pipeline.process(img) for img in train_images])
svm = SVC(kernel='rbf').fit(features, labels)

# Predict
test_features = np.array([pipeline.process(img) for img in test_images])
predictions = svm.predict(test_features)
```

### Batch Processing
```python
from preprocessing import batch_process, ImagePipeline

pipeline = ImagePipeline([...])
features_batch = batch_process(images_list, pipeline)
# Shape: (num_images, feature_dim)
```

### Vector Reduction (classical models)
```python
from preprocessing import batch_process, ImagePipeline

pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {}),                                  # flatten to vectors
    ('reduce', {'method': 'vec-pca', 'n_components': 128}),
])
X = batch_process(images_list, pipeline)  # (num_images, 128)
```

### Matrix Reduction (CNN / ViT inputs)
```python
from preprocessing import batch_process, ImagePipeline

# Grayscale matrices
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    # NO 'vectorize' — keep matrices
    ('reduce', {'method': 'mat-pca', 'n_components': 32}),
])
M = batch_process(images_list, pipeline)  # (num_images, 128, 32) — rows preserved

# Colour matrices (keep channels for CNNs/ViTs): drop 'grayscale' too
colour_pipeline = ImagePipeline([
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    # NO 'grayscale', NO 'vectorize' — keep (H, W, C)
    ('reduce', {'method': 'mat-pca', 'n_components': 32}),
])
C = batch_process(images_list, colour_pipeline)  # (num_images, 128, 32, 3) — rows + channels preserved
```

### Multiple Sizes (Experiment)
```python
fast = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (64, 64)}), ('vectorize', {})])
medium = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (128, 128)}), ('vectorize', {})])
hq = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (224, 224)}), ('vectorize', {})])

for pipeline, name in [(fast, '64'), (medium, '128'), (hq, '224')]:
    features = batch_process(images, pipeline)
    score = train_and_evaluate(features, labels)
    print(f"{name}x{name}: {score}")
```

## Function Reference

### Main Operations

| Function | Default | Output |
|----------|---------|--------|
| `vectorize_image(img)` | - | 1D array (float32) |
| `normalize_image(img, 'minmax')` | minmax | Normalized [0, 1] |
| `resize_image(img, (64,64))` | - | Resized array |
| `to_grayscale(img)` | - | 2D grayscale |
| `reduce_noise(img, 'bilateral')` | bilateral | Denoised array |

### Vectorization Methods (optional step)
- `'flat'`: Raw pixel flatten (default)
- `'vgg16'`: VGG16 embedding from block5_pool (requires tensorflow)

Omit the `('vectorize', {})` op entirely to keep each image as a 2D/3D matrix
for CNN/ViT inputs.

### Dimensionality Reduction Methods (`'reduce'` op)
Vector subgroup — needs a `('vectorize', {})` step before it; input
`(n_samples, n_features)`, output `(n_samples, n_components)`:
- `'vec-pca'`: PCA, variance-preserving (requires scikit-learn)
- `'vec-jl'`: Johnson–Lindenstrauss random projection (requires scikit-learn)

Matrix subgroup — used **without** `vectorize`; reduces only the width
(column) axis, preserving rows **and** colour channels. Pure NumPy. Accepts
grayscale or multi-channel stacks:
- grayscale: input `(n_samples, height, width)` → output `(n_samples, height, n_components)`
- colour:    input `(n_samples, height, width, channels)` → output `(n_samples, height, n_components, channels)`

Methods:
- `'mat-pca'`: two-dimensional PCA on the column axis (one shared basis pooled across channels)
- `'mat-jl'`: two-dimensional random projection of the column axis (shared across channels)

- `None` (default): bypass — return input unchanged.

Aliases: `'pca'` → `'vec-pca'`, `'jl'` / `'johnson-lindenstrauss'` → `'vec-jl'`.

### Normalization Methods
- `'minmax'`: [0, 1] range
- `'standard'`: Zero mean, unit variance
- `'histogram'`: Histogram equalization

### Noise Methods
- `'bilateral'`: Edge-preserving (default)
- `'gaussian'`: Simple blur
- `'morphological'`: Morphological ops
- `'median'`: Salt-and-pepper removal

### Interpolation Methods
- `'nearest'`: Fast, blocky
- `'bilinear'`: Default, balanced
- `'bicubic'`: Slower, smoother
- `'lanczos'`: Highest quality

## Feature Dimensions

```
Input Size          Grayscale       RGB
64×64               4,096           12,288
128×128             16,384          49,152
224×224             50,176          150,528
```

## Common Errors & Fixes

### TypeError: Expected np.ndarray
```python
# ❌ Wrong: passing PIL Image
preprocessing.vectorize_image(pil_image)

# ✅ Correct: convert to numpy array
import numpy as np
preprocessing.vectorize_image(np.array(pil_image))
```

### ValueError: Image must be 2D or 3D
```python
# ❌ Wrong: 4D array
pipeline.process(image_array.reshape(1, 224, 224, 3))

# ✅ Correct: single 3D image
pipeline.process(image_array)
```

### Pipeline Operation Not Found
```python
# ❌ Wrong: typo in operation name
ImagePipeline([('normailize', {})])  # typo!

# ✅ Correct: exact operation name
ImagePipeline([('normalize', {})])
```

## Performance Tips

### Fast Processing (Real-time)
```python
fast_pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('vectorize', {})
])
# ~100 images/sec
```

### Accurate Processing (Batch)
```python
accurate_pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (224, 224)}),
    ('denoise', {'method': 'bilateral'}),
    ('normalize', {'method': 'standard'}),
    ('vectorize', {})
])
# ~10-20 images/sec
```

### Memory Efficient
```python
# Process in batches, don't load all images
for batch in get_image_batches(size=32):
    features = batch_process(batch, pipeline)
    model.predict(features)
```

## Debugging Tips

### Check intermediate outputs
```python
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('vectorize', {})
])

image = cv2.imread('test.jpg')
print(f"Original: {image.shape}, dtype={image.dtype}")

gray = to_grayscale(image)
print(f"After grayscale: {gray.shape}")

resized = resize_image(gray, (64, 64))
print(f"After resize: {resized.shape}")

features = vectorize_image(resized)
print(f"After vectorize: {features.shape}, range=[{features.min()}, {features.max()}]")
```

### Profile timing
```python
import time
start = time.time()
for _ in range(100):
    features = pipeline.process(image)
elapsed = time.time() - start
print(f"{elapsed:.2f}s for 100 images = {100/elapsed:.1f} img/sec")
```

### Validate features
```python
features = pipeline.process(image)
print(f"Shape: {features.shape}")
print(f"Type: {features.dtype}")
print(f"Range: [{features.min()}, {features.max()}]")
print(f"NaNs: {np.isnan(features).sum()}")
print(f"Infs: {np.isinf(features).sum()}")
```

## File Organization

```
TDSPyProject/
├── preprocessing/                   # Implementation package — the single public import path
│   ├── __init__.py                  # Public API: re-exports the whole surface (see __all__)
│   ├── transforms.py                # grayscale / resize / normalize / denoise
│   ├── vectorize.py                 # vectorize_image (optional step)
│   ├── reduce.py                    # reduce_dimensions (vector + matrix)
│   ├── pipeline.py                  # ImagePipeline, batch_process, compose
│   └── io.py                        # image loaders
├── prebuilt_pipelines.py            # Named ready-made pipelines
├── integration_example.py           # Data/feature/train/eval helpers
├── download_dataset.py              # Downloads the deepfake-vs-real-20k images into datasets/
├── create_split.py                  # Scans Real/Deepfake folders -> datasets/dataset_split.csv (70/15/15)
├── extract_features.py              # Manifest-driven image streamer (reads datasets/dataset_split.csv)
├── tests/                           # pytest suite
└── Docs/
    ├── IMAGE_PREPROCESSING_GUIDE.md # Full documentation
    └── QUICK_REFERENCE.md           # This file
```

## Import Everything You Need

```python
from preprocessing import (
    # Core functions
    vectorize_image,
    normalize_image,
    resize_image,
    to_grayscale,
    reduce_noise,
    reduce_dimensions,   # vector + matrix dimensionality reduction

    # Pipeline classes
    ImagePipeline,
    
    # Composition utilities
    compose,
    pipeline_decorator,
    
    # Batch utilities
    batch_process,
    load_image_from_file,
    load_image_from_pil
)
```

## Prebuilt Pipeline Templates (from prebuilt_pipelines.py)

```python
from prebuilt_pipelines import PrebuiltPipelines

# --- Vector outputs (classical models) ---
fast = PrebuiltPipelines.fast_pipeline()            # 64×64 -> 4,096
balanced = PrebuiltPipelines.svm_pipeline()         # 128×128 -> 16,384
hq = PrebuiltPipelines.hq_pipeline()                # 224×224 -> 50,176
no_denoise = PrebuiltPipelines.no_denoise_pipeline()

# --- Vector reduction ---
bypass = PrebuiltPipelines.reduction_bypass_pipeline()   # 'reduce' slot = None
vec_pca = PrebuiltPipelines.vec_pca_pipeline(128)        # -> (n, 128)
vec_jl = PrebuiltPipelines.vec_jl_pipeline(256)          # -> (n, 256)

# --- Matrix reduction (CNN / ViT) ---
mat_pca = PrebuiltPipelines.mat_pca_pipeline(32)         # -> (n, 128, 32)
mat_jl = PrebuiltPipelines.mat_jl_pipeline(64)           # -> (n, 128, 64)
```

## Getting Help

```python
# Function docstring
from preprocessing import vectorize_image
help(vectorize_image)

# Class methods
from preprocessing import ImagePipeline
help(ImagePipeline.process)

# All available operations
print(ImagePipeline.OPERATIONS.keys())

# Pipeline string representation
pipeline = ImagePipeline([...])
print(pipeline)
```

## Testing Your Installation

```bash
# Run the full pytest suite
pytest
```

This exercises every transform, vectorization, both reduction subgroups, and
the composition patterns, and should complete without errors.

---

**For detailed documentation, see:** `IMAGE_PREPROCESSING_GUIDE.md`  
**For ready-made pipelines, see:** `prebuilt_pipelines.py`  
**For data/train/eval helpers, see:** `integration_example.py`
