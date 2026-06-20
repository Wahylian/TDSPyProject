# Image Preprocessing Pipeline - Usage Guide

## Overview

The image preprocessing layer provides a production-ready, modular pipeline serving both classical ML models (SVM, Random Forest) and matrix-native models (CNNs, ViTs). It features:

✅ **Optional image vectorization** — flatten to vectors for classical models, or omit it to keep matrices for CNNs/ViTs  
✅ **4 powerful per-image preprocessing operations** (normalization, resizing, grayscale, denoising)  
✅ **Optional dimensionality reduction in two subgroups** — vector (`vec-pca` / `vec-jl`) and matrix (`mat-pca` / `mat-jl`), plus a `None` bypass  
✅ **3 composition patterns** for flexible pipeline configuration  
✅ **Full type hints and comprehensive docstrings**  
✅ **Batch processing support**  
✅ **Production-ready error handling**

---

## File organization

The implementation lives in a small `preprocessing/` package. Import **only** from the `preprocessing` package — its `__init__.py` re-exports the whole public API from one stable path.

```
TDSPyProject/
└── preprocessing/                # Implementation package — the single public import path
    ├── __init__.py               # Public API: re-exports everything below (see __all__)
    ├── transforms.py             # to_grayscale, resize_image, normalize_image, reduce_noise
    ├── vectorize.py              # vectorize_image (flat + VGG16) — optional step
    ├── reduce.py                 # reduce_dimensions (vector + matrix subgroups, + bypass)
    ├── scale.py                  # standardize_features (per-feature zero-mean / unit-variance)
    ├── pipeline.py               # ImagePipeline, batch_process, compose, pipeline_decorator
    └── io.py                     # load_image_from_{bytes,file,pil}
```

Import the public API from the package:

```python
# Public API (preferred — one path to remember):
from preprocessing import ImagePipeline, vectorize_image, batch_process

# A specific submodule can still be imported directly when needed:
from preprocessing.transforms import to_grayscale
```

---

## Pipeline architecture

The pipeline distinguishes two kinds of operations:

| Kind | Examples | Applied by |
|------|----------|------------|
| **Per-image** | `grayscale`, `resize`, `denoise`, `normalize`, `vectorize` | `ImagePipeline.process(image)` — runs on each image independently |
| **Batch-level** | `reduce` (`vec-*` / `mat-*`), `scale` | `batch_process(images, pipeline)` — runs once on the stacked batch |

`batch_process` splits a pipeline into its per-image and batch-level segments, applies the per-image ops to each image, and stacks the results. **Vectorization is optional** and decides the stacked shape:

- **With** a `'vectorize'` step, each image is a 1D vector, so the stack is a `(n_samples, n_features)` matrix → pair with a **vector** reduce method (`vec-pca` / `vec-jl`).
- **Without** `'vectorize'`, each image stays a matrix, so the stack is a `(n_samples, height, width)` grayscale stack (or `(n_samples, height, width, channels)` if `'grayscale'` is also omitted) → pair with a **matrix** reduce method (`mat-pca` / `mat-jl`), which preserves the channel axis.

The batch-level ops then run once on that stack, in order: `'reduce'` (if present) projects it, and `'scale'` (if present) standardizes the resulting flat features per column. Pipelines with neither simply return the stacked vectors or matrices.

Both batch-level ops **learn statistics across samples** (a projection for `reduce`, per-feature mean/std for `scale`), so for a train/val/test split fit them once on train and reuse them — use `ImagePipeline.fit_transform(train)` then `ImagePipeline.transform(val/test)` (see **Train/test correctness** under `reduce_dimensions()`), since `batch_process` refits on every call.

---

## Installation

### Dependencies

```bash
pip install opencv-python scikit-image numpy pillow
```

For VGG16 embeddings, also install TensorFlow:

```bash
pip install tensorflow
```

For the **vector** dimensionality-reduction methods (`vec-pca` / `vec-jl`):

```bash
pip install scikit-learn
```

`scikit-learn` is lazy-imported only when `method='vec-pca'` or `method='vec-jl'`. The **matrix** methods (`mat-pca` / `mat-jl`) are pure NumPy and need no extra install, and the `None` bypass (the default) needs nothing either.

---

## Core Function: `vectorize_image()`

Converts 2D/3D image arrays into 1D feature vectors for ML models. This step is
**optional**: include it to feed classical vector models (SVM, Random Forest);
omit it to keep each image as a 2D/3D matrix for CNNs and ViTs.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_array` | np.ndarray | - | Input image (2D grayscale or 3D color) |
| `method` | str | `'flat'` | `'flat'` for raw pixels, `'vgg16'` for VGG16 embeddings |
| `preserve_structure` | bool | False | For `method='flat'`, flatten each channel separately |
| `input_size` | tuple | `(224, 224)` | For `method='vgg16'`, target (height, width) |

### Methods

| Method | Output Size (224×224 RGB) | Best For |
|--------|---------------------------|----------|
| `'flat'` | 150,528 (H×W×C pixels) | SVM, Random Forest, fast baselines |
| `'vgg16'` | 25,088 (7×7×512 from block5_pool) | Transfer learning, richer semantic features |

### Examples

```python
import numpy as np
from preprocessing import vectorize_image

image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Flat pixel vectorization (default)
features = vectorize_image(image)
print(features.shape)  # (150528,)

# Grayscale image
gray_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
features_gray = vectorize_image(gray_image, method='flat')
print(features_gray.shape)  # (50176,)

# VGG16 embedding (requires: pip install tensorflow)
vgg_features = vectorize_image(image, method='vgg16')
print(vgg_features.shape)  # (25088,)
```

For VGG16 pipelines, resize first and skip `normalize_image()` — VGG16 applies its own preprocessing:

```python
from preprocessing import ImagePipeline

vgg_pipeline = ImagePipeline([
    ('resize', {'target_size': (224, 224), 'preserve_aspect': False}),
    ('vectorize', {'method': 'vgg16'})
])
features = vgg_pipeline.process(image)
```

---

## Preprocessing Functions

### 1. `normalize_image()` - Pixel Value Normalization

Scales pixel intensities to improve model convergence.

**Methods:**
- `'minmax'` (default): Scale to [0, 1]
- `'standard'`: Zero-mean, unit-variance
- `'histogram'`: Histogram equalization (grayscale only)

```python
from preprocessing import normalize_image

# MinMax normalization (0-1 range)
normalized = normalize_image(image, method='minmax')
print(normalized.min(), normalized.max())  # 0.0, 1.0

# Standard normalization
standardized = normalize_image(image, method='standard')
print(standardized.mean(), standardized.std())  # ~0.0, ~1.0

# Histogram equalization (grayscale)
gray = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
equalized = normalize_image(gray, method='histogram')
```

---

### 2. `resize_image()` - Flexible Image Resizing

Resize with optional aspect ratio preservation and multiple interpolation methods.

**Parameters:**
- `target_size`: (height, width)
- `preserve_aspect`: Maintain aspect ratio with padding
- `interpolation`: 'nearest', 'bilinear' (default), 'bicubic', 'lanczos'

```python
from preprocessing import resize_image

# Resize to 64x64 with aspect ratio preservation
resized = resize_image(image, (64, 64), preserve_aspect=True)
print(resized.shape)  # (64, 64, 3)

# Stretch to exact size without padding
stretched = resize_image(image, (128, 128), preserve_aspect=False, 
                         interpolation='bicubic')

# High-quality resize for small images
hq_resized = resize_image(image, (256, 256), interpolation='lanczos')
```

---

### 3. `to_grayscale()` - Grayscale Conversion

Convert color images to single-channel grayscale.

```python
from preprocessing import to_grayscale

# Convert RGB to grayscale
gray = to_grayscale(image)
print(gray.shape)  # (224, 224)

# Useful for reducing feature dimensionality
# RGB image: 224*224*3 = 150,528 features
# Grayscale: 224*224 = 50,176 features
```

---

### 4. `reduce_noise()` - Noise Reduction

Multiple denoising techniques for different noise types.

**Methods:**
- `'bilateral'` (default): Edge-preserving, best for most cases
- `'gaussian'`: Simple Gaussian blur
- `'morphological'`: Morphological opening/closing
- `'median'`: Median filter (good for salt-and-pepper noise)

```python
from preprocessing import reduce_noise

# Bilateral filtering (recommended)
denoised = reduce_noise(image, method='bilateral', kernel_size=5)

# Gaussian blur
smooth = reduce_noise(image, method='gaussian', kernel_size=7)

# Median filtering for salt-and-pepper noise
median_filtered = reduce_noise(image, method='median', kernel_size=5)
```

---

### 5. `reduce_dimensions()` — Dimensionality Reduction

Optional final pipeline step that compresses the feature representation before
training. It comes in **two subgroups** selected by `method`, so the same step
serves both classical vector models and matrix-native models:

#### Vector subgroup — for flat vectors `(n_samples, n_features)`

Pair these with a pipeline that **includes** `'vectorize'`.

| `method` value | What it does | Data-dependent? | Library |
|---|---|---|---|
| `'vec-pca'` | Principal Component Analysis: keeps directions of maximum variance. | Yes (fit on a batch). | `scikit-learn` |
| `'vec-jl'` | Gaussian random projection (`GaussianRandomProjection`). Distortion of pairwise distances bounded by the Johnson–Lindenstrauss lemma. | No (random matrix). | `scikit-learn` |

Output shape: `(n_samples, n_components)`.

#### Matrix subgroup — for image stacks (grayscale or colour)

Pair these with a pipeline that **omits** `'vectorize'` (each image stays a
matrix). Only the width (column) axis is reduced, so the row layout CNNs/ViTs
rely on — **and the colour channels** — are preserved. Both are pure NumPy — no
`scikit-learn` needed. Grayscale and multi-channel (BGR/RGB) stacks are both
accepted; the channel axis, when present, is carried through untouched:

| Input | Stack shape | Output shape |
|---|---|---|
| Grayscale (drop `vectorize`, keep `grayscale`) | `(n_samples, height, width)` | `(n_samples, height, n_components)` |
| Colour (drop both `grayscale` and `vectorize`) | `(n_samples, height, width, channels)` | `(n_samples, height, n_components, channels)` |

| `method` value | What it does | Data-dependent? | Library |
|---|---|---|---|
| `'mat-pca'` | Two-dimensional PCA: projects the column axis onto the leading eigenvectors of the batch column-covariance matrix. For colour input the covariance is pooled across channels, so one shared basis is applied to every channel. | Yes (fit on a batch). | none (NumPy) |
| `'mat-jl'` | Two-dimensional Gaussian random projection of the column axis, shared across channels. | No (random matrix). | none (NumPy) |

#### Bypass

| `method` value | What it does |
|---|---|
| `None` (default) or `'none'` | **Bypass** — returns the input unchanged (vector *or* matrix). A no-op slot for easy A/B comparison. |

**Aliases for ease of use:** matching is case-insensitive and treats `-`/`_` the
same, and the terse aliases `'pca'` → `'vec-pca'`, `'jl'` /
`'johnson-lindenstrauss'` → `'vec-jl'` are accepted, so configs stay concise.

**Important shape rules:**
- The `method` subgroup decides how the input rank is read:
  - Vector methods: 2D `(n_samples, n_features)` is a batch; a bare 1D vector is a single sample.
  - Matrix methods: a grayscale batch is 3D `(n_samples, height, width)` and a colour batch is 4D `(n_samples, height, width, channels)`; a single image is one rank lower (`(height, width)` grayscale or `(height, width, channels)` colour). A fitted reducer records whether it was trained on grayscale or colour, so it reads single-sample rank correctly at inference time.
- `vec-pca` needs `n_samples ≥ n_components`; it clamps `n_components` to `min(n_samples, n_features)`. The random projections clamp `n_components` to the input width.
- Inside `ImagePipeline.process(image)` (a single image):
  - `method=None` → pass-through.
  - A fitting method (`vec-*` / `mat-*`) without a pre-fit `reducer` → raises `ValueError` (cannot fit on one sample). Use `batch_process`, or pass an already-fitted `reducer` via the op kwargs.

**Direct usage:**

```python
from preprocessing import reduce_dimensions
import numpy as np

# --- Vector subgroup: flat (n_samples, n_features) features ---
X_train = np.random.rand(200, 4096).astype(np.float32)
X_test  = np.random.rand( 50, 4096).astype(np.float32)

# Bypass — no-op.
assert reduce_dimensions(X_train, method=None) is X_train

# PCA — fit on train, RE-APPLY to test using the same reducer.
X_train_pca, pca = reduce_dimensions(
    X_train, method='vec-pca', n_components=64,
    random_state=42, return_reducer=True,
)
X_test_pca = reduce_dimensions(X_test, reducer=pca)  # transform-only

# Johnson–Lindenstrauss random projection.
X_train_jl, jl = reduce_dimensions(
    X_train, method='vec-jl', n_components=256,
    random_state=42, return_reducer=True,
)
X_test_jl = reduce_dimensions(X_test, reducer=jl)

# --- Matrix subgroup: image stacks (n_samples, height, width) ---
imgs_train = np.random.rand(200, 128, 128).astype(np.float32)
imgs_test  = np.random.rand( 50, 128, 128).astype(np.float32)

# 2D-PCA — each image -> (128, 32), rows preserved.
mat_train, mat_pca = reduce_dimensions(
    imgs_train, method='mat-pca', n_components=32, return_reducer=True,
)
mat_test = reduce_dimensions(imgs_test, reducer=mat_pca)
print(mat_train.shape, mat_test.shape)  # (200, 128, 32) (50, 128, 32)

# --- Matrix subgroup, COLOUR: stacks (n_samples, height, width, channels) ---
rgb_train = np.random.rand(200, 128, 128, 3).astype(np.float32)
rgb_test  = np.random.rand( 50, 128, 128, 3).astype(np.float32)

# Width reduced, channels preserved -> each image (128, 32, 3).
rgb_r, rgb_pca = reduce_dimensions(
    rgb_train, method='mat-pca', n_components=32, return_reducer=True,
)
rgb_test_r = reduce_dimensions(rgb_test, reducer=rgb_pca)
print(rgb_r.shape, rgb_test_r.shape)  # (200, 128, 32, 3) (50, 128, 32, 3)
```

**Via `ImagePipeline` (the recommended end-to-end form):**

```python
from preprocessing import ImagePipeline, batch_process

# Vector pipeline: vectorize, then reduce the flat vectors.
vector_pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {}),
    ('reduce', {'method': 'vec-pca', 'n_components': 128, 'random_state': 42}),
])
X = batch_process(images, vector_pipeline)
print(X.shape)  # (n_samples, 128)

# Matrix pipeline: NO vectorize — reduce the grayscale image matrices directly.
matrix_pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
    ('normalize', {'method': 'minmax'}),
    ('reduce', {'method': 'mat-pca', 'n_components': 32, 'random_state': 42}),
])
M = batch_process(images, matrix_pipeline)
print(M.shape)  # (n_samples, 128, 32) — CNN/ViT-ready matrices

# Colour matrix pipeline: drop BOTH grayscale and vectorize — keep channels.
colour_pipeline = ImagePipeline([
    ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
    ('normalize', {'method': 'minmax'}),
    ('reduce', {'method': 'mat-pca', 'n_components': 32, 'random_state': 42}),
])
C = batch_process(images, colour_pipeline)
print(C.shape)  # (n_samples, 128, 32, 3) — channels preserved for CNN/ViT
```

**Choosing a method:**

| Situation | Recommended `method` |
|---|---|
| Classical model (SVM / RF) and you can afford one PCA fit on the training data. | `'vec-pca'` |
| Classical model, zero training cost, only need approximate distance preservation. | `'vec-jl'` |
| Matrix-native model (CNN / ViT), want to keep spatial structure while shrinking width, variance-preserving. | `'mat-pca'` |
| Matrix-native model, want a free, data-independent width reduction. | `'mat-jl'` |
| No reduction (full features) or a placeholder slot for A/B comparison. | `None` |

**Train/test correctness:**

Across train/test splits, **always fit the reducer on the training set and
re-apply it to the test set** — never refit per split. The pattern is identical
for both subgroups; only the method name and input shape differ:

```python
from preprocessing import ImagePipeline, batch_process, reduce_dimensions

# 1. Build a per-image-only sub-pipeline (drop the 'reduce' op).
full = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {}),
    ('reduce', {'method': 'vec-pca', 'n_components': 128}),
])
per_image = ImagePipeline(full.per_image_operations())

# 2. Run the per-image stages on both splits.
X_train_raw = batch_process(train_images, per_image)
X_test_raw  = batch_process(test_images,  per_image)

# 3. Fit the reducer on train, transform both with the *same* fitted reducer.
X_train, reducer = reduce_dimensions(
    X_train_raw, method='vec-pca', n_components=128,
    random_state=42, return_reducer=True,
)
X_test = reduce_dimensions(X_test_raw, reducer=reducer)
```

The same fit-on-train/reuse rule applies to the `'scale'` step (`standardize_features`),
which standardizes flat features per column for scale-sensitive models (e.g. SVM).
When a pipeline chains both — `… → ('reduce', …) → ('scale', {})` — the cleanest
way to honour the rule for *both* at once is the pipeline's own
`fit_transform` / `transform` pair, which fits every batch-level op on train and
reuses each on held-out data:

```python
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {}),
    ('reduce', {'method': 'vec-pca', 'n_components': 128, 'random_state': 42}),
    ('scale', {}),                       # zero-mean / unit-variance per feature
])
X_train = pipeline.fit_transform(train_images)   # fits PCA + scaler on train
X_val   = pipeline.transform(val_images)         # reuses both
X_test  = pipeline.transform(test_images)        # reuses both
```

Note `'scale'` (per-feature, across the dataset, after `reduce`) is distinct from
the per-image `normalize` step (within one image, before `vectorize`): they act on
different axes and are complementary, not redundant.

---

## Composition Patterns

### Pattern 1: Pipeline Class (Recommended for Config-driven Workflows)

Best when: You want to configure pipelines via config files or during runtime.

```python
from preprocessing import ImagePipeline
import numpy as np

# Create pipeline
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
    ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])

# Process images
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
features = pipeline.process(image)
print(features.shape)  # (4096,)

# Add operations dynamically
pipeline.add_operation('normalize', {'method': 'standard'})

# String representation for debugging
print(pipeline)
```

**Advantages:**
- Easy to configure via YAML/JSON
- Dynamic operation addition
- Clear, readable operation chain
- Good for experimentation

---

### Pattern 2: Functional Composition (Elegant, Pythonic)

Best when: You want explicit function composition without class overhead.

```python
from preprocessing import compose, vectorize_image, normalize_image, resize_image, to_grayscale
from functools import partial

# Define partial functions
gray = to_grayscale
resize_64 = partial(resize_image, target_size=(64, 64), preserve_aspect=True)
normalize = partial(normalize_image, method='minmax')
vectorize = vectorize_image

# Compose functions (applied right-to-left)
pipeline = compose(vectorize, normalize, resize_64, gray)

# Process
features = pipeline(image)
print(features.shape)  # (4096,)
```

**Advantages:**
- Functional programming style
- No class overhead
- Easy to understand flow
- Highly reusable

---

### Pattern 3: Pipeline Decorator (Function Wrapping)

Best when: You want to wrap custom feature extraction functions.

```python
from preprocessing import pipeline_decorator, to_grayscale, resize_image, vectorize_image
from functools import partial

@pipeline_decorator(
    (to_grayscale, {}),
    (lambda x: resize_image(x, (64, 64)), {}),
    (vectorize_image, {})
)
def extract_ml_features(image):
    """Custom feature extraction with automatic preprocessing."""
    return image  # Image already preprocessed by decorator

# Use
features = extract_ml_features(image)
print(features.shape)  # (4096,)
```

**Advantages:**
- Clean, declarative syntax
- Preprocessing transparent to function
- Good for API design

---

## Integration with Your Project

### 1. In `extract_features.py`

Add the pipeline to your feature extraction workflow. `get_feature_stream`
yields decoded BGR images read from local storage, one `(image, label)` pair at
a time:

```python
from preprocessing import ImagePipeline
from extract_features import get_feature_stream
import numpy as np

# Define your standard preprocessing pipeline
ML_PIPELINE = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])

def extract_image_features(image_array: np.ndarray) -> np.ndarray:
    """Extract 1D feature vector from image for ML model."""
    return ML_PIPELINE.process(image_array)

# Stream pre-downloaded images straight into the pipeline
for image, label in get_feature_stream("train", base_dir="./datasets"):
    features = extract_image_features(image)
    # ... train model
```

### 2. For Batch Processing

```python
from preprocessing import batch_process, ImagePipeline
import numpy as np

pipeline = ImagePipeline([...])

# Process multiple images at once
images = [load_image(path) for path in image_paths]
features_batch = batch_process(images, pipeline)
print(features_batch.shape)  # (num_images, feature_dim)

# Feed to sklearn models
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(features_batch, labels)
```

### 3. For SVM Training (Flat Features)

```python
from preprocessing import ImagePipeline
from sklearn.svm import SVC
import numpy as np

# Prepare pipeline optimized for SVM with flat pixel features
svm_pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('denoise', {'method': 'bilateral'}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])

# Extract features from training set
train_features = np.array([svm_pipeline.process(img) for img in train_images])
print(f"Feature matrix shape: {train_features.shape}")  # (n_samples, n_features)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(train_features, train_labels)

# Predict on test set
test_features = np.array([svm_pipeline.process(img) for img in test_images])
predictions = svm.predict(test_features)
```

### 4. For SVM Training (VGG16 Embeddings)

```python
from preprocessing import ImagePipeline, batch_process
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# VGG16 feature extraction pipeline
vgg_pipeline = ImagePipeline([
    ('resize', {'target_size': (224, 224), 'preserve_aspect': False}),
    ('vectorize', {'method': 'vgg16'})
])

train_features = batch_process(train_images, vgg_pipeline)
test_features = batch_process(test_images, vgg_pipeline)
print(f"Feature matrix shape: {train_features.shape}")  # (n_samples, 25088)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(train_scaled, train_labels)
predictions = svm.predict(test_scaled)
```

---

## Performance Considerations

### Feature Dimension Reduction

For faster training, reduce image size:

```python
# High resolution (many features)
pipeline_hq = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (224, 224)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])
# Output: 224*224 = 50,176 features

# Medium resolution (balanced)
pipeline_med = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])
# Output: 128*128 = 16,384 features

# Low resolution (fast training)
pipeline_fast = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('normalize', {'method': 'minmax'}),
    ('vectorize', {})
])
# Output: 64*64 = 4,096 features
```

### Flat vs VGG16 Feature Trade-offs

| Approach | Feature Dim | Speed | Semantic Richness |
|----------|-------------|-------|-------------------|
| Flat 64×64 grayscale | 4,096 | Fastest | Low (raw pixels) |
| Flat 224×224 RGB | 150,528 | Moderate | Low (raw pixels) |
| VGG16 embedding | 25,088 | Slower (GPU helps) | High (pre-trained CNN) |

VGG16 is preferred when you have limited labeled data and want transferable visual features. Flat vectorization is preferred for fast iteration and when images are already low-dimensional (e.g., 64×64 grayscale).

### Preprocessing Time Comparison

```python
import time
from preprocessing import ImagePipeline

pipelines = {
    'fast': ImagePipeline([
        ('grayscale', {}),
        ('resize', {'target_size': (64, 64)}),
        ('vectorize', {})
    ]),
    'balanced': ImagePipeline([
        ('grayscale', {}),
        ('resize', {'target_size': (128, 128)}),
        ('normalize', {'method': 'minmax'}),
        ('vectorize', {})
    ])
}

for name, pipeline in pipelines.items():
    start = time.time()
    for _ in range(100):
        pipeline.process(large_image)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s for 100 images")
```

---

## Error Handling

All functions include robust error handling:

```python
from preprocessing import ImagePipeline, vectorize_image

# Type checking
try:
    vectorize_image([1, 2, 3])  # Not an array
except TypeError as e:
    print(f"Error: {e}")

# Dimension checking
try:
    vectorize_image(np.ones((10, 10, 3, 2)))  # 4D array
except ValueError as e:
    print(f"Error: {e}")

# Pipeline operation errors
try:
    bad_pipeline = ImagePipeline([
        ('unknown_op', {})
    ])
except ValueError as e:
    print(f"Error: {e}")
```

---

## Advanced: Custom Pipeline Operations

Extend the pipeline with custom functions:

```python
from preprocessing import ImagePipeline
import numpy as np

def custom_edge_detection(image: np.ndarray) -> np.ndarray:
    """Custom edge detection operation."""
    import cv2
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(image, 100, 200)

# Create custom pipeline
class CustomPipeline(ImagePipeline):
    OPERATIONS = {**ImagePipeline.OPERATIONS}
    
    @staticmethod
    def custom_edges(image, **kwargs):
        return custom_edge_detection(image)

CustomPipeline.OPERATIONS['edges'] = CustomPipeline.custom_edges

# Use
pipeline = CustomPipeline([
    ('grayscale', {}),
    ('edges', {}),
    ('vectorize', {})
])
```

---

## Testing

The full behaviour of the API is covered by the pytest suite under `tests/`:

```bash
pytest
```

The suite asserts shapes, dtypes and value ranges for the individual transforms,
vectorization, both reduction subgroups (vector and matrix), the per-image vs
batch-level split, and the composition patterns.

---

## API Reference

### Functions

| Function | Purpose | Returns | Module |
|----------|---------|---------|--------|
| `vectorize_image()` | Convert image to 1D vector (flat or vgg16); optional step | np.ndarray (1D) | `preprocessing.vectorize` |
| `normalize_image()` | Normalize pixel values | np.ndarray (float32) | `preprocessing.transforms` |
| `resize_image()` | Resize image | np.ndarray (resized) | `preprocessing.transforms` |
| `to_grayscale()` | Convert to grayscale | np.ndarray (2D) | `preprocessing.transforms` |
| `reduce_noise()` | Denoise image | np.ndarray (denoised) | `preprocessing.transforms` |
| `reduce_dimensions()` | Reduce features — vector (`vec-pca`/`vec-jl`), matrix (`mat-pca`/`mat-jl`), or `None` bypass | np.ndarray, or (np.ndarray, reducer) | `preprocessing.reduce` |
| `standardize_features()` | Standardize flat features per column (zero-mean / unit-variance), fit-once/reuse | np.ndarray, or (np.ndarray, scaler) | `preprocessing.scale` |
| `compose()` | Functional composition | Callable | `preprocessing.pipeline` |
| `pipeline_decorator()` | Decorator for pipelines | Callable (decorator) | `preprocessing.pipeline` |
| `batch_process()` | Process multiple images (and apply batch-level reduce) | np.ndarray (batch) | `preprocessing.pipeline` |

### Classes

| Class | Purpose |
|-------|---------|
| `ImagePipeline` | Composable operation chaining |

---

## Questions & Support

For issues or customization needs, refer to the comprehensive docstrings in the module:

```python
from preprocessing import ImagePipeline
help(ImagePipeline)
help(ImagePipeline.process)
```

---

**Version:** 1.6  
**Last Updated:** 2026-06-20  
**Production Ready:** ✅

### Changelog

- **1.6 (2026-06-20)** — Added a batch-level `'scale'` step (`standardize_features`,
  pure NumPy) that standardizes flat features per column (zero-mean / unit-variance),
  fit on train and reused on val/test via the same fit-once/reuse protocol as
  `reduce_dimensions`. Chain it after `'reduce'` (`… → ('reduce', …) → ('scale', {})`)
  to condition features for scale-sensitive models (e.g. SVM) inside the pipeline,
  rather than with a separate scikit-learn `StandardScaler`.
- **1.5 (2026-06-17)** — The `preprocessing/` package `__init__.py` is now the
  single public API. The standalone `image_preprocessing.py` facade was removed;
  `__all__` and all re-exports moved into the package, so callers import directly
  from `preprocessing` (`from preprocessing import ImagePipeline, batch_process`).
- **1.4 (2026-06-12)** — Matrix reduction (`'mat-pca'` / `'mat-jl'`) now accepts
  multi-channel (BGR/RGB) image stacks. A colour stack
  `(n_samples, height, width, channels)` reduces to
  `(n_samples, height, n_components, channels)` — only the width axis is
  projected (one shared basis pooled across channels for `mat-pca`), so rows and
  colour channels are preserved for CNN/ViT inputs. Fitted reducers record
  whether they were trained on grayscale or colour and read single-sample rank
  accordingly. The vector subgroup is already channel-agnostic (vectorization
  folds channels into the feature axis).
- **1.3 (2026-06-12)** — Vectorization is now explicitly optional, and
  `reduce_dimensions()` is split into two subgroups: **vector** (`'vec-pca'` /
  `'vec-jl'`, on flat `(n_samples, n_features)` matrices) and **matrix**
  (`'mat-pca'` / `'mat-jl'`, on `(n_samples, height, width)` image stacks,
  reducing the width axis while preserving rows for CNN/ViT inputs). The matrix
  methods are pure NumPy. Short aliases `'pca'` → `'vec-pca'` and
  `'johnson-lindenstrauss'` → `'vec-jl'` are accepted.
- **1.2 (2026-06-10)** — Split `image_preprocessing.py` into the
  `preprocessing/` package (transforms / vectorize / reduce / pipeline / io) and
  added the `reduce_dimensions()` function and `'reduce'` pipeline op.
- **1.1 (2026-06-09)** — VGG16 vectorization support.
