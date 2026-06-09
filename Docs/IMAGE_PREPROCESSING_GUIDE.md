# Image Preprocessing Pipeline - Usage Guide

## Overview

The `image_preprocessing.py` module provides a production-ready, modular image preprocessing pipeline designed for traditional ML models (SVM, Random Forest, etc.). It features:

✅ **Flexible image vectorization** — flat pixel vectors or VGG16 deep embeddings  
✅ **4 powerful preprocessing operations** (normalization, resizing, grayscale, denoising)  
✅ **3 composition patterns** for flexible pipeline configuration  
✅ **Full type hints and comprehensive docstrings**  
✅ **Batch processing support**  
✅ **Production-ready error handling**

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

These are typically already in your project environment if you're using computer vision tasks.

---

## Core Function: `vectorize_image()`

Converts 2D/3D image arrays into 1D feature vectors for ML models.

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
from image_preprocessing import vectorize_image

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
from image_preprocessing import ImagePipeline

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
from image_preprocessing import normalize_image

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
from image_preprocessing import resize_image

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
from image_preprocessing import to_grayscale

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
from image_preprocessing import reduce_noise

# Bilateral filtering (recommended)
denoised = reduce_noise(image, method='bilateral', kernel_size=5)

# Gaussian blur
smooth = reduce_noise(image, method='gaussian', kernel_size=7)

# Median filtering for salt-and-pepper noise
median_filtered = reduce_noise(image, method='median', kernel_size=5)
```

---

## Composition Patterns

### Pattern 1: Pipeline Class (Recommended for Config-driven Workflows)

Best when: You want to configure pipelines via config files or during runtime.

```python
from image_preprocessing import ImagePipeline
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
from image_preprocessing import compose, vectorize_image, normalize_image, resize_image, to_grayscale
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
from image_preprocessing import pipeline_decorator, to_grayscale, resize_image, vectorize_image
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

### 1. In `extractfeatures.py`

Add the pipeline to your feature extraction workflow:

```python
from image_preprocessing import ImagePipeline
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

# Use in your existing pipeline
for url, label in get_data_stream():
    image = fetch_image(url)
    features = extract_image_features(image)
    # ... train model
```

### 2. For Batch Processing

```python
from image_preprocessing import batch_process, ImagePipeline
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
from image_preprocessing import ImagePipeline
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
from image_preprocessing import ImagePipeline, batch_process
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
from image_preprocessing import ImagePipeline

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
from image_preprocessing import ImagePipeline, vectorize_image

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
from image_preprocessing import ImagePipeline
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

Run the included examples:

```bash
python image_preprocessing.py
```

This will output shape and value range information for:
- Individual function demonstrations
- Pipeline class usage
- Functional composition
- Batch processing

---

## API Reference

### Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `vectorize_image()` | Convert image to 1D vector (flat or vgg16) | np.ndarray (1D) |
| `normalize_image()` | Normalize pixel values | np.ndarray (float32) |
| `resize_image()` | Resize image | np.ndarray (resized) |
| `to_grayscale()` | Convert to grayscale | np.ndarray (2D) |
| `reduce_noise()` | Denoise image | np.ndarray (denoised) |
| `compose()` | Functional composition | Callable |
| `pipeline_decorator()` | Decorator for pipelines | Callable (decorator) |
| `batch_process()` | Process multiple images | np.ndarray (batch) |

### Classes

| Class | Purpose |
|-------|---------|
| `ImagePipeline` | Composable operation chaining |

---

## Questions & Support

For issues or customization needs, refer to the comprehensive docstrings in the module:

```python
from image_preprocessing import ImagePipeline
help(ImagePipeline)
help(ImagePipeline.process)
```

---

**Version:** 1.1  
**Last Updated:** 2026-06-09  
**Production Ready:** ✅
