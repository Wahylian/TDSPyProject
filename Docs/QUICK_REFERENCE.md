# Image Preprocessing Pipeline - Quick Reference

## One-Liner Installation
```bash
pip install opencv-python scikit-image
```

## Quickstart (3 lines)
```python
from image_preprocessing import ImagePipeline
pipeline = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (64, 64)}), ('normalize', {'method': 'minmax'}), ('flatten', {})])
features = pipeline.process(image)  # (4096,) float32 vector
```

## 3 Ways to Compose

### 1. Pipeline Class
```python
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (64, 64)}),
    ('normalize', {'method': 'minmax'}),
    ('flatten', {})
])
features = pipeline.process(image)
```

### 2. Functional Composition
```python
from functools import partial
from image_preprocessing import compose
pipeline = compose(
    flatten_image,
    partial(normalize_image, method='minmax'),
    partial(resize_image, target_size=(64, 64)),
    to_grayscale
)
features = pipeline(image)
```

### 3. Decorator
```python
from image_preprocessing import pipeline_decorator, to_grayscale, resize_image, flatten_image

@pipeline_decorator(
    (to_grayscale, {}),
    (lambda x: resize_image(x, (64, 64)), {}),
    (flatten_image, {})
)
def extract_features(image):
    return image

features = extract_features(image)
```

## Common Patterns

### SVM Training Pipeline
```python
from image_preprocessing import ImagePipeline
from sklearn.svm import SVC

# Preprocess
pipeline = ImagePipeline([
    ('grayscale', {}),
    ('resize', {'target_size': (128, 128)}),
    ('denoise', {'method': 'bilateral'}),
    ('normalize', {'method': 'minmax'}),
    ('flatten', {})
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
from image_preprocessing import batch_process, ImagePipeline

pipeline = ImagePipeline([...])
features_batch = batch_process(images_list, pipeline)
# Shape: (num_images, feature_dim)
```

### Multiple Sizes (Experiment)
```python
fast = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (64, 64)}), ('flatten', {})])
medium = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (128, 128)}), ('flatten', {})])
hq = ImagePipeline([('grayscale', {}), ('resize', {'target_size': (224, 224)}), ('flatten', {})])

for pipeline, name in [(fast, '64'), (medium, '128'), (hq, '224')]:
    features = batch_process(images, pipeline)
    score = train_and_evaluate(features, labels)
    print(f"{name}x{name}: {score}")
```

## Function Reference

### Main Operations

| Function | Default | Output |
|----------|---------|--------|
| `flatten_image(img)` | - | 1D array (float32) |
| `normalize_image(img, 'minmax')` | minmax | Normalized [0, 1] |
| `resize_image(img, (64,64))` | - | Resized array |
| `to_grayscale(img)` | - | 2D grayscale |
| `reduce_noise(img, 'bilateral')` | bilateral | Denoised array |

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
image_preprocessing.flatten_image(pil_image)

# ✅ Correct: convert to numpy array
import numpy as np
image_preprocessing.flatten_image(np.array(pil_image))
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
    ('flatten', {})
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
    ('flatten', {})
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
    ('flatten', {})
])

image = cv2.imread('test.jpg')
print(f"Original: {image.shape}, dtype={image.dtype}")

gray = to_grayscale(image)
print(f"After grayscale: {gray.shape}")

resized = resize_image(gray, (64, 64))
print(f"After resize: {resized.shape}")

flat = flatten_image(resized)
print(f"After flatten: {flat.shape}, range=[{flat.min()}, {flat.max()}]")
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
your_project/
├── image_preprocessing.py           # Main module
├── IMAGE_PREPROCESSING_GUIDE.md     # Full documentation
├── integration_example.py           # Working examples
├── PREPROCESSING_DELIVERABLE.md     # Completion report
├── QUICK_REFERENCE.md              # This file
├── extractfeatures.py              # Your feature extraction
└── datasets/
    └── images/
```

## Import Everything You Need

```python
from image_preprocessing import (
    # Core functions
    flatten_image,
    normalize_image,
    resize_image,
    to_grayscale,
    reduce_noise,
    
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

## Prebuilt Pipeline Templates (from integration_example.py)

```python
from integration_example import PrebuiltPipelines

# Fast (64×64)
fast = PrebuiltPipelines.fast_pipeline()

# Balanced (128×128)
balanced = PrebuiltPipelines.svm_pipeline()

# High Quality (224×224)
hq = PrebuiltPipelines.hq_pipeline()

# Custom without denoising
no_denoise = PrebuiltPipelines.no_denoise_pipeline()
```

## Getting Help

```python
# Function docstring
from image_preprocessing import flatten_image
help(flatten_image)

# Class methods
from image_preprocessing import ImagePipeline
help(ImagePipeline.process)

# All available operations
print(ImagePipeline.OPERATIONS.keys())

# Pipeline string representation
pipeline = ImagePipeline([...])
print(pipeline)
```

## Testing Your Installation

```bash
# Test all functions
python image_preprocessing.py

# Test integration example
python integration_example.py
```

Both should complete without errors and show example outputs.

---

**For detailed documentation, see:** `IMAGE_PREPROCESSING_GUIDE.md`  
**For examples, see:** `integration_example.py`  
**For full report, see:** `PREPROCESSING_DELIVERABLE.md`
