"""
preprocessing — Unified public API for the image-preprocessing package.
=======================================================================

This package is the **single front door** to every image-preprocessing
capability in the project. Application code should import *only* from here::

    from preprocessing import ImagePipeline, batch_process, vectorize_image

and never reach into the internal submodules directly. The heavy lifting is
split across several focused submodules for maintainability, while this
``__init__`` presents one small, stable, well-documented surface to the rest of
the codebase.

--------------------------------------------------------------------------------
Why a single package API?
--------------------------------------------------------------------------------
The implementation is deliberately broken into cohesive submodules. This
``__init__`` re-exports their public symbols so that:

  * downstream code has **one import path** to remember (`preprocessing`);
  * the internal package layout can be reorganized freely (split a module, rename
    a file, move a function) **without breaking a single caller**, as long as this
    package keeps re-exporting the same names;
  * the public API is made **explicit and discoverable** via ``__all__`` below —
    anything not listed there is an internal detail and may change without notice.

--------------------------------------------------------------------------------
Routing map — where each exported symbol actually lives
--------------------------------------------------------------------------------
The package imports directly from its internal submodules, so the routing is
explicit and there is exactly one hop from this file to the real implementation:

    Symbol                     Submodule                     Responsibility
    -------------------------  ----------------------------  --------------------------------
    to_grayscale               preprocessing/transforms.py   Per-image: color -> grayscale
    resize_image               preprocessing/transforms.py   Per-image: resize (+aspect/pad)
    normalize_image            preprocessing/transforms.py   Per-image: pixel-value scaling
    reduce_noise               preprocessing/transforms.py   Per-image: denoising filters
    vectorize_image            preprocessing/vectorize.py    Per-image: image -> 1D vector (optional)
    reduce_dimensions          preprocessing/reduce.py       Batch-level: vector / matrix / bypass
    ImagePipeline              preprocessing/pipeline.py     Config-driven op chain
    batch_process              preprocessing/pipeline.py     Run a pipeline over many images
    compose                    preprocessing/pipeline.py     Right-to-left functional compose
    pipeline_decorator         preprocessing/pipeline.py     Decorator-style preprocessing
    BATCH_LEVEL_OPS            preprocessing/pipeline.py     Set of ops that run on a batch
    load_image_from_bytes      preprocessing/io.py           I/O: raw bytes  -> ndarray (RGB)
    load_image_from_file       preprocessing/io.py           I/O: file path  -> ndarray (BGR)
    load_image_from_pil        preprocessing/io.py           I/O: PIL.Image  -> ndarray

--------------------------------------------------------------------------------
How to use this API
--------------------------------------------------------------------------------
1. Load an image into a NumPy array::

       from preprocessing import load_image_from_file
       img = load_image_from_file("cat.jpg")            # BGR, as OpenCV loads it

2. Build a declarative pipeline and run it on one image::

       from preprocessing import ImagePipeline
       pipeline = ImagePipeline([
           ('grayscale', {'force': True}),
           ('resize',    {'target_size': (64, 64)}),
           ('normalize', {'method': 'minmax'}),
           ('vectorize', {}),                              # optional — omit for matrices
           ('reduce',    {'method': 'vec-pca', 'n_components': 64}),  # optional
       ])
       features = pipeline.process(img)                  # per-image steps only

3. Run the same pipeline across a batch. ``batch_process`` fits a *fresh*
   batch-level ``'reduce'`` reducer on the images it is given — handy for a
   one-shot transform::

       from preprocessing import batch_process
       X = batch_process([img] * 200, pipeline)          # (200, n_components)

4. For a train/val/test workflow, fit the ``'reduce'`` projection once on the
   training batch and reuse it on held-out data with the scikit-learn-style
   ``fit_transform`` / ``transform`` methods::

       X_train = pipeline.fit_transform(train_images)    # fits PCA/JL on train
       X_val   = pipeline.transform(val_images)          # reuses the train basis
       X_test  = pipeline.transform(test_images)         # reuses the train basis

   After fitting, ``pipeline.process(single_image)`` also reuses the stored
   projection, so single-image inference matches the batched output.

Vectorize is optional:
    Include ``'vectorize'`` to get flat vectors for classical models (SVM,
    Random Forest); pair it with a ``'vec-*'`` reduction method. Omit it to keep
    each image a 2D matrix for CNNs/ViTs; pair that with a ``'mat-*'`` reduction
    method, which compresses the matrix while preserving its row structure.

Per-image vs batch-level routing (important):
    Every transform except ``'reduce'`` operates on a single image. ``'reduce'``
    needs the whole batch to fit, so:
      * ``pipeline.process(image)`` treats ``('reduce', {'method': None})`` as a
        no-op. For a fitting method on a lone image it raises a helpful
        ``ValueError`` *unless the pipeline has already been fitted* (via
        ``fit`` / ``fit_transform``), in which case it reuses the stored reducer.
      * ``batch_process(images, pipeline)`` splits the chain, runs per-image ops
        on each image, stacks the results, then fits a fresh reducer once over
        the full batch. Use this for a one-shot batch transform.
      * ``pipeline.fit_transform(train)`` then ``pipeline.transform(val/test)``
        fit the reducer once and reuse that same projection across splits — the
        right choice whenever a fitting ``'reduce'`` step must be consistent
        across train/val/test (e.g. before training a classifier).

--------------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------------
    - numpy
    - opencv-python (cv2)
    - scikit-image
    - Pillow
    - tensorflow / keras   (optional; only for method='vgg16' embeddings)
    - scikit-learn         (optional; only for the 'vec-pca' / 'vec-jl' reductions —
                            the 'mat-pca' / 'mat-jl' matrix reductions are pure NumPy)
"""

# =============================================================================
# Public API imports — routed directly to the internal submodules.
#
# Grouping the imports by source submodule keeps the routing obvious: each block
# below maps one-to-one to a submodule of this package. To add a new public
# symbol, import it in the matching block and list it in ``__all__``.
# =============================================================================

# --- Per-image transforms ......................... preprocessing/transforms.py
from preprocessing.transforms import (
    normalize_image,
    reduce_noise,
    resize_image,
    to_grayscale,
)

# --- Vectorization (image -> 1D feature vector) ... preprocessing/vectorize.py
from preprocessing.vectorize import vectorize_image

# --- Batch-level dimensionality reduction ......... preprocessing/reduce.py
from preprocessing.reduce import reduce_dimensions

# --- Pipeline composition & batching .............. preprocessing/pipeline.py
from preprocessing.pipeline import (
    BATCH_LEVEL_OPS,
    ImagePipeline,
    batch_process,
    compose,
    pipeline_decorator,
)

# --- Image I/O helpers ............................ preprocessing/io.py
from preprocessing.io import (
    load_image_from_bytes,
    load_image_from_file,
    load_image_from_pil,
)

# ``_vgg16_models`` is the module-level VGG16 weight cache — a private
# implementation detail (note the leading underscore) deliberately kept out of
# ``__all__``. It is re-exported here so tests and advanced callers can seed or
# inspect the cache directly.
from preprocessing.vectorize import _vgg16_models  # noqa: F401

# =============================================================================
# Explicit public API.
#
# ``__all__`` defines exactly what ``from preprocessing import *`` exposes and
# documents the supported surface. Symbols not listed here (e.g. the
# ``_vgg16_models`` cache) are internal and may change without notice.
# =============================================================================
__all__ = [
    # transforms (per-image)
    'to_grayscale',
    'resize_image',
    'normalize_image',
    'reduce_noise',
    # vectorize (per-image)
    'vectorize_image',
    # reduce (batch-level)
    'reduce_dimensions',
    # pipeline / composition
    'ImagePipeline',
    'batch_process',
    'compose',
    'pipeline_decorator',
    'BATCH_LEVEL_OPS',
    # io
    'load_image_from_bytes',
    'load_image_from_file',
    'load_image_from_pil',
]
