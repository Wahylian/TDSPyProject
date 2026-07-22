"""Shared pytest fixtures for the test suite.

All data is synthetic, in-memory, and seeded so tests run in milliseconds
without touching disk, network, or the VGG16 weights. Heavy/optional paths are
mocked (see _FakeVGG16 and the fake_vgg16 fixture).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest


# Fixed seed so every fixture-derived array is reproducible across runs.
SEED = 20260611


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy random Generator for reproducible test data."""
    return np.random.default_rng(SEED)


@pytest.fixture
def color_image(rng) -> np.ndarray:
    """A standard 224x224x3 uint8 colour image (the canonical pipeline input)."""
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def gray_image(rng) -> np.ndarray:
    """A 224x224 uint8 single-channel image."""
    return rng.integers(0, 256, size=(224, 224), dtype=np.uint8)


@pytest.fixture
def small_color_image(rng) -> np.ndarray:
    """A tiny 32x32x3 image — keeps pipeline/vectorize tests fast."""
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.fixture
def image_batch(rng) -> List[np.ndarray]:
    """A list of 6 same-shaped colour images, suitable for batch_process."""
    return [
        rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        for _ in range(6)
    ]


@pytest.fixture
def feature_matrix(rng) -> np.ndarray:
    """A (40, 256) float32 feature matrix for vector reduce_dimensions tests."""
    return rng.random(size=(40, 256), dtype=np.float64).astype(np.float32)


@pytest.fixture
def matrix_stack(rng) -> np.ndarray:
    """A (24, 32, 32) float32 grayscale image stack for the mat-* reducers."""
    return rng.random(size=(24, 32, 32), dtype=np.float64).astype(np.float32)


@pytest.fixture
def color_matrix_stack(rng) -> np.ndarray:
    """A (24, 32, 32, 3) float32 colour image stack for the mat-* colour path."""
    return rng.random(size=(24, 32, 32, 3), dtype=np.float64).astype(np.float32)


@pytest.fixture
def tmp_image_file(tmp_path, color_image) -> str:
    """Write a real PNG to a temp dir and return its path, for load_image_from_file."""
    import cv2

    path = os.path.join(str(tmp_path), "sample.png")
    cv2.imwrite(path, color_image)
    return path


class _FakeVGG16:
    """Stub keras VGG16(include_top=False): returns a (n, 7, 7, 512) random map.

    Reproduces the real block5_pool output shape (25,088 flattened) cheaply,
    without loading ImageNet weights.
    """

    output_size = 7 * 7 * 512  # 25,088, matching block5_pool for 224x224

    def predict(self, batch, verbose=0):  # noqa: D401 - mimics keras signature
        n = batch.shape[0]
        return np.random.default_rng(0).random((n, 7, 7, 512)).astype(np.float32)


@pytest.fixture
def fake_vgg16(monkeypatch):
    """Seed the VGG16 cache with the stub so method='vgg16' skips keras entirely.

    Returns the expected flattened output length for assertions.
    """
    from preprocessing import vectorize

    monkeypatch.setitem(vectorize._vgg16_models, (224, 224), _FakeVGG16())
    return _FakeVGG16.output_size


# trainbase fixtures: tiny seeded model-ready data so fitting/grid search stay instant.


@pytest.fixture
def feature_split(rng) -> SimpleNamespace:
    """A tiny linearly-separable 2-class train/val/test feature split.

    Tight 5-D Gaussian blobs at -1.5 (class 0) and +1.5 (class 1), float32
    features and int labels, sized 16/8/10 to keep GridSearchCV instant.
    """

    def block(center: float, n: int) -> np.ndarray:
        # Tight blob (std 0.4) keeps the classes well separated.
        return (rng.standard_normal((n, 5)).astype(np.float32) * 0.4 + center)

    def split(n_per: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([block(-1.5, n_per), block(1.5, n_per)]).astype(np.float32)
        y = np.array([0] * n_per + [1] * n_per, dtype=int)
        return X, y

    X_train, y_train = split(8)   # 16 rows
    X_val, y_val = split(4)       #  8 rows
    X_test, y_test = split(5)     # 10 rows
    return SimpleNamespace(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


@pytest.fixture
def pixel_split(rng) -> SimpleNamespace:
    """Tiny separable flat-pixel splits for the torch models.

    Emulates the pixels pipeline: flat 8x8 grayscale vectors in [0, 1], class 0
    dim (~0.2) and class 1 bright (~0.8), so a small CNN/ViT separates them fast.
    """
    side = 8
    f = side * side

    def block(level: float, n: int) -> np.ndarray:
        x = rng.normal(level, 0.05, size=(n, f)).astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    def split(n: int):
        X = np.vstack([block(0.2, n), block(0.8, n)]).astype(np.float32)
        y = np.array([0] * n + [1] * n, dtype=int)
        return X, y

    X_train, y_train = split(8)
    X_val, y_val = split(4)
    X_test, y_test = split(4)
    return SimpleNamespace(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        image_shape=(1, side, side),
    )


@pytest.fixture
def image_label_pairs(rng) -> List[Tuple[np.ndarray, int]]:
    """7 (uint8 BGR image, int label) pairs backing a mocked feature stream.

    Mirrors get_feature_stream so load_images can be tested without a manifest.
    Seven pairs lets a max_samples cap fall strictly inside the stream.
    """
    return [
        (rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8), i % 2)
        for i in range(7)
    ]
