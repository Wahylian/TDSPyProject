"""Feature front-end: turn a dataset split into a model-ready feature matrix.

build_feature_pipeline resolves the ImagePipeline (a registry name or a verbatim
JSON spec); it appends nothing, so any needed 'reduce'/'scale' steps must already
be in the pipeline. fit_features fits it on train (learning batch-level PCA and
scaling stats); transform_features reuses the fitted pipeline on held-out splits
so train/val/test stay in one feature space. Both cache output (and, for train,
the fitted pipeline), keyed by a caller prefix plus the per-split sample cap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import joblib

# Project building blocks (imported, not reimplemented).
from extract_features import get_feature_stream
from preprocessing import ImagePipeline

from .model_registry import RANDOM_STATE
from .pipeline_registry import PIPELINE_REGISTRY

logger = logging.getLogger(__name__)


def build_feature_pipeline(
    pipeline_name: Optional[str] = None,
    pipeline_spec: Optional[str] = None,
) -> ImagePipeline:
    """Resolve the unfitted feature pipeline from a registry name or a JSON spec.

    A given pipeline_spec wins and is parsed verbatim; otherwise the named
    registry factory runs. Either way nothing is appended.
    """
    if pipeline_spec is not None:
        return ImagePipeline(_parse_pipeline_spec(pipeline_spec))
    if pipeline_name is None:
        raise ValueError("Provide either a pipeline_name or a pipeline_spec.")
    return PIPELINE_REGISTRY[pipeline_name]()


def _parse_pipeline_spec(spec: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse a JSON spec into (operation_name, kwargs) tuples for ImagePipeline.

    Expects a non-empty JSON array of [operation_name, kwargs] pairs. Op-name
    validation is left to ImagePipeline; this only checks the pair structure.
    """
    try:
        raw = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--pipeline-spec is not valid JSON: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "--pipeline-spec must be a non-empty JSON array of "
            "[operation_name, kwargs] pairs."
        )

    operations: List[Tuple[str, Dict[str, Any]]] = []
    for index, item in enumerate(raw):
        # Each entry must be a 2-element pair: a string op name and a kwargs map.
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"--pipeline-spec entry {index} must be a [operation_name, "
                f"kwargs] pair, got: {item!r}"
            )
        name, kwargs = item
        if not isinstance(name, str):
            raise ValueError(
                f"--pipeline-spec entry {index}: operation name must be a "
                f"string, got: {name!r}"
            )
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"--pipeline-spec entry {index}: kwargs must be a JSON object, "
                f"got: {kwargs!r}"
            )
        operations.append((name, kwargs))
    return operations


# Decoded images held per streamed batch, bounding peak RAM regardless of split size.
FEATURE_BATCH_SIZE = 512


def _stream_feature_batches(
    split: str, max_samples: int, batch_size: int = FEATURE_BATCH_SIZE
) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
    """Yield (images, labels) chunks of at most batch_size from the feature stream.

    Groups the seeded, shuffled (image, label) stream into bounded batches so at
    most batch_size decoded images coexist in memory. max_samples (0 = all) caps
    the total across batches.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    total = 0
    for image, label in get_feature_stream(split, random_seed=RANDOM_STATE):
        images.append(image)
        labels.append(label)
        total += 1
        if len(images) >= batch_size:
            yield images, labels
            images, labels = [], []
        if max_samples and total >= max_samples:
            break
    if images:
        yield images, labels


def load_images(split: str, max_samples: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load one split's images and int labels into memory, subsampled to max_samples.

    The stream is seeded and shuffled, so taking the first max_samples (0 = all)
    is an unbiased subsample. Decoding is batched, but the full split is returned.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    for batch_images, batch_labels in _stream_feature_batches(split, max_samples):
        images.extend(batch_images)
        labels.extend(batch_labels)
        logger.info("  ... %d images loaded", len(images))

    if not images:
        raise RuntimeError(f"No usable images found for split '{split}'.")
    return images, np.asarray(labels, dtype=int)


def fit_features(
    pipeline: ImagePipeline,
    max_samples: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
) -> Tuple[ImagePipeline, np.ndarray, np.ndarray]:
    """Fit the pipeline on TRAIN and return (fitted_pipeline, X_train, y_train).

    fit_transform learns and stores any batch-level PCA/scaling stats for reuse
    on val/test. The fitted pipeline and train features are cached together,
    keyed by cache_prefix and the sample cap, so a rerun skips re-decoding.
    """
    feat_path, pipe_path = _train_cache_paths(cache_dir, cache_prefix, max_samples)
    if feat_path and feat_path.is_file() and pipe_path.is_file():
        logger.info("Loading cached fitted pipeline + train features from %s", feat_path)
        data = np.load(feat_path)
        return joblib.load(pipe_path), data["X"], data["y"]

    images, y = load_images("train", max_samples)
    logger.info("Fitting feature pipeline on %d train images...", len(images))
    X = pipeline.fit_transform(images)
    logger.info("  -> train features %s", X.shape)

    if feat_path:
        np.savez_compressed(feat_path, X=X, y=y)
        joblib.dump(pipeline, pipe_path)
        logger.info("Cached train features + fitted pipeline to %s", cache_dir)
    return pipeline, X, y


def transform_features(
    split: str,
    pipeline: ImagePipeline,
    max_samples: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a held-out split with the fitted pipeline, returning (X, y).

    Projects val/test with the same batch-level statistics learned on train.
    Features are cached per split.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_prefix}_{split}_n{max_samples}.npz"
        if cache_path.is_file():
            logger.info("Loading cached features for '%s' from %s", split, cache_path)
            data = np.load(cache_path)
            return data["X"], data["y"]

    # Transform in bounded batches and concatenate. Because the fitted
    # reducer/scaler is a fixed projection, per-batch results match the whole split.
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    total = 0
    for images, labels in _stream_feature_batches(split, max_samples):
        X_parts.append(pipeline.transform(images))
        y_parts.append(np.asarray(labels, dtype=int))
        total += len(images)

    if not X_parts:
        raise RuntimeError(f"No usable images found for split '{split}'.")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    logger.info("Transformed %d '%s' images with the fitted pipeline -> %s", total, split, X.shape)

    if cache_path is not None:
        np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def _train_cache_paths(
    cache_dir: Optional[Path], cache_prefix: str, max_samples: int
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return the (features, fitted-pipeline) cache paths for the train split."""
    if cache_dir is None:
        return None, None
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cache_prefix}_train_n{max_samples}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}_pipeline.joblib"
