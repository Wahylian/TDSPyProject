"""Feature front-end for the training pipeline: build, extract, cache.

This module owns everything between a dataset split and the model-ready feature
matrix:

  * :func:`build_feature_pipeline` resolves the *which transforms* question —
    either a named pipeline from :data:`PIPELINE_REGISTRY` or a fully custom one
    parsed verbatim from a JSON spec. The resulting :class:`ImagePipeline` is
    used exactly as defined; no reduction or scaling steps are appended. A
    pipeline that needs PCA / standardization (e.g. for a scale-sensitive SVM)
    is expected to carry those ``'reduce'`` / ``'scale'`` steps itself — the
    prebuilt registry pipelines do.
  * :func:`fit_features` fits that pipeline on the training split (learning any
    batch-level PCA basis and scaling statistics) and returns the reduced train
    features.
  * :func:`transform_features` reuses the *fitted* pipeline to project held-out
    splits into the same feature space, so train/val/test never diverge.

Both extraction steps cache their output (and, for train, the fitted pipeline)
so a rerun skips re-decoding images. The cache is keyed by a caller-supplied
prefix that must capture the feature-defining configuration (which pipeline /
spec), plus the per-split sample cap folded into each file name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

# Project building blocks (imported, not reimplemented).
from extract_features import get_feature_stream
from preprocessing import ImagePipeline

from .model_registry import RANDOM_STATE
from .pipeline_registry import PIPELINE_REGISTRY

logger = logging.getLogger(__name__)


# =============================================================================
# Feature pipeline construction
# =============================================================================
def build_feature_pipeline(
    pipeline_name: Optional[str] = None,
    pipeline_spec: Optional[str] = None,
) -> ImagePipeline:
    """Resolve the feature pipeline to use, from a registry name or a JSON spec.

    Exactly one source is used. When ``pipeline_spec`` is given it wins and is
    parsed into a custom :class:`ImagePipeline` *verbatim* (see
    :func:`_parse_pipeline_spec`); otherwise the named factory in
    :data:`PIPELINE_REGISTRY` is invoked. Either way the pipeline is returned as
    defined — the runner appends nothing, so any required ``'reduce'`` /
    ``'scale'`` steps must already be part of the pipeline.

    Args:
        pipeline_name: Key into :data:`PIPELINE_REGISTRY`. Used only when
            ``pipeline_spec`` is ``None``.
        pipeline_spec: A JSON string describing a custom pipeline as a list of
            ``[operation_name, kwargs]`` pairs (mirroring
            :class:`ImagePipeline`'s constructor). Overrides ``pipeline_name``.

    Returns:
        An unfitted :class:`ImagePipeline`.

    Raises:
        KeyError: If ``pipeline_name`` is not registered (and no spec is given).
        ValueError: If ``pipeline_spec`` is malformed or names an unknown op.
    """
    if pipeline_spec is not None:
        return ImagePipeline(_parse_pipeline_spec(pipeline_spec))
    if pipeline_name is None:
        raise ValueError("Provide either a pipeline_name or a pipeline_spec.")
    return PIPELINE_REGISTRY[pipeline_name]()


def _parse_pipeline_spec(spec: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse a custom-pipeline JSON spec into ``ImagePipeline`` operations.

    The expected structure mirrors :class:`ImagePipeline`'s constructor: a JSON
    array whose every element is a two-item ``[operation_name, kwargs]`` pair,
    e.g.::

        [["grayscale", {}],
         ["resize", {"target_size": [128, 128], "preserve_aspect": true}],
         ["normalize", {"method": "minmax"}],
         ["vectorize", {}]]

    JSON arrays in the kwargs (such as ``target_size``) stay as lists, which the
    transforms accept. Operation-name validation is left to
    :class:`ImagePipeline`, which raises on an unknown op.

    Args:
        spec: The JSON string from ``--pipeline-spec``.

    Returns:
        A list of ``(operation_name, kwargs)`` tuples ready for
        :class:`ImagePipeline`.

    Raises:
        ValueError: If the JSON is invalid or does not match the expected
            list-of-``[str, dict]``-pairs structure.
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


# =============================================================================
# Data loading & feature extraction
# =============================================================================
def load_images(split: str, max_samples: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Stream one split's raw images + labels into memory (subsampled).

    Images come from :func:`extract_features.get_feature_stream`, which yields
    ``(image, label)`` pairs in a seeded, shuffled order — so taking the first
    ``max_samples`` items is an unbiased random subsample.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_samples: Cap on the number of images to read; ``0`` means "all".

    Returns:
        ``(images, y)`` — a list of decoded BGR image arrays and an int label
        array aligned with it.

    Raises:
        FileNotFoundError: If the manifest CSV is missing (from the stream).
        RuntimeError: If the split yields no usable images.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    for image, label in get_feature_stream(split, random_seed=RANDOM_STATE):
        images.append(image)
        labels.append(label)
        if len(images) % 1000 == 0:
            logger.info("  ... %d images loaded", len(images))
        if max_samples and len(images) >= max_samples:
            break

    if not images:
        raise RuntimeError(f"No usable images found for split '{split}'.")
    return images, np.asarray(labels, dtype=int)


def fit_features(
    pipeline: ImagePipeline,
    max_samples: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
) -> Tuple[ImagePipeline, np.ndarray, np.ndarray]:
    """Fit the feature pipeline on TRAIN and return ``(pipeline, X, y)``.

    Runs ``pipeline.fit_transform`` on the training images: the per-image
    transforms produce their features, and any batch-level steps the pipeline
    carries (a ``'reduce'`` PCA/JL basis, a ``'scale'`` standardizer) are fit on
    that batch and stored on the pipeline for later reuse on val/test.

    Caches the fitted pipeline *and* the train features together, so a rerun
    reloads both without re-decoding images (and the reloaded pipeline can still
    transform val/test). The cache is keyed by ``cache_prefix`` (the
    feature-defining config) plus the sample cap, so changing either recomputes.

    Returns:
        ``(fitted_pipeline, X_train, y_train)``.
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
    """Transform a held-out split with the already-fitted pipeline.

    Runs ``pipeline.transform`` so val/test images are projected with the *same*
    batch-level statistics (PCA basis, scaling) learned on train. Features are
    cached per split.

    Returns:
        ``(X, y)`` for the requested split.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_prefix}_{split}_n{max_samples}.npz"
        if cache_path.is_file():
            logger.info("Loading cached features for '%s' from %s", split, cache_path)
            data = np.load(cache_path)
            return data["X"], data["y"]

    images, y = load_images(split, max_samples)
    logger.info("Transforming %d '%s' images with the fitted pipeline...", len(images), split)
    X = pipeline.transform(images)
    logger.info("  -> %s features %s", split, X.shape)

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
