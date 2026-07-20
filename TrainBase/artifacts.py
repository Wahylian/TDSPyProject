"""Persist the trained run: feature pipeline, classifier, and metrics.

A run is reproducible from two joblib artifacts plus a metrics JSON. Together
the fitted feature pipeline and the fitted classifier capture the full
inference path from a raw image.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
from sklearn.base import BaseEstimator

from preprocessing import ImagePipeline

logger = logging.getLogger(__name__)


def save_artifacts(
    model: BaseEstimator,
    feature_pipeline: ImagePipeline,
    metrics: Dict[str, object],
    output_dir: Path,
    model_name: str,
) -> None:
    """Serialize the fitted feature pipeline, the model, and the metrics JSON.

    Two joblib artifacts together capture the full inference path from a raw image: 
    the fitted ``feature_pipeline`` (the project ImagePipeline that does
    the per-image transforms *and* holds any trained PCA basis *and* scaling
    statistics) turns an image into a model-ready vector via
    ``feature_pipeline.process(image)``, and the fitted sklearn ``model`` (the classifier) scores it.
    End to end:
    ``model.predict([feature_pipeline.process(image)])``.

    Args:
        model: The fitted best estimator (a one-step ``Pipeline``: classifier).
        feature_pipeline: The fitted :class:`ImagePipeline`.
        metrics: The evaluation dict from :func:`trainbase.evaluation.evaluate`
            (plus baseline and run metadata).
        output_dir: Directory to write into (created if missing).
        model_name: Used in the artifact filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}_model.joblib"
    pipeline_path = output_dir / f"{model_name}_feature_pipeline.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_pipeline, pipeline_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved model     -> %s", model_path)
    logger.info("Saved pipeline  -> %s", pipeline_path)
    logger.info("Saved metrics   -> %s", metrics_path)
