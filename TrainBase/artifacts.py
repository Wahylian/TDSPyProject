"""Persist a training run: feature pipeline, classifier, and metadata JSON.

Each run writes an isolated bundle to ``<base_dir>/<model_name>/<run_id>/`` so
reruns of the same model never overwrite one another:

    <base_dir>/<model_name>/<run_id>/
        model.joblib             # the fitted best estimator
        feature_pipeline.joblib  # the fitted ImagePipeline (PCA basis + scaling)
        metadata.json            # the uniform run record (see build_metadata)

The two joblib artifacts together capture the full inference path from a raw
image: ``model.predict([feature_pipeline.process(image)])``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from sklearn.base import BaseEstimator

from preprocessing import ImagePipeline

logger = logging.getLogger(__name__)

# The standardized headline metrics captured uniformly for every model, so runs
# stay directly comparable regardless of estimator.
HEADLINE_METRICS = ["accuracy", "precision", "recall", "pr_auc", "roc_auc"]


def _headline(metrics: Dict[str, object]) -> Dict[str, object]:
    """Pull just the standardized headline metrics out of a full metrics dict."""
    return {key: metrics.get(key) for key in HEADLINE_METRICS}


def build_metadata(
    *,
    model_name: str,
    run_id: str,
    timestamp: str,
    pipeline_used: str,
    pipeline_spec: Optional[str],
    pipeline_steps: List,
    scoring: str,
    sample_sizes: Dict[str, int],
    hyperparameters: Dict[str, object],
    best_val_score: float,
    test_metrics: Dict[str, object],
    baseline_metrics: Dict[str, object],
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    """Assemble the uniform ``metadata.json`` record for a run.

    The schema's top-level keys are identical for every model. ``evaluation_metrics``
    and ``baseline_metrics`` hold the five standardized headline scores; the
    confusion matrix and per-class report are folded into ``diagnostics`` next to
    the estimator-specific diagnostics (feature importances, OOB, curves), whose
    keys are always present (``None`` when a diagnostic does not apply).

    Args:
        model_name: Registry name of the trained classifier.
        run_id: Per-run identifier (also the run subdirectory name).
        timestamp: ISO-8601 run start time.
        pipeline_used: Feature-pipeline registry name, or ``"custom"``.
        pipeline_spec: The verbatim custom JSON spec, or ``None``.
        pipeline_steps: The fitted ImagePipeline's operation list (reproducibility).
        scoring: Metric optimized during tuning.
        sample_sizes: ``{"train": n, "val": n, "test": n}`` actually used.
        hyperparameters: The selected best hyperparameters.
        best_val_score: Validation score of the selected configuration.
        test_metrics: Full :func:`trainbase.evaluation.evaluate` dict for the model.
        baseline_metrics: Full evaluate dict for the naive baseline.
        diagnostics: :func:`trainbase.diagnostics.collect_diagnostics` output.

    Returns:
        A JSON-serializable dict ready to write as ``metadata.json``.
    """
    run_diagnostics: Dict[str, object] = {
        "confusion_matrix": test_metrics.get("confusion_matrix"),
        "classification_report": test_metrics.get("classification_report"),
    }
    run_diagnostics.update(diagnostics)

    return {
        "model_name": model_name,
        "run_id": run_id,
        "timestamp": timestamp,
        "pipeline_used": pipeline_used,
        "pipeline_spec": pipeline_spec,
        "pipeline_steps": pipeline_steps,
        "scoring": scoring,
        "sample_sizes": sample_sizes,
        "hyperparameters": hyperparameters,
        "best_val_score": best_val_score,
        "evaluation_metrics": _headline(test_metrics),
        "baseline_metrics": _headline(baseline_metrics),
        "diagnostics": run_diagnostics,
    }


def save_artifacts(
    model: BaseEstimator,
    feature_pipeline: ImagePipeline,
    metadata: Dict[str, object],
    base_dir: Path,
    model_name: str,
    run_id: str,
) -> Path:
    """Serialize the fitted pipeline, the model, and the metadata into a run dir.

    Writes ``model.joblib``, ``feature_pipeline.joblib`` and ``metadata.json``
    into ``base_dir/model_name/run_id/`` (created if missing). The fitted
    ``feature_pipeline`` (the project ImagePipeline holding the per-image
    transforms *and* the trained PCA basis *and* scaling statistics) turns an
    image into a model-ready vector via ``feature_pipeline.process(image)``, and
    the fitted sklearn ``model`` scores it.

    Args:
        model: The fitted best estimator (a one-step ``clf`` ``Pipeline``).
        feature_pipeline: The fitted :class:`ImagePipeline`.
        metadata: The uniform run record from :func:`build_metadata`.
        base_dir: Root artifacts directory (created if missing).
        model_name: Names the per-model subdirectory.
        run_id: Names the per-run subdirectory.

    Returns:
        The run directory the artifacts were written to.
    """
    run_dir = base_dir / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    pipeline_path = run_dir / "feature_pipeline.joblib"
    metadata_path = run_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_pipeline, pipeline_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model     -> %s", model_path)
    logger.info("Saved pipeline  -> %s", pipeline_path)
    logger.info("Saved metadata  -> %s", metadata_path)
    return run_dir
