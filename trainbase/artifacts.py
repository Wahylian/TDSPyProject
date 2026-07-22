"""Persist a training run: feature pipeline, classifier, and metadata JSON.

Each run writes model.joblib, feature_pipeline.joblib, and metadata.json into an
isolated <base_dir>/<model_name>/<run_id>/ so reruns never overwrite. The two
joblib artifacts capture the full inference path:
model.predict([feature_pipeline.process(image)]).
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

# Headline metrics captured uniformly so runs stay comparable across estimators.
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
    """Assemble the uniform metadata.json record for a run.

    Top-level keys are identical for every model. evaluation_metrics and
    baseline_metrics hold the five headline scores; the confusion matrix and
    per-class report are folded into diagnostics alongside the estimator-specific
    ones (feature importances, OOB, curves), all always present (None if absent).
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
    """Write model.joblib, feature_pipeline.joblib, and metadata.json to the run dir.

    Creates base_dir/model_name/run_id/ if missing and returns it.
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
