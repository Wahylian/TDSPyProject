"""RunRecord: a typed, frozen view of one training run's metadata.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def _reconstructed_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Copy a headline-metrics dict, adding f1 from precision/recall if absent.

    trainbase.artifacts.HEADLINE_METRICS deliberately omits f1 (see
    tests/test_artifacts.py::test_headline_metrics_extracted), so it is
    reconstructed here as 2PR/(P+R) — identical to sklearn's binary f1_score.
    """
    out = dict(metrics)
    if "f1" not in out:
        precision, recall = out.get("precision"), out.get("recall")
        if precision is None or recall is None:
            out["f1"] = None
        elif precision + recall == 0:
            out["f1"] = 0.0
        else:
            out["f1"] = 2 * precision * recall / (precision + recall)
    return out


@dataclass(frozen=True)
class RunRecord:
    """One training run: mirrors the metadata.json schema written by trainbase."""

    model_name: str
    run_id: str
    timestamp: str
    pipeline_used: str
    pipeline_spec: Optional[str]
    pipeline_steps: Tuple[Tuple[str, Dict[str, Any]], ...]
    scoring: str
    sample_sizes: Dict[str, int]
    hyperparameters: Dict[str, Any]
    best_val_score: float
    evaluation_metrics: Dict[str, Optional[float]]
    baseline_metrics: Dict[str, Optional[float]]
    diagnostics: Dict[str, Any]
    source_path: Path

    @staticmethod
    def from_metadata(path: Union[str, Path]) -> "RunRecord":
        """Parse one metadata.json file into a RunRecord."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        return RunRecord(
            model_name=data["model_name"],
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            pipeline_used=data["pipeline_used"],
            pipeline_spec=data.get("pipeline_spec"),
            pipeline_steps=tuple(tuple(step) for step in data.get("pipeline_steps", [])),
            scoring=data["scoring"],
            sample_sizes=data["sample_sizes"],
            hyperparameters=data["hyperparameters"],
            best_val_score=data["best_val_score"],
            evaluation_metrics=_reconstructed_metrics(data["evaluation_metrics"]),
            baseline_metrics=_reconstructed_metrics(data["baseline_metrics"]),
            diagnostics=data.get("diagnostics", {}),
            source_path=path,
        )

    def metric(self, name: str) -> Optional[float]:
        """Look up a test-set headline metric by name (e.g. 'f1', 'accuracy')."""
        return self.evaluation_metrics.get(name)

    def baseline(self, name: str) -> Optional[float]:
        """Look up the naive-baseline counterpart of a headline metric."""
        return self.baseline_metrics.get(name)

    def lift(self, name: str) -> Optional[float]:
        """Test metric minus its naive-baseline value, or None if either is missing."""
        value, base = self.metric(name), self.baseline(name)
        if value is None or base is None:
            return None
        return value - base

    def has_reduce_step(self) -> bool:
        """Whether this run's pipeline included a batch-level 'reduce' op."""
        return any(name == "reduce" for name, _ in self.pipeline_steps)
