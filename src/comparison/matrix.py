"""ComparisonMatrix: N×1 leaderboards, 1×N resilience sweeps, N×M grids."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .records import RunRecord
from .registry_utils import model_family, pipeline_family


class ComparisonMatrix:
    """Builds comparison tables over a fixed set of RunRecords.

    Assumes records were produced by RunLoader (so model_name/pipeline_used are
    already validated against the live registries).
    """

    def __init__(self, records: List[RunRecord]):
        self.records = list(records)

    def leaderboard(self, metric: str = "f1", pipeline: Optional[str] = None) -> pd.DataFrame:
        """N×1: rank models by a metric, optionally fixed to one pipeline.

        Without `pipeline`, each model's single best run (across whichever
        pipelines it has) represents that model.
        """
        rows = self.records if pipeline is None else [
            r for r in self.records if r.pipeline_used == pipeline
        ]
        best = _best_per_group(rows, "model_name", metric)
        return _to_frame(best, metric, index_name="model_name")

    def resilience(self, model: str, metric: str = "f1") -> pd.DataFrame:
        """1×N: how one model's score varies across the pipelines it was run on."""
        rows = [r for r in self.records if r.model_name == model]
        best = _best_per_group(rows, "pipeline_used", metric)
        return _to_frame(best, metric, index_name="pipeline_used")

    def grid(self, metric: str = "f1") -> pd.DataFrame:
        """N×M: model x pipeline pivot of a metric.

        Classical models are only ever pivoted against classical pipelines, and
        torch models only against torch (raw-pixel) pipelines — the two blocks
        never cross, matching the pairing rule. Missing combinations are NaN.
        """
        cells = [
            (r.model_name, r.pipeline_used, r.metric(metric))
            for r in self.records
            if r.pipeline_used != "custom"
            and model_family(r.model_name) == pipeline_family(r.pipeline_used)
        ]
        if not cells:
            return pd.DataFrame()
        frame = pd.DataFrame(cells, columns=["model", "pipeline", metric])
        return frame.pivot_table(index="model", columns="pipeline", values=metric, aggfunc="max")


def _best_per_group(rows: List[RunRecord], key: str, metric: str) -> Dict[str, RunRecord]:
    """The highest-metric record per distinct value of `key`, dropping None metrics."""
    best: Dict[str, RunRecord] = {}
    for record in rows:
        value = record.metric(metric)
        if value is None:
            continue
        group = getattr(record, key)
        if group not in best or value > best[group].metric(metric):
            best[group] = record
    return best


def _to_frame(best: Dict[str, RunRecord], metric: str, index_name: str) -> pd.DataFrame:
    """Assemble a sorted DataFrame of metric/baseline/lift from the best-per-group map."""
    rows = [
        {
            index_name: key,
            metric: record.metric(metric),
            f"baseline_{metric}": record.baseline(metric),
            f"lift_{metric}": record.lift(metric),
            "run_id": record.run_id,
        }
        for key, record in best.items()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(metric, ascending=False).reset_index(drop=True)
