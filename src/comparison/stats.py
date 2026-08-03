"""StatisticalComparison: ranking, paired significance tests, and lift deltas."""

from __future__ import annotations

from typing import Dict, List

from .records import RunRecord

try:
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:  # pragma: no cover - scipy ships with scikit-learn but stay optional
    _wilcoxon = None


class StatisticalComparison:
    """Ranks and pairwise-compares a fixed set of RunRecords by run_id."""

    def __init__(self, records: List[RunRecord]):
        self.records = list(records)
        self._by_run_id = {r.run_id: r for r in records}

    def rank(self, metric: str = "f1") -> List[RunRecord]:
        """Records sorted by a metric, descending; those missing it are dropped."""
        scored = [r for r in self.records if r.metric(metric) is not None]
        return sorted(scored, key=lambda r: r.metric(metric), reverse=True)

    def compare(self, run_id_a: str, run_id_b: str, metric: str = "f1") -> Dict[str, object]:
        """Compare two runs by run_id.

        Runs a Wilcoxon signed-rank test when both log equal-length per-sample
        scores under diagnostics['per_sample_scores']; the current metadata.json
        schema does not capture these, so in practice this falls back to
        comparing each run's lift over its own baseline.
        """
        a, b = self._by_run_id[run_id_a], self._by_run_id[run_id_b]
        samples_a = a.diagnostics.get("per_sample_scores")
        samples_b = b.diagnostics.get("per_sample_scores")

        if (
            _wilcoxon is not None
            and samples_a is not None
            and samples_b is not None
            and len(samples_a) == len(samples_b)
        ):
            statistic, p_value = _wilcoxon(samples_a, samples_b)
            return {
                "method": "wilcoxon_paired",
                "statistic": float(statistic),
                "p_value": float(p_value),
                run_id_a: a.metric(metric),
                run_id_b: b.metric(metric),
            }

        lift_a, lift_b = a.lift(metric), b.lift(metric)
        return {
            "method": "lift_delta",
            f"{run_id_a}_lift": lift_a,
            f"{run_id_b}_lift": lift_b,
            "lift_delta": (lift_a - lift_b) if lift_a is not None and lift_b is not None else None,
        }
