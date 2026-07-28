"""RunLoader: discover and parse training-run metadata under an artifacts root."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from trainbase import MODEL_REGISTRY, PIPELINE_REGISTRY

from .records import RunRecord

logger = logging.getLogger(__name__)


class RunLoader:
    """Globs <root>/**/metadata.json into validated RunRecord objects.

    Validation is against the live MODEL_REGISTRY/PIPELINE_REGISTRY (queried at
    call time, never hardcoded), so a run whose model or pipeline has since been
    removed from the registry is skipped with a warning rather than raised.
    """

    def __init__(self, root: Union[str, Path] = "artifacts"):
        self.root = Path(root)

    def load(
        self,
        model: Optional[str] = None,
        pipeline: Optional[str] = None,
        require_metric: Optional[str] = None,
    ) -> List[RunRecord]:
        """Load all valid runs, optionally filtered by model, pipeline, or metric.

        require_metric keeps only runs whose evaluation_metrics contains that key
        with a non-None value.
        """
        records: List[RunRecord] = []
        for path in sorted(self.root.glob("**/metadata.json")):
            record = self._parse(path)
            if record is None:
                continue
            if model is not None and record.model_name != model:
                continue
            if pipeline is not None and record.pipeline_used != pipeline:
                continue
            if require_metric is not None and record.metric(require_metric) is None:
                continue
            records.append(record)
        return records

    def _parse(self, path: Path) -> Optional[RunRecord]:
        """Parse and registry-validate one metadata.json, warning and skipping on failure."""
        try:
            record = RunRecord.from_metadata(path)
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("Skipping malformed metadata at %s: %s", path, exc)
            return None

        if record.model_name not in MODEL_REGISTRY:
            logger.warning(
                "Skipping %s: model %r is not in the current MODEL_REGISTRY",
                path, record.model_name,
            )
            return None
        # pipeline_used is "custom" for --pipeline-spec runs, which have no registry entry.
        if record.pipeline_used != "custom" and record.pipeline_used not in PIPELINE_REGISTRY:
            logger.warning(
                "Skipping %s: pipeline %r is not in the current PIPELINE_REGISTRY",
                path, record.pipeline_used,
            )
            return None
        return record
