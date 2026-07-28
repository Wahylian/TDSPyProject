"""Read-side comparison package: scans artifacts/**/metadata.json and reports.

Consumes metadata.json only; never imports or executes training code. See
Docs/model_comparison_plan.md for the architecture this implements.
"""

from .records import RunRecord
from .loader import RunLoader
from .matrix import ComparisonMatrix
from .registry_utils import model_family, pipeline_family
from .stats import StatisticalComparison
from .report import Reporter

__all__ = [
    "RunRecord", "RunLoader", "ComparisonMatrix", "model_family", "pipeline_family",
    "StatisticalComparison", "Reporter",
]
