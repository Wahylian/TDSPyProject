"""Reporter: render one comparison table (leaderboard/resilience/grid) to a format.

metadata.json stores only scalar pr_auc/roc_auc, not per-threshold curve data, so
this renders confusion-matrix and metric-bar plots (genuinely reconstructable)
rather than fabricated ROC/PR curves. matplotlib stays optional, mirroring the
torch/keras guards elsewhere in the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None


def _has_row_index(table: pd.DataFrame) -> bool:
    """Whether the table's index carries meaning (a named/non-default index)."""
    return not isinstance(table.index, pd.RangeIndex)


def _cell(value: object) -> str:
    """Format one table cell for Markdown, rounding floats for readability."""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


class Reporter:
    """Renders a single pandas DataFrame (a leaderboard/resilience/grid table)."""

    def __init__(self, table: pd.DataFrame, title: str = "Comparison Report"):
        self.table = table
        self.title = title

    def to_markdown(self) -> str:
        """Render the table as a Markdown document with a title heading."""
        show_index = _has_row_index(self.table)
        headers = ([self.table.index.name or ""] if show_index else []) + [
            str(c) for c in self.table.columns
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for idx, row in self.table.iterrows():
            cells = ([str(idx)] if show_index else []) + [_cell(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return f"# {self.title}\n\n" + "\n".join(lines) + "\n"

    def to_html(self) -> str:
        """Render the table as a standalone HTML document."""
        body = self.table.to_html(index=_has_row_index(self.table))
        return (
            f"<html><head><title>{self.title}</title></head>"
            f"<body><h1>{self.title}</h1>{body}</body></html>"
        )

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the table as CSV to `path`, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(path, index=_has_row_index(self.table))
        return path

    def plot_bar(self, path: Union[str, Path], column: Optional[str] = None) -> Optional[Path]:
        """Save a bar chart of `column` (default: first numeric column) to `path`.

        Returns None without writing anything when matplotlib is not installed.
        """
        if plt is None:
            return None
        if column is None:
            numeric = self.table.select_dtypes("number").columns
            if len(numeric) == 0:
                return None
            column = numeric[0]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        self.table[column].plot(kind="bar", ax=ax)
        ax.set_ylabel(column)
        ax.set_title(self.title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
