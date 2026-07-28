"""Render one RunRecord's diagnostics block for --diagnostics CLI output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .records import RunRecord

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional
    plt = None

_DIAGNOSTIC_KEYS = (
    "confusion_matrix", "classification_report", "feature_importances",
    "oob_score", "hyperparameter_scores", "learning_curve",
)


def diagnostics_markdown(record: RunRecord) -> str:
    """Render the available diagnostics for one run as a Markdown section."""
    d = record.diagnostics
    lines = [
        f"## Diagnostics: {record.model_name} / {record.pipeline_used} ({record.run_id})", "",
    ]

    if d.get("confusion_matrix") is not None:
        lines += ["**Confusion matrix** (rows=true, cols=pred):", "```",
                   str(d["confusion_matrix"]), "```", ""]
    if d.get("classification_report"):
        lines += ["**Classification report:**", "```", d["classification_report"], "```", ""]
    if d.get("feature_importances") is not None:
        top = sorted(enumerate(d["feature_importances"]), key=lambda kv: kv[1], reverse=True)[:10]
        lines += ["**Top 10 feature importances:**"]
        lines += [f"- feature[{i}]: {v:.4f}" for i, v in top]
        lines.append("")
    if d.get("oob_score") is not None:
        lines += [f"**OOB score:** {d['oob_score']:.4f}", ""]
    if d.get("hyperparameter_scores"):
        best = sorted(d["hyperparameter_scores"], key=lambda s: s["mean_val_score"], reverse=True)[:5]
        lines += ["**Top 5 hyperparameter configurations (by mean val score):**"]
        lines += [
            f"- {s['params']}: {s['mean_val_score']:.4f} (+/- {s['std_val_score']:.4f})"
            for s in best
        ]
        lines.append("")
    if d.get("learning_curve") is not None:
        lc = d["learning_curve"]
        lines += ["**Learning curve (train size -> val score):**"]
        lines += [f"- {n}: {v:.4f}" for n, v in zip(lc["train_sizes"], lc["val_scores_mean"])]
        lines.append("")
    if not any(d.get(k) is not None for k in _DIAGNOSTIC_KEYS):
        lines.append("_No diagnostics recorded for this run._")

    return "\n".join(lines) + "\n"


def plot_confusion_matrix(record: RunRecord, path: Union[str, Path]) -> Optional[Path]:
    """Save a confusion-matrix heatmap for one run.

    Returns None (writes nothing) when matplotlib is absent or the run has no
    confusion matrix.
    """
    cm = record.diagnostics.get("confusion_matrix")
    if plt is None or cm is None:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    ax.set_title(f"Confusion matrix: {record.model_name}/{record.pipeline_used}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
