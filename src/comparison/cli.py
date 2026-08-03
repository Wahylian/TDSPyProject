"""python -m comparison: thin CLI orchestrator over RunLoader/ComparisonMatrix/Reporter."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .diagnostics_report import diagnostics_markdown, plot_confusion_matrix
from .loader import RunLoader
from .matrix import ComparisonMatrix
from .report import Reporter


def _add_common_args(subparser: argparse.ArgumentParser) -> None:
    """Attach the flags shared by every shape to one subparser.

    Attached per-subparser (not on the top-level parser) so they can follow the
    shape name on the command line, e.g. `leaderboard --root ... --metric ...`.
    """
    subparser.add_argument("--root", default="artifacts", help="Artifacts root to scan.")
    subparser.add_argument("--metric", default="f1", help="Metric to rank/pivot on.")
    subparser.add_argument(
        "--output-dir", default="reports",
        help="Reports are written to <output-dir>/<timestamp>/.",
    )
    subparser.add_argument(
        "--diagnostics", action="store_true",
        help="Also emit a diagnostics section (confusion matrix, hyperparameter "
        "scores, feature importances/OOB, learning curve) for each run shown. "
        "Not available for the 'grid' shape (many cells, no single run per cell).",
    )
    subparser.add_argument(
        "--plot", action="store_true",
        help="Also save PNG plots (a metric bar chart, or a heatmap for 'grid', "
        "plus confusion matrices) if matplotlib is installed; skipped silently "
        "otherwise.",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the CLI: one subcommand per evaluation shape."""
    parser = argparse.ArgumentParser(
        prog="python -m comparison",
        description="Compare trained registry models against prebuilt pipelines "
        "by reading artifacts/**/metadata.json.",
    )
    sub = parser.add_subparsers(dest="shape", required=True)

    leaderboard = sub.add_parser("leaderboard", help="N x 1: rank models on one pipeline.")
    leaderboard.add_argument("--pipeline", default=None, help="Fix comparison to one pipeline.")
    _add_common_args(leaderboard)

    resilience = sub.add_parser("resilience", help="1 x N: sweep one model across pipelines.")
    resilience.add_argument("--model", required=True, help="Model to sweep across pipelines.")
    _add_common_args(resilience)

    grid = sub.add_parser("grid", help="N x M: model x pipeline pivot (classical/torch blocks separate).")
    _add_common_args(grid)

    return parser.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> Path:
    """Load runs, build the requested table, and write Markdown/HTML/CSV to a timestamped dir."""
    args = args or parse_args()
    records = RunLoader(args.root).load()
    matrix = ComparisonMatrix(records)

    if args.shape == "leaderboard":
        table = matrix.leaderboard(metric=args.metric, pipeline=args.pipeline)
        title = f"Leaderboard ({args.metric})" + (f" on {args.pipeline}" if args.pipeline else "")
    elif args.shape == "resilience":
        table = matrix.resilience(args.model, metric=args.metric)
        title = f"Resilience: {args.model} ({args.metric})"
    else:
        table = matrix.grid(metric=args.metric)
        title = f"Grid ({args.metric})"

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    reporter = Reporter(table, title=title)
    reporter.to_csv(run_dir / f"{args.shape}.csv")
    (run_dir / f"{args.shape}.md").write_text(reporter.to_markdown(), encoding="utf-8")
    (run_dir / f"{args.shape}.html").write_text(reporter.to_html(), encoding="utf-8")

    # The grid is a model x pipeline matrix; the other two shapes are one ranked
    # column, so each gets the plot form that shows all of its cells.
    if args.plot:
        plot_path = run_dir / f"{args.shape}.png"
        if args.shape == "grid":
            reporter.plot_matrix(plot_path)
        else:
            reporter.plot_bar(plot_path)

    # run_id only identifies a single run for leaderboard/resilience tables, not
    # the many-celled grid, so diagnostics are only emitted for those two shapes.
    if args.diagnostics and "run_id" in table.columns:
        wanted = [r for r in records if r.run_id in set(table["run_id"])]
        text = "\n".join(diagnostics_markdown(r) for r in wanted)
        (run_dir / f"{args.shape}_diagnostics.md").write_text(text, encoding="utf-8")
        if args.plot:
            for r in wanted:
                plot_confusion_matrix(r, run_dir / f"confusion_{r.model_name}_{r.run_id}.png")

    return run_dir


if __name__ == "__main__":
    main()
