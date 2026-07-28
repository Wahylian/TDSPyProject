"""Tests for comparison/cli.py: argument parsing and the end-to-end orchestrator."""

from __future__ import annotations

import json

import pytest

from comparison.cli import main, parse_args


def _write_metadata(run_dir, model_name, pipeline_used, **overrides):
    """Write a minimal, schema-shaped metadata.json into <run_dir>/metadata.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "run_id": run_dir.name,
        "timestamp": "2026-01-01T00:00:00",
        "pipeline_used": pipeline_used,
        "pipeline_spec": None,
        "pipeline_steps": [["reduce", {}]],
        "scoring": "f1",
        "sample_sizes": {"train": 10, "val": 5, "test": 5},
        "hyperparameters": {},
        "best_val_score": 0.7,
        "evaluation_metrics": {"accuracy": 0.7, "precision": 0.6, "recall": 0.8},
        "baseline_metrics": {"accuracy": 0.5, "precision": 0.0, "recall": 0.0},
        "diagnostics": {"confusion_matrix": [[2, 1], [1, 2]]},
    }
    metadata.update(overrides)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class TestParseArgs:
    """CLI argument parsing per subcommand."""

    def test_leaderboard_defaults(self):
        args = parse_args(["leaderboard"])
        assert args.shape == "leaderboard"
        assert args.metric == "f1"
        assert args.pipeline is None

    def test_resilience_requires_model(self):
        args = parse_args(["resilience", "--model", "svm"])
        assert args.shape == "resilience"
        assert args.model == "svm"

    def test_grid_shape(self):
        assert parse_args(["grid"]).shape == "grid"

    def test_missing_shape_errors(self):
        with pytest.raises(SystemExit):
            parse_args([])


class TestMain:
    """End-to-end: load synthetic artifacts, write reports to a timestamped dir."""

    def test_leaderboard_writes_three_formats(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        _write_metadata(artifacts / "svm" / "run_a", "svm", "svm")
        _write_metadata(artifacts / "logreg" / "run_b", "logreg", "svm")
        reports = tmp_path / "reports"

        args = parse_args(["leaderboard", "--root", str(artifacts), "--output-dir", str(reports)])
        run_dir = main(args)

        assert (run_dir / "leaderboard.csv").is_file()
        assert (run_dir / "leaderboard.md").is_file()
        assert (run_dir / "leaderboard.html").is_file()

    def test_diagnostics_flag_writes_diagnostics_file(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        _write_metadata(artifacts / "svm" / "run_a", "svm", "svm")
        reports = tmp_path / "reports"

        args = parse_args([
            "leaderboard", "--root", str(artifacts), "--output-dir", str(reports), "--diagnostics",
        ])
        run_dir = main(args)

        assert (run_dir / "leaderboard_diagnostics.md").is_file()
        assert "Confusion matrix" in (run_dir / "leaderboard_diagnostics.md").read_text()

    def test_grid_shape_skips_diagnostics_even_when_requested(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        _write_metadata(artifacts / "svm" / "run_a", "svm", "svm")
        reports = tmp_path / "reports"

        args = parse_args(["grid", "--root", str(artifacts), "--output-dir", str(reports), "--diagnostics"])
        run_dir = main(args)

        assert not (run_dir / "grid_diagnostics.md").exists()

    def test_resilience_shape(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        _write_metadata(artifacts / "svm" / "run_a", "svm", "svm")
        _write_metadata(artifacts / "svm" / "run_b", "svm", "fast")
        reports = tmp_path / "reports"

        args = parse_args([
            "resilience", "--model", "svm", "--root", str(artifacts), "--output-dir", str(reports),
        ])
        run_dir = main(args)

        content = (run_dir / "resilience.csv").read_text()
        assert "svm" in content and "fast" in content
