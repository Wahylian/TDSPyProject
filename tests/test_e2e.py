"""End-to-end test for train_model.main plus an artifact leak scan.

Drives the full load -> tune -> evaluate -> save path on a tiny synthetic split
(the feature stream is monkeypatched, so no manifest or real images are needed)
and asserts the artifact bundle exists, metadata carries the uniform schema, the
joblib artifacts reload, and the emitted files leak no machine-specific paths or
credential-like tokens. A minimal custom pipeline keeps the run instant while
still exercising the fitted batch-level reducer/scaler on val/test.
"""

from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pytest

import train_model
from trainbase.artifacts import HEADLINE_METRICS

# A tiny self-contained feature pipeline: 64 pixels -> PCA(8) -> standardized.
SPEC = json.dumps([
    ["grayscale", {}],
    ["resize", {"target_size": [8, 8], "preserve_aspect": False}],
    ["normalize", {"method": "minmax"}],
    ["vectorize", {}],
    ["reduce", {"method": "vec-pca", "n_components": 8, "random_state": 42}],
    ["scale", {}],
])

# Top-level metadata schema every run must emit (mirrors trainbase.artifacts).
EXPECTED_METADATA_KEYS = {
    "model_name", "run_id", "timestamp", "pipeline_used", "pipeline_spec",
    "pipeline_steps", "scoring", "sample_sizes", "hyperparameters",
    "best_val_score", "evaluation_metrics", "baseline_metrics", "diagnostics",
}


def _fake_feature_stream(n: int = 50):
    """A get_feature_stream stand-in yielding a separable 2-class split.

    Class 0 is dark (~40), class 1 bright (~200), so any linear model separates
    them. A fresh generator is produced per call, matching the real stream.
    """
    def stream(split, csv_path=None, random_seed=None):
        rng = np.random.default_rng(0)
        for i in range(n):
            label = i % 2
            base = 200 if label == 1 else 40
            image = np.clip(
                rng.normal(base, 15, size=(16, 16, 3)), 0, 255
            ).astype(np.uint8)
            yield image, label
    return stream


@pytest.fixture
def synthetic_run_artifact_dir(tmp_path, monkeypatch):
    """Run train_model.main on the synthetic stream; return the run dir."""
    monkeypatch.setattr(
        "trainbase.features.get_feature_stream", _fake_feature_stream()
    )

    output_root = tmp_path / "artifacts"
    args = train_model.parse_args([
        "--model", "logreg",
        "--pipeline-spec", SPEC,
        "--max-train-samples", "40",
        "--max-val-samples", "20",
        "--max-test-samples", "20",
        "--output-dir", str(output_root),
        "--cache-dir", "",  # disable the feature cache for a clean run
    ])
    train_model.main(args)

    run_dirs = list((output_root / "logreg").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    return run_dirs[0]


class TestEndToEndBundle:
    """The generated run bundle and its metadata schema."""

    def test_bundle_files_exist(self, synthetic_run_artifact_dir):
        """All three artifacts land in the run dir."""
        run = synthetic_run_artifact_dir
        assert (run / "model.joblib").is_file()
        assert (run / "feature_pipeline.joblib").is_file()
        assert (run / "metadata.json").is_file()

    def test_metadata_schema_and_content(self, synthetic_run_artifact_dir):
        """metadata.json has the uniform schema and coherent content."""
        metadata = json.loads(
            (synthetic_run_artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert set(metadata) == EXPECTED_METADATA_KEYS
        assert metadata["model_name"] == "logreg"
        assert metadata["pipeline_used"] == "custom"
        assert set(metadata["evaluation_metrics"]) == set(HEADLINE_METRICS)
        sizes = metadata["sample_sizes"]
        assert sizes["train"] == 40 and sizes["val"] == 20 and sizes["test"] == 20

    def test_saved_artifacts_reload_and_predict(self, synthetic_run_artifact_dir):
        """The joblib model + pipeline reload into working objects."""
        run = synthetic_run_artifact_dir
        pipeline = joblib.load(run / "feature_pipeline.joblib")
        model = joblib.load(run / "model.joblib")
        assert len(pipeline.operations) >= 1
        preds = model.predict(np.zeros((3, 8), dtype=np.float32))
        assert len(preds) == 3


def test_synthetic_run_artifact_has_no_security_or_privacy_leaks(
    synthetic_run_artifact_dir,
):
    """No machine-specific paths or credential tokens leak into the run output.

    Scoped to the freshly generated run directory: scans the metadata and the
    serialized model/pipeline bytes so the check is portable across hosts.
    """
    path_pattern = re.compile(
        r'(?:/home/|/Users/|C:\\Users\\|/var/|/tmp/)', re.IGNORECASE
    )
    secret_pattern = re.compile(
        r'(?:api_key|secret_token|password)\s*[:=]', re.IGNORECASE
    )

    metadata_content = (
        synthetic_run_artifact_dir / "metadata.json"
    ).read_text(encoding="utf-8")
    assert not path_pattern.search(metadata_content), \
        "Machine-specific absolute path detected in generated artifact metadata."
    assert not secret_pattern.search(metadata_content), \
        "Potential credential leak detected in generated artifact metadata."

    # Also scan the binary artifacts: decode leniently and look for path roots.
    for artifact in synthetic_run_artifact_dir.iterdir():
        text = artifact.read_bytes().decode("latin-1", errors="ignore")
        assert not path_pattern.search(text), \
            f"Machine-specific absolute path detected in {artifact.name}."
