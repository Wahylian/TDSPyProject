"""Tests for train_model.py: CLI parsing, validation, and the cache-key hash.

main()'s happy path (load -> tune -> evaluate -> save) is covered end-to-end by
test_e2e.py; these focus on parse_args's contract and the argument-validation /
data-loading failure modes that end-to-end test doesn't exercise, plus
_cache_prefix's naming logic.
"""

from __future__ import annotations

import json

import pytest

import train_model


class TestParseArgs:
    """CLI argument parsing and its defaults/constraints."""

    def test_model_is_required(self):
        with pytest.raises(SystemExit):
            train_model.parse_args([])

    def test_pipeline_defaults_to_svm(self):
        args = train_model.parse_args(["--model", "svm"])
        assert args.pipeline == "svm"
        assert args.pipeline_spec is None

    def test_pipeline_and_pipeline_spec_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            train_model.parse_args([
                "--model", "svm", "--pipeline", "fast", "--pipeline-spec", "[]",
            ])

    def test_sample_caps_have_documented_defaults(self):
        args = train_model.parse_args(["--model", "svm"])
        assert args.max_train_samples == 5000
        assert args.max_val_samples == 2000
        assert args.max_test_samples == 5000

    def test_diagnostics_flag_defaults_off(self):
        args = train_model.parse_args(["--model", "svm"])
        assert args.diagnostics is False


class TestMainValidation:
    """main()'s fail-fast validation, each surfacing as SystemExit."""

    def test_unknown_model_raises_systemexit(self):
        args = train_model.parse_args(["--model", "not-a-real-model"])
        with pytest.raises(SystemExit, match="Unknown --model"):
            train_model.main(args)

    def test_unknown_pipeline_raises_systemexit(self):
        args = train_model.parse_args([
            "--model", "svm", "--pipeline", "not-a-real-pipeline",
        ])
        with pytest.raises(SystemExit, match="Unknown --pipeline"):
            train_model.main(args)

    def test_malformed_pipeline_spec_raises_systemexit(self):
        args = train_model.parse_args([
            "--model", "svm", "--pipeline-spec", "{not valid json",
        ])
        with pytest.raises(SystemExit, match="Invalid feature pipeline"):
            train_model.main(args)

    def test_missing_manifest_raises_systemexit(self, monkeypatch):
        """A missing dataset manifest surfaces as SystemExit, not a raw traceback."""
        def _raise(*args, **kwargs):
            raise FileNotFoundError("dataset_split.csv not found")

        monkeypatch.setattr("trainbase.features.get_feature_stream", _raise)
        args = train_model.parse_args(["--model", "logreg", "--cache-dir", ""])
        with pytest.raises(SystemExit, match="Could not load data"):
            train_model.main(args)


class TestCachePrefix:
    """The feature-cache key derived from the resolved pipeline definition."""

    def test_named_pipeline_prefix_starts_with_its_name(self):
        args = train_model.parse_args(["--model", "svm", "--pipeline", "fast"])
        assert train_model._cache_prefix(args).startswith("fast_")

    def test_custom_spec_prefix_starts_with_custom(self):
        spec = json.dumps([["grayscale", {}]])
        args = train_model.parse_args(["--model", "svm", "--pipeline-spec", spec])
        assert train_model._cache_prefix(args).startswith("custom_")

    def test_different_specs_hash_to_different_prefixes(self):
        spec_a = json.dumps([["grayscale", {}]])
        spec_b = json.dumps([["grayscale", {}], ["vectorize", {}]])
        args_a = train_model.parse_args(["--model", "svm", "--pipeline-spec", spec_a])
        args_b = train_model.parse_args(["--model", "svm", "--pipeline-spec", spec_b])
        assert train_model._cache_prefix(args_a) != train_model._cache_prefix(args_b)

    def test_same_registry_pipeline_is_deterministic(self):
        args1 = train_model.parse_args(["--model", "svm", "--pipeline", "svm"])
        args2 = train_model.parse_args(["--model", "svm", "--pipeline", "svm"])
        assert train_model._cache_prefix(args1) == train_model._cache_prefix(args2)
