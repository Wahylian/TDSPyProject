"""Tests for comparison/registry_utils.py: dynamic model/pipeline family classification."""

from __future__ import annotations

from trainbase import MODEL_REGISTRY, PIPELINE_REGISTRY

from comparison.registry_utils import model_family, pipeline_family


def _torch_installed() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestModelFamily:
    """Classifying registered models as 'torch' or 'classical'."""

    def test_every_model_classified_matches_torch_models_contribution(self):
        """model_family agrees with what trainbase.torch_models itself contributes."""
        if _torch_installed():
            from trainbase.torch_models import build_torch_registry
            torch_names = set(build_torch_registry())
        else:
            torch_names = set()

        for name in MODEL_REGISTRY:
            expected = "torch" if name in torch_names else "classical"
            assert model_family(name) == expected

    def test_classical_model_is_classical(self):
        """A model absent from torch_models's contribution is 'classical'."""
        assert model_family("svm") == "classical"


class TestPipelineFamily:
    """Classifying registered pipelines as 'torch' (raw-pixel) or 'classical'."""

    def test_reduced_pipeline_is_classical(self):
        """A pipeline with a 'reduce' op (e.g. svm) is 'classical'."""
        assert pipeline_family("svm") == "classical"

    def test_raw_pixel_pipeline_is_torch(self):
        """A pipeline with no 'reduce' op (e.g. pixels) is 'torch'."""
        assert pipeline_family("pixels") == "torch"

    def test_every_pipeline_family_matches_reduce_step_presence(self):
        """pipeline_family agrees with whether the built pipeline has a 'reduce' op."""
        for name, factory in PIPELINE_REGISTRY.items():
            has_reduce = any(op == "reduce" for op, _ in factory().operations)
            expected = "classical" if has_reduce else "torch"
            assert pipeline_family(name) == expected
