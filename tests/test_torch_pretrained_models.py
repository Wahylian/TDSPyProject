"""Tests for trainbase.torch_pretrained_models — pretrained ResNet18/ViT-B/16.

Skipped entirely when torch or torchvision is not installed. Every model is
built with pretrained=False so no ImageNet weights are downloaded; that keeps
the suite offline-safe while still exercising fit/predict on a real (random-
init) instance of the actual torchvision architecture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from trainbase.torch_pretrained_models import (
    CNNPretrainedClassifier,
    ViTPretrainedClassifier,
    build_pretrained_torch_registry,
)

pytestmark = pytest.mark.slow

MODELS = [CNNPretrainedClassifier, ViTPretrainedClassifier]


@pytest.mark.parametrize("Model", MODELS)
def test_unfitted_raises(Model):
    with pytest.raises(NotFittedError):
        check_is_fitted(Model())


@pytest.mark.parametrize("Model", MODELS)
def test_is_base_estimator_and_clones(Model):
    est = Model(epochs=1, pretrained=False)
    assert isinstance(est, BaseEstimator)
    assert clone(est).get_params() == est.get_params()


@pytest.mark.parametrize("Model", MODELS)
def test_fit_predict_shapes_and_labels(Model, pretrained_pixel_split):
    est = Model(epochs=1, pretrained=False, batch_size=4).fit(
        pretrained_pixel_split.X_train, pretrained_pixel_split.y_train)
    preds = est.predict(pretrained_pixel_split.X_test)
    assert preds.shape == (len(pretrained_pixel_split.y_test),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.parametrize("Model", MODELS)
def test_predict_proba_is_distribution(Model, pretrained_pixel_split):
    est = Model(epochs=1, pretrained=False, batch_size=4).fit(
        pretrained_pixel_split.X_train, pretrained_pixel_split.y_train)
    proba = est.predict_proba(pretrained_pixel_split.X_test)
    assert proba.shape == (len(pretrained_pixel_split.y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("Model", MODELS)
def test_joblib_roundtrip_predicts_identically(Model, pretrained_pixel_split, tmp_path):
    import joblib
    est = Model(epochs=1, pretrained=False, batch_size=4).fit(
        pretrained_pixel_split.X_train, pretrained_pixel_split.y_train)
    before = est.predict(pretrained_pixel_split.X_test)
    path = tmp_path / "m.joblib"
    joblib.dump(est, path)
    after = joblib.load(path).predict(pretrained_pixel_split.X_test)
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("Model", MODELS)
def test_module_stays_cpu_resident_after_predict(Model, pretrained_pixel_split):
    """Inference may use the GPU internally but leaves module_ CPU-resident
    afterward -- the case that matters most here: a full ResNet18/ViT-B/16
    forward pass over a real test split must not silently run on CPU."""
    est = Model(epochs=1, pretrained=False, batch_size=4).fit(
        pretrained_pixel_split.X_train, pretrained_pixel_split.y_train)
    est.predict(pretrained_pixel_split.X_test)
    assert next(est.module_.parameters()).device.type == "cpu"


@pytest.mark.parametrize("Model", MODELS)
def test_freeze_backbone_leaves_only_head_trainable(Model):
    est = Model(pretrained=False, freeze_backbone=True)
    module = est._build_module((3, 224, 224), 2)
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    assert trainable  # the replacement head
    assert all(("fc" in n or "heads" in n) for n in trainable)


@pytest.mark.parametrize("Model", MODELS)
def test_unfrozen_backbone_trains_every_parameter(Model):
    est = Model(pretrained=False, freeze_backbone=False)
    module = est._build_module((3, 224, 224), 2)
    assert all(p.requires_grad for p in module.parameters())


@pytest.mark.parametrize("Model", MODELS)
def test_device_auto_selects_gpu_when_available(Model, monkeypatch):
    """With ``device=None`` and a GPU present, training targets CUDA (no warning)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Model()._device().type == "cuda"


@pytest.mark.parametrize("Model", MODELS)
def test_device_falls_back_to_cpu_with_warning(Model, monkeypatch):
    """With ``device=None`` and no GPU, training falls back to CPU and warns."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CPU"):
        assert Model()._device().type == "cpu"


def test_build_pretrained_torch_registry_shape():
    from trainbase.model_registry import ModelSpec
    reg = build_pretrained_torch_registry()
    assert set(reg) == {"cnn_pretrained", "vit_pretrained"}
    for spec in reg.values():
        assert isinstance(spec, ModelSpec)
        est = spec.factory()
        assert est.get_params()["random_state"] == 42
        assert est.get_params()["pretrained"] is True
        assert est.get_params()["freeze_backbone"] is True
        assert spec.param_grid and all(k.startswith("clf__") for k in spec.param_grid)
