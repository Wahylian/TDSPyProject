"""Tests for trainbase.torch_models — the CNN/ViT wrappers on raw pixels.

Skipped entirely when torch is not installed. Everything runs on a tiny
synthetic pixel split with 1 epoch so it stays fast and deterministic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from trainbase.torch_models import CNNClassifier, ViTClassifier, build_torch_registry

pytestmark = pytest.mark.slow

MODELS = [CNNClassifier, ViTClassifier]


@pytest.mark.parametrize("Model", MODELS)
def test_unfitted_raises(Model):
    with pytest.raises(NotFittedError):
        check_is_fitted(Model())


@pytest.mark.parametrize("Model", MODELS)
def test_is_base_estimator_and_clones(Model):
    est = Model(epochs=1)
    assert isinstance(est, BaseEstimator)
    assert clone(est).get_params() == est.get_params()


@pytest.mark.parametrize("Model", MODELS)
def test_fit_predict_shapes_and_labels(Model, pixel_split):
    est = Model(epochs=2).fit(pixel_split.X_train, pixel_split.y_train)
    preds = est.predict(pixel_split.X_test)
    assert preds.shape == (len(pixel_split.y_test),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.parametrize("Model", MODELS)
def test_predict_proba_is_distribution(Model, pixel_split):
    est = Model(epochs=1).fit(pixel_split.X_train, pixel_split.y_train)
    proba = est.predict_proba(pixel_split.X_test)
    assert proba.shape == (len(pixel_split.y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("Model", MODELS)
def test_infers_square_grayscale_shape(Model, pixel_split):
    est = Model(epochs=1).fit(pixel_split.X_train, pixel_split.y_train)
    assert est.image_shape_ == (1, 8, 8)
    assert est.n_features_in_ == 64


@pytest.mark.parametrize("Model", MODELS)
def test_explicit_image_shape_respected(Model, pixel_split):
    est = Model(epochs=1, image_shape=(1, 8, 8)).fit(
        pixel_split.X_train, pixel_split.y_train)
    assert est.image_shape_ == (1, 8, 8)


@pytest.mark.parametrize("Model", MODELS)
def test_mismatched_width_raises(Model):
    X = np.random.default_rng(0).random((4, 63)).astype(np.float32)  # not a square
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        Model(epochs=1).fit(X, y)


@pytest.mark.parametrize("Model", MODELS)
def test_joblib_roundtrip_predicts_identically(Model, pixel_split, tmp_path):
    import joblib
    est = Model(epochs=2).fit(pixel_split.X_train, pixel_split.y_train)
    before = est.predict(pixel_split.X_test)
    path = tmp_path / "m.joblib"
    joblib.dump(est, path)
    after = joblib.load(path).predict(pixel_split.X_test)
    np.testing.assert_array_equal(before, after)


def test_learns_separable_pixels(pixel_split):
    # A couple of epochs should clear chance on the easy dim-vs-bright split.
    est = CNNClassifier(epochs=15, lr=1e-2).fit(
        pixel_split.X_train, pixel_split.y_train)
    acc = (est.predict(pixel_split.X_test) == pixel_split.y_test).mean()
    assert acc > 0.75


@pytest.mark.parametrize("Model", MODELS)
def test_device_auto_selects_gpu_when_available(Model, monkeypatch):
    """With ``device=None`` and a GPU present, training targets CUDA (no warning)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail here
        assert Model()._device().type == "cuda"


@pytest.mark.parametrize("Model", MODELS)
def test_device_falls_back_to_cpu_with_warning(Model, monkeypatch):
    """With ``device=None`` and no GPU, training falls back to CPU and warns."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CPU"):
        assert Model()._device().type == "cpu"


@pytest.mark.parametrize("Model", MODELS)
def test_explicit_device_is_honored_silently(Model, monkeypatch):
    """An explicit ``device`` is used as-is, with no GPU probing or warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Model(device="cpu")._device().type == "cpu"


@pytest.mark.parametrize("Model", MODELS)
def test_inference_device_prefers_cuda_without_warning(Model, monkeypatch):
    """_inference_device resolves like _device but never warns (fit() already did)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Model()._inference_device().type == "cuda"


@pytest.mark.parametrize("Model", MODELS)
def test_inference_device_falls_back_to_cpu_without_warning(Model, monkeypatch):
    """Unlike _device, a missing GPU doesn't make _inference_device warn."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Model()._inference_device().type == "cpu"


@pytest.mark.parametrize("Model", MODELS)
def test_module_stays_cpu_resident_after_predict(Model, pixel_split):
    """predict() may use the real device internally but leaves module_ on CPU
    afterward, preserving fit()'s 'portable pickling' invariant."""
    est = Model(epochs=1).fit(pixel_split.X_train, pixel_split.y_train)
    est.predict(pixel_split.X_test)
    assert next(est.module_.parameters()).device.type == "cpu"


@pytest.mark.parametrize("Model", MODELS)
def test_predict_does_not_warn_about_missing_gpu(Model, pixel_split, monkeypatch):
    """Unlike fit(), predict() doesn't re-warn about a missing GPU on every call."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    est = Model(epochs=1).fit(pixel_split.X_train, pixel_split.y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.predict(pixel_split.X_test)


def test_build_torch_registry_shape():
    from trainbase.model_registry import ModelSpec
    reg = build_torch_registry()
    assert set(reg) == {"cnn", "cnn_deep", "vit", "vit_deep"}
    for spec in reg.values():
        assert isinstance(spec, ModelSpec)
        est = spec.factory()
        assert est.get_params()["random_state"] == 42
        assert spec.param_grid and all(k.startswith("clf__") for k in spec.param_grid)
