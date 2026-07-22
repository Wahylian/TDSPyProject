"""Tests for trainbase/model_registry.py, the classifier catalogue.

Pins the contract every entry honours: the catalogue is non-empty ModelSpecs,
each factory returns a fresh unfitted estimator seeded with RANDOM_STATE, and
every param_grid key is 'clf__'-prefixed. Construction-level only; no fitting.
"""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from trainbase.model_registry import (
    MODEL_REGISTRY,
    RANDOM_STATE,
    ModelSpec,
)


class TestRegistryContents:
    """Top-level shape of the registry."""

    def test_random_state_is_documented_seed(self):
        """RANDOM_STATE is the fixed reproducibility seed (42)."""
        assert RANDOM_STATE == 42

    def test_registry_is_non_empty(self):
        """The catalogue ships at least one model (guards vacuous sweeps below)."""
        assert MODEL_REGISTRY

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_every_entry_is_a_model_spec(self, key):
        """Each registry value is a ModelSpec."""
        assert isinstance(MODEL_REGISTRY[key], ModelSpec)


class TestFactoryContract:
    """Guarantees about the factory callable on each spec."""

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_factory_returns_fresh_unfitted_estimator(self, key):
        """factory() yields distinct, brand-new, unfitted estimators each call."""
        factory = MODEL_REGISTRY[key].factory
        first, second = factory(), factory()

        assert isinstance(first, BaseEstimator)
        assert first is not second

        with pytest.raises(NotFittedError):
            check_is_fitted(first)

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_factory_threads_random_state(self, key):
        """Each randomised estimator is seeded with RANDOM_STATE."""
        params = MODEL_REGISTRY[key].factory().get_params()
        assert params.get("random_state") == RANDOM_STATE


class TestParamGrids:
    """The hyperparameter grids handed to GridSearchCV."""

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_grid_keys_are_clf_prefixed_and_map_to_lists(self, key):
        """Every grid key targets the 'clf' step and maps to a non-empty list."""
        grid = MODEL_REGISTRY[key].param_grid
        assert grid, f"{key} ships a non-empty tuning grid"
        for param, values in grid.items():
            assert param.startswith("clf__"), param
            assert isinstance(values, list) and values

    def test_model_spec_default_param_grid_is_independent_empty_dict(self):
        """ModelSpec defaults param_grid to a fresh empty dict, not a shared one."""
        a = ModelSpec(factory=lambda: BaseEstimator())
        b = ModelSpec(factory=lambda: BaseEstimator())
        assert a.param_grid == {}
        a.param_grid["clf__C"] = [1.0]
        assert b.param_grid == {}


class TestNewClassicalModels:
    """The added classical estimators and their realizations."""

    def test_hard_svm_is_linearsvc_with_large_C(self):
        from sklearn.svm import LinearSVC
        est = MODEL_REGISTRY["hard_svm"].factory()
        assert isinstance(est, LinearSVC)
        assert est.get_params()["C"] >= 1e4

    def test_hard_svm_kernel_is_linear_svc_large_C(self):
        from sklearn.svm import SVC
        est = MODEL_REGISTRY["hard_svm_kernel"].factory()
        assert isinstance(est, SVC)
        params = est.get_params()
        assert params["kernel"] == "linear" and params["C"] >= 1e4

    def test_ridge_is_ridge_classifier(self):
        from sklearn.linear_model import RidgeClassifier
        assert isinstance(MODEL_REGISTRY["ridge"].factory(), RidgeClassifier)

    def test_hard_svm_separates_and_exposes_decision_function(self, feature_split):
        est = MODEL_REGISTRY["hard_svm"].factory()
        est.fit(feature_split.X_train, feature_split.y_train)
        assert hasattr(est, "decision_function")
        acc = est.score(feature_split.X_test, feature_split.y_test)
        assert acc > 0.8


class TestHistGradientBoosting:
    """The added histogram-based gradient boosting estimator (hgb)."""

    def test_hgb_is_hist_gradient_boosting(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        assert isinstance(MODEL_REGISTRY["hgb"].factory(), HistGradientBoostingClassifier)


class TestMLP:
    """The added iterative multi-layer perceptron estimator (mlp)."""

    def test_mlp_is_mlp_classifier(self):
        from sklearn.neural_network import MLPClassifier
        assert isinstance(MODEL_REGISTRY["mlp"].factory(), MLPClassifier)

    def test_mlp_exposes_loss_curve_after_fit(self, feature_split):
        """A fitted mlp records per-epoch loss in loss_curve_ (diagnostic hook)."""
        est = MODEL_REGISTRY["mlp"].factory()
        est.fit(feature_split.X_train, feature_split.y_train)
        assert hasattr(est, "loss_curve_") and len(est.loss_curve_) >= 1

    def test_mlp_separates_feature_split(self, feature_split):
        est = MODEL_REGISTRY["mlp"].factory()
        est.fit(feature_split.X_train, feature_split.y_train)
        assert est.score(feature_split.X_test, feature_split.y_test) > 0.8


class TestConditionalTorchRegistration:
    """cnn/vit register only when torch is importable."""

    def test_torch_models_present_iff_torch(self):
        torch_installed = True
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_installed = False
        torch_keys = {"cnn", "cnn_deep", "vit", "vit_deep"}
        present = torch_keys & set(MODEL_REGISTRY)
        if torch_installed:
            assert present == torch_keys
        else:
            assert present == set()
