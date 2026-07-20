"""
Tests for ``trainbase/model_registry.py`` — the classifier catalogue.

The registry is the single place a new estimator is added, so these tests pin
the *contract* every entry must honour rather than any one model's behaviour:

* the catalogue is non-empty and every entry is a :class:`ModelSpec`;
* each ``factory`` is a true factory — it returns a *fresh, unfitted* estimator
  every call, so runs never share fitted state;
* the shared :data:`RANDOM_STATE` is threaded into every randomised estimator;
* every ``param_grid`` key is ``clf__``-prefixed (the grids target the ``"clf"``
  step of the pipeline built in ``build_estimator``) and maps to a list.

No model is actually fitted here; construction-level guarantees only.
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
        """``RANDOM_STATE`` is the fixed reproducibility seed (42).

        The whole package threads this one constant through every randomised
        operation; pinning it guards the reproducibility guarantee.
        """
        assert RANDOM_STATE == 42

    def test_registry_is_non_empty(self):
        """The catalogue ships at least one model.

        Sweeps below parametrise over the live registry, so a broken or empty
        import would silently collect *zero* cases and pass vacuously; this
        guards that ``--model`` always has something to select.
        """
        assert MODEL_REGISTRY

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_every_entry_is_a_model_spec(self, key):
        """Each registry value is a :class:`ModelSpec` (the common entry type).

        Args:
            key: A registered model name from the sweep.
        """
        assert isinstance(MODEL_REGISTRY[key], ModelSpec)


class TestFactoryContract:
    """Guarantees about the ``factory`` callable on each spec."""

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_factory_returns_fresh_unfitted_estimator(self, key):
        """``factory()`` yields a brand-new, unfitted sklearn estimator.

        Two calls must return *distinct* objects (so concurrent/repeated runs
        never share state), and the returned estimator must not yet be fitted —
        ``check_is_fitted`` should raise ``NotFittedError``.

        Args:
            key: A registered model name from the sweep.
        """
        factory = MODEL_REGISTRY[key].factory
        first, second = factory(), factory()

        # Distinct, genuine estimators.
        assert isinstance(first, BaseEstimator)
        assert first is not second

        # A fresh estimator carries no fitted attributes yet.
        with pytest.raises(NotFittedError):
            check_is_fitted(first)

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_factory_threads_random_state(self, key):
        """Each randomised estimator is seeded with :data:`RANDOM_STATE`.

        Every registry estimator exposes a ``random_state`` param and must carry
        the shared seed, so the run is reproducible end to end.

        Args:
            key: A registered model name from the sweep.
        """
        params = MODEL_REGISTRY[key].factory().get_params()
        assert params.get("random_state") == RANDOM_STATE


class TestParamGrids:
    """The hyperparameter grids handed to ``GridSearchCV``."""

    @pytest.mark.parametrize("key", sorted(MODEL_REGISTRY))
    def test_grid_keys_are_clf_prefixed_and_map_to_lists(self, key):
        """Every grid key targets the ``"clf"`` step and maps to a list of values.

        ``build_estimator`` wraps the classifier as the ``"clf"`` step of a
        ``Pipeline``, so ``GridSearchCV`` only finds these params if the keys are
        ``clf__``-prefixed; the values must be iterables of candidates (lists).

        Args:
            key: A registered model name from the sweep.
        """
        grid = MODEL_REGISTRY[key].param_grid
        assert grid, f"{key} ships a non-empty tuning grid"
        for param, values in grid.items():
            assert param.startswith("clf__"), param
            assert isinstance(values, list) and values

    def test_model_spec_default_param_grid_is_independent_empty_dict(self):
        """``ModelSpec`` defaults ``param_grid`` to a *fresh* empty dict.

        The dataclass uses ``default_factory=dict`` precisely so two specs never
        share one mutable grid object — mutating one default must not leak into
        another.
        """
        a = ModelSpec(factory=lambda: BaseEstimator())
        b = ModelSpec(factory=lambda: BaseEstimator())
        assert a.param_grid == {}
        a.param_grid["clf__C"] = [1.0]
        # b's default grid is untouched by mutating a's.
        assert b.param_grid == {}
