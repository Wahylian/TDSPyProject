# New Models (ViT, CNN, Hard-SVM, Linear Regression) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new models — Hard-SVM, Linear Regression, CNN, and ViT — to the registry-driven classifier catalogue without changing the project's abstraction.

**Architecture:** Classical models (`hard_svm`, `hard_svm_kernel`, `ridge`, `linreg`) are plain sklearn estimators added to `MODEL_REGISTRY`. Deep models (`cnn`/`cnn_deep`, `vit`/`vit_deep`) are sklearn-compatible torch wrappers in a new optional module, registered only when torch imports. Both consume the existing flat feature matrix; the torch wrappers reshape flat pixels back to `(C,H,W)` internally, fed by new no-PCA pixel pipelines.

**Tech Stack:** Python, numpy, scikit-learn (existing); PyTorch/torchvision (new, optional, CPU wheels).

## Global Constraints

- Every `MODEL_REGISTRY` entry must satisfy `tests/test_model_registry.py`: a `ModelSpec`; `factory()` returns a fresh, **unfitted** `BaseEstimator` (new object each call; `check_is_fitted` raises `NotFittedError`); estimator exposes `random_state == RANDOM_STATE` (42); `param_grid` is **non-empty** with `clf__`-prefixed keys mapping to lists.
- `RANDOM_STATE = 42`, imported from `trainbase.model_registry`. Thread it into every randomised/estimator param.
- Do **not** modify `train_model.py`, `trainbase/training.py`, `trainbase/evaluation.py`, `trainbase/diagnostics.py`, or `trainbase/artifacts.py`.
- Torch is optional: importing `trainbase` and running any non-torch model must work with torch absent.
- Tests live in `./tests/`. Run with `.venv/Scripts/python.exe -m pytest`. Torch tests use `pytest.importorskip("torch")` and the `slow` marker (registered in `pytest.ini`).
- Follow existing file style: module docstring, thorough docstrings, `from __future__ import annotations`.

---

### Task 1: No-PCA pixel pipelines (`pixels`, `pixels_hq`)

**Files:**
- Modify: `prebuilt_pipelines.py` (add two static methods after `fast_embedding_pipeline`)
- Modify: `trainbase/pipeline_registry.py` (register two keys)
- Test: `tests/test_prebuilt_pipelines.py` (add cases)

**Interfaces:**
- Produces: `PrebuiltPipelines.pixels_pipeline() -> ImagePipeline` (64×64 grayscale → 4096 flat pixels, no reduce/scale); `PrebuiltPipelines.pixels_hq_pipeline() -> ImagePipeline` (128×128 → 16384). Registry keys `"pixels"`, `"pixels_hq"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prebuilt_pipelines.py`:

```python
class TestPixelPipelines:
    """No-PCA pixel pipelines that feed the raw-image torch models."""

    def test_pixels_pipeline_emits_flat_4096_no_reduce(self, image_batch):
        from prebuilt_pipelines import PrebuiltPipelines
        pipe = PrebuiltPipelines.pixels_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 64 * 64
        assert 0.0 <= float(X.min()) and float(X.max()) <= 1.0
        ops = [name for name, _ in pipe.operations]
        assert "reduce" not in ops and "scale" not in ops

    def test_pixels_hq_pipeline_emits_flat_16384(self, image_batch):
        from prebuilt_pipelines import PrebuiltPipelines
        pipe = PrebuiltPipelines.pixels_hq_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 128 * 128

    def test_both_registered(self):
        from trainbase.pipeline_registry import PIPELINE_REGISTRY
        assert "pixels" in PIPELINE_REGISTRY
        assert "pixels_hq" in PIPELINE_REGISTRY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prebuilt_pipelines.py::TestPixelPipelines -v`
Expected: FAIL (`AttributeError: ... pixels_pipeline` / KeyError on registry).

- [ ] **Step 3: Write minimal implementation**

In `prebuilt_pipelines.py`, add after `fast_embedding_pipeline`:

```python
    # ----------------------------------------------------------------------
    # Raw-pixel pipelines (no dimensionality reduction).
    # These deliberately OMIT 'reduce'/'scale': they emit the full flattened
    # pixel vector so a spatial model (CNN/ViT) can reshape it back to an image.
    # ----------------------------------------------------------------------

    @staticmethod
    def pixels_pipeline() -> ImagePipeline:
        """Raw 64x64 grayscale pixels, flattened — for CNN/ViT on real pixels.

        No PCA or standardization: the 64x64 grayscale image is min-max
        normalized to [0, 1] and flattened to a 4,096-length vector. The torch
        image models reshape this flat vector back to ``(1, 64, 64)`` internally,
        so the convolution / patch-embedding layers see genuine spatial pixels.

        Output: 4,096 flat pixel features per image (64x64 grayscale).
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
        ])

    @staticmethod
    def pixels_hq_pipeline() -> ImagePipeline:
        """Raw 128x128 grayscale pixels, flattened — for the deep CNN/ViT presets.

        Same idea as :meth:`pixels_pipeline` at higher resolution: 128x128
        grayscale, min-max normalized, flattened to 16,384 pixels. The torch
        models reshape it to ``(1, 128, 128)``.

        Output: 16,384 flat pixel features per image (128x128 grayscale).
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
        ])
```

In `trainbase/pipeline_registry.py`, extend the registry dict:

```python
    "no_denoise": PrebuiltPipelines.no_denoise_pipeline,  # svm minus denoise -> 150 PCA features
    # Raw-pixel pipelines (no PCA) for the torch image models (CNN/ViT).
    "pixels": PrebuiltPipelines.pixels_pipeline,        # 64x64 grayscale   -> 4096 flat pixels
    "pixels_hq": PrebuiltPipelines.pixels_hq_pipeline,  # 128x128 grayscale -> 16384 flat pixels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prebuilt_pipelines.py::TestPixelPipelines -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prebuilt_pipelines.py trainbase/pipeline_registry.py tests/test_prebuilt_pipelines.py
git commit -m "feat: add no-PCA pixel pipelines (pixels, pixels_hq) for image models"
```

---

### Task 2: Classical SVM + Ridge registry entries (`hard_svm`, `hard_svm_kernel`, `ridge`)

**Files:**
- Modify: `trainbase/model_registry.py` (imports + three registry entries)
- Test: `tests/test_model_registry.py` (add targeted class)

**Interfaces:**
- Produces: `MODEL_REGISTRY` keys `"hard_svm"` (`LinearSVC(C=1e6)`), `"hard_svm_kernel"` (`SVC(kernel="linear", C=1e6)`), `"ridge"` (`RidgeClassifier`). All threaded with `RANDOM_STATE`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_registry.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_registry.py::TestNewClassicalModels -v`
Expected: FAIL (`KeyError: 'hard_svm'`).

- [ ] **Step 3: Write minimal implementation**

In `trainbase/model_registry.py`, extend the sklearn imports:

```python
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
```

Add these entries inside `MODEL_REGISTRY` (after `logreg`):

```python
    # Hard-margin SVM — a huge C drives the soft margin toward the hard-margin
    # limit (no slack). LinearSVC is the fast, purpose-built linear realization;
    # it exposes decision_function for ROC/PR-AUC.
    "hard_svm": ModelSpec(
        factory=lambda: LinearSVC(C=1e6, random_state=RANDOM_STATE),
        param_grid={"clf__C": [1e4, 1e6]},
    ),
    # Same hard margin via the kernel SVC with a linear kernel — mirrors the
    # existing 'svm' entry's style for a like-for-like comparison.
    "hard_svm_kernel": ModelSpec(
        factory=lambda: SVC(kernel="linear", C=1e6, random_state=RANDOM_STATE),
        param_grid={"clf__C": [1e4, 1e6]},
    ),
    # Ridge (least-squares) classifier — "linear regression as a classifier":
    # it regresses the class targets and thresholds. Exposes decision_function.
    "ridge": ModelSpec(
        factory=lambda: RidgeClassifier(random_state=RANDOM_STATE),
        param_grid={"clf__alpha": [0.1, 1.0, 10.0]},
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_registry.py -v`
Expected: PASS (new class + the existing parametrized sweep now covering the three keys).

- [ ] **Step 5: Commit**

```bash
git add trainbase/model_registry.py tests/test_model_registry.py
git commit -m "feat: add hard_svm, hard_svm_kernel, ridge classifiers to the registry"
```

---

### Task 3: `ThresholdedLinearRegression` + `linreg` entry

**Files:**
- Create: `trainbase/linear_models.py`
- Modify: `trainbase/model_registry.py` (import + `linreg` entry)
- Test: `tests/test_linear_models.py`

**Interfaces:**
- Consumes: `RANDOM_STATE` from `trainbase.model_registry`.
- Produces: `trainbase.linear_models.ThresholdedLinearRegression(fit_intercept=True, threshold=0.5, random_state=RANDOM_STATE)` — a `BaseEstimator, ClassifierMixin` with `fit`, `predict`, `decision_function`, `classes_`, `reg_`. `MODEL_REGISTRY["linreg"]` uses it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_linear_models.py`:

```python
"""Tests for trainbase.linear_models.ThresholdedLinearRegression.

The wrapper turns plain least-squares LinearRegression into a binary classifier
(regress 0/1 targets, threshold at 0.5), exposing the sklearn classifier surface
the registry contract and evaluation suite expect.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from trainbase.linear_models import ThresholdedLinearRegression
from trainbase.model_registry import RANDOM_STATE


def test_unfitted_raises_not_fitted():
    with pytest.raises(NotFittedError):
        check_is_fitted(ThresholdedLinearRegression())


def test_exposes_random_state_default_seed():
    assert ThresholdedLinearRegression().get_params()["random_state"] == RANDOM_STATE


def test_fit_predict_separates_classes(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    acc = (clf.predict(feature_split.X_test) == feature_split.y_test).mean()
    assert acc > 0.8


def test_decision_function_shape_and_ordering(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    scores = clf.decision_function(feature_split.X_test)
    assert scores.shape == (len(feature_split.y_test),)
    # class-1 test points should score higher on average than class-0 ones.
    y = feature_split.y_test
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_predict_labels_are_from_classes(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    assert set(np.unique(clf.predict(feature_split.X_test))).issubset({0, 1})


def test_clone_and_get_params_roundtrip():
    clf = ThresholdedLinearRegression(fit_intercept=False, threshold=0.3)
    twin = clone(clf)
    assert twin.get_params() == clf.get_params()


def test_rejects_non_binary():
    X = np.random.default_rng(0).random((9, 3))
    y = np.array([0, 1, 2] * 3)
    with pytest.raises(ValueError):
        ThresholdedLinearRegression().fit(X, y)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_linear_models.py -v`
Expected: FAIL (`ModuleNotFoundError: trainbase.linear_models`).

- [ ] **Step 3: Write minimal implementation**

Create `trainbase/linear_models.py`:

```python
"""Linear-regression-as-classifier estimator for the model registry.

``ThresholdedLinearRegression`` wraps plain ordinary-least-squares
:class:`~sklearn.linear_model.LinearRegression` into a binary classifier: it
regresses 0/1 class targets and thresholds the continuous output at 0.5. This
gives the project a literal "linear regression" classifier alongside the
least-squares :class:`~sklearn.linear_model.RidgeClassifier`, while exposing the
sklearn classifier surface (``predict`` / ``decision_function`` / ``classes_``)
that the registry contract and the evaluation suite rely on.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from .model_registry import RANDOM_STATE


class ThresholdedLinearRegression(BaseEstimator, ClassifierMixin):
    """Binary classifier: least-squares regression thresholded at ``threshold``.

    Args:
        fit_intercept: Passed through to the internal ``LinearRegression``.
        threshold: Decision boundary on the regression output; predictions are
            the positive class where ``raw >= threshold``.
        random_state: Held only to satisfy the registry contract (every entry
            exposes the shared seed). OLS is deterministic, so it is unused.
    """

    def __init__(self, fit_intercept: bool = True, threshold: float = 0.5,
                 random_state: int = RANDOM_STATE):
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Fit OLS on 0/1-encoded targets; store ``classes_`` and ``reg_``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "ThresholdedLinearRegression supports binary classification "
                f"only; got {len(self.classes_)} classes."
            )
        # Regress the indicator of the positive (second) class.
        y01 = (y == self.classes_[1]).astype(float)
        self.reg_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.reg_.fit(X, y01)
        return self

    def decision_function(self, X):
        """Continuous score, centered on the threshold (for PR-AUC / ROC-AUC)."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return self.reg_.predict(X) - self.threshold

    def predict(self, X):
        """Predict class labels from ``classes_`` by thresholding the score."""
        idx = (self.decision_function(X) >= 0).astype(int)
        return self.classes_[idx]
```

In `trainbase/model_registry.py`, add the import near the other project imports (below the sklearn block):

```python
from .linear_models import ThresholdedLinearRegression
```

Add the entry to `MODEL_REGISTRY` (after `ridge`):

```python
    # Plain linear regression used as a classifier: regress 0/1 targets and
    # threshold at 0.5. The most literal "linear regression" baseline.
    "linreg": ModelSpec(
        factory=lambda: ThresholdedLinearRegression(random_state=RANDOM_STATE),
        param_grid={"clf__fit_intercept": [True, False]},
    ),
```

Note: place the `from .linear_models import ThresholdedLinearRegression` line **after** `RANDOM_STATE` and `ModelSpec` are defined is not required (linear_models only needs `RANDOM_STATE`, which is defined at import time of model_registry before this import executes). Put the import at the **bottom** of `model_registry.py`, just above the `MODEL_REGISTRY` literal is NOT possible (MODEL_REGISTRY uses the name). Instead: define `RANDOM_STATE` and `ModelSpec` (already near top), then add the import immediately before the `MODEL_REGISTRY` dict literal so the name is available inside it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_linear_models.py tests/test_model_registry.py -v`
Expected: PASS (including the registry sweep now covering `linreg`).

- [ ] **Step 5: Commit**

```bash
git add trainbase/linear_models.py trainbase/model_registry.py tests/test_linear_models.py
git commit -m "feat: add ThresholdedLinearRegression (linreg) classifier to the registry"
```

---

### Task 4: Torch image classifiers (CNN + ViT wrappers)

**Files:**
- Create: `trainbase/torch_models.py`
- Modify: `tests/conftest.py` (add `pixel_split` fixture)
- Test: `tests/test_torch_models.py`

**Interfaces:**
- Consumes: `RANDOM_STATE`, `ModelSpec` from `trainbase.model_registry`.
- Produces:
  - `trainbase.torch_models.CNNClassifier(epochs=10, lr=1e-3, batch_size=32, weight_decay=0.0, image_shape=None, device=None, random_state=RANDOM_STATE, channels=(16, 32), n_blocks=2)`
  - `trainbase.torch_models.ViTClassifier(epochs=10, lr=1e-3, batch_size=32, weight_decay=0.0, image_shape=None, device=None, random_state=RANDOM_STATE, patch_size=8, embed_dim=64, depth=2, n_heads=4, mlp_dim=128)`
  - Both are `BaseEstimator, ClassifierMixin` with `fit`, `predict`, `predict_proba`, `classes_`, `module_`, `image_shape_`, `n_features_in_`.
  - `trainbase.torch_models.build_torch_registry() -> Dict[str, ModelSpec]` (used in Task 5).

**Setup (once, before Step 1):** install CPU-only torch into the venv:

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Confirm: `.venv/Scripts/python.exe -c "import torch, torchvision; print(torch.__version__)"`

- [ ] **Step 1: Write the failing test**

Add the `pixel_split` fixture to `tests/conftest.py` (after `feature_split`):

```python
@pytest.fixture
def pixel_split(rng) -> SimpleNamespace:
    """Tiny separable flat-pixel splits for the torch image models.

    Emulates the ``pixels`` pipeline output: flat 8x8 grayscale vectors (64
    features) in [0, 1]. Class 0 is dim (~0.2), class 1 is bright (~0.8), so a
    small CNN/ViT separates them in a couple of epochs. Deliberately tiny so a
    forward/backward pass runs in milliseconds.
    """
    side = 8
    f = side * side

    def block(level: float, n: int) -> np.ndarray:
        x = rng.normal(level, 0.05, size=(n, f)).astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    def split(n: int):
        X = np.vstack([block(0.2, n), block(0.8, n)]).astype(np.float32)
        y = np.array([0] * n + [1] * n, dtype=int)
        return X, y

    X_train, y_train = split(8)
    X_val, y_val = split(4)
    X_test, y_test = split(4)
    return SimpleNamespace(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        image_shape=(1, side, side),
    )
```

Create `tests/test_torch_models.py`:

```python
"""Tests for trainbase.torch_models — the CNN/ViT wrappers on raw pixels.

Skipped entirely when torch is not installed. Everything runs on a tiny
synthetic pixel split with 1 epoch so it stays fast and deterministic.
"""

from __future__ import annotations

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


def test_build_torch_registry_shape():
    from trainbase.model_registry import ModelSpec
    reg = build_torch_registry()
    assert set(reg) == {"cnn", "cnn_deep", "vit", "vit_deep"}
    for spec in reg.values():
        assert isinstance(spec, ModelSpec)
        est = spec.factory()
        assert est.get_params()["random_state"] == 42
        assert spec.param_grid and all(k.startswith("clf__") for k in spec.param_grid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_torch_models.py -v`
Expected: FAIL (`ModuleNotFoundError: trainbase.torch_models`).

- [ ] **Step 3: Write minimal implementation**

Create `trainbase/torch_models.py`:

```python
"""Torch-backed image classifiers (CNN, ViT) for the model registry.

These wrap small PyTorch networks in the sklearn estimator interface so they
drop into :data:`trainbase.model_registry.MODEL_REGISTRY` like any other
classifier — selected by ``--model``, tuned by ``GridSearchCV``, scored by the
shared evaluation suite. Torch is an OPTIONAL dependency: this module is only
imported (and its models only registered) when ``import torch`` succeeds.

Raw-pixel faithfulness
----------------------
The feature front-end hands the estimator a flat feature matrix ``X``. Paired
with a no-PCA pixel pipeline (``pixels`` / ``pixels_hq``), each row is the
flattened grayscale image. :meth:`_TorchImageClassifier.fit` reshapes it back to
``(N, C, H, W)`` before the first conv / patch-embedding layer, so the network
operates on genuine spatial pixels. When ``image_shape`` is not given it is
inferred as square grayscale from the vector width.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import torch
from torch import nn

from .model_registry import RANDOM_STATE, ModelSpec

logger = logging.getLogger(__name__)


def _resolve_image_shape(
    n_features: int, image_shape: Optional[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Resolve/validate the ``(C, H, W)`` a flat vector of width ``n_features`` maps to.

    If ``image_shape`` is given, validate ``C*H*W == n_features``. Otherwise infer
    a square single-channel image (``C=1, H=W=sqrt(n_features)``).

    Raises:
        ValueError: if the explicit shape does not match, or a square grayscale
            image cannot be inferred.
    """
    if image_shape is not None:
        c, h, w = image_shape
        if c * h * w != n_features:
            raise ValueError(
                f"image_shape {image_shape} has {c * h * w} elements but the "
                f"feature width is {n_features}."
            )
        return int(c), int(h), int(w)
    side = int(round(n_features ** 0.5))
    if side * side != n_features:
        raise ValueError(
            f"Cannot infer a square grayscale image from feature width "
            f"{n_features}; pass image_shape=(C, H, W) explicitly."
        )
    return 1, side, side


class _TorchImageClassifier(BaseEstimator, ClassifierMixin):
    """Base sklearn wrapper: reshape flat pixels, train a torch module, predict.

    Subclasses implement :meth:`_build_module`. All constructor arguments are
    plain attributes so sklearn ``clone`` / ``get_params`` / ``GridSearchCV``
    work. Fitted state (``classes_``, ``module_``, ``image_shape_``,
    ``n_features_in_``) is set only in :meth:`fit`.
    """

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.image_shape = image_shape
        self.device = device
        self.random_state = random_state

    # --- subclass hook ----------------------------------------------------
    def _build_module(self, in_shape: Tuple[int, int, int], n_classes: int) -> nn.Module:
        raise NotImplementedError

    def _device(self) -> "torch.device":
        return torch.device(self.device) if self.device is not None else torch.device("cpu")

    # --- sklearn API ------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.image_shape_ = _resolve_image_shape(self.n_features_in_, self.image_shape)

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = self._device()

        module = self._build_module(self.image_shape_, n_classes).to(device)
        c, h, w = self.image_shape_
        X_t = torch.from_numpy(X).view(-1, c, h, w).to(device)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)
        y_t = torch.from_numpy(y_idx).to(device)

        opt = torch.optim.Adam(module.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n = X_t.shape[0]
        bs = max(1, int(self.batch_size))
        module.train()
        for _ in range(int(self.epochs)):
            for start in range(0, n, bs):
                idx = rng.permutation(n)[start:start + bs] if False else None
                # deterministic shuffled minibatches
            perm = rng.permutation(n)
            for start in range(0, n, bs):
                sel = perm[start:start + bs]
                xb, yb = X_t[sel], y_t[sel]
                opt.zero_grad()
                loss = loss_fn(module(xb), yb)
                loss.backward()
                opt.step()
        module.eval()
        # Store on CPU so the fitted estimator pickles/loads without a GPU.
        self.module_ = module.to(torch.device("cpu"))
        return self

    def _logits(self, X) -> "torch.Tensor":
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; estimator was fit on "
                f"{self.n_features_in_}."
            )
        c, h, w = self.image_shape_
        X_t = torch.from_numpy(X).view(-1, c, h, w)
        self.module_.eval()
        with torch.no_grad():
            return self.module_(X_t)

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(self._logits(X), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _CNNModule(nn.Module):
    """Small conv stack (Conv→ReLU→MaxPool blocks) → linear head."""

    def __init__(self, in_shape, n_classes, channels=(16, 32), n_blocks=2):
        super().__init__()
        c, h, w = in_shape
        layers = []
        prev = c
        for out_c in channels[:n_blocks]:
            layers += [nn.Conv2d(prev, out_c, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            prev, h, w = out_c, h // 2, w // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(prev * h * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNClassifier(_TorchImageClassifier):
    """A small convolutional image classifier."""

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE,
                 channels: Tuple[int, ...] = (16, 32), n_blocks: int = 2):
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, image_shape=image_shape,
                         device=device, random_state=random_state)
        self.channels = channels
        self.n_blocks = n_blocks

    def _build_module(self, in_shape, n_classes):
        return _CNNModule(in_shape, n_classes, self.channels, self.n_blocks)


class _ViTModule(nn.Module):
    """Minimal Vision Transformer: patch embed → transformer encoder → CLS head."""

    def __init__(self, in_shape, n_classes, patch_size=8, embed_dim=64,
                 depth=2, n_heads=4, mlp_dim=128):
        super().__init__()
        c, h, w = in_shape
        if h % patch_size or w % patch_size:
            raise ValueError(
                f"image size ({h}x{w}) must be divisible by patch_size {patch_size}."
            )
        self.patch = nn.Conv2d(c, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (h // patch_size) * (w // patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        b = x.shape[0]
        p = self.patch(x).flatten(2).transpose(1, 2)      # (b, n_patches, embed)
        z = torch.cat([self.cls.expand(b, -1, -1), p], dim=1) + self.pos
        return self.head(self.encoder(z)[:, 0])


class ViTClassifier(_TorchImageClassifier):
    """A small Vision Transformer image classifier."""

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE,
                 patch_size: int = 8, embed_dim: int = 64, depth: int = 2,
                 n_heads: int = 4, mlp_dim: int = 128):
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, image_shape=image_shape,
                         device=device, random_state=random_state)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim

    def _build_module(self, in_shape, n_classes):
        return _ViTModule(in_shape, n_classes, self.patch_size, self.embed_dim,
                          self.depth, self.n_heads, self.mlp_dim)


def build_torch_registry() -> Dict[str, ModelSpec]:
    """Return the torch model entries for :data:`MODEL_REGISTRY`.

    Two presets per architecture: a light default (CPU-friendly, ~10 epochs,
    64x64 via the ``pixels`` pipeline) and a heavier variant (~20-25 epochs,
    128x128 via ``pixels_hq``). Each ships a tiny ``clf__lr`` grid so tuning
    stays cheap while honoring the non-empty-grid registry contract.
    """
    return {
        "cnn": ModelSpec(
            factory=lambda: CNNClassifier(epochs=10, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "cnn_deep": ModelSpec(
            factory=lambda: CNNClassifier(
                epochs=25, channels=(32, 64, 128), n_blocks=3,
                random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "vit": ModelSpec(
            factory=lambda: ViTClassifier(epochs=10, depth=2, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "vit_deep": ModelSpec(
            factory=lambda: ViTClassifier(
                epochs=20, depth=4, embed_dim=96, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
    }
```

Note: delete the dead `if False` scaffolding line before finalizing — the real minibatch loop is the `perm = rng.permutation(n)` block. (Written explicitly here to flag it; the implementation should contain only the `perm`-based loop.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_torch_models.py -v`
Expected: PASS (all parametrized CNN/ViT cases + registry-shape test).

- [ ] **Step 5: Commit**

```bash
git add trainbase/torch_models.py tests/test_torch_models.py tests/conftest.py
git commit -m "feat: add CNN and ViT torch image classifiers (sklearn-wrapped)"
```

---

### Task 5: Conditional registration + optional dependency + docs

**Files:**
- Modify: `trainbase/model_registry.py` (guarded `MODEL_REGISTRY.update(build_torch_registry())`)
- Modify: `requirements.txt` (optional torch block)
- Modify: `trainbase/model_registry.py` module docstring (mention new models)
- Test: `tests/test_model_registry.py` (conditional-registration test)

**Interfaces:**
- Consumes: `build_torch_registry` from `trainbase.torch_models` (Task 4).
- Produces: `MODEL_REGISTRY` contains `cnn`/`cnn_deep`/`vit`/`vit_deep` iff torch is importable.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model_registry.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_registry.py::TestConditionalTorchRegistration -v`
Expected: FAIL (torch installed but keys absent — registration not wired yet).

- [ ] **Step 3: Write minimal implementation**

At the **bottom** of `trainbase/model_registry.py` (after the `MODEL_REGISTRY` literal), add:

```python
# --- Optional deep models (CNN / ViT) -------------------------------------
# Registered only when torch is importable, so the project imports and runs
# unchanged without the optional PyTorch dependency. Membership is dynamic by
# design (see the package docstring): with torch installed, `--model cnn/vit`
# become available automatically.
try:
    from .torch_models import build_torch_registry

    MODEL_REGISTRY.update(build_torch_registry())
except ImportError:  # pragma: no cover - exercised only when torch is absent
    pass
```

Update the module docstring's "how to add" example region to note the optional torch models (append a short paragraph after the existing example):

```python
#     The registry also carries optional deep models — ``cnn``/``cnn_deep`` and
#     ``vit``/``vit_deep`` — which are registered only when PyTorch is installed
#     (see ``trainbase/torch_models.py``). Pair them with a raw-pixel pipeline:
#     ``--model cnn --pipeline pixels`` (or ``--model cnn_deep --pipeline pixels_hq``).
```

In `requirements.txt`, add after the keras/tensorflow optional block:

```
# Optional: CNN / ViT deep models (trainbase/torch_models.py). CPU wheels are
# fine; only needed if you run --model cnn/cnn_deep/vit/vit_deep. Install with:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# torch
# torchvision
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_registry.py -v`
Expected: PASS. The registry sweep tests now also cover cnn/vit (fresh/unfitted, random_state, grid shape) since torch is installed.

- [ ] **Step 5: Commit**

```bash
git add trainbase/model_registry.py requirements.txt tests/test_model_registry.py
git commit -m "feat: conditionally register CNN/ViT; document optional torch dependency"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite (including slow torch tests)**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (existing + new). Note the count of new tests.

- [ ] **Step 2: Run the suite excluding slow (verifies torch-free path shape)**

Run: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
Expected: pass; torch tests deselected.

- [ ] **Step 3: Smoke-check the registries import cleanly**

Run:
```bash
.venv/Scripts/python.exe -c "from trainbase import MODEL_REGISTRY, PIPELINE_REGISTRY; print(sorted(MODEL_REGISTRY)); print(sorted(PIPELINE_REGISTRY))"
```
Expected: model list includes `hard_svm, hard_svm_kernel, ridge, linreg, cnn, cnn_deep, vit, vit_deep`; pipeline list includes `pixels, pixels_hq`.

- [ ] **Step 4: Commit (if any doc tweaks)**

```bash
git add -A && git commit -m "test: verify full suite green with new models" || echo "nothing to commit"
```

## Self-Review

**Spec coverage:**
- Hard-SVM (both realizations) → Task 2 (`hard_svm`, `hard_svm_kernel`). ✓
- Linear Regression (both realizations: Ridge + thresholded) → Task 2 (`ridge`) + Task 3 (`linreg`). ✓
- CNN (light + deep) → Task 4 + Task 5 registration. ✓
- ViT (light + deep) → Task 4 + Task 5 registration. ✓
- Raw-pixel faithfulness via reshape + pixel pipelines → Task 1 + Task 4. ✓
- Optional/conditional torch → Task 4 setup + Task 5. ✓
- Registry contract (fresh/unfitted, random_state, non-empty clf__ grid) → honored in Tasks 2–5. ✓
- Tests under ./tests/ → every task adds tests. ✓
- No changes to train_model/evaluation/etc. → respected. ✓

**Placeholder scan:** One intentional flag in Task 4 Step 3 (the `if False` scaffolding line) with an explicit instruction to remove it; the correct loop is shown. No other TBD/placeholder.

**Type consistency:** `build_torch_registry` signature matches between Task 4 (produced) and Task 5 (consumed). `ThresholdedLinearRegression` constructor/attributes match between Task 3 definition and its test. Estimator fitted attributes (`classes_`, `module_`, `image_shape_`, `n_features_in_`, `reg_`) are consistent across implementation and tests.
