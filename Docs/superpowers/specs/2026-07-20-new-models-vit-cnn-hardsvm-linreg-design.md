# Design: Add ViT, CNN, Hard-SVM, and Linear Regression models

**Date:** 2026-07-20
**Status:** Approved design → implementation

## Goal

Add four new models to the project as additions to the existing registry-driven
classifier catalogue, **without changing the project's abstraction level**:

- **ViT** (Vision Transformer) — deep model on raw pixels
- **CNN** (Convolutional network) — deep model on raw pixels
- **Hard-SVM** (hard-margin support vector machine) — classical
- **Linear Regression** (as a classifier) — classical

All four are selected exactly like every existing model: by name via
`--model <key>`, resolved through `MODEL_REGISTRY`. `train_model.py`,
`build_estimator`, `tune_hyperparameters`, `evaluate`, `collect_diagnostics`,
and `save_artifacts` are **unchanged** — the models plug into the existing
`ModelSpec(factory, param_grid)` contract.

## Background: the abstraction being preserved

The project is a binary image classifier (`real` vs `fake`). Its core contract:

- `MODEL_REGISTRY: Dict[str, ModelSpec]`, where `ModelSpec` bundles a
  zero-arg `factory()` returning a **fresh, unfitted sklearn `BaseEstimator`**
  plus a `param_grid` of `clf__`-prefixed hyperparameters.
- The feature front-end (`trainbase/features.py`) turns each split's images into
  a **flat numpy feature matrix** `X` (fit on train, reused on val/test).
- `build_estimator` wraps the chosen classifier as the single `"clf"` step of a
  `Pipeline`; `tune_hyperparameters` grid-searches it on the val split; `evaluate`
  computes classification metrics (accuracy/precision/recall/F1/PR-AUC/ROC-AUC,
  confusion matrix, per-class report). AUC metrics use `predict_proba` if present,
  else `decision_function`.

**Existing test contract** (`tests/test_model_registry.py`) that every new entry
must honor:
1. Each entry is a `ModelSpec`.
2. `factory()` returns a fresh, **unfitted** `BaseEstimator` (a new object each call;
   `check_is_fitted` raises `NotFittedError`).
3. Each estimator exposes `random_state` equal to `RANDOM_STATE` (42).
4. Each `param_grid` is **non-empty** with `clf__`-prefixed keys mapping to lists.

## Design

### Group A — Classical models (natural sklearn estimators)

Added directly to `trainbase/model_registry.py`. "Do both" realizations were
requested, giving four keys:

| Key | Estimator | Rationale |
|-----|-----------|-----------|
| `hard_svm` | `LinearSVC(C=1e6, random_state=RANDOM_STATE)` | Hard margin = very large `C`, linear. Fast. Exposes `decision_function` → ROC/PR-AUC. |
| `hard_svm_kernel` | `SVC(kernel="linear", C=1e6, random_state=RANDOM_STATE)` | Kernel-SVC realization of the same hard margin (matches the existing `svm` style). |
| `ridge` | `RidgeClassifier(random_state=RANDOM_STATE)` | Least-squares linear classification — the standard "linear regression as a classifier." Exposes `decision_function`. |
| `linreg` | `ThresholdedLinearRegression(random_state=RANDOM_STATE)` | Literal linear regression thresholded at 0.5. |

`param_grid`s (non-empty, per contract), e.g.:
- `hard_svm`: `{"clf__C": [1e4, 1e6]}`
- `hard_svm_kernel`: `{"clf__C": [1e4, 1e6]}`
- `ridge`: `{"clf__alpha": [0.1, 1.0, 10.0]}`
- `linreg`: `{"clf__fit_intercept": [True, False]}`

**`ThresholdedLinearRegression`** — a thin `BaseEstimator, ClassifierMixin` in
`trainbase/model_registry.py` (or a small `trainbase/linear_models.py`):
- `__init__(self, fit_intercept=True, threshold=0.5, random_state=RANDOM_STATE)`
  (holds `random_state` only to satisfy the registry contract; it does not affect
  the deterministic least-squares fit).
- `fit(X, y)`: store `self.classes_ = unique(y)`; fit an internal
  `LinearRegression(fit_intercept=...)` on `X` against `y` as floats.
- `decision_function(X)`: the continuous regression output (centered at
  `threshold`, i.e. `raw - threshold`) — used for PR-AUC / ROC-AUC.
- `predict(X)`: `(raw >= threshold).astype(int)` mapped through `classes_`.
- No `predict_proba` (so `evaluate` falls back to `decision_function`, which is fine).
- No trailing-underscore attributes are set in `__init__` (keeps it "unfitted"
  until `fit`, per the contract).

### Group B — Deep models on raw pixels (ViT, CNN)

New module `trainbase/torch_models.py`. Torch is an **optional** dependency.

**Base class** `_TorchImageClassifier(BaseEstimator, ClassifierMixin)`:
- `__init__` params (all plain, so sklearn `clone`/`get_params`/`set_params` and
  `GridSearchCV` work): `epochs`, `lr`, `batch_size`, `weight_decay`,
  `image_shape=None`, `device=None`, `random_state=RANDOM_STATE`, plus
  architecture knobs used by subclasses (see below).
- **Input reshaping (faithfulness):** flat `X` of width `F` is reshaped to
  `(N, C, H, W)`. If `image_shape` is given, use it (validating `C*H*W == F`).
  Otherwise **infer** square grayscale: `C=1`, `H=W=round(sqrt(F))`, validating
  `H*W == F`. The conv/patch layers therefore see genuine spatial pixels; the flat
  vector is only transport through the sklearn `X`-matrix interface.
- `fit(X, y)`: seed all RNGs from `random_state`; build the module via
  `_build_module(in_shape, n_classes)`; train with Adam + cross-entropy for
  `epochs`; set `self.classes_`, `self.module_`, `self.n_features_in_`; move
  `module_` to CPU at the end for portable `joblib` persistence. Returns `self`.
- `predict_proba(X)`: softmax over logits (→ real probabilities, so ROC/PR-AUC
  use calibrated-ish scores rather than a bare margin).
- `predict(X)`: argmax → `classes_`.
- Persistence: `module_` is a `torch.nn.Module`, picklable by `joblib`. Trained on
  `device`, stored on CPU; `predict`/`predict_proba` run on CPU by default.

**Subclasses** override only `_build_module`:
- `CNNClassifier`: 2 conv blocks (Conv2d→ReLU→MaxPool) → flatten → linear head.
- `ViTClassifier`: patch embedding (Conv2d stride=patch) → learnable CLS token +
  positional embedding → `nn.TransformerEncoder` (depth `L`) → CLS → linear head.

Both keep the base `__init__` (no override), so `get_params` introspects the
shared signature and `clone` works.

**Two presets per architecture** ("create both options" — light + heavier),
four registry keys total:

| Key | Preset | Approx. config |
|-----|--------|----------------|
| `cnn` | light | 2 conv blocks, ~10 epochs, 64×64 grayscale (via `pixels`) |
| `cnn_deep` | heavier | 3 conv blocks + wider, ~25 epochs, 128×128 (via `pixels_hq`) |
| `vit` | light | patch=8, depth=2, embed≈64, ~10 epochs, 64×64 |
| `vit_deep` | heavier | patch=8, depth=4, wider, ~20 epochs, 128×128 |

Each ships a small non-empty `clf__`-prefixed grid (e.g. `{"clf__lr": [1e-3, 3e-4]}`)
to satisfy the registry contract while keeping tuning cheap (1–2 candidates).

**Conditional registration.** `cnn`/`cnn_deep`/`vit`/`vit_deep` are added to
`MODEL_REGISTRY` **only if `import torch` succeeds**. Without torch the project
imports and runs exactly as today. This matches the documented philosophy
("the set of available models is dynamic … whatever the registries currently
contain") and the existing keras/tensorflow-optional precedent. Implementation:
in `model_registry.py`, a guarded block
`try: from .torch_models import build_torch_registry; MODEL_REGISTRY.update(build_torch_registry()) except ImportError: pass`.

### Group C — Pixel pipelines (no PCA)

CNN/ViT need raw pixels, not PCA components. Add to `prebuilt_pipelines.py` and
register in `PIPELINE_REGISTRY`:

- `pixels`: grayscale → resize 64×64 → normalize(minmax) → vectorize → **4096 flat
  pixels** (no `reduce`/`scale`). Pairs with `cnn`/`vit` (infer → `(1,64,64)`).
- `pixels_hq`: grayscale → resize 128×128 → normalize → vectorize → **16384 flat
  pixels**. Pairs with `cnn_deep`/`vit_deep` (infer → `(1,128,128)`).

Run examples:
```
python train_model.py --model cnn       --pipeline pixels
python train_model.py --model vit_deep   --pipeline pixels_hq
python train_model.py --model hard_svm   --pipeline fast
python train_model.py --model linreg     --pipeline fast
```

### Dependencies

`requirements.txt` gains a commented **optional** block (mirroring the keras/
tensorflow block):
```
# Optional: CNN / ViT deep models (trainbase/torch_models.py). CPU wheels are
# fine; install only if you use --model cnn/vit.
# torch
# torchvision
```
CPU-only torch will be installed into `.venv` for development and to run the tests.

## Testing (all under `./tests/`)

TDD: tests are written first for each unit, then implementation to green.

1. **`tests/test_model_registry.py`** — the existing parametrized sweep now covers
   the four classical keys automatically (fresh/unfitted, `random_state`, grid
   shape). Add targeted cases: `hard_svm` is `LinearSVC` with large `C`; `ridge`
   is `RidgeClassifier`; `linreg` predicts on the separable `feature_split` and
   beats chance; `decision_function` exists where claimed.
2. **`tests/test_linear_models.py`** — `ThresholdedLinearRegression`: unfitted →
   `NotFittedError`; fit/predict on `feature_split` separates classes;
   `decision_function` shape; `clone`/`get_params` round-trip; threshold behavior.
3. **`tests/test_torch_models.py`** — `pytest.importorskip("torch")`, marked
   `slow`. On a tiny synthetic pixel matrix (e.g. 16×4096) with `epochs=1`:
   - `fit → predict` returns labels in `classes_`, correct length;
   - `predict_proba` rows sum to 1, shape `(n, 2)`;
   - `image_shape` inference reshapes correctly; explicit `image_shape` respected;
     mismatched width raises `ValueError`;
   - unfitted → `NotFittedError`; `clone`/`get_params` work;
   - joblib `dump`/`load` round-trip predicts identically;
   - a `build_estimator("cnn") + tune_hyperparameters(...)` end-to-end smoke test
     on the separable pixel data completes and yields a fitted `best_estimator_`;
   - conditional registration: if torch present, `cnn`/`vit` keys exist.
4. **`tests/test_prebuilt_pipelines.py`** — add `pixels`/`pixels_hq`: output is 2D
   `(n, 4096)` / `(n, 16384)`, float, no PCA step present.

Fixtures reuse `feature_split` (classical) and add a tiny pixel-matrix fixture
(torch) in `conftest.py`.

## Non-goals / YAGNI

- No changes to `train_model.py` CLI, evaluation, diagnostics, or artifact format.
- No GPU-specific tuning, mixed precision, data augmentation, or pretrained
  weights — the deep models are trained from scratch, small, and CPU-runnable.
- No new preprocessing operations; pixel pipelines reuse existing ops.

## Process

Sub-agent driven development + TDD:
1. This spec → implementation plan (writing-plans).
2. Execute independent tasks via subagents that write tests first, then implement
   to green, per the plan. Classical models, linear wrapper, torch models, and
   pixel pipelines are largely independent units.
