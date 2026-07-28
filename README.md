# TDSPyProject — Real vs. Fake Image Classifier

A from-scratch, registry-driven image-classification pipeline that distinguishes
real photos from AI-generated/deepfake images. Every stage — image preprocessing,
feature extraction, model choice, hyperparameter tuning, evaluation, and
run comparison — is built on NumPy/OpenCV/scikit-learn (with optional
Keras/VGG16 and PyTorch/torchvision paths), with no black-box AutoML.

## 1. Overview

Given a folder of `real/` and `fake/` images, the project:

1. Streams images per split (train/val/test) from a seeded manifest CSV.
2. Turns each image into a model-ready feature vector through a configurable
   **feature pipeline** (grayscale → resize → denoise → normalize → vectorize →
   reduce → scale).
3. Trains and tunes a classifier chosen from a **model registry**, selecting
   hyperparameters on a held-out validation split (not k-fold CV).
4. Evaluates on the test split against a naive majority-class baseline and
   writes a uniform, self-contained **artifact bundle** per run.
5. Compares runs across models and pipelines (leaderboards, resilience sweeps,
   grids) purely by reading those artifact bundles back.

The design is registry-driven: a run selects a model and a feature pipeline
**by name**, so adding either is a new registry entry, not a new code path.

## 2. Architecture

```
Ingestion (top-level scripts)
  download_dataset.py    -> download + restructure the Kaggle dataset into real/ + fake/
  create_split.py        -> write a seeded 70/15/15 manifest: datasets/dataset_split.csv
  extract_features.py    -> stream (image, label) pairs per split from the manifest

preprocessing/           -> image -> vector building blocks (the public API)
  io.py                  -> load images (PIL loaders = RGB, load_image_from_file = BGR)
  transforms.py          -> grayscale, resize, normalize, denoise
  vectorize.py           -> flat-pixel vectorization or a VGG16 embedding
  reduce.py              -> vec-pca / vec-jl (flat) or mat-pca / mat-jl (2D-preserving)
  scale.py               -> per-feature standardization
  pipeline.py            -> ImagePipeline: chains ops; fit_transform / transform / process
  batching.py            -> batch_process: one-shot batch run (refits per call)
  composition.py         -> compose() / pipeline_decorator() functional helpers

trainbase/               -> training backend used by train_model.py
  model_registry.py      -> MODEL_REGISTRY: classical, kernel, and (optional) deep models
  pipeline_registry.py   -> PIPELINE_REGISTRY: prebuilt feature pipelines
  torch_models.py        -> CNNClassifier / ViTClassifier, trained from scratch (needs torch)
  torch_pretrained_models.py -> frozen-backbone ResNet18 / ViT-B/16 (needs torchvision)
  features.py            -> build/fit/transform features, with an on-disk feature cache
  training.py            -> build_estimator + validation-holdout GridSearchCV tuning
  evaluation.py          -> accuracy/precision/recall/F1/PR-AUC/ROC-AUC + baseline
  diagnostics.py         -> feature importances, OOB score, hyperparameter/learning curves
  artifacts.py           -> build_metadata + save_artifacts (the run bundle)

train_model.py           -> CLI: load -> tune -> evaluate -> save one run

comparison/              -> read-only: scans artifacts/**/metadata.json and reports
  loader.py, records.py  -> discover + parse runs into typed RunRecords
  matrix.py, stats.py    -> leaderboard/resilience/grid tables, paired significance tests
  report.py, diagnostics_report.py -> render tables/diagnostics to Markdown/HTML/CSV/PNG
  cli.py (python -m comparison) -> leaderboard / resilience / grid subcommands
```

### Model registry

| Name | Estimator | Notes |
|---|---|---|
| `svm`, `hard_svm`, `hard_svm_kernel` | kernel/linear SVM | `hard_svm*` use a huge `C` for a hard margin |
| `rf` | Random Forest | exposes `oob_score_` |
| `logreg`, `ridge`, `linreg` | linear classifiers | `linreg` thresholds OLS regression |
| `hgb` | Histogram Gradient Boosting | |
| `mlp` | MLP (scikit-learn) | exposes `loss_curve_` |
| `cnn`, `cnn_deep`, `vit`, `vit_deep` | from-scratch PyTorch | registered only if `torch` is installed |
| `cnn_pretrained`, `vit_pretrained` | frozen ImageNet ResNet18/ViT-B/16 | registered only if `torchvision` is installed |

### Feature pipeline registry

`svm`, `fast`, `hq`, `no_denoise`, `svm_jl` (classical, PCA/JL-reduced + scaled);
`pixels`, `pixels_hq` (raw grayscale pixels for `cnn`/`vit`);
`pixels_pretrained` (raw RGB pixels for the pretrained backbones);
`embedding_pca`, `embedding_jl` (VGG16 embeddings, registered only if `keras` is
installed).

### Data flow (one training run)

```
manifest CSV --> get_feature_stream(split) --> images
    train images --> feature_pipeline.fit_transform --> X_train (basis/scaler LEARNED here)
    val/test     --> feature_pipeline.transform      --> X_val / X_test (same feature space)
X_train/X_val --> GridSearchCV (tuned on val, refit on train+val) --> best_model
best_model + X_test --> evaluate (+ naive baseline) --> metrics
best_model + search --> collect_diagnostics --> diagnostics
everything --> build_metadata --> save_artifacts --> artifacts/<model>/<run_id>/
```

## 3. Setup & Installation

The project uses a flat layout (`pytest.ini` sets `pythonpath = .`), so no
package install step is needed — run scripts from the project root with the
dependencies below installed into your Python environment (a virtualenv is
recommended).

```bash
# Default: core deps + the GPU (CUDA) torch/torchvision build.
pip install -r requirements.txt

# No NVIDIA/CUDA GPU, or the cu126 wheels aren't available for your platform?
pip install -r requirements-cpu.txt

# Or let it try GPU first and fall back to CPU automatically:
python install.py
```

`pip` has no built-in "try this wheel, else that one" logic, so
`requirements.txt` alone can't fall back from GPU to CPU by itself — that's
what `install.py` is for. `requirements-base.txt` holds the deps common to
both (`numpy`, `pandas`, `opencv-python`, `Pillow`, `scikit-learn`, `joblib`).

Everything below is optional and guarded at import time, so the project runs
without it — uncomment the matching line in `requirements.txt`, or install directly:

| Extra | Enables | Install |
|---|---|---|
| `keras` (torch backend, no tensorflow) | VGG16 embedding pipelines | `pip install keras` |
| `kagglehub` | `download_dataset.py` | `pip install kagglehub` |
| `pytest` | running the test suite | `pip install pytest` |

## 4. Usage

### Prepare the dataset

```bash
python download_dataset.py     # downloads + restructures the Kaggle dataset into real/, fake/
python create_split.py         # writes the seeded manifest: datasets/dataset_split.csv
```

`download_dataset.py` only touches the network when run directly (importing it
has no side effects) and needs Kaggle credentials configured for `kagglehub`.

### Train a model

`train_model.py` is the single training entry point; `--model` is required.

```bash
# Random Forest on the fast (64x64) pipeline:
python train_model.py --model rf --pipeline fast

# SVM on the default pipeline, capping sample sizes (0 = use all):
python train_model.py --model svm --max-train-samples 5000 --max-test-samples 5000

# A from-scratch CNN on raw pixels, with the extra learning-curve diagnostic:
python train_model.py --model cnn --pipeline pixels --diagnostics

# Fully custom feature pipeline (inline JSON; include reduce/scale yourself):
python train_model.py --model svm --pipeline-spec '[
  ["grayscale", {}],
  ["resize", {"target_size": [128, 128], "preserve_aspect": true}],
  ["normalize", {"method": "minmax"}],
  ["vectorize", {}],
  ["reduce", {"method": "vec-pca", "n_components": 150}],
  ["scale", {}]
]'
```

Each run writes an isolated bundle so reruns never overwrite one another:

```
artifacts/<model_name>/<run_id>/      # run_id = YYYYMMDD_HHMMSS
    model.joblib                      # the fitted classifier
    feature_pipeline.joblib           # the fitted ImagePipeline (basis + scaler)
    metadata.json                     # the uniform run record (metrics, hyperparameters, diagnostics)
```

Inference from a raw image reuses both artifacts:
`model.predict([feature_pipeline.process(image)])`.

### Compare runs

`python -m comparison` reads every `artifacts/**/metadata.json` and writes
Markdown/HTML/CSV reports (plus optional PNG plots) to a timestamped directory
under `Docs/reports/`:

```bash
# Rank every model on one pipeline:
python -m comparison leaderboard --pipeline svm --diagnostics --plot

# How resilient is one model across the pipelines it's been run on:
python -m comparison resilience --model svm

# Full model x pipeline grid (classical and torch families kept separate):
python -m comparison grid
```

## 5. Running Tests

```bash
python -m pytest -q
```

Tests run against tiny, seeded, in-memory fixtures — no dataset, network, or
GPU required — and cover the preprocessing API, both registries, feature
extraction/caching, tuning, evaluation, diagnostics, artifact persistence, and
the comparison package. Optional-dependency paths (torch, torchvision, keras)
are skipped automatically when that dependency isn't installed. Slower tests
(real torch model fits) are marked `slow` and can be excluded:

```bash
python -m pytest -q -m "not slow"
```

CI (`.github/workflows/test.yml`) runs the full suite on Ubuntu with the CPU
torch build and `keras` installed, so both optional-dependency paths are
exercised on every push and pull request.
