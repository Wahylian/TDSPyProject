# Deepfake-Detect — Real vs. Fake Image Classifier

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

## 2. Project Layout

A standard `src/` layout: all importable code lives under `src/`, and the
generated data, run artifacts, and comparison reports each have their own
top-level directory.

```
TDSPyProject/
├── src/                    -> all importable code (see the breakdown below)
├── tests/                  -> pytest suite (synthetic fixtures; no data/network)
├── datasets/               -> raw dataset + manifest CSV (git-ignored, created by the ingestion scripts)
├── artifacts/              -> per-run bundles: artifacts/<model>/<run_id>/
├── reports/                -> comparison output: reports/<timestamp>/
├── docs/                   -> design docs (project_guide.md, specs/plans)
├── pyproject.toml          -> packaging for the src/ layout (`pip install -e .`)
├── pytest.ini              -> pythonpath = src, markers, discovery
├── requirements*.txt       -> dependency sets (base / GPU / CPU)
└── install.py              -> GPU-then-CPU torch install fallback
```

### `src/` breakdown

```
Ingestion (entry-point scripts)
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
  prebuilt_pipelines.py  -> PrebuiltPipelines: named factories backing PIPELINE_REGISTRY
                             (import via `from trainbase import PrebuiltPipelines`)
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

Install the dependencies, then install the project in editable mode so the
`src/` layout is on the import path (this is what lets `python -m comparison`
and the `trainbase`/`preprocessing`/`comparison` imports resolve from the repo
root). A virtualenv is recommended.

```bash
# 1. Dependencies. Default: core deps + the GPU (CUDA) torch/torchvision build.
pip install -r requirements.txt

# No NVIDIA/CUDA GPU, or the cu126 wheels aren't available for your platform?
pip install -r requirements-cpu.txt

# Or let it try GPU first and fall back to CPU automatically:
python install.py

# 2. The project itself (src/ layout, editable). Pulls no dependencies of its
#    own — it only wires src/ onto the import path.
pip install -e .
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
| `kagglehub` | `src/download_dataset.py` | `pip install kagglehub` |
| `pytest` | running the test suite | `pip install pytest` |

## 4. Usage

Run every command from the repo root. The ingestion and training scripts live
under `src/`, so they are invoked as `python src/<script>.py`; the comparison
package is invoked as a module (`python -m comparison`), which works from the
root once `pip install -e .` has run.

### Prepare the dataset

```bash
python src/download_dataset.py     # downloads + restructures the Kaggle dataset into real/, fake/
python src/create_split.py         # writes the seeded manifest: datasets/dataset_split.csv
```

`download_dataset.py` only touches the network when run directly (importing it
has no side effects) and needs Kaggle credentials configured for `kagglehub`.

### Generate the run artifacts

Every artifact bundle under `artifacts/<model>/<run_id>/` is produced by a
`train_model.py` run. The complete, authoritative set of those commands — the 16
baseline exploratory runs plus the 53 full-matrix cells (69 valid model ×
pipeline pairings) — lives in a single batch script,
[`scripts/generate_artifacts.sh`](scripts/generate_artifacts.sh). Prepare the
dataset first (above), then run the whole batch:

```bash
# Linux / macOS / CI:
bash scripts/generate_artifacts.sh
```

On Windows, `bash` resolves to WSL2 and cannot see the Windows `.venv`, so drive
the commands through PowerShell instead (comment and blank lines are filtered
out):

```powershell
.\.venv\Scripts\Activate.ps1
Get-Content scripts/generate_artifacts.sh |
  Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } |
  ForEach-Object { Write-Host ">>> $_"; Invoke-Expression $_ }
```

`--scoring f1` is the default, `--diagnostics` is limited to the `mlp` rows, and
`--max-train-samples 5000` / `RANDOM_STATE=42` are left at their defaults so
every cell shares one train/val/test budget and seed. The feature cache
(`--cache-dir feature_cache`, on by default) means each pipeline's extraction
cost is paid once and amortized across its column.

### Train a single model

`train_model.py` is the single training entry point; `--model` is required.

```bash
# Random Forest on the fast (64x64) pipeline:
python src/train_model.py --model rf --pipeline fast

# SVM on the default pipeline, capping sample sizes (0 = use all):
python src/train_model.py --model svm --max-train-samples 5000 --max-test-samples 5000

# A from-scratch CNN on raw pixels, with the extra learning-curve diagnostic:
python src/train_model.py --model cnn --pipeline pixels --diagnostics

# Fully custom feature pipeline (inline JSON; include reduce/scale yourself):
python src/train_model.py --model svm --pipeline-spec '[
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
under `reports/`. It never touches training code, so it is safe to re-run
at any time. Three evaluation shapes, one subcommand each:

| Shape | Subcommand | Question it answers |
|---|---|---|
| N×1 | `leaderboard` | Which model wins on a fixed front-end? |
| 1×N | `resilience` | How sensitive is one model to its front-end? |
| N×M | `grid` | Which model/pipeline *combination* wins? |

**Leaderboards (N×1).** With `--pipeline`, every model is scored on an identical
feature space; without it, each model is represented by its own best run across
whichever pipelines it has.

```bash
# Rank every model on one pipeline (like-for-like):
python -m comparison leaderboard --pipeline svm --diagnostics --plot

# Same ranking on the other classical front-ends:
python -m comparison leaderboard --pipeline hq
python -m comparison leaderboard --pipeline embedding_pca
python -m comparison leaderboard --pipeline svm_jl

# Best-of-all-pipelines per model (no --pipeline), ranked by a different metric:
python -m comparison leaderboard
python -m comparison leaderboard --metric roc_auc
python -m comparison leaderboard --metric recall --plot

# The torch block: each deep model has exactly one legal pipeline.
python -m comparison leaderboard --pipeline pixels
python -m comparison leaderboard --pipeline pixels_hq
python -m comparison leaderboard --pipeline pixels_pretrained
```

**Resilience sweeps (1×N).** Every classical model has been run on all seven
classical pipelines, so this isolates how much of a score is the classifier
versus the front-end:

```bash
python -m comparison resilience --model svm
python -m comparison resilience --model rf
python -m comparison resilience --model hgb --metric pr_auc
python -m comparison resilience --model mlp --diagnostics --plot

# Hard-margin pair, to check they degrade the same way:
python -m comparison resilience --model hard_svm
python -m comparison resilience --model hard_svm_kernel
```

**Full grid (N×M).** Classical models pivot only against classical pipelines and
torch models only against raw-pixel pipelines — the two blocks never cross, and
uncovered combinations show as `NaN`:

```bash
python -m comparison grid
python -m comparison grid --metric roc_auc
python -m comparison grid --metric accuracy --plot

# Scan an alternate artifacts root and write elsewhere:
python -m comparison grid --root artifacts --output-dir reports/full_matrix
```

**Flags** (shared by all three shapes):

| Flag | Default | Effect |
|---|---|---|
| `--root` | `artifacts` | Artifacts root to scan. |
| `--metric` | `f1` | Ranks/pivots on `accuracy`, `precision`, `recall`, `f1`, `pr_auc`, or `roc_auc`. |
| `--output-dir` | `reports` | Reports land in `<output-dir>/<timestamp>/`. |
| `--diagnostics` | off | Adds confusion matrix, hyperparameter scores, importances/OOB, and learning curve per run shown. |
| `--plot` | off | Saves PNG plots when `matplotlib` is installed; skipped silently otherwise. |

Two caveats: `--diagnostics` is accepted but ignored for `grid` (a cell has no
single `run_id` to attribute diagnostics to), and a `resilience` sweep of a torch
model returns a single row, since each deep tier has exactly one legal pipeline.

## 5. Running Tests

```bash
python -m pytest -q
```

`pytest.ini` sets `pythonpath = src`, so the suite runs whether or not
`pip install -e .` has been done. Tests run against tiny, seeded, in-memory
fixtures — no dataset, network, or GPU required — and cover the preprocessing
API, both registries, feature extraction/caching, tuning, evaluation,
diagnostics, artifact persistence, and the comparison package.
Optional-dependency paths (torch, torchvision, keras) are skipped automatically
when that dependency isn't installed. Slower tests (real torch model fits) are
marked `slow` and can be excluded:

```bash
python -m pytest -q -m "not slow"
```

CI (`.github/workflows/test.yml`) installs the project (`pip install -e .`) and
runs the full suite on Ubuntu with the CPU torch build and `keras` installed, so
both optional-dependency paths are exercised on every push and pull request.
