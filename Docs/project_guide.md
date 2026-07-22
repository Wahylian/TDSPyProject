# Project Guide — Real vs. Fake Image Classifier

An onboarding guide for the `TDSPyProject` deepfake (real vs. fake) image
classifier: a classical-ML pipeline built on NumPy, OpenCV, scikit-learn and
joblib, with an optional Keras/VGG16 embedding path.

## 1. What this project does

Given a folder of `real/` and `fake/` images, the project:

1. Streams images per split (train/val/test) from a seeded manifest.
2. Turns each image into a model-ready feature vector through a configurable
   **feature pipeline** (grayscale → resize → denoise → normalize → vectorize →
   PCA → standardize).
3. Trains and tunes a classifier chosen from a **model registry** (`svm`, `rf`,
   `logreg`), selecting hyperparameters on the validation split.
4. Evaluates on the test split against a naive baseline and writes a
   self-contained, reproducible **artifact bundle** per run.

## 2. Architecture

The design is registry-driven: a training run selects a model and a feature
pipeline **by name**, so adding either is a one-line registry entry, not a code
change.

```
Ingestion (top-level scripts)
  download_dataset.py   -> download + restructure the Kaggle dataset to real/ + fake/
  create_split.py       -> write a seeded 70/15/15 manifest: datasets/dataset_split.csv
  extract_features.py   -> stream (image, label) pairs per split

preprocessing/          -> image -> vector building blocks
  io.py                 -> load images (PIL=RGB, OpenCV=BGR; see the color-space note)
  transforms.py         -> grayscale, resize, normalize, denoise
  vectorize.py          -> flat vector or VGG16 embedding
  reduce.py             -> PCA / Johnson-Lindenstrauss dimensionality reduction
  scale.py              -> standardization
  pipeline.py           -> ImagePipeline: compose ops; fit_transform / transform / process

trainbase/              -> training backend used by train_model.py
  pipeline_registry.py  -> PIPELINE_REGISTRY: svm / fast / hq / no_denoise
  model_registry.py     -> MODEL_REGISTRY: svm / rf / logreg (+ hyperparameter grids)
  features.py           -> build/fit/transform features (+ on-disk feature cache)
  training.py           -> build_estimator + validation-holdout GridSearchCV tuning
  evaluation.py         -> accuracy/precision/recall/F1/PR-AUC/ROC-AUC + baseline
  diagnostics.py        -> feature importances, OOB, hyperparameter & learning curves
  artifacts.py          -> build_metadata + save_artifacts (the run bundle)

train_model.py          -> thin CLI orchestrator: load -> tune -> evaluate -> save
```

### Data flow (one run)

```
manifest CSV --> get_feature_stream(split) --> images
    train images --> feature_pipeline.fit_transform --> X_train (PCA basis + scaler LEARNED here)
    val/test     --> feature_pipeline.transform      --> X_val / X_test (SAME feature space, no leakage)
X_train/X_val --> GridSearchCV (tune on val) --> best_model
best_model + X_test --> evaluate (+ baseline) --> metrics
best_model + search --> collect_diagnostics --> diagnostics
everything --> build_metadata --> save_artifacts --> artifacts/<model>/<run_id>/
```

## 3. Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `opencv-python`, `Pillow`, `scikit-learn`,
`joblib`. Optional extras (commented in `requirements.txt`): `keras` for VGG16
embeddings (runs on the torch backend — no tensorflow needed), `kagglehub` for
the dataset download, `pytest` for tests.

The project uses a flat layout (`pytest.ini` sets `pythonpath = .`), so no
installation step is needed — run scripts from the project root.

## 4. Prepare the dataset

```bash
python download_dataset.py     # downloads + restructures into datasets/.../real, fake
python create_split.py         # writes the seeded manifest: datasets/dataset_split.csv
```

`download_dataset.py` performs the network download only when run directly
(importing it has no side effects). It requires Kaggle credentials configured for
`kagglehub`.

## 5. Train and evaluate

`train_model.py` is the single entry point. `--model` is required.

```bash
# Random Forest on the fast (64x64) pipeline:
python train_model.py --model rf --pipeline fast

# SVM on the default svm pipeline, capping sample sizes (0 = use all):
python train_model.py --model svm --max-train-samples 5000 --max-test-samples 5000

# Add the (more expensive) sample-size learning curve to the diagnostics:
python train_model.py --model rf --diagnostics

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

Key flags: `--pipeline` (registry name) or `--pipeline-spec` (custom JSON, mutually
exclusive); `--max-{train,val,test}-samples` (`0` = all); `--scoring` (tuning
metric, default `f1`); `--diagnostics` (add the learning curve); `--output-dir`
(artifacts root, default `artifacts`); `--cache-dir` (feature cache, `''` to
disable).

## 6. Run outputs

Each run writes an isolated bundle so reruns never overwrite one another:

```
artifacts/<model_name>/<run_id>/      # run_id = YYYYMMDD_HHMMSS
    model.joblib                      # the fitted classifier
    feature_pipeline.joblib           # the fitted ImagePipeline (PCA basis + scaler)
    metadata.json                     # the uniform run record (below)
```

Inference from a raw image reuses both artifacts:
`model.predict([feature_pipeline.process(image)])`.

### `metadata.json` schema (uniform across all models)

```json
{
  "model_name": "rf",
  "run_id": "20260720_164200",
  "timestamp": "2026-07-20T16:42:00",
  "pipeline_used": "fast",
  "pipeline_spec": null,
  "pipeline_steps": [["grayscale", {}], ...],
  "scoring": "f1",
  "sample_sizes": {"train": 200, "val": 100, "test": 100},
  "hyperparameters": {"clf__n_estimators": 200, "clf__max_depth": null},
  "best_val_score": 0.61,
  "evaluation_metrics": {"accuracy": .., "precision": .., "recall": .., "pr_auc": .., "roc_auc": ..},
  "baseline_metrics":   {"accuracy": .., "precision": .., "recall": .., "pr_auc": .., "roc_auc": ..},
  "diagnostics": {
    "confusion_matrix": [[..], [..]],
    "classification_report": "...",
    "feature_importances": [...] ,   // tree models only, else null
    "oob_score": 0.62,               // tree models only, else null
    "hyperparameter_scores": [{"params": {...}, "mean_val_score": .., "std_val_score": ..}],
    "learning_curve": {"train_sizes": [...], "train_scores_mean": [...], "val_scores_mean": [...]}  // --diagnostics only, else null
  }
}
```

The `diagnostics` keys are always present (`null` when a diagnostic does not
apply), so runs stay directly comparable regardless of estimator. Note: there is
currently no neural-network model, so no per-epoch loss/metric history is
recorded — see the roadmap.

## 7. Run the tests

```bash
python -m pytest -q
```

Tests use tiny synthetic fixtures (no dataset or network needed) and cover the
preprocessing API, the registries, feature extraction, tuning, evaluation,
diagnostics, and artifact persistence.
