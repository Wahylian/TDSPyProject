# Model Comparison Plan

Actionable blueprint to train and evaluate the registry models against the
prebuilt feature pipelines, and a design for a `comparison` package to automate
it. All keys below are the **verified** registry keys and `clf__` parameter
names as implemented and tested (`pytest -q` → 400 passed).

## 0. Verified registry keys

**Models** (`trainbase/model_registry.py`):

| Key | Estimator | Tuned params (`clf__…`) | Family |
|-----|-----------|--------------------------|--------|
| `svm` | `SVC` | `C`, `kernel`, `gamma` | kernel |
| `hard_svm` | `LinearSVC(C=1e6)` | `C` | linear (hard margin) |
| `hard_svm_kernel` | `SVC(kernel="linear", C=1e6)` | `C` | kernel (hard margin) |
| `logreg` | `LogisticRegression` | `C` | linear |
| `ridge` | `RidgeClassifier` | `alpha` | linear (least-squares) |
| `linreg` | `ThresholdedLinearRegression` | `fit_intercept` | linear (OLS threshold) |
| `rf` | `RandomForestClassifier` | `n_estimators`, `max_depth` | bagging |
| `hgb` | `HistGradientBoostingClassifier` | `learning_rate`, `max_iter` | boosting |
| `mlp` | `MLPClassifier(max_iter=500)` | `alpha`, `hidden_layer_sizes` | iterative NN |
| `cnn` / `cnn_deep` | `CNNClassifier` | `lr` | deep (torch, optional) |
| `vit` / `vit_deep` | `ViTClassifier` | `lr` | deep (torch, optional) |

`cnn*`/`vit*` register only when `torch` is importable.

**Pipelines** (`trainbase/pipeline_registry.py`):

| Key | Output | Reduce | For |
|-----|--------|--------|-----|
| `svm` | 128×128 gray → 150 | PCA | classical |
| `svm_jl` | 128×128 gray → 150 | JL (random proj.) | classical |
| `fast` | 64×64 gray → 150 | PCA | classical |
| `hq` | 224×224 gray → 300 | PCA | classical |
| `no_denoise` | 128×128 gray → 150 | PCA (no denoise) | classical |
| `embedding_pca` | VGG16 block5 → 150 | PCA | classical (needs keras) |
| `embedding_jl` | VGG16 block5 → 150 | JL (random proj.) | classical (needs keras) |
| `pixels` | 64×64 gray → 4096 flat | none | torch (cnn/vit) |
| `pixels_hq` | 128×128 gray → 16384 flat | none | torch (\*_deep) |

`embedding_pca`/`embedding_jl` register only when `keras` is importable.

**Pairing rule.** Classical models need a reduced+scaled front-end
(`svm`/`svm_jl`/`fast`/`hq`/`no_denoise`, or `embedding_pca`/`embedding_jl`).
Torch models need a
**square raw-pixel** pipeline (`pixels` for `cnn`/`vit`, `pixels_hq` for
`cnn_deep`/`vit_deep`) — they reshape the flat vector back to a square image, so
a PCA pipeline's 150-wide output (not a perfect square) will raise at fit time.

## 1. Training execution roadmap

Entry point: `python train_model.py --model <M> --pipeline <P> [flags]`.
Prereq: dataset prepared (`python download_dataset.py` then
`python create_split.py`). Each run writes
`artifacts/<model>/<run_id>/{model.joblib,feature_pipeline.joblib,metadata.json}`.

```bash
# Classical baselines on the default svm (PCA-150) front-end
python train_model.py --model svm    --pipeline svm
python train_model.py --model logreg --pipeline svm
python train_model.py --model ridge  --pipeline svm
python train_model.py --model linreg --pipeline svm
python train_model.py --model rf     --pipeline svm
python train_model.py --model hgb    --pipeline svm
python train_model.py --model mlp    --pipeline svm --diagnostics   # loss_curve_ available

# Hard-margin SVM pair (like-for-like)
python train_model.py --model hard_svm        --pipeline svm
python train_model.py --model hard_svm_kernel --pipeline svm

# Reduction-method resilience (same model, PCA vs JL vs learned embedding)
python train_model.py --model svm --pipeline svm
python train_model.py --model svm --pipeline svm_jl
python train_model.py --model svm --pipeline embedding_pca
python train_model.py --model svm --pipeline embedding_jl

# Torch image models on raw square pixels
python train_model.py --model cnn      --pipeline pixels
python train_model.py --model vit      --pipeline pixels
python train_model.py --model cnn_deep --pipeline pixels_hq
python train_model.py --model vit_deep --pipeline pixels_hq
```

Useful flags (defaults in parentheses): `--scoring` (`f1`),
`--max-train-samples` (`5000`, `0`=all), `--max-val-samples` (`2000`),
`--max-test-samples` (`5000`), `--diagnostics` (off; adds sample-size learning
curve), `--output-dir` (`artifacts`), `--cache-dir` (`feature_cache`, `''` to
disable).

## 2. Hyperparameter specifications

Baselines are the registry grids (already tuned on the val split via
`GridSearchCV`). Widen only for a deeper sweep; keep keys `clf__`-prefixed.

| Model | Baseline grid (as registered) | Suggested wider search |
|-------|-------------------------------|------------------------|
| `svm` | `C∈{0.1,1,10}`, `kernel∈{rbf,linear}`, `gamma∈{scale,auto}` | `C∈{0.01,…,100}` (log), `gamma∈{scale,auto,0.01,0.1}` |
| `hard_svm` / `hard_svm_kernel` | `C∈{1e4,1e6}` | `C∈{1e3,1e4,1e5,1e6}` |
| `logreg` | `C∈{0.1,1,10}` | `C∈{0.01,…,100}` (log) |
| `ridge` | `alpha∈{0.1,1,10}` | `alpha∈{0.01,…,100}` (log) |
| `linreg` | `fit_intercept∈{True,False}` | (fully specified) |
| `rf` | `n_estimators∈{200,400}`, `max_depth∈{None,20}` | `+ max_features∈{sqrt,log2}`, `min_samples_leaf∈{1,5}` |
| `hgb` | `learning_rate∈{0.05,0.1}`, `max_iter∈{100,200}` | `+ max_leaf_nodes∈{15,31,63}`, `l2_regularization∈{0,1}` |
| `mlp` | `alpha∈{1e-4,1e-3}`, `hidden_layer_sizes∈{(100,),(64,32)}` | `+ learning_rate_init∈{1e-3,1e-2}` |
| `cnn*` / `vit*` | `lr∈{1e-3,3e-4}` | `+ weight_decay∈{0,1e-4}`, `epochs∈{10,25}` |

**Caveats for fair comparison.**
- `hgb` default `min_samples_leaf=20`: needs at least a few hundred train
  samples to split at all (the default `--max-train-samples 5000` is fine; a
  tiny cap makes it predict the majority class).
- Keep `--scoring f1` and the seed fixed (`RANDOM_STATE=42`, threaded
  everywhere) across all runs so numbers are comparable.
- Reuse one pipeline key across models being compared so they share an identical
  feature space.

## 3. Comparison methodology

The uniform `metadata.json` (headline metrics + baseline + diagnostics) makes
every run directly comparable. Three evaluation shapes:

**N×1 — cross-model on one pipeline (which model wins on a fixed front-end).**
Fix `--pipeline svm`; sweep every classical `--model`
(`svm, hard_svm, hard_svm_kernel, logreg, ridge, linreg, rf, hgb, mlp`). Rank by
test F1; inspect precision/recall/PR-AUC/ROC-AUC.

**1×N — single-model resilience across pipelines (front-end sensitivity).**
Fix a model (e.g. `svm`); sweep `--pipeline` over
`{svm, svm_jl, fast, hq, no_denoise, embedding_pca, embedding_jl}`. `svm` vs
`svm_jl` (and `embedding_pca` vs `embedding_jl`) isolates PCA-vs-JL reduction
(identical stages otherwise); `svm` vs `no_denoise` isolates denoising;
`fast`/`hq` vary resolution; `embedding_*` swaps to learned VGG16 features.

**N×M — full matrix.** Classical models × classical pipelines; torch models ×
pixel pipelines (the two blocks don't cross — see the pairing rule). Emit a
model-by-pipeline grid of the primary metric.

**Metrics to log per cell** (all already in `metadata.json`):
`evaluation_metrics` = accuracy, precision, recall, f1, pr_auc, roc_auc; plus
`baseline_metrics` (naive majority) and `best_val_score`. Report the test metric
**and** its lift over baseline. `diagnostics` adds confusion matrix,
classification report, hyperparameter-grid scores, feature importances (tree
models: `rf`), OOB (`rf`), and the learning curve (with `--diagnostics`).
`mlp.loss_curve_` is available for a future per-epoch `training_history` panel.

## 4. Comparison package architecture

A standalone read-side `comparison/` package that scans the artifact bundles and
produces leaderboards, matrices, and statistical comparisons. It changes no
training code — it consumes `metadata.json` only.

```
comparison/
  loader.py       -> RunLoader: scan artifacts/**/metadata.json -> list[RunRecord]
  records.py      -> RunRecord: typed view of one metadata.json (frozen dataclass)
  matrix.py       -> ComparisonMatrix: build N×1 / 1×N / N×M metric grids
  stats.py        -> StatisticalComparison: rank + paired significance tests
  report.py       -> Reporter: render leaderboard / matrix to Markdown/HTML/CSV
  cli.py          -> `python -m comparison ...` thin orchestrator
```

**Interfaces (sketch).**
- `RunRecord` — frozen dataclass mirroring the metadata schema: `model_name`,
  `pipeline_used`, `run_id`, `evaluation_metrics`, `baseline_metrics`,
  `best_val_score`, `diagnostics`. One factory `from_metadata(path)`.
- `RunLoader(root="artifacts").load() -> list[RunRecord]` — globs
  `**/metadata.json`, parses, skips malformed with a warning. Optional filters
  by model/pipeline/metric presence.
- `ComparisonMatrix(records)` — `.leaderboard(metric="f1")` (N×1),
  `.resilience(model, metric)` (1×N), `.grid(metric)` (N×M pivot:
  models × pipelines). Returns plain tables (list/`pandas`).
- `StatisticalComparison(records)` — `.rank(metric)`; `.compare(a, b)` for a
  paired test where per-sample scores exist, else a lift-over-baseline delta.
- `Reporter(matrix).to_markdown()/.to_html()/.to_csv(path)` — self-contained
  output; optional ROC/PR/confusion plots when `matplotlib` is present (kept
  optional, mirroring the torch/keras guards).

**Data logging flow.**
```
artifacts/<model>/<run_id>/metadata.json
    -> RunLoader.load()            (parse + validate)
    -> [RunRecord, ...]            (typed, in memory)
    -> ComparisonMatrix            (N×1 / 1×N / N×M selection)
    -> StatisticalComparison       (rank, significance / baseline lift)
    -> Reporter                    (Markdown / HTML / CSV [+ optional plots])
    -> Docs/reports/<timestamp>/   (committed comparison output)
```

**Build order (next step).** `records.py` + `loader.py` first (pure parsing,
unit-testable against fixture metadata), then `matrix.py`, then `stats.py`, then
`report.py`/`cli.py`. Tests under `tests/` using tiny synthetic `metadata.json`
fixtures — same convention as the existing suite (`pytest`, no network/dataset).
</content>
