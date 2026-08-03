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
| `cnn_pretrained` | `CNNPretrainedClassifier` (ResNet18) | `lr` | deep (torchvision, optional) |
| `vit_pretrained` | `ViTPretrainedClassifier` (ViT-B/16) | `lr` | deep (torchvision, optional) |

`cnn`/`cnn_deep`/`vit`/`vit_deep` register only when `torch` is importable;
`cnn_pretrained`/`vit_pretrained` additionally need `torchvision`. 15 keys total.

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
| `pixels_pretrained` | 224×224 RGB → 150528 flat | none | torch (\*_pretrained) |

`embedding_pca`/`embedding_jl` register only when `keras` is importable. 10 keys
total. Note the pretrained key is `pixels_pretrained` (plural `pixels`).

**Pairing rule.** Classical models need a reduced+scaled front-end
(`svm`/`svm_jl`/`fast`/`hq`/`no_denoise`, or `embedding_pca`/`embedding_jl`).
Torch models need a **raw-pixel** pipeline whose width matches the
architecture's expected image: `pixels` for `cnn`/`vit`, `pixels_hq` for
`cnn_deep`/`vit_deep`, `pixels_pretrained` for
`cnn_pretrained`/`vit_pretrained`. The grayscale variants reshape the flat
vector back to a *square* image, so a PCA pipeline's 150-wide output (not a
perfect square) will raise at fit time; `pixels_pretrained` is reshaped with the
estimator's explicit `image_shape=(3, 224, 224)` instead. The three torch tiers
do not cross either: a `*_pretrained` model on `pixels` gets 4096 features where
it expects 150528 and raises.

## 1. Training execution roadmap

Entry point: `python src/train_model.py --model <M> --pipeline <P> [flags]`.
Prereq: dataset prepared (`python src/download_dataset.py` then
`python src/create_split.py`). Each run writes
`artifacts/<model>/<run_id>/{model.joblib,feature_pipeline.joblib,metadata.json}`.

### 1a. Baseline runs (16 unique)

The original exploratory set. `svm --pipeline svm` appears twice below (once as a
classical baseline, once as the reduction-method reference), so the 17 lines are
16 distinct runs.

```bash
# Classical baselines on the default svm (PCA-150) front-end
python src/train_model.py --model svm    --pipeline svm
python src/train_model.py --model logreg --pipeline svm
python src/train_model.py --model ridge  --pipeline svm
python src/train_model.py --model linreg --pipeline svm
python src/train_model.py --model rf     --pipeline svm
python src/train_model.py --model hgb    --pipeline svm
python src/train_model.py --model mlp    --pipeline svm --diagnostics   # loss_curve_ available

# Hard-margin SVM pair (like-for-like)
python src/train_model.py --model hard_svm        --pipeline svm
python src/train_model.py --model hard_svm_kernel --pipeline svm

# Reduction-method resilience (same model, PCA vs JL vs learned embedding)
python src/train_model.py --model svm --pipeline svm
python src/train_model.py --model svm --pipeline svm_jl
python src/train_model.py --model svm --pipeline embedding_pca
python src/train_model.py --model svm --pipeline embedding_jl

# Torch image models on raw square pixels
python src/train_model.py --model cnn      --pipeline pixels
python src/train_model.py --model vit      --pipeline pixels
python src/train_model.py --model cnn_deep --pipeline pixels_hq
python src/train_model.py --model vit_deep --pipeline pixels_hq
```

### 1b. Full-matrix completion (53 new runs)

The remaining cells needed for complete N×M coverage: every classical model ×
every classical pipeline (9 × 7 = 63, of which 12 are already covered in §1a),
plus the two pretrained torch pairings §1a never ran. Mirrored verbatim in
`run_new_comparisons.txt` at the repo root.

**Running the batch on Windows.** Do not run it with `bash`. On this machine
`bash` resolves to `C:\Windows\system32\bash.exe` — WSL2 Linux, a separate
filesystem with no access to the Windows `.venv` — so every line fails with
`python: command not found`. Run it from PowerShell with the venv activated:

```powershell
.\.venv\Scripts\Activate.ps1
Get-Content run_new_comparisons.txt | ForEach-Object { Write-Host ">>> $_"; Invoke-Expression $_ }
```

The commands in the file stay interpreter-agnostic (identical to the block
below, and still correct under `bash` on Linux/CI) rather than hardcoding a venv
path into all 53 lines.

Flag convention for the whole matrix: `--scoring f1` is stated explicitly on
every command (it is also the default) so no cell can silently optimize a
different metric, and `--diagnostics` is limited to the `mlp` rows — the only
estimator whose extra cost buys something the cheap diagnostics don't already
give (`loss_curve_` plus the sample-size learning curve). All other defaults
(`--max-train-samples 5000`, `RANDOM_STATE=42`) are left untouched so every cell
shares one train/val/test budget and seed.

```bash
# svm_jl column (JL reduce) — svm x svm_jl already in §1a
python src/train_model.py --model hard_svm        --pipeline svm_jl --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline svm_jl --scoring f1
python src/train_model.py --model logreg          --pipeline svm_jl --scoring f1
python src/train_model.py --model ridge           --pipeline svm_jl --scoring f1
python src/train_model.py --model linreg          --pipeline svm_jl --scoring f1
python src/train_model.py --model rf              --pipeline svm_jl --scoring f1
python src/train_model.py --model hgb             --pipeline svm_jl --scoring f1
python src/train_model.py --model mlp             --pipeline svm_jl --scoring f1 --diagnostics

# fast column (64x64, PCA-150) — full 9
python src/train_model.py --model svm             --pipeline fast --scoring f1
python src/train_model.py --model hard_svm        --pipeline fast --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline fast --scoring f1
python src/train_model.py --model logreg          --pipeline fast --scoring f1
python src/train_model.py --model ridge           --pipeline fast --scoring f1
python src/train_model.py --model linreg          --pipeline fast --scoring f1
python src/train_model.py --model rf              --pipeline fast --scoring f1
python src/train_model.py --model hgb             --pipeline fast --scoring f1
python src/train_model.py --model mlp             --pipeline fast --scoring f1 --diagnostics

# hq column (224x224, PCA-300) — full 9
python src/train_model.py --model svm             --pipeline hq --scoring f1
python src/train_model.py --model hard_svm        --pipeline hq --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline hq --scoring f1
python src/train_model.py --model logreg          --pipeline hq --scoring f1
python src/train_model.py --model ridge           --pipeline hq --scoring f1
python src/train_model.py --model linreg          --pipeline hq --scoring f1
python src/train_model.py --model rf              --pipeline hq --scoring f1
python src/train_model.py --model hgb             --pipeline hq --scoring f1
python src/train_model.py --model mlp             --pipeline hq --scoring f1 --diagnostics

# no_denoise column (svm minus denoise) — full 9
python src/train_model.py --model svm             --pipeline no_denoise --scoring f1
python src/train_model.py --model hard_svm        --pipeline no_denoise --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline no_denoise --scoring f1
python src/train_model.py --model logreg          --pipeline no_denoise --scoring f1
python src/train_model.py --model ridge           --pipeline no_denoise --scoring f1
python src/train_model.py --model linreg          --pipeline no_denoise --scoring f1
python src/train_model.py --model rf              --pipeline no_denoise --scoring f1
python src/train_model.py --model hgb             --pipeline no_denoise --scoring f1
python src/train_model.py --model mlp             --pipeline no_denoise --scoring f1 --diagnostics

# embedding_pca column (VGG16 + PCA) — svm x embedding_pca already in §1a
python src/train_model.py --model hard_svm        --pipeline embedding_pca --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline embedding_pca --scoring f1
python src/train_model.py --model logreg          --pipeline embedding_pca --scoring f1
python src/train_model.py --model ridge           --pipeline embedding_pca --scoring f1
python src/train_model.py --model linreg          --pipeline embedding_pca --scoring f1
python src/train_model.py --model rf              --pipeline embedding_pca --scoring f1
python src/train_model.py --model hgb             --pipeline embedding_pca --scoring f1
python src/train_model.py --model mlp             --pipeline embedding_pca --scoring f1 --diagnostics

# embedding_jl column (VGG16 + JL) — svm x embedding_jl already in §1a
python src/train_model.py --model hard_svm        --pipeline embedding_jl --scoring f1
python src/train_model.py --model hard_svm_kernel --pipeline embedding_jl --scoring f1
python src/train_model.py --model logreg          --pipeline embedding_jl --scoring f1
python src/train_model.py --model ridge           --pipeline embedding_jl --scoring f1
python src/train_model.py --model linreg          --pipeline embedding_jl --scoring f1
python src/train_model.py --model rf              --pipeline embedding_jl --scoring f1
python src/train_model.py --model hgb             --pipeline embedding_jl --scoring f1
python src/train_model.py --model mlp             --pipeline embedding_jl --scoring f1 --diagnostics

# Pretrained torch backbones on 224x224 RGB pixels
python src/train_model.py --model cnn_pretrained  --pipeline pixels_pretrained --scoring f1
python src/train_model.py --model vit_pretrained  --pipeline pixels_pretrained --scoring f1
```

### 1c. Coverage ledger

| Block | Cells | In §1a | New in §1b |
|-------|-------|--------|------------|
| classical models × `svm` | 9 | 9 | 0 |
| classical models × `svm_jl` | 9 | 1 | 8 |
| classical models × `fast` | 9 | 0 | 9 |
| classical models × `hq` | 9 | 0 | 9 |
| classical models × `no_denoise` | 9 | 0 | 9 |
| classical models × `embedding_pca` | 9 | 1 | 8 |
| classical models × `embedding_jl` | 9 | 1 | 8 |
| **classical subtotal (9 × 7)** | **63** | **12** | **51** |
| `cnn`, `vit` × `pixels` | 2 | 2 | 0 |
| `cnn_deep`, `vit_deep` × `pixels_hq` | 2 | 2 | 0 |
| `cnn_pretrained`, `vit_pretrained` × `pixels_pretrained` | 2 | 0 | 2 |
| **torch subtotal** | **6** | **4** | **2** |
| **total valid pairings** | **69** | **16** | **53** |

69 of the 15 × 10 = 150 nominal combinations are valid; the other 81 are barred
by the pairing rule (classical × pixel, torch × reduced, and cross-tier torch).

**Cost note.** The `hq` and `embedding_*` columns dominate wall-clock (224×224
decode, and a VGG16 forward pass per image respectively), and `vit_pretrained`
is the single most expensive run. Run §1b top-to-bottom to get the cheap columns
banked first; the feature cache (`--cache-dir feature_cache`, on by default)
means each pipeline's extraction cost is paid once and amortized across the 8–9
models in its column.

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
model-by-pipeline grid of the primary metric. §1a + §1b together fill every
valid cell, so the grid is complete rather than sparse:

| | `svm` | `svm_jl` | `fast` | `hq` | `no_denoise` | `embedding_pca` | `embedding_jl` |
|---|---|---|---|---|---|---|---|
| `svm` | §1a | §1a | §1b | §1b | §1b | §1a | §1a |
| `hard_svm` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `hard_svm_kernel` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `logreg` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `ridge` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `linreg` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `rf` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `hgb` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |
| `mlp` | §1a | §1b | §1b | §1b | §1b | §1b | §1b |

The torch block is diagonal by construction — one legal pipeline per tier — so
it is a 6-cell list, not a grid: `cnn`/`vit` × `pixels` and `cnn_deep`/`vit_deep`
× `pixels_hq` (§1a), `cnn_pretrained`/`vit_pretrained` × `pixels_pretrained`
(§1b).

**Reading the completed grid.** With every cell present, three comparisons that
the 16-run baseline could not support become available:
- *Row-wise (front-end sensitivity per model).* Each model now has all 7
  classical pipelines, so §1a's `svm`-only 1×N generalizes: does the PCA-vs-JL
  gap hold for tree models (`rf`, `hgb`) the way it does for margin models, or
  is it specific to distance-based estimators?
- *Column-wise (model ranking per front-end).* The N×1 leaderboard can be
  recomputed on all 7 front-ends and the rankings compared. A ranking that is
  stable across columns is a property of the models; one that reorders is
  evidence the front-end, not the classifier, is doing the work.
- *Interaction effects.* Full coverage means row and column effects can be
  separated — e.g. whether `hq`'s extra resolution helps every model uniformly
  or only the higher-capacity ones. This is exactly the pivot `ComparisonMatrix
  .grid(metric)` (§4) emits.

Cells are keyed `(model_name, pipeline_used)` from `metadata.json`, so
`RunLoader` reconstructs the grid without any run-ordering assumptions, and
re-running a cell simply adds a newer `run_id` under the same key.

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
    -> reports/<timestamp>/        (committed comparison output)
```

**Build order (next step).** `records.py` + `loader.py` first (pure parsing,
unit-testable against fixture metadata), then `matrix.py`, then `stats.py`, then
`report.py`/`cli.py`. Tests under `tests/` using tiny synthetic `metadata.json`
fixtures — same convention as the existing suite (`pytest`, no network/dataset).
</content>
