# Recommended Next Steps

A technical roadmap for `TDSPyProject`, split into new work (Section A) and known
technical debt to pay down later (Section B).

## Section A — New Features & Expansions

### A1. Add an iterative model (and per-epoch history)
The registry currently holds only classical sklearn estimators (`svm`, `rf`,
`logreg`), so `metadata.json` records no per-epoch loss/metric history. Adding an
iterative model — an MLP (`sklearn.neural_network.MLPClassifier`, which exposes
`loss_curve_`) or a small PyTorch/Keras network — would populate that currently
unused branch. `trainbase/diagnostics.py` is structured to accept a new
per-model diagnostic block; the natural extension is a `training_history`
(loss/metric per epoch) key alongside the existing curves.

### A2. Visualization / comparison dashboard
Every run now emits a uniform `metadata.json`. A small tool (Streamlit, or a
static HTML report) could scan `artifacts/**/metadata.json` and plot ROC/PR
curves, confusion matrices, feature importances, learning curves, and a
leaderboard across runs and models. The uniform schema makes this a pure
read-side feature — no training-code changes required.

### A3. Automated hyperparameter sweeps
Tuning is a single `GridSearchCV` over a hand-written grid. Options: wire in a
smarter search (`RandomizedSearchCV`, Optuna) behind the existing
`tune_hyperparameters` seam, and/or a batch-sweep runner that trains several
`--model`/`--pipeline` combinations and writes them all under `artifacts/` for
comparison via A2.

### A4. Richer feature front-ends
The optional VGG16 embedding pipeline exists but is not registered. Registering
an embedding-based pipeline (and other reducers already implemented in
`preprocessing/reduce.py`) would broaden the feature-space options selectable by
name.

## Section B — Existing Technical Debt & Problems to Fix Later

These are known, deliberately deferred issues. Document and schedule; do not fix
opportunistically.

### B1. ~~In-memory bottleneck in `trainbase.load_images`~~ — DONE
`trainbase/features.py::load_images` eagerly appends every decoded image into a
Python list and returns it whole (`List[np.ndarray]`), materializing the entire
split in RAM. This defeats the streaming design of
`extract_features.get_feature_stream` and caps the usable dataset size at
whatever fits in memory. Fix direction: process the stream in bounded batches
(incremental `partial_fit` / batched transform) so peak memory is independent of
split size.

### B2. ~~Weak cache invalidation in the feature cache~~ — DONE
The feature cache key (`train_model._cache_prefix`) is the pipeline **name**
alone for registry pipelines. Editing a registry pipeline's steps while keeping
its name silently reuses stale cached features from the old definition. Fix
direction: key the cache on a hash of the fully-resolved pipeline definition (its
operation list), not just the name — mirroring how custom `--pipeline-spec` runs
are already hashed.

### B3. ~~No end-to-end test for `train_model.main` and no CI~~ — DONE
`train_model.main` (the full load → tune → evaluate → save path) has no automated
end-to-end test; only its components are unit-tested, and artifact persistence is
covered at the `save_artifacts`/`build_metadata` level. There is also no CI
configuration. Fix direction: an end-to-end test driving `main` on a tiny
synthetic split (monkeypatching the feature stream) that asserts the
`artifacts/<model>/<run_id>/` bundle and `metadata.json` schema, plus a CI
workflow running `pytest` on push.

### B4. ~~`pipeline.py` exceeds the file-size guideline~~ — DONE
`preprocessing/pipeline.py` is over the project's ~200-line-per-file guideline and
was intentionally left unsplit for now (splitting it was explicitly out of scope
for the current work). Fix direction: extract cohesive concerns (e.g. the
operation registry/dispatch vs. the `ImagePipeline` class vs. batch helpers) into
separate modules behind the same public `preprocessing` API, so no import sites
change.
