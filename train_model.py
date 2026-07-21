"""
Train a model of your choice from a registry of models to classify Real vs. Fake images.

This script is a thin orchestrator. It wires the project's building blocks —
exposed through the ``trainbase`` package — into one reproducible training run:

    extract_features.get_feature_stream   ->  streams (image, label) per split
    trainbase.build_feature_pipeline      ->  the ImagePipeline that turns each
                                              image into a model-ready vector
    ImagePipeline.fit_transform/transform ->  fit the feature space on train,
                                              reuse it unchanged on val/test
    trainbase.build_estimator / tune      ->  assemble & tune the classifier
    trainbase.evaluate / save_artifacts   ->  score on test and persist the run

Pipeline stages (per run):
    * Extract a seeded, shuffled (image, label) stream for each split.
    * Run it through a feature pipeline that is FIT on the train split and
      REUSED on the val/test splits, so every split shares one feature space
      (no leakage).
    * Tune the model's hyperparameters against the validation split.
    * Evaluate the tuned model on the test split (against a naive baseline).
    * Save the fitted feature pipeline, the model, and a uniform metadata.json.

Feature pipelines are self-contained
-------------------------------------
Each pipeline fully defines its own feature transform, including any
dimensionality reduction (``vec-pca``) and per-feature standardization
(``scale``) it needs. This script appends nothing to it. A scale-sensitive model
such as the SVM therefore requires a pipeline that already carries a ``reduce`` +
``scale`` tail; the prebuilt registry pipelines (``svm``, ``fast``, ``hq``,
``no_denoise``) all do.

Models
------
The classifier is chosen by name from the model registry (``--model``), which is required. 
To list the available models or add a new one, see ``trainbase/model_registry.py``. 
Swapping the classifier needs no other change — the feature front-end, tuning, evaluation, and saving are shared.

How to Run (Example):
    NOTE: --model is required, so include it in every run. It is omitted from
    some single-flag examples below only for brevity.

    #   Choose the classifier (required; see trainbase/model_registry.py):
        python train_model.py --model rf

    #   Choose a Preprocessing Pipeline — a prebuilt one OR a fully custom one.
        *   Choose a PreBuilt Pipeline (see trainbase/pipeline_registry.py):
            python train_model.py --pipeline fast

        *   Customize Your Pipeline:
            Pass --pipeline-spec with inline JSON describing the pipeline as a
            list of [operation_name, kwargs] pairs — the same structure
            ImagePipeline's constructor takes. It is used verbatim (nothing is
            appended), so include the 'reduce'/'scale' steps yourself if your
            model needs them. --pipeline-spec is mutually exclusive with
            --pipeline:

            python train_model.py --pipeline-spec '[
                ["grayscale", {}],
                ["resize", {"target_size": [128, 128], "preserve_aspect": true}],
                ["normalize", {"method": "minmax"}],
                ["vectorize", {}],
                ["reduce", {"method": "vec-pca", "n_components": 150}],
                ["scale", {}]
            ]'

            (Pass it as one argument. The single quotes above are bash-style;
            on Windows PowerShell wrap the JSON in single quotes likewise, or
            escape the inner double quotes.)

    #   Choosing Maximum Training Sample Sizes:
        python train_model.py --max-train-samples 10000
        NOTE: For the maximal sample size do --max-train-samples 0

    #   Choosing Maximum Testing Sample Sizes:
        python train_model.py --max-test-samples 10000
        NOTE: For the maximal sample size do --max-test-samples 0

    #   Saving Diagnostics (adds the sample-size learning curve; extra fits):
        python train_model.py --model rf --diagnostics

    #   Choosing the Artifacts Root (a folder name or path; created if missing):
        python train_model.py --output-dir my_runs
        NOTE: Defaults to `artifacts/`.

Each run writes an isolated bundle to ``artifacts/<model>/<run_id>/`` (override
the root with ``--output-dir``), so reruns of the same model never overwrite one
another. The bundle holds the fitted classifier (``model.joblib``), the fitted
feature pipeline (``feature_pipeline.joblib`` — which holds the PCA basis *and*
the scaling statistics), and a uniform ``metadata.json`` (configuration,
hyperparameters, headline metrics, and diagnostics).

How to Add a New Model to the Registry: see 'trainbase/model_registry.py'.

Requirements:
    pip install numpy opencv-python scikit-learn joblib
    (plus this project's ``preprocessing`` and ``trainbase`` packages and their
    dependencies)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from trainbase import (
    MODEL_REGISTRY,
    PIPELINE_REGISTRY,
    baseline_metrics,
    build_estimator,
    build_feature_pipeline,
    build_metadata,
    collect_diagnostics,
    evaluate,
    fit_features,
    save_artifacts,
    transform_features,
    tune_hyperparameters,
)

logger = logging.getLogger("train_model")


def _cache_prefix(args: argparse.Namespace) -> str:
    """Derive the feature-cache key for this run's feature space.

    Both sources key on a short hash of the *fully-resolved* pipeline definition,
    so editing the underlying operation list invalidates any stale cache instead
    of silently reusing features from the old definition:

    * A custom ``--pipeline-spec`` hashes its JSON directly.
    * A registry pipeline hashes its resolved operation list (name + kwargs),
      keeping the readable pipeline name as a prefix.
    """
    if args.pipeline_spec is not None:
        digest = hashlib.sha1(args.pipeline_spec.encode("utf-8")).hexdigest()[:10]
        return f"custom_{digest}"
    operations = PIPELINE_REGISTRY[args.pipeline]().operations
    digest = hashlib.sha1(
        json.dumps(operations, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    return f"{args.pipeline}_{digest}"


def main(args: argparse.Namespace) -> None:
    """Run the end-to-end pipeline: load -> tune -> evaluate -> save.

    Each step is verified before the next proceeds: feature extraction must
    yield samples, tuning must select a model, and evaluation is compared
    against a naive baseline (logged with a warning if the model fails to beat
    it).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Validate selections up front so we fail fast with a clear message. ---
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown --model '{args.model}'. Choices: {sorted(MODEL_REGISTRY)}"
        )
    custom_pipeline = args.pipeline_spec is not None
    if not custom_pipeline and args.pipeline not in PIPELINE_REGISTRY:
        raise SystemExit(
            f"Unknown --pipeline '{args.pipeline}'. Choices: {sorted(PIPELINE_REGISTRY)}"
        )

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_prefix = _cache_prefix(args)

    # A per-run identifier isolates each run's artifacts so reruns never overwrite.
    run_start = datetime.now()
    run_id = run_start.strftime("%Y%m%d_%H%M%S")

    # --- Build the feature pipeline (a registry name or a custom JSON spec).
    #     It is used exactly as defined: any required reduce/scale steps must
    #     already be part of it.
    try:
        feature_pipeline = build_feature_pipeline(
            pipeline_name=None if custom_pipeline else args.pipeline,
            pipeline_spec=args.pipeline_spec,
        )
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid feature pipeline: {exc}")
    logger.info("Feature pipeline: %r", feature_pipeline)

    # --- 1. Build features. The pipeline is FIT on train (learning any PCA
    #        basis / scaling statistics) and REUSED (transform) on val/test, so
    #        all splits share one consistent feature space — no leakage.
    try:
        feature_pipeline, X_train, y_train = fit_features(
            feature_pipeline, args.max_train_samples, cache_dir, cache_prefix
        )
        X_val, y_val = transform_features(
            "val", feature_pipeline, args.max_val_samples, cache_dir, cache_prefix
        )
        X_test, y_test = transform_features(
            "test", feature_pipeline, args.max_test_samples, cache_dir, cache_prefix
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not load data: {exc}")

    # --- 2. Tune hyperparameters on the validation split. --------------------
    estimator = build_estimator(args.model)
    search = tune_hyperparameters(
        estimator,
        MODEL_REGISTRY[args.model].param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scoring=args.scoring,
    )
    best_model = search.best_estimator_

    # --- 3. Evaluate: naive baseline first, then the tuned model. ------------
    baseline = baseline_metrics(X_train, y_train, X_test, y_test)
    model_metrics = evaluate(best_model, X_test, y_test, model_label=args.model)

    # Quick sanity check the model actually beats "always predict majority".
    if model_metrics["accuracy"] <= baseline["accuracy"]:
        logger.warning(
            "Model accuracy (%.4f) does NOT beat the naive baseline (%.4f).",
            model_metrics["accuracy"],
            baseline["accuracy"],
        )

    # --- 4. Collect diagnostics and save the run bundle. ---------------------
    #     Cheap diagnostics (feature importances, OOB, hyperparameter grid
    #     scores) are always captured; the sample-size learning curve is gated
    #     behind --diagnostics because it costs extra model fits.
    diagnostics = collect_diagnostics(
        search, best_model, X_train, y_train,
        include_curves=args.diagnostics, scoring=args.scoring,
    )
    metadata = build_metadata(
        model_name=args.model,
        run_id=run_id,
        timestamp=run_start.isoformat(timespec="seconds"),
        # Record how the feature space was specified: a registry name, or the
        # verbatim custom spec (so the run is reproducible from the metadata).
        pipeline_used="custom" if custom_pipeline else args.pipeline,
        pipeline_spec=args.pipeline_spec,
        pipeline_steps=feature_pipeline.operations,
        scoring=args.scoring,
        sample_sizes={
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        hyperparameters=search.best_params_,
        best_val_score=float(search.best_score_),
        test_metrics=model_metrics,
        baseline_metrics=baseline,
        diagnostics=diagnostics,
    )
    save_artifacts(best_model, feature_pipeline, metadata, output_dir, args.model, run_id)
    logger.info("Done.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    All sample caps accept ``0`` to mean "use the entire split". The feature
    pipeline is chosen either by registry name (``--pipeline``) or as a custom
    inline-JSON spec (``--pipeline-spec``); the two are mutually exclusive.
    """
    parser = argparse.ArgumentParser(
        description="Train a registered model to classify images as real vs. fake.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Classifier to train (required; no default). "
        f"Choices: {sorted(MODEL_REGISTRY)}.",
    )

    # The feature pipeline comes from exactly one source: a registry name or a
    # custom inline-JSON spec. A mutually exclusive group lets argparse reject
    # "both given" for us; --pipeline keeps a default so a bare run still works.
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument(
        "--pipeline",
        default="svm",
        help=f"Prebuilt feature pipeline. Choices: {sorted(PIPELINE_REGISTRY)}.",
    )
    pipeline_group.add_argument(
        "--pipeline-spec",
        default=None,
        help="Custom feature pipeline as inline JSON: a list of "
        "[operation_name, kwargs] pairs, mirroring ImagePipeline's constructor. "
        "Used verbatim (no reduce/scale appended). Overrides --pipeline.",
    )

    parser.add_argument(
        "--max-train-samples", type=int, default=5000,
        help="Cap on training images (0 = all). Kept small so an expensive model "
        "(e.g. a kernel SVM) stays tractable; raise it if you have the compute.",
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=2000,
        help="Cap on validation images used for tuning (0 = all).",
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=5000,
        help="Cap on test images for evaluation (0 = all).",
    )
    parser.add_argument(
        "--scoring", default="f1",
        help="Metric GridSearchCV optimizes on the validation split.",
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Also compute the sample-size learning curve (extra model fits, "
        "off by default). Cheap diagnostics (feature importances, OOB, "
        "hyperparameter-grid scores) are always saved regardless.",
    )
    parser.add_argument(
        "--output-dir", default="artifacts",
        help="Root directory for run artifacts, written to "
        "<output-dir>/<model>/<run_id>/.",
    )
    parser.add_argument(
        "--cache-dir", default="feature_cache",
        help="Directory for the feature cache (set empty '' to disable).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
