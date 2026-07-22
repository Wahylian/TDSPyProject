"""Train a registry model to classify real vs. fake images.

Thin orchestrator over the trainbase package. Per run it extracts a seeded
(image, label) stream per split, fits a feature pipeline on train and reuses it
on val/test (one leak-free feature space), tunes hyperparameters on val,
evaluates on test against a naive baseline, and saves the fitted pipeline, model,
and a uniform metadata.json to artifacts/<model>/<run_id>/.

The classifier is chosen by --model (required). Feature pipelines are
self-contained: this script appends nothing, so a scale-sensitive model needs a
pipeline that already carries its own reduce/scale tail. See --help for flags and
trainbase/{model,pipeline}_registry.py for the available choices.
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
    """Derive the feature-cache key from a hash of the resolved pipeline definition.

    Hashing the fully-resolved operation list means editing it invalidates any
    stale cache. A custom spec hashes its JSON; a registry pipeline keeps its
    readable name as a prefix.
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
    """Run the end-to-end pipeline: load, tune, evaluate, save.

    Evaluation is compared against a naive baseline, logging a warning if the
    model fails to beat it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate selections up front to fail fast with a clear message.
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

    # Build the feature pipeline (registry name or custom spec), used as defined.
    try:
        feature_pipeline = build_feature_pipeline(
            pipeline_name=None if custom_pipeline else args.pipeline,
            pipeline_spec=args.pipeline_spec,
        )
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid feature pipeline: {exc}")
    logger.info("Feature pipeline: %r", feature_pipeline)

    # 1. Build features: fit the pipeline on train, reuse it on val/test so all
    #    splits share one feature space with no leakage.
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

    # 2. Tune hyperparameters on the validation split.
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

    # 3. Evaluate: naive baseline first, then the tuned model.
    baseline = baseline_metrics(X_train, y_train, X_test, y_test)
    model_metrics = evaluate(best_model, X_test, y_test, model_label=args.model)

    # Sanity check the model beats "always predict majority".
    if model_metrics["accuracy"] <= baseline["accuracy"]:
        logger.warning(
            "Model accuracy (%.4f) does NOT beat the naive baseline (%.4f).",
            model_metrics["accuracy"],
            baseline["accuracy"],
        )

    # 4. Collect diagnostics and save the run bundle. The learning curve is gated
    #    behind --diagnostics since it costs extra fits; the rest are always captured.
    diagnostics = collect_diagnostics(
        search, best_model, X_train, y_train,
        include_curves=args.diagnostics, scoring=args.scoring,
    )
    metadata = build_metadata(
        model_name=args.model,
        run_id=run_id,
        timestamp=run_start.isoformat(timespec="seconds"),
        # Record the feature-space source so the run is reproducible from metadata.
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
    """Define and parse the CLI.

    Sample caps accept 0 for the whole split. The feature pipeline is chosen by
    --pipeline or --pipeline-spec (mutually exclusive).
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

    # Registry name or custom spec, one source only. The exclusive group rejects
    # "both given"; --pipeline keeps a default so a bare run still works.
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
