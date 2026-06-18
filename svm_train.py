"""
svm_train.py — an end-to-end, *teachable* training pipeline for a Hard Margin SVM.

This script is written first and foremost to be **read**. It walks through the
four stages every classical-ML training job shares, in order, with one small
function per stage so the same skeleton can be copy-pasted to train a different
algorithm (just swap the model in stage 3):

    1. LOAD     stream (image, label) pairs for a split  ->  extract_features.py
    2. PREPROCESS  turn raw images into a feature matrix  ->  prebuilt_pipelines.py
    3. TRAIN    fit the model on (X, y)                   ->  scikit-learn SVC
    4. EVALUATE measure accuracy on a held-out split

------------------------------------------------------------------------------
Why a *Hard Margin* SVM?
------------------------------------------------------------------------------
A soft-margin SVM lets some points sit inside (or on the wrong side of) the
margin, trading a few mistakes for a wider, more robust boundary. The penalty
parameter ``C`` controls that trade-off: small ``C`` = soft (tolerant), large
``C`` = hard (intolerant).

A *hard* margin permits **zero** violations — it demands a hyperplane that
separates the two classes perfectly with the widest possible gap. In the SVM
optimisation this is the limit ``C -> infinity``. scikit-learn has no literal
"infinity" option, so we approximate it with a very large ``C`` (1e10): the
solver is so heavily penalised for any margin violation that it will not allow
one if a separating hyperplane exists at all.

Caveat worth teaching: a hard margin only *has a solution* when the data are
linearly separable in the feature space. On messy real-world features the
solver may strain to fit (and overfit). That fragility is exactly why soft
margins dominate in practice — but seeing the hard-margin case makes the role
of ``C`` concrete.

------------------------------------------------------------------------------
Run it
------------------------------------------------------------------------------
    python svm_train.py
    python svm_train.py --base-dir ./datasets --train-split train --eval-split test
    python svm_train.py --max-samples 100        # cap downloads per split

Requirements: numpy, scikit-learn, plus the deps of extract_features /
preprocessing (requests, pillow, opencv-python).
"""

import argparse
from typing import List, Tuple

import numpy as np
from sklearn.svm import SVC

# Project modules. These are the integration points the task asks us to wire
# together — we treat each as a black box behind its public function.
from extract_features import get_feature_stream      # stage 1: the data stream
from prebuilt_pipelines import PrebuiltPipelines      # stage 2: preprocessing recipe
from preprocessing import ImagePipeline, batch_process  # stage 2: pipeline runner

# A hard margin is the C -> infinity limit of the SVM. We can't pass infinity to
# scikit-learn, so we use a huge finite penalty: any margin violation costs so
# much that the solver avoids it whenever a separating hyperplane exists.
HARD_MARGIN_C = 1e10


# ============================================================================
# STAGE 1 — LOAD: stream (image, label) pairs and collect them in order
# ============================================================================

def load_split(
    split: str,
    base_dir: str,
    max_samples: int | None = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Pull one dataset split into memory as ``(images, labels)``.

    ``get_feature_stream`` yields ``(image, label)`` one pair at a time,
    downloading lazily. We append the image and its label *together* on every
    iteration, so the two lists stay perfectly index-aligned: ``images[i]`` is
    always described by ``labels[i]``. Broken URLs are skipped *inside* the
    stream (the pair is never yielded), so no misalignment can sneak in here.

    Args:
        split: Which partition to load — "train", "val", or "test".
        base_dir: Folder holding the split metadata (e.g. ``split_dataset.csv``).
        max_samples: Optional cap on how many images to pull. ``None`` loads the
            whole split. A cap keeps the demo runnable, since each sample is a
            network download.

    Returns:
        ``(images, labels)`` where ``images`` is a list of BGR ``uint8`` arrays
        and ``labels`` is an int array of class ids (from ``label_numeric``).
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    for image, label in get_feature_stream(split, base_dir=base_dir):
        images.append(image)
        # Labels arrive as raw strings from the CSV ("0"/"1"); SVMs need a
        # numeric target vector, so cast once here at the boundary.
        labels.append(int(label))

        if max_samples is not None and len(images) >= max_samples:
            break

    if not images:
        raise RuntimeError(
            f"No images were loaded for split '{split}'. Check that "
            f"'{base_dir}' contains the split metadata and that the URLs resolve."
        )

    return images, np.array(labels)


# ============================================================================
# STAGE 2 — PREPROCESS: raw images -> a numeric feature matrix X
# ============================================================================

def extract_feature_matrix(
    images: List[np.ndarray],
    pipeline: ImagePipeline,
) -> np.ndarray:
    """
    Convert a list of images into an ``(n_samples, n_features)`` matrix.

    ``batch_process`` runs the whole pipeline over the batch and stacks the
    per-image results. The chosen ``svm_pipeline`` grayscales, resizes to
    128x128, denoises, min-max normalises to ``[0, 1]``, then flattens — giving
    16,384 already-scaled features per image. Because the pipeline normalises,
    we deliberately skip a separate StandardScaler: feeding the SVM features
    that already share a common scale is enough.

    The row order out of ``batch_process`` matches the input order, so X stays
    aligned with the labels collected in stage 1.
    """
    return batch_process(images, pipeline)


# ============================================================================
# STAGE 3 — TRAIN: fit the Hard Margin SVM on (X, y)
# ============================================================================

def train_hard_margin_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    """
    Fit a hard-margin SVM and return the trained model.

    - ``kernel="linear"``: the classic hard-margin SVM looks for a separating
      *hyperplane*, i.e. a linear boundary. (Kernels generalise this to curved
      boundaries, but the linear case is the canonical "maximum-margin" picture.)
    - ``C=HARD_MARGIN_C`` (1e10): the near-infinite penalty that turns the
      soft-margin solver into a hard-margin one — no margin violations tolerated.

    The support vectors (``model.support_vectors_``) are the handful of training
    points that touch the margin and alone determine the boundary; everything
    else could be deleted without changing the fit. Printing their count is a
    quick sanity check on how "tight" the problem is.
    """
    print(f"\nTraining Hard Margin SVM on {X.shape[0]} samples, "
          f"{X.shape[1]} features (C={HARD_MARGIN_C:g}, kernel=linear)...")

    model = SVC(kernel="linear", C=HARD_MARGIN_C)
    model.fit(X, y)

    print(f"  Done. {len(model.support_vectors_)} support vectors define the boundary.")
    return model


# ============================================================================
# STAGE 4 — EVALUATE: accuracy on a held-out split
# ============================================================================

def evaluate(model: SVC, X: np.ndarray, y: np.ndarray, split_name: str) -> float:
    """
    Report plain accuracy on a split the model did not train on.

    Accuracy = fraction of correct predictions. We keep evaluation minimal on
    purpose; the point of the script is the *methodology*, and ``model.score``
    is the one-line way to close the loop.
    """
    accuracy = model.score(X, y)
    print(f"  {split_name} accuracy: {accuracy:.4f} ({X.shape[0]} samples)")
    return accuracy


# ============================================================================
# ORCHESTRATION — wire the four stages together
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Hard Margin SVM on the image dataset."
    )
    parser.add_argument("--base-dir", default="./datasets",
                        help="Folder with the split metadata (default: ./datasets).")
    parser.add_argument("--train-split", default="train",
                        help="Split to train on (default: train).")
    parser.add_argument("--eval-split", default="test",
                        help="Split to evaluate on (default: test).")
    parser.add_argument("--max-samples", type=int, default=60,
                        help="Cap images loaded per split; keeps the demo "
                             "runnable. Pass 0 to load the full split.")
    args = parser.parse_args()

    # argparse can't express "None"; treat 0 as "no cap".
    max_samples = args.max_samples or None

    # One shared preprocessing recipe is fit-free per image, so the *same*
    # pipeline object can transform both the train and eval images consistently.
    pipeline = PrebuiltPipelines.svm_pipeline()

    # Stage 1 + 2 for the training split.
    print(f"Loading '{args.train_split}' split from {args.base_dir}...")
    train_images, y_train = load_split(args.train_split, args.base_dir, max_samples)
    X_train = extract_feature_matrix(train_images, pipeline)

    # Stage 3.
    model = train_hard_margin_svm(X_train, y_train)

    # Close the loop with training accuracy, then the held-out split.
    print("\nResults:")
    evaluate(model, X_train, y_train, args.train_split)

    print(f"Loading '{args.eval_split}' split from {args.base_dir}...")
    eval_images, y_eval = load_split(args.eval_split, args.base_dir, max_samples)
    X_eval = extract_feature_matrix(eval_images, pipeline)
    evaluate(model, X_eval, y_eval, args.eval_split)


if __name__ == "__main__":
    main()
