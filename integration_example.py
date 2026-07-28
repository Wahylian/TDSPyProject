"""Runnable smoke-test wiring the public preprocessing API end to end.

Loads a sample dataset, builds a pipeline, extracts features in batches, and
optionally trains and evaluates a classifier, so the whole flow can be checked
by eye. Not a pytest target.
"""

import numpy as np
from typing import Tuple, List

from preprocessing import (
    ImagePipeline,
    batch_process,
)

# sklearn is optional and only the training helpers need it; a missing install
# must not break importing the pipelines/loaders above. Tests guard with importorskip.
try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    pass


def load_sample_dataset(num_samples: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
    """Generate a synthetic (images, labels) dataset for the demo.

    Alternating rows are real-like (uniform) and fake-like (noisier). Replace
    with real loading via the create_split.py manifest in production.
    """
    print(f"Generating {num_samples} synthetic images for demonstration...")

    images = []
    labels = []

    for i in range(num_samples):
        if i % 2 == 0:
            # Real-like: more uniform, less noise.
            image = np.random.normal(128, 20, (224, 224, 3)).astype(np.uint8)
            label = 0
        else:
            # Fake-like: more variation.
            image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            label = 1

        images.append(image)
        labels.append(label)

    return images, np.array(labels)


def extract_features_batch(
    images: List[np.ndarray],
    pipeline: ImagePipeline,
    batch_size: int = 8,
    verbose: bool = True
) -> np.ndarray:
    """Extract features from images, returning a (num_images, feature_dim) matrix.

    A pipeline with batch-level ops is run in a single batch_process call, since
    chunking would refit the reducer per chunk into inconsistent feature spaces.
    """
    num_images = len(images)

    if pipeline.batch_operations():
        if verbose:
            print(f"  Pipeline contains batch-level ops "
                  f"({[n for n, _ in pipeline.batch_operations()]}); "
                  f"processing all {num_images} images in one batch.")
        return batch_process(images, pipeline)

    # Ceiling division so the last partial batch still runs.
    num_batches = (num_images + batch_size - 1) // batch_size

    all_features = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)

        batch_images = images[start_idx:end_idx]
        batch_features = batch_process(batch_images, pipeline)
        all_features.append(batch_features)

        if verbose:
            print(f"  Processed batch {batch_idx + 1}/{num_batches} "
                  f"({end_idx}/{num_images} images)")

    features = np.vstack(all_features)
    return features


def extract_train_eval_features(
    train_images: List[np.ndarray],
    eval_images: List[np.ndarray],
    pipeline: ImagePipeline,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract train/eval features sharing one fitted reducer.

    fit_transform learns any 'reduce' projection on train; transform reuses it on
    the held-out split so both land in the same feature space. batch_process can't
    do this (it refits per call). Works too for a pipeline with no 'reduce' step.
    """
    if verbose:
        print(f"Fitting pipeline on {len(train_images)} train images...")
    # Learn the optional reducer on train and reduce the train features.
    X_train = pipeline.fit_transform(train_images)

    if verbose:
        print(f"Transforming {len(eval_images)} eval images with the train basis...")
    # Reuse the fitted projection; no refitting on the held-out split.
    X_eval = pipeline.transform(eval_images)

    if verbose:
        print(f"  -> train {X_train.shape}, eval {X_eval.shape} "
              f"(shared {X_train.shape[1:]} feature space)")
    return X_train, X_eval


def train_svm_model(
    features: np.ndarray,
    labels: np.ndarray,
    kernel: str = 'rbf',
    C: float = 1.0
) -> 'SVC':
    """Train an SVM on already-standardized pipeline features (no extra scaler)."""

    print(f"\nTraining SVM with {features.shape[0]} samples, {features.shape[1]} features...")

    # gamma='scale' is a solid default; higher C means a harder margin.
    svm = SVC(kernel=kernel, C=C, gamma='scale', verbose=0)
    svm.fit(features, labels)

    print(f"✓ SVM trained. Support vectors: {len(svm.support_vectors_)}")
    return svm


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100
) -> 'RandomForestClassifier':
    """Train a Random Forest on extracted features."""

    print(f"\nTraining Random Forest with {features.shape[0]} samples...")

    # Seeded for reproducible splits; n_jobs=-1 builds trees in parallel.
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(features, labels)
    
    print(f"✓ Random Forest trained with {n_estimators} trees")
    return rf


def evaluate_model(
    model,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    model_name: str = "Model"
) -> dict:
    """Evaluate a model and return accuracy/precision/recall/f1.

    Test features should come from the same pipeline used in training.
    """

    predictions = model.predict(features_test)

    # zero_division=0 avoids warnings when a class has no predicted samples.
    metrics = {
        'accuracy': accuracy_score(labels_test, predictions),
        'precision': precision_score(labels_test, predictions, zero_division=0),
        'recall': recall_score(labels_test, predictions, zero_division=0),
        'f1': f1_score(labels_test, predictions, zero_division=0)
    }
    
    print(f"\n{model_name} Evaluation:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    return metrics
