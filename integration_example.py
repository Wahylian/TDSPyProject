"""
integration_example.py — manual end-to-end check of the feature-extraction and
preprocessing-pipeline API.

This is a runnable *demonstration / smoke-test* script (not a pytest target):
it wires the public ``preprocessing`` API together the way real calling
code would — load a sample dataset, build a pipeline, extract features in
batches, and (optionally) train and evaluate a classifier — so you can sanity
-check the whole flow by eye with ``python integration_example.py``-style use.

The reusable, named pipeline configurations it relied on now live in their own
module, :mod:`prebuilt_pipelines`, and are re-exported here for backward
compatibility (``from integration_example import PrebuiltPipelines`` still
works). The remaining helpers below are the data/feature/train/eval building
blocks used by that manual flow.
"""

import numpy as np
from typing import Tuple, List

# Import pipeline components.
# The names below form the public API of the `preprocessing` package; underneath
# they live in its submodules (transforms.py, vectorize.py, reduce.py,
# pipeline.py, io.py).
from preprocessing import (
    ImagePipeline,
    batch_process,
)

# The named pipeline factories moved to their own module. Re-exported here so
# existing ``from integration_example import PrebuiltPipelines`` imports keep
# working.
from prebuilt_pipelines import PrebuiltPipelines

# scikit-learn is optional and only the model-training helpers below need it.
# The tests that call those helpers guard themselves with ``importorskip``, so
# this import is kept non-fatal: a missing sklearn must not break importing the
# pipelines / data loaders above (which carry no sklearn dependency).
try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    pass


# ============================================================================
# PART 1: Data Loading Utilities
# ============================================================================

def load_sample_dataset(num_samples: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic dataset for demonstration.
    
    In production, replace this with actual image loading from:
    datasets/chuneeb/deepfake-detection-dataset-2026/FINAL_DATASET.csv
    
    Args:
        num_samples: Number of synthetic images to generate
    
    Returns:
        Tuple of (images, labels)
    """
    print(f"Generating {num_samples} synthetic images for demonstration...")
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # Generate synthetic images (in practice, load real images)
        # Real: real frame, Fake: generated/deepfake
        if i % 2 == 0:
            # Real-like image: more uniform, less noise
            image = np.random.normal(128, 20, (224, 224, 3)).astype(np.uint8)
            label = 0  # Real
        else:
            # Fake-like image: more variation, different patterns
            image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            label = 1  # Fake
        
        images.append(image)
        labels.append(label)
    
    return images, np.array(labels)


# ============================================================================
# PART 2: Feature Extraction
# ============================================================================

def extract_features_batch(
    images: List[np.ndarray],
    pipeline: ImagePipeline,
    batch_size: int = 8,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract features from batch of images with progress tracking.
    
    Args:
        images: List of image arrays
        pipeline: ImagePipeline instance
        batch_size: Number of images to process at once
        verbose: Print progress
    
    Returns:
        Feature matrix of shape (num_images, feature_dim)
    """
    num_images = len(images)

    # If the pipeline has any batch-level step (e.g. PCA / Johnson-Lindenstrauss
    # via the 'reduce' op), chunking would refit the reducer on each chunk and
    # produce inconsistent feature spaces across chunks. Process everything in
    # one batch_process call so the reducer is fit once over the full set.
    if pipeline.batch_operations():
        if verbose:
            print(f"  Pipeline contains batch-level ops "
                  f"({[n for n, _ in pipeline.batch_operations()]}); "
                  f"processing all {num_images} images in one batch.")
        return batch_process(images, pipeline)

    # Ceiling division: ensures the last partial batch is still processed
    num_batches = (num_images + batch_size - 1) // batch_size

    all_features = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        # Clamp end_idx so the final batch doesn't overflow the list
        end_idx = min((batch_idx + 1) * batch_size, num_images)

        batch_images = images[start_idx:end_idx]
        batch_features = batch_process(batch_images, pipeline)
        all_features.append(batch_features)
        
        if verbose:
            print(f"  Processed batch {batch_idx + 1}/{num_batches} "
                  f"({end_idx}/{num_images} images)")
    
    # Stack all batch results into a single (num_images, feature_dim) matrix
    features = np.vstack(all_features)
    return features


def extract_train_eval_features(
    train_images: List[np.ndarray],
    eval_images: List[np.ndarray],
    pipeline: ImagePipeline,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for a train/eval split sharing one fitted reducer.

    When the pipeline ends in a *fitting* ``'reduce'`` step (``vec-pca`` /
    ``vec-jl`` / ``mat-pca`` / ``mat-jl``), the projection must be learned on the
    training data and then reused unchanged on the held-out split — otherwise the
    two splits land in different feature spaces and the model sees inconsistent
    inputs. ``batch_process`` cannot do this: it refits a fresh reducer on every
    call. The pipeline's scikit-learn-style ``fit_transform`` / ``transform``
    pair is the right tool:

        * ``fit_transform(train_images)`` fits the reducer on train and returns
          the reduced train features, storing the fitted projection on the
          pipeline.
        * ``transform(eval_images)`` reuses that same projection (and so would
          ``pipeline.process(single_image)`` afterwards).

    For a pipeline *without* a ``'reduce'`` step this still works — it simply
    stacks the per-image vectors, with nothing to fit.

    Args:
        train_images: Images used to fit the reducer.
        eval_images: Held-out images (val/test) to project with the train basis.
        pipeline: ImagePipeline instance, typically ending in a ``'reduce'`` step.
        verbose: Print progress.

    Returns:
        ``(X_train, X_eval)`` feature matrices sharing the same feature space.

    Example:
        >>> from prebuilt_pipelines import PrebuiltPipelines
        >>> pipeline = PrebuiltPipelines.vec_pca_pipeline(n_components=64)
        >>> X_train, X_test = extract_train_eval_features(train_imgs, test_imgs, pipeline)
        >>> X_train.shape[1] == X_test.shape[1]  # same 64-d space
        True
    """
    if verbose:
        print(f"Fitting pipeline on {len(train_images)} train images...")
    # Learn the (optional) reducer on train and reduce the train features.
    X_train = pipeline.fit_transform(train_images)

    if verbose:
        print(f"Transforming {len(eval_images)} eval images with the train basis...")
    # Reuse the fitted projection — no refitting on the held-out split.
    X_eval = pipeline.transform(eval_images)

    if verbose:
        print(f"  -> train {X_train.shape}, eval {X_eval.shape} "
              f"(shared {X_train.shape[1:]} feature space)")
    return X_train, X_eval


# ============================================================================
# PART 3: Training and Evaluation (requires scikit-learn)
# ============================================================================

def train_svm_model(
    features: np.ndarray,
    labels: np.ndarray,
    kernel: str = 'rbf',
    C: float = 1.0
) -> Tuple['SVC', 'StandardScaler']:
    """Train SVM classifier on extracted features."""
    
    print(f"\nTraining SVM with {features.shape[0]} samples, {features.shape[1]} features...")
    
    # SVM is sensitive to feature scale; StandardScaler normalizes to zero mean, unit variance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # RBF kernel with gamma='scale' (1 / (n_features * X.var())) is a solid default;
    # C controls the margin softness — higher C = less tolerance for misclassification
    svm = SVC(kernel=kernel, C=C, gamma='scale', verbose=0)
    svm.fit(features_scaled, labels)
    
    print(f"✓ SVM trained. Support vectors: {len(svm.support_vectors_)}")
    # Return scaler alongside the model so test features can be transformed consistently
    return svm, scaler


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100
) -> 'RandomForestClassifier':
    """Train Random Forest classifier on extracted features."""
    
    print(f"\nTraining Random Forest with {features.shape[0]} samples...")
    
    # random_state=42 ensures reproducible tree splits across runs;
    # n_jobs=-1 uses all available CPU cores for parallel tree building
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(features, labels)
    
    print(f"✓ Random Forest trained with {n_estimators} trees")
    return rf


def evaluate_model(
    model,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    scaler=None,
    model_name: str = "Model"
) -> dict:
    """Evaluate model and return metrics."""
    
    # Apply the same scaling used during training (fit only on train set)
    if scaler is not None:
        features_test = scaler.transform(features_test)
    
    predictions = model.predict(features_test)
    
    # zero_division=0 suppresses warnings when a class has no predicted samples
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
