"""
Integration Example: Using Image Preprocessing Pipeline with ML Models

This script demonstrates how to integrate the image_preprocessing module
with traditional ML models (SVM, Random Forest) for deepfake detection.

Shows:
1. Loading images from the dataset
2. Building preprocessing pipelines
3. Training and evaluating ML models
4. Batch feature extraction
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
import sys

# Import pipeline components
from image_preprocessing import ImagePipeline, batch_process, vectorize_image

# Optional: ML library imports (comment out if not installed)
try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    HAS_SKLEARN = False
    # Provide fallback for train_test_split
    def train_test_split(*args, **kwargs):
        """Fallback simple split (80/20)."""
        test_size = kwargs.get('test_size', 0.2)
        data = args[0]
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:], args[1][:split_idx], args[1][split_idx:]


# ============================================================================
# PART 1: Predefined Pipelines for Different Use Cases
# ============================================================================

class PrebuiltPipelines:
    """Collection of optimized pipelines for different scenarios."""
    
    @staticmethod
    def svm_pipeline() -> ImagePipeline:
        """
        Optimized for SVM training: medium resolution, normalized.
        Output: 16,384 features per image (128x128 grayscale)
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False})
        ])
    
    @staticmethod
    def fast_pipeline() -> ImagePipeline:
        """
        Fast training pipeline: low resolution, minimal preprocessing.
        Output: 4,096 features per image (64x64 grayscale)
        Best for quick experimentation or when data is limited.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {})
        ])
    
    @staticmethod
    def hq_pipeline() -> ImagePipeline:
        """
        High-quality pipeline: high resolution, comprehensive preprocessing.
        Output: 50,176 features per image (224x224 grayscale)
        Best when accuracy is critical and computation time is not.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (224, 224), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 7}),
            ('normalize', {'method': 'standard'}),  # Standard normalization for better SVM
            ('vectorize', {})
        ])
    
    @staticmethod
    def no_denoise_pipeline() -> ImagePipeline:
        """Pipeline without denoising for comparison."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {})
        ])
    
    @staticmethod
    def fast_embedding_pipeline() -> ImagePipeline:
        """Fast embedding pipeline: low resolution, using vector embedding for the images.
        Output: 4,096 features per image (64x64 grayscale)
        Best for quick experimentation with vector embeddings."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {'method': "vgg16"})
        ])


# ============================================================================
# PART 2: Data Loading Utilities
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
# PART 3: Feature Extraction
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


# ============================================================================
# PART 4: Training and Evaluation (requires scikit-learn)
# ============================================================================

def train_svm_model(
    features: np.ndarray,
    labels: np.ndarray,
    kernel: str = 'rbf',
    C: float = 1.0
) -> Tuple['SVC', 'StandardScaler']:
    """Train SVM classifier on extracted features."""
    
    print(f"\nTraining SVM with {features.shape[0]} samples, {features.shape[1]} features...")
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train SVM
    svm = SVC(kernel=kernel, C=C, gamma='scale', verbose=0)
    svm.fit(features_scaled, labels)
    
    print(f"✓ SVM trained. Support vectors: {len(svm.support_vectors_)}")
    return svm, scaler


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 100
) -> 'RandomForestClassifier':
    """Train Random Forest classifier on extracted features."""
    
    print(f"\nTraining Random Forest with {features.shape[0]} samples...")
    
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
    
    if scaler is not None:
        features_test = scaler.transform(features_test)
    
    predictions = model.predict(features_test)
    
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


# ============================================================================
# PART 5: Pipeline Comparison
# ============================================================================

def compare_pipelines(
    images_train: List[np.ndarray],
    labels_train: np.ndarray,
    images_test: List[np.ndarray],
    labels_test: np.ndarray
) -> None:
    """Compare performance of different preprocessing pipelines."""
    
    if not HAS_SKLEARN:
        print("Skipping pipeline comparison: scikit-learn not installed")
        return
    
    pipelines = {
        'Fast (64x64)': PrebuiltPipelines.fast_pipeline(),
        'Balanced (128x128)': PrebuiltPipelines.svm_pipeline(),
        'High-Quality (224x224)': PrebuiltPipelines.hq_pipeline(),
        'Fast Vector-Embedding (64x64)': PrebuiltPipelines.fast_embedding_pipeline(),
    }
    
    results = {}
    
    for pipeline_name, pipeline in pipelines.items():
        print(f"\n{'='*70}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*70}")
        print(str(pipeline))
        
        # Extract features
        print(f"\nExtracting training features...")
        features_train = extract_features_batch(images_train, pipeline, verbose=False)
        print(f"Train features shape: {features_train.shape}")
        
        print(f"Extracting test features...")
        features_test = extract_features_batch(images_test, pipeline, verbose=False)
        print(f"Test features shape: {features_test.shape}")
        
        # Train and evaluate
        if len(np.unique(labels_train)) > 1:  # Need both classes
            svm, scaler = train_svm_model(features_train, labels_train)
            metrics = evaluate_model(svm, features_test, labels_test, scaler, pipeline_name)
            results[pipeline_name] = metrics
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("PIPELINE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pipeline':<30} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 54)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f}")


# ============================================================================
# PART 6: vectorize_image Demonstration
# ============================================================================

def demonstrate_vectorize_image() -> None:
    """
    Demonstrate vectorize_image directly, covering all methods and options:
      - 'flat' on a color image
      - 'flat' on a grayscale image
      - 'flat' with preserve_structure=True (channels concatenated separately)
      - 'vgg16' embedding (skipped gracefully if Keras not installed)
    """
    print(f"\n{'='*70}")
    print("PART 6: vectorize_image Demonstration")
    print(f"{'='*70}")

    color_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    gray_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    # --- flat, color ---
    print("\n[6a] method='flat', color image (224x224x3)")
    vec = vectorize_image(color_image, method='flat')
    print(f"  Input shape:  {color_image.shape}")
    print(f"  Output shape: {vec.shape}  (224*224*3 = {224*224*3})")
    print(f"  dtype: {vec.dtype}")

    # --- flat, grayscale ---
    print("\n[6b] method='flat', grayscale image (224x224)")
    vec_gray = vectorize_image(gray_image, method='flat')
    print(f"  Input shape:  {gray_image.shape}")
    print(f"  Output shape: {vec_gray.shape}  (224*224 = {224*224})")
    print(f"  dtype: {vec_gray.dtype}")

    # --- flat, preserve_structure=True ---
    print("\n[6c] method='flat', preserve_structure=True (channels concatenated separately)")
    vec_structured = vectorize_image(color_image, method='flat', preserve_structure=True)
    print(f"  Input shape:  {color_image.shape}")
    print(f"  Output shape: {vec_structured.shape}  (same total, channel order preserved)")
    # Verify the first channel block matches channel 0 flattened
    expected_ch0 = color_image[:, :, 0].flatten().astype(np.float32)
    match = np.array_equal(vec_structured[:224*224], expected_ch0)
    print(f"  First channel block matches channel 0: {match}")

    # --- vgg16 ---
    print("\n[6d] method='vgg16' (requires Keras/TensorFlow)")
    try:
        vec_vgg = vectorize_image(color_image, method='vgg16', input_size=(224, 224))
        print(f"  Input shape:  {color_image.shape}")
        print(f"  Output shape: {vec_vgg.shape}")
        print(f"  dtype: {vec_vgg.dtype}")
    except ImportError as e:
        print(f"  Skipped — {e}")


# ============================================================================
# MAIN: Full Workflow Example
# ============================================================================

def main():
    """Complete example: load data → extract features → train model → evaluate."""
    
    print("=" * 70)
    print("IMAGE PREPROCESSING PIPELINE - INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    images, labels = load_sample_dataset(num_samples=20)
    print(f"✓ Loaded {len(images)} images")
    print(f"  Class distribution: Real={sum(labels==0)}, Fake={sum(labels==1)}")
    
    # Step 2: Split into train/test
    print("\n[Step 2] Splitting dataset...")
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"✓ Train: {len(images_train)} images")
    print(f"✓ Test:  {len(images_test)} images")
    
    # Step 3: Single pipeline example
    print("\n[Step 3] Extracting features with single pipeline...")
    pipeline = PrebuiltPipelines.svm_pipeline()
    print(f"Pipeline: {pipeline}")
    
    features_train = extract_features_batch(images_train, pipeline)
    features_test = extract_features_batch(images_test, pipeline)
    
    print(f"✓ Train features: {features_train.shape}")
    print(f"✓ Test features:  {features_test.shape}")
    print(f"✓ Feature statistics (train):")
    print(f"    Mean: {features_train.mean():.4f}, Std: {features_train.std():.4f}")
    print(f"    Min:  {features_train.min():.4f}, Max: {features_train.max():.4f}")
    
    # Step 4: Train models (if scikit-learn available)
    if HAS_SKLEARN:
        print("\n[Step 4] Training and evaluating models...")
        
        # SVM
        svm, scaler = train_svm_model(features_train, labels_train)
        svm_metrics = evaluate_model(svm, features_test, labels_test, scaler, "SVM")
        
        # Random Forest
        rf = train_random_forest(features_train, labels_train)
        rf_metrics = evaluate_model(rf, features_test, labels_test, model_name="Random Forest")
        
        # Step 5: Compare pipelines
        print("\n[Step 5] Comparing different preprocessing pipelines...")
        compare_pipelines(images_train, labels_train, images_test, labels_test)
    else:
        print("\n[Step 4-5] Skipped (scikit-learn not installed)")
        print("  Install with: pip install scikit-learn")
    
    # Step 6: vectorize_image demonstration
    demonstrate_vectorize_image()

    print("\n" + "=" * 70)
    print("✓ Integration example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()