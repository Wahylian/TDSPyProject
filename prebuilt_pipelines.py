"""
prebuilt_pipelines.py — ready-made :class:`ImagePipeline` configurations.

A small library of named, opinionated preprocessing pipelines for common
scenarios (SVM-ready features, fast experimentation, high-quality extraction,
VGG16 embeddings, and the dimensionality-reduction variants). Each factory
returns a fresh :class:`ImagePipeline`, so callers can safely mutate or reuse
the result without affecting others.

Usage::

    from prebuilt_pipelines import PrebuiltPipelines
    from image_preprocessing import batch_process

    pipeline = PrebuiltPipelines.fast_pipeline()      # 64x64 grayscale -> 4096
    features = batch_process(images, pipeline)

The pipelines are built only from the public ``image_preprocessing`` facade, so
this module has no dependency on the internal ``preprocessing/`` layout.
"""

from image_preprocessing import ImagePipeline


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

    # ----------------------------------------------------------------------
    # NEW (2026-06): Pipelines exercising the dimensionality-reduction step.
    # All three are identical apart from the trailing ('reduce', {...}) op.
    # ----------------------------------------------------------------------

    @staticmethod
    def reduction_bypass_pipeline() -> ImagePipeline:
        """
        Baseline with the new 'reduce' step present but set to ``None`` (bypass).

        This is functionally identical to ``svm_pipeline`` — appending a
        ``('reduce', {'method': None})`` op is the formal way to document
        "no dimensionality reduction" while keeping the slot available for
        easy A/B comparison against PCA / JL.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': None}),
        ])

    @staticmethod
    def pca_pipeline(n_components: int = 128) -> ImagePipeline:
        """
        Pipeline ending with PCA dimensionality reduction.

        PCA is a *data-dependent* linear projection that keeps the directions
        of maximum variance. It typically yields the most compact features for
        a given accuracy target, at the cost of needing to fit a covariance
        decomposition on the training batch.

        Args:
            n_components: Target post-PCA dimensionality. Must be ≤
                ``min(n_samples, n_features)`` of the batch fed to
                ``batch_process``.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {
                'method': 'pca',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])

    @staticmethod
    def jl_pipeline(n_components: int = 256) -> ImagePipeline:
        """
        Pipeline ending with Johnson–Lindenstrauss random projection.

        JL projection is *data-independent*: the projection matrix is drawn
        from a Gaussian and does not depend on the training data. This makes
        fitting trivially fast (essentially free) and means the same matrix
        can be applied to streaming data without re-fitting. Distortion of
        pairwise distances is bounded by the Johnson–Lindenstrauss lemma.

        Args:
            n_components: Target post-projection dimensionality.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {
                'method': 'johnson_lindenstrauss',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])
