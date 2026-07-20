"""
prebuilt_pipelines.py — ready-made :class:`ImagePipeline` configurations.

A small library of named, opinionated preprocessing pipelines for common
scenarios (SVM-ready features, fast experimentation, high-quality extraction,
VGG16 embeddings, and the dimensionality-reduction variants). Each factory
returns a fresh :class:`ImagePipeline`, so callers can safely mutate or reuse
the result without affecting others.

Usage::

    from prebuilt_pipelines import PrebuiltPipelines
    from preprocessing import batch_process

    pipeline = PrebuiltPipelines.fast_pipeline()      # 64x64 grayscale -> 4096
    features = batch_process(images, pipeline)

The pipelines are built only from the public ``preprocessing`` package API, so
this module depends only on that stable surface, not the internal submodule
layout.
"""

from preprocessing import ImagePipeline

# A single seed threaded through every random operation (subsampling, PCA, the
# estimators) so the whole run is reproducible.
RANDOM_STATE = 42


class PrebuiltPipelines:
    """Collection of optimized pipelines for different scenarios."""

    @staticmethod
    def svm_pipeline() -> ImagePipeline:
        """
        Optimized for SVM training: medium resolution, normalized.

        Self-contained feature front-end: the 128x128 grayscale image (16,384
        raw pixels) is flattened, compressed with PCA to 150 components, and the
        components are standardized. The trailing ``vec-pca`` + ``scale`` steps
        are batch-level — they are fit on the training batch (via
        ``fit_transform``) and reused on val/test — so the emitted features are
        ready for a scale-sensitive kernel SVM with no further conditioning.

        Output: 150 standardized PCA features per image.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
            ('reduce', {'method': 'vec-pca', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def fast_pipeline() -> ImagePipeline:
        """
        Fast training pipeline: low resolution, minimal per-image preprocessing.

        Keeps the cheap 64x64 grayscale front-end (4,096 raw pixels) for quick
        experimentation, then reduces to 150 PCA components and standardizes them
        so the output drops straight into a scale-sensitive classifier. The
        ``vec-pca`` + ``scale`` steps are batch-level (fit on train, reused on
        val/test).

        Output: 150 standardized PCA features per image.
        Best for quick experimentation or when data is limited.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': 'vec-pca', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def hq_pipeline() -> ImagePipeline:
        """
        High-quality pipeline: high resolution, comprehensive preprocessing.

        The 224x224 grayscale front-end yields 50,176 raw pixels — far too wide
        to feed a kernel SVM directly — so it is reduced with PCA to 300
        components (more than the lower-resolution pipelines, to retain detail
        befitting the "accuracy critical" intent) and then standardized. The
        ``vec-pca`` + ``scale`` steps are batch-level (fit on train, reused on
        val/test).

        Output: 300 standardized PCA features per image.
        Best when accuracy is critical and computation time is not.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (224, 224), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 7}),
            ('normalize', {'method': 'standard'}),  # Standard normalization for better SVM
            ('vectorize', {}),
            ('reduce', {'method': 'vec-pca', 'n_components': 300, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def no_denoise_pipeline() -> ImagePipeline:
        """Denoise-ablation of :meth:`svm_pipeline` for comparison.

        Identical to ``svm_pipeline`` in every stage *except* the missing
        ``denoise`` step — same 128x128 grayscale front-end, same ``vec-pca`` to
        150 components, same ``scale`` — so an A/B run isolates the effect of
        denoising alone.

        Output: 150 standardized PCA features per image.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': 'vec-pca', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
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
    # Vector dimensionality reduction.
    # Each pipeline vectorizes the image, then reduces the flat feature vector.
    # All three share the same per-image stages and differ only in the trailing
    # ('reduce', {...}) op, for easy A/B comparison.
    # ----------------------------------------------------------------------

    @staticmethod
    def reduction_bypass_pipeline() -> ImagePipeline:
        """
        Baseline with the 'reduce' step present but set to ``None`` (bypass).

        Functionally identical to ``svm_pipeline`` — appending a
        ``('reduce', {'method': None})`` op documents "no dimensionality
        reduction" while keeping the slot available for A/B comparison against
        the vector or matrix reducers.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': None}),
        ])

    @staticmethod
    def vec_pca_pipeline(n_components: int = 128) -> ImagePipeline:
        """
        Vectorize, then reduce the flat vectors with PCA (``'vec-pca'``).

        PCA is a data-dependent linear projection that keeps the directions of
        maximum variance. It yields the most compact features for a given
        accuracy target, at the cost of fitting a covariance decomposition on
        the training batch.

        Args:
            n_components: Target post-PCA width. Clamped to
                ``min(n_samples, n_features)`` of the batch fed to
                ``batch_process``.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {
                'method': 'vec-pca',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])

    @staticmethod
    def vec_jl_pipeline(n_components: int = 256) -> ImagePipeline:
        """
        Vectorize, then reduce the flat vectors with a JL random projection
        (``'vec-jl'``).

        The Johnson–Lindenstrauss projection is data-independent: the matrix is
        drawn from a Gaussian and does not depend on the training data, so
        fitting is essentially free and the same matrix applies to streaming
        data without re-fitting. Pairwise-distance distortion is bounded by the
        Johnson–Lindenstrauss lemma.

        Args:
            n_components: Target post-projection width.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {
                'method': 'vec-jl',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])

    # ----------------------------------------------------------------------
    # Matrix dimensionality reduction.
    # These omit 'vectorize', so each image stays a 2D matrix and the reducer
    # compresses its column axis — producing (height, n_components) features
    # that keep the spatial layout CNNs and ViTs expect.
    # ----------------------------------------------------------------------

    @staticmethod
    def mat_pca_pipeline(n_components: int = 32) -> ImagePipeline:
        """
        Reduce grayscale image matrices with two-dimensional PCA (``'mat-pca'``).

        No ``'vectorize'`` step: each 128×128 image is projected along its
        column axis to ``(128, n_components)``, preserving rows for matrix-native
        models. PCA keeps the directions of maximum variance in the column space.

        Args:
            n_components: Target reduced width. Clamped to the image width.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('reduce', {
                'method': 'mat-pca',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])

    @staticmethod
    def mat_jl_pipeline(n_components: int = 64) -> ImagePipeline:
        """
        Reduce grayscale image matrices with a 2D JL random projection
        (``'mat-jl'``).

        No ``'vectorize'`` step: each 128×128 image's column axis is projected
        by a fixed Gaussian matrix to ``(128, n_components)``. Data-independent
        and cheap to fit, keeping the spatial layout intact for CNN/ViT inputs.

        Args:
            n_components: Target reduced width. Clamped to the image width.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('reduce', {
                'method': 'mat-jl',
                'n_components': n_components,
                'random_state': 42,
            }),
        ])
