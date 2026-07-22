"""Ready-made ImagePipeline configurations for common training scenarios.

Each factory returns a fresh ImagePipeline built only from the public
preprocessing API, so callers can mutate or reuse the result freely.
"""

from preprocessing import ImagePipeline

# One seed threaded through every random operation for reproducibility.
RANDOM_STATE = 42


class PrebuiltPipelines:
    """Named preprocessing pipelines for different scenarios."""

    @staticmethod
    def svm_pipeline() -> ImagePipeline:
        """128x128 grayscale, denoised, minmax-normalized, PCA to 150, standardized.

        The trailing vec-pca + scale steps fit on train and reuse on val/test,
        emitting 150 features ready for a scale-sensitive kernel SVM.
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
    def svm_jl_pipeline() -> ImagePipeline:
        """JL-reduction twin of svm_pipeline for a PCA-vs-JL comparison.

        Identical to svm_pipeline except the reduce step is a data-independent
        Johnson-Lindenstrauss projection (vec-jl), so an A/B run isolates the
        reduction method.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
            ('reduce', {'method': 'vec-jl', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def fast_pipeline() -> ImagePipeline:
        """Cheap 64x64 grayscale front-end, PCA to 150, standardized.

        For quick experimentation or limited data. Emits 150 features.
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
        """224x224 grayscale, denoised, standard-normalized, PCA to 300, standardized.

        More components than the lower-res pipelines to retain detail. Best when
        accuracy matters more than speed.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (224, 224), 'preserve_aspect': True}),
            ('denoise', {'method': 'bilateral', 'kernel_size': 7}),
            ('normalize', {'method': 'standard'}),
            ('vectorize', {}),
            ('reduce', {'method': 'vec-pca', 'n_components': 300, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def no_denoise_pipeline() -> ImagePipeline:
        """svm_pipeline without the denoise step, to isolate denoising's effect."""
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
        """64x64 grayscale VGG16 block5 embedding (25,088-dim), no reduction.

        No pre-embedding normalize: VGG16's preprocess_input owns input scaling,
        and normalizing to [0, 1] first would truncate to black in the uint8 path.
        """
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('vectorize', {'method': "vgg16"})
        ])

    # VGG16-embedding pipelines, reduced and standardized. The two variants differ
    # only in reduce method (PCA vs JL). No pre-embedding normalize: VGG16's
    # preprocess_input owns input scaling, so a [0, 1] normalize would truncate to
    # black in the uint8 path; 'scale' standardizes the embedding at the end.

    @staticmethod
    def embedding_pca_pipeline() -> ImagePipeline:
        """224x224 VGG16 block5 embedding, PCA to 150, standardized."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (224, 224), 'preserve_aspect': True}),
            ('vectorize', {'method': 'vgg16'}),
            ('reduce', {'method': 'vec-pca', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    @staticmethod
    def embedding_jl_pipeline() -> ImagePipeline:
        """embedding_pca_pipeline but with a JL projection instead of PCA."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (224, 224), 'preserve_aspect': True}),
            ('vectorize', {'method': 'vgg16'}),
            ('reduce', {'method': 'vec-jl', 'n_components': 150, 'random_state': RANDOM_STATE}),
            ('scale', {}),
        ])

    # Raw-pixel pipelines omit 'reduce'/'scale', emitting the full flattened pixel
    # vector so a CNN/ViT can reshape it back to an image.

    @staticmethod
    def pixels_pipeline() -> ImagePipeline:
        """64x64 grayscale, minmax-normalized, flattened to 4,096 pixels for CNN/ViT."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
        ])

    @staticmethod
    def pixels_hq_pipeline() -> ImagePipeline:
        """pixels_pipeline at 128x128, flattened to 16,384 pixels for the deep presets."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax', 'value_range': (0.0, 1.0)}),
            ('vectorize', {'preserve_structure': False}),
        ])

    # Vector reduction: vectorize, then reduce the flat vector. These share the
    # per-image stages and differ only in the trailing ('reduce', {...}) op.

    @staticmethod
    def reduction_bypass_pipeline() -> ImagePipeline:
        """Baseline with 'reduce' present but set to None, documenting no reduction."""
        return ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (128, 128), 'preserve_aspect': True}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': None}),
        ])

    @staticmethod
    def vec_pca_pipeline(n_components: int = 128) -> ImagePipeline:
        """Vectorize, then PCA-reduce to n_components (clamped to the batch rank limit)."""
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
        """Vectorize, then reduce with a data-independent JL projection to n_components."""
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

    # Matrix reduction omits 'vectorize', so each image stays a 2D matrix and the
    # reducer compresses its column axis to (height, n_components), keeping the
    # spatial layout CNNs and ViTs expect.

    @staticmethod
    def mat_pca_pipeline(n_components: int = 32) -> ImagePipeline:
        """Reduce 128x128 grayscale matrices along the column axis with 2D PCA."""
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
        """Reduce 128x128 grayscale matrices along the column axis with a 2D JL projection."""
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
