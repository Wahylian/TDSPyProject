"""Registry of prebuilt preprocessing pipelines for train_model.py."""


from typing import Callable, Dict, List, Optional, Tuple

from .prebuilt_pipelines import PrebuiltPipelines
from preprocessing import ImagePipeline

# CLI name to a PrebuiltPipelines factory. Each returns a self-contained pipeline
# that extracts and reduces end to end (its own 'reduce'/'scale' tail), so
# build_feature_pipeline appends nothing. Comments give the emitted feature width.
PIPELINE_REGISTRY: Dict[str, Callable[[], ImagePipeline]] = {
    "svm": PrebuiltPipelines.svm_pipeline,      # 128x128 grayscale -> 150 PCA features
    "fast": PrebuiltPipelines.fast_pipeline,    # 64x64 grayscale   -> 150 PCA features
    "hq": PrebuiltPipelines.hq_pipeline,        # 224x224 grayscale -> 300 PCA features
    "no_denoise": PrebuiltPipelines.no_denoise_pipeline,  # svm minus denoise -> 150 PCA features
    "svm_jl": PrebuiltPipelines.svm_jl_pipeline,  # svm with JL (not PCA) reduce -> 150 JL features
    # Raw-pixel pipelines (no PCA) for the torch image models (CNN/ViT).
    "pixels": PrebuiltPipelines.pixels_pipeline,        # 64x64 grayscale   -> 4096 flat pixels
    "pixels_hq": PrebuiltPipelines.pixels_hq_pipeline,  # 128x128 grayscale -> 16384 flat pixels
    # RGB pixels for the pretrained torchvision backbones (cnn_pretrained/vit_pretrained).
    "pixels_pretrained": PrebuiltPipelines.pixels_pretrained_pipeline,  # 224x224 RGB -> 150528 flat pixels
}


# VGG16 embedding pipelines register only when keras is importable; the two
# variants differ only in the reduce method (PCA vs JL).
try:
    import keras  # noqa: F401

    PIPELINE_REGISTRY["embedding_pca"] = PrebuiltPipelines.embedding_pca_pipeline
    PIPELINE_REGISTRY["embedding_jl"] = PrebuiltPipelines.embedding_jl_pipeline
except ImportError:  # pragma: no cover - exercised only when keras is absent
    pass