"""
Helper For 'train_model.py' 

Contains the Registry of prebuilt preprocessing pipelines that the project can use during training.
"""


from typing import Callable, Dict, List, Optional, Tuple

# -- Custom Classes -----------------------------------------------------------
from prebuilt_pipelines import PrebuiltPipelines
from preprocessing import ImagePipeline

# =============================================================================
# Feature pipeline registry — the per-image image -> vector transforms.
# =============================================================================
# Maps a CLI name to a PrebuiltPipelines factory. Each factory returns a
# self-contained ImagePipeline that does extraction *and* dimensionality
# reduction end to end: it already carries its own 'reduce' (vec-pca) and
# 'scale' tail, so build_feature_pipeline() appends nothing. The comments below
# give the final emitted feature width (post-PCA), not the raw pixel count. To
# add one, point at any PrebuiltPipelines factory.
PIPELINE_REGISTRY: Dict[str, Callable[[], ImagePipeline]] = {
    "svm": PrebuiltPipelines.svm_pipeline,      # 128x128 grayscale -> 150 PCA features
    "fast": PrebuiltPipelines.fast_pipeline,    # 64x64 grayscale   -> 150 PCA features
    "hq": PrebuiltPipelines.hq_pipeline,        # 224x224 grayscale -> 300 PCA features
    "no_denoise": PrebuiltPipelines.no_denoise_pipeline,  # svm minus denoise -> 150 PCA features
    "svm_jl": PrebuiltPipelines.svm_jl_pipeline,  # svm with JL (not PCA) reduce -> 150 JL features
    # Raw-pixel pipelines (no PCA) for the torch image models (CNN/ViT).
    "pixels": PrebuiltPipelines.pixels_pipeline,        # 64x64 grayscale   -> 4096 flat pixels
    "pixels_hq": PrebuiltPipelines.pixels_hq_pipeline,  # 128x128 grayscale -> 16384 flat pixels
}


# --- Optional embedding pipelines (VGG16) ---------------------------------
# Registered only when Keras is importable, mirroring the optional deep-model
# registration in model_registry.py: the VGG16 embedding front-end needs
# keras/tensorflow, so these are offered by name only when that optional
# dependency is present. Two variants differ only in the reduce method (PCA vs
# JL). With keras installed, `--pipeline embedding_pca` / `embedding_jl` become
# available automatically.
try:
    import keras  # noqa: F401

    PIPELINE_REGISTRY["embedding_pca"] = PrebuiltPipelines.embedding_pca_pipeline
    PIPELINE_REGISTRY["embedding_jl"] = PrebuiltPipelines.embedding_jl_pipeline
except ImportError:  # pragma: no cover - exercised only when keras is absent
    pass