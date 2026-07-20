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
# Maps a CLI name to a PrebuiltPipelines factory. These are the per-image *base*
# pipelines (image -> flat vector). build_feature_pipeline() appends the project's
# batch-level 'vec-pca' reduce step to the chosen base, so a single ImagePipeline
# does extraction *and* PCA reduction end to end. To add one, point at any
# per-image PrebuiltPipelines factory.
PIPELINE_REGISTRY: Dict[str, Callable[[], ImagePipeline]] = {
    "svm": PrebuiltPipelines.svm_pipeline,      # 128x128 grayscale -> 16,384
    "fast": PrebuiltPipelines.fast_pipeline,    # 64x64 grayscale  ->  4,096
    "hq": PrebuiltPipelines.hq_pipeline,        # 224x224 grayscale -> 50,176
    "no_denoise": PrebuiltPipelines.no_denoise_pipeline,
}