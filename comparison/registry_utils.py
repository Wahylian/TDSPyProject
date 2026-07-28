"""Dynamic classification of registered models/pipelines into paired families.

Everything here is derived structurally from the live trainbase registries at
call time — no model or pipeline name is ever hardcoded — so a newly
registered entry is classified automatically.
"""

from __future__ import annotations

from typing import FrozenSet

from trainbase import PIPELINE_REGISTRY
from trainbase.model_registry import optional_torch_registries

TORCH = "torch"
CLASSICAL = "classical"


def _torch_model_names() -> FrozenSet[str]:
    """Names trainbase's torch-backed registries contribute to MODEL_REGISTRY:
    the from-scratch cnn/vit and the pretrained-backbone cnn_pretrained/
    vit_pretrained. Each is present only when its own optional dependency
    (torch, then torchvision) is installed; see optional_torch_registries.
    """
    return frozenset(optional_torch_registries())


def model_family(name: str) -> str:
    """Classify a registered model name as 'torch' or 'classical'.

    A model is 'torch' iff trainbase.torch_models itself contributes that name
    (diffed against its own build_torch_registry, not a fixed name list).
    """
    return TORCH if name in _torch_model_names() else CLASSICAL


def pipeline_family(name: str) -> str:
    """Classify a registered pipeline name as 'torch' (raw-pixel) or 'classical'.

    Built by constructing the pipeline and checking its operations: one with a
    batch-level 'reduce' step is a reduced+scaled classical front-end; one
    without emits flat raw pixels for a torch model to reshape back to square.
    """
    operations = PIPELINE_REGISTRY[name]().operations
    has_reduce = any(op_name == "reduce" for op_name, _ in operations)
    return CLASSICAL if has_reduce else TORCH
