"""Dynamic classification of registered models/pipelines into paired families.

Everything here is derived structurally from the live trainbase registries at
call time — no model or pipeline name is ever hardcoded — so a newly
registered entry is classified automatically.
"""

from __future__ import annotations

from typing import FrozenSet

from trainbase import PIPELINE_REGISTRY

TORCH = "torch"
CLASSICAL = "classical"


def _torch_model_names() -> FrozenSet[str]:
    """Names trainbase's torch-backed registries contribute to MODEL_REGISTRY:
    the from-scratch cnn/vit (torch_models) and the pretrained-backbone
    cnn_pretrained/vit_pretrained (torch_pretrained_models). Each is present
    only when its own optional dependency (torch, then torchvision) is installed.
    """
    names: set = set()
    try:
        from trainbase.torch_models import build_torch_registry
        names.update(build_torch_registry())
    except ImportError:
        pass
    try:
        from trainbase.torch_pretrained_models import build_pretrained_torch_registry
        names.update(build_pretrained_torch_registry())
    except ImportError:
        pass
    return frozenset(names)


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
