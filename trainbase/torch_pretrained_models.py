"""Pretrained torchvision backbones (ResNet18, ViT-B/16), fine-tuned on real-vs-fake.

Unlike CNNClassifier/ViTClassifier in torch_models.py (trained from scratch on
~5,000 images), these start from ImageNet weights: the backbone is frozen and
only its classification head is trained, so the model adapts existing visual
features to the task instead of learning them from a small dataset. Pairs with
the 'pixels_pretrained' pipeline (224x224 RGB, no PCA/JL reduce).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torchvision.models import (
    ViT_B_16_Weights,
    ResNet18_Weights,
    resnet18,
    vit_b_16,
)

from .model_registry import RANDOM_STATE, ModelSpec
from .torch_models import _TorchImageClassifier

# ImageNet per-channel normalization the pretrained backbones were trained
# with. Pipelines hand over [0, 1]-scaled pixels (see vectorize_image), so this
# is applied inside the module, mirroring how vectorize.py's vgg16 method owns
# its own preprocess_input rather than pushing it into the generic pipeline.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ImageNetNormalize(nn.Module):
    """Per-channel ImageNet normalization as the network's first layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class _PretrainedImageClassifier(_TorchImageClassifier):
    """Shared plumbing for a torchvision-pretrained backbone with a fresh head.

    freeze_backbone (default True) trains only the replacement classification
    head on top of frozen ImageNet features -- the small training set here
    isn't enough to fine-tune a whole ImageNet-scale backbone without
    overfitting. pretrained=False (real weights skipped, random init instead)
    exists for fast tests that shouldn't download ImageNet weights.
    """

    def __init__(self, epochs: int = 5, lr: float = 1e-4, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Tuple[int, int, int] = (3, 224, 224),
                 device: Optional[str] = None, random_state: int = RANDOM_STATE,
                 pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, image_shape=image_shape,
                         device=device, random_state=random_state)
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

    def __getstate__(self) -> dict:
        """Drop the frozen backbone from the pickle when it's exactly reconstructible.

        With freeze_backbone=True the optimizer skips every backbone parameter
        (requires_grad=False), so those weights never move from the torchvision
        checkpoint _build_module loads on rebuild; only the trainable head's
        parameters -- and any buffers, e.g. BatchNorm running stats, which drift
        during training regardless of requires_grad -- need to survive the
        pickle. This shrinks a ~330MB ViT-B/16 (or ~45MB ResNet18) artifact to a
        few tens of KB. Skipped when pretrained=False, since a random-init
        backbone isn't reconstructible without replaying the exact training-time
        RNG state.
        """
        state = self.__dict__.copy()
        module = state.get("module_")
        if module is not None and self.pretrained and self.freeze_backbone:
            state["module_"] = None
            state["_slim_module_state"] = {
                **{n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad},
                **{n: b.detach().cpu() for n, b in module.named_buffers()},
            }
        return state

    def __setstate__(self, state: dict) -> None:
        slim_module_state = state.pop("_slim_module_state", None)
        self.__dict__.update(state)
        if slim_module_state is not None:
            module = self._build_module(self.image_shape_, len(self.classes_))
            module.load_state_dict(slim_module_state, strict=False)
            module.eval()
            self.module_ = module


class CNNPretrainedClassifier(_PretrainedImageClassifier):
    """ResNet18 pretrained on ImageNet, with its head fine-tuned here."""

    def _build_module(self, in_shape, n_classes):
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        backbone = resnet18(weights=weights)
        if self.freeze_backbone:
            _freeze(backbone)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        return nn.Sequential(_ImageNetNormalize(), backbone)


class ViTPretrainedClassifier(_PretrainedImageClassifier):
    """ViT-B/16 pretrained on ImageNet, with its head fine-tuned here."""

    def _build_module(self, in_shape, n_classes):
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if self.pretrained else None
        backbone = vit_b_16(weights=weights)
        if self.freeze_backbone:
            _freeze(backbone)
        backbone.heads.head = nn.Linear(backbone.hidden_dim, n_classes)
        return nn.Sequential(_ImageNetNormalize(), backbone)


def build_pretrained_torch_registry() -> dict:
    """Return the pretrained-backbone MODEL_REGISTRY entries.

    Only lr is grid-searched, matching the from-scratch cnn/vit entries; with
    the backbone frozen there are far fewer trainable parameters to tune.
    """
    return {
        "cnn_pretrained": ModelSpec(
            factory=lambda: CNNPretrainedClassifier(random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 1e-4]},
        ),
        "vit_pretrained": ModelSpec(
            factory=lambda: ViTPretrainedClassifier(random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 1e-4]},
        ),
    }
