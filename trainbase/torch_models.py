"""Torch-backed image classifiers (CNN, ViT) for the model registry.

Small PyTorch networks wrapped in the sklearn estimator interface so they drop
into MODEL_REGISTRY like any other classifier. Torch is optional: this module is
imported only when import torch succeeds.

The feature front-end hands over a flat matrix; paired with a no-PCA pixel
pipeline each row is a flattened grayscale image, which fit reshapes back to
(N, C, H, W) so the network sees real pixels. image_shape defaults to square
grayscale inferred from the vector width.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import torch
from torch import nn

from .model_registry import RANDOM_STATE, ModelSpec


def _resolve_image_shape(
    n_features: int, image_shape: Optional[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Resolve the (C, H, W) a flat vector of width n_features maps to.

    A given image_shape is validated against n_features; otherwise a square
    single-channel image is inferred. Raises if neither holds.
    """
    if image_shape is not None:
        c, h, w = image_shape
        if c * h * w != n_features:
            raise ValueError(
                f"image_shape {image_shape} has {c * h * w} elements but the "
                f"feature width is {n_features}."
            )
        return int(c), int(h), int(w)
    side = int(round(n_features ** 0.5))
    if side * side != n_features:
        raise ValueError(
            f"Cannot infer a square grayscale image from feature width "
            f"{n_features}; pass image_shape=(C, H, W) explicitly."
        )
    return 1, side, side


class _TorchImageClassifier(BaseEstimator, ClassifierMixin):
    """Base sklearn wrapper: reshape flat pixels, train a torch module, predict.

    Subclasses implement _build_module. Constructor args are plain attributes so
    sklearn clone/get_params/GridSearchCV work; fitted state is set only in fit.
    """

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.image_shape = image_shape
        self.device = device
        self.random_state = random_state

    def _build_module(self, in_shape: Tuple[int, int, int], n_classes: int) -> nn.Module:
        raise NotImplementedError

    def _device(self) -> "torch.device":
        """Resolve the training device: explicit device, else CUDA, else CPU with a warning."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "No CUDA GPU available; training on CPU (this may be slow).",
            stacklevel=2,
        )
        return torch.device("cpu")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.image_shape_ = _resolve_image_shape(self.n_features_in_, self.image_shape)

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = self._device()

        module = self._build_module(self.image_shape_, n_classes).to(device)
        c, h, w = self.image_shape_
        X_t = torch.from_numpy(X).view(-1, c, h, w).to(device)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)
        y_t = torch.from_numpy(y_idx).to(device)

        opt = torch.optim.Adam(module.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n = X_t.shape[0]
        bs = max(1, int(self.batch_size))
        module.train()
        for _ in range(int(self.epochs)):
            perm = rng.permutation(n)  # shuffled, seeded minibatches
            for start in range(0, n, bs):
                sel = perm[start:start + bs]
                xb, yb = X_t[sel], y_t[sel]
                opt.zero_grad()
                loss = loss_fn(module(xb), yb)
                loss.backward()
                opt.step()
        module.eval()
        # Store on CPU so the fitted estimator pickles/loads without a GPU.
        self.module_ = module.to(torch.device("cpu"))
        return self

    def _logits(self, X) -> "torch.Tensor":
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; estimator was fit on "
                f"{self.n_features_in_}."
            )
        c, h, w = self.image_shape_
        X_t = torch.from_numpy(X).view(-1, c, h, w)
        self.module_.eval()
        with torch.no_grad():
            return self.module_(X_t)

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(self._logits(X), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _CNNModule(nn.Module):
    """Small conv stack (Conv→ReLU→MaxPool blocks) → linear head."""

    def __init__(self, in_shape, n_classes, channels=(16, 32), n_blocks=2):
        super().__init__()
        if n_blocks > len(channels):
            raise ValueError(
                f"n_blocks ({n_blocks}) exceeds available channel widths "
                f"({len(channels)}); provide at least n_blocks channel values."
            )
        c, h, w = in_shape
        layers = []
        prev = c
        for out_c in channels[:n_blocks]:
            layers += [nn.Conv2d(prev, out_c, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            prev, h, w = out_c, h // 2, w // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(prev * h * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNClassifier(_TorchImageClassifier):
    """A small convolutional image classifier."""

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE,
                 channels: Tuple[int, ...] = (16, 32), n_blocks: int = 2):
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, image_shape=image_shape,
                         device=device, random_state=random_state)
        self.channels = channels
        self.n_blocks = n_blocks

    def _build_module(self, in_shape, n_classes):
        return _CNNModule(in_shape, n_classes, self.channels, self.n_blocks)


class _ViTModule(nn.Module):
    """Minimal Vision Transformer: patch embed → transformer encoder → CLS head."""

    def __init__(self, in_shape, n_classes, patch_size=8, embed_dim=64,
                 depth=2, n_heads=4, mlp_dim=128):
        super().__init__()
        c, h, w = in_shape
        if h % patch_size or w % patch_size:
            raise ValueError(
                f"image size ({h}x{w}) must be divisible by patch_size {patch_size}."
            )
        self.patch = nn.Conv2d(c, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (h // patch_size) * (w // patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        b = x.shape[0]
        p = self.patch(x).flatten(2).transpose(1, 2)      # (b, n_patches, embed)
        z = torch.cat([self.cls.expand(b, -1, -1), p], dim=1) + self.pos
        return self.head(self.encoder(z)[:, 0])


class ViTClassifier(_TorchImageClassifier):
    """A small Vision Transformer image classifier."""

    def __init__(self, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                 weight_decay: float = 0.0,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 device: Optional[str] = None, random_state: int = RANDOM_STATE,
                 patch_size: int = 8, embed_dim: int = 64, depth: int = 2,
                 n_heads: int = 4, mlp_dim: int = 128):
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, image_shape=image_shape,
                         device=device, random_state=random_state)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim

    def _build_module(self, in_shape, n_classes):
        return _ViTModule(in_shape, n_classes, self.patch_size, self.embed_dim,
                          self.depth, self.n_heads, self.mlp_dim)


def build_torch_registry() -> Dict[str, ModelSpec]:
    """Return the torch MODEL_REGISTRY entries.

    Two presets per architecture: a light default and a heavier deep variant,
    each with a tiny clf__lr grid so tuning stays cheap.
    """
    return {
        "cnn": ModelSpec(
            factory=lambda: CNNClassifier(epochs=10, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "cnn_deep": ModelSpec(
            factory=lambda: CNNClassifier(
                epochs=25, channels=(32, 64, 128), n_blocks=3,
                random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "vit": ModelSpec(
            factory=lambda: ViTClassifier(epochs=10, depth=2, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
        "vit_deep": ModelSpec(
            factory=lambda: ViTClassifier(
                epochs=20, depth=4, embed_dim=96, random_state=RANDOM_STATE),
            param_grid={"clf__lr": [1e-3, 3e-4]},
        ),
    }
