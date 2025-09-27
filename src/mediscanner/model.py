from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Base_Weights,
    EfficientNet_V2_M_Weights,
    Swin_V2_B_Weights,
)


def _build_convnext(num_classes: int, dropout: float) -> Tuple[nn.Module, Callable]:
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    backbone = models.convnext_base(weights=weights)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(
        nn.LayerNorm((in_features,), eps=1e-6, elementwise_affine=True),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone, weights.transforms()


def _build_efficientnet(num_classes: int, dropout: float) -> Tuple[nn.Module, Callable]:
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_v2_m(weights=weights)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone, weights.transforms()


def _build_swin(num_classes: int, dropout: float) -> Tuple[nn.Module, Callable]:
    weights = Swin_V2_B_Weights.IMAGENET1K_V1
    backbone = models.swin_v2_b(weights=weights)
    in_features = backbone.head.in_features
    backbone.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone, weights.transforms()


BUILDERS: Dict[str, Callable[[int, float], Tuple[nn.Module, Callable]]] = {
    "convnext_base": _build_convnext,
    "efficientnet_v2_m": _build_efficientnet,
    "swin_v2_b": _build_swin,
}


def build_model(model_name: str, num_classes: int, dropout: float = 0.2) -> Tuple[nn.Module, Callable]:
    if model_name not in BUILDERS:
        supported = ", ".join(sorted(BUILDERS))
        raise KeyError(f"Unsupported model '{model_name}'. Choose from {supported}.")
    model, transform = BUILDERS[model_name](num_classes, dropout)
    return model, transform
