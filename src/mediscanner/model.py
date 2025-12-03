from __future__ import annotations

from typing import Callable, Dict, Tuple, Optional, Mapping

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
    # ConvNeXt classifier input after avgpool is (N, C, 1, 1). Ensure flatten first.
    backbone.classifier = nn.Sequential(
        nn.Flatten(1),
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


class MultiTaskClassifier(nn.Module):
    """
    Wrapper around torchvision backbones to support multiple heads:
      - disease: multiclass classification (CrossEntropy)
      - severity: multiclass classification (CrossEntropy), optional
      - infectious: binary (BCEWithLogits), optional
      - urgent: binary (BCEWithLogits), optional

    Forward returns a dict with logits per head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        dropout: float,
        disease_classes: int,
        severity_classes: Optional[int] = None,
        enable_infectious: bool = False,
        enable_urgent: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pre_head = nn.Sequential(
            nn.LayerNorm((in_features,), eps=1e-6, elementwise_affine=True),
            nn.Dropout(dropout),
        )
        self.disease_head = nn.Linear(in_features, disease_classes)
        self.severity_head = nn.Linear(in_features, severity_classes) if severity_classes else None
        self.infectious_head = nn.Linear(in_features, 1) if enable_infectious else None
        self.urgent_head = nn.Linear(in_features, 1) if enable_urgent else None

    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        feats = self.backbone(x)  # expected to be penultimate features (N, F)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        z = self.pre_head(feats)
        out = {"disease": self.disease_head(z)}
        if self.severity_head is not None:
            out["severity"] = self.severity_head(z)
        if self.infectious_head is not None:
            out["infectious"] = self.infectious_head(z).squeeze(-1)
        if self.urgent_head is not None:
            out["urgent"] = self.urgent_head(z).squeeze(-1)
        return out


def build_multitask_model(
    model_name: str,
    disease_classes: int,
    dropout: float = 0.2,
    severity_classes: Optional[int] = 3,
    enable_infectious: bool = True,
    enable_urgent: bool = True,
) -> Tuple[nn.Module, Callable]:
    """
    Build a backbone configured to output penultimate features, then attach multi-head tops.
    Returns (model, transform).
    """
    name = model_name
    if name == "convnext_base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        backbone = models.convnext_base(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        model = MultiTaskClassifier(
            backbone=backbone,
            in_features=in_features,
            dropout=dropout,
            disease_classes=disease_classes,
            severity_classes=severity_classes,
            enable_infectious=enable_infectious,
            enable_urgent=enable_urgent,
        )
        return model, weights.transforms()
    elif name == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_v2_m(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        model = MultiTaskClassifier(
            backbone=backbone,
            in_features=in_features,
            dropout=dropout,
            disease_classes=disease_classes,
            severity_classes=severity_classes,
            enable_infectious=enable_infectious,
            enable_urgent=enable_urgent,
        )
        return model, weights.transforms()
    elif name == "swin_v2_b":
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        backbone = models.swin_v2_b(weights=weights)
        in_features = backbone.head.in_features
        backbone.head = nn.Identity()
        model = MultiTaskClassifier(
            backbone=backbone,
            in_features=in_features,
            dropout=dropout,
            disease_classes=disease_classes,
            severity_classes=severity_classes,
            enable_infectious=enable_infectious,
            enable_urgent=enable_urgent,
        )
        return model, weights.transforms()
    else:
        supported = ", ".join(sorted(BUILDERS))
        raise KeyError(f"Unsupported model '{model_name}'. Choose from {supported}.")
