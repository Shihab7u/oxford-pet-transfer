"""MobileNetV3-Small construction and the three adaptation strategies."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CLASSES = 37
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
WEIGHTS_NAME = "MobileNet_V3_Small_Weights.DEFAULT (ImageNet-1K)"
FEATURE_DIM = 576  # output of model.features + global average pooling


def build_model(mode: str = "full", num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Small with a fresh 37-way classifier head.

    mode="full"  : every parameter is trainable.
    mode="head"  : only ``classifier[0]`` and ``classifier[3]`` (the two FC layers) train.
    mode="frozen": everything frozen (used for feature extraction).
    """
    weights = WEIGHTS if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if mode in {"head", "frozen"}:
        for param in model.parameters():
            param.requires_grad = False
        if mode == "head":
            for layer in (model.classifier[0], model.classifier[3]):
                for param in layer.parameters():
                    param.requires_grad = True
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")
    return model


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


class FeatureExtractor(nn.Module):
    """Frozen backbone features + global average pooling -> 576-d vector."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
