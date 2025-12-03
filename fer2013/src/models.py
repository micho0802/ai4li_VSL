# src/models.py
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torchvision import models


def create_resnet50(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ResNet-50 classifier pretrained on ImageNet-1K,
    with a custom classification head.

    Args:
        num_classes: Number of output classes for the FER task.
        pretrained: If True, load ImageNet-1K pretrained weights.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, freeze all layers except the final head.

    Returns:
        A torch.nn.Module ready for training/evaluation.
    """
    weights = (
        models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    )
    model = models.resnet50(weights=weights)

    # Optionally freeze all backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model
