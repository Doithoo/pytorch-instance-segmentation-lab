"""An explicit external Mask R-CNN factory."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from instance_segmenter.models.torchvision_models import build_maskrcnn_resnet50_fpn


def build_model(*, num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    """Customize params or replace this factory while preserving the model contract."""
    return build_maskrcnn_resnet50_fpn(num_classes, weights, params)
