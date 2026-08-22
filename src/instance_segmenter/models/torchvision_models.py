"""Torchvision Mask R-CNN factories with explicit weight policies."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn_resnet50_fpn(num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    """Build Mask R-CNN and replace both heads for the requested label space."""
    _validate_num_classes(num_classes)
    kwargs = dict(params)
    if weights == "none":
        return maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs)
    if weights == "coco_v1":
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1, **kwargs)
        return _replace_predictors(model, num_classes)
    raise ValueError(f"unsupported maskrcnn_resnet50_fpn weight policy {weights!r}")


def build_maskrcnn_resnet50_fpn_v2(num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    """Build the v2 variant when installed torchvision exposes it."""
    _validate_num_classes(num_classes)
    from torchvision.models import detection as detection_models

    constructor = getattr(detection_models, "maskrcnn_resnet50_fpn_v2", None)
    weights_enum = getattr(detection_models, "MaskRCNN_ResNet50_FPN_V2_Weights", None)
    if constructor is None or weights_enum is None:
        raise RuntimeError("maskrcnn_resnet50_fpn_v2 requires a newer torchvision build")
    kwargs = dict(params)
    if weights == "none":
        return constructor(weights=None, weights_backbone=None, num_classes=num_classes, **kwargs)
    if weights == "coco_v1":
        model = constructor(weights=weights_enum.COCO_V1, **kwargs)
        return _replace_predictors(model, num_classes)
    raise ValueError(f"unsupported maskrcnn_resnet50_fpn_v2 weight policy {weights!r}")


def _replace_predictors(model: nn.Module, num_classes: int) -> nn.Module:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        raise TypeError("expected torchvision Mask R-CNN with roi_heads")
    box_predictor = roi_heads.box_predictor
    mask_predictor = roi_heads.mask_predictor
    roi_heads.box_predictor = FastRCNNPredictor(box_predictor.cls_score.in_features, num_classes)
    roi_heads.mask_predictor = MaskRCNNPredictor(mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model


def _validate_num_classes(num_classes: int) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
