"""Torchvision Mask R-CNN factories with explicit weight policies."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


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


def build_maskrcnn_mobilenet_v3_large(num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    """Build a single-feature-map Mask R-CNN with an ImageNet MobileNetV3 backbone."""
    _validate_num_classes(num_classes)
    if weights == "none":
        backbone_weights = None
    elif weights == "imagenet_v2":
        backbone_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    else:
        raise ValueError(f"unsupported maskrcnn_mobilenet_v3_large weight policy {weights!r}")
    backbone = mobilenet_v3_large(weights=backbone_weights).features
    backbone.out_channels = 960
    anchors = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    box_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    mask_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)
    return MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchors,
        box_roi_pool=box_pool,
        mask_roi_pool=mask_pool,
        **dict(params),
    )


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
