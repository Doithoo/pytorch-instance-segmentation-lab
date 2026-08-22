"""Built-in instance segmentation model registry."""

from __future__ import annotations

from instance_segmenter.models.spec import ModelSpec
from instance_segmenter.models.torchvision_models import (
    build_maskrcnn_mobilenet_v3_large,
    build_maskrcnn_resnet50_fpn,
    build_maskrcnn_resnet50_fpn_v2,
)

_REGISTRY: dict[str, ModelSpec] = {
    "maskrcnn_mobilenet_v3_large": ModelSpec(
        name="maskrcnn_mobilenet_v3_large",
        factory=build_maskrcnn_mobilenet_v3_large,
        description="Educational Mask R-CNN with a lightweight MobileNetV3-Large backbone.",
        supported_weights=("none", "imagenet_v2"),
        input_notes=("Uses one backbone feature map and custom anchors/ROI pooling.",),
        parameters={"min_size": "Optional shortest side.", "max_size": "Optional longest side."},
    ),
    "maskrcnn_resnet50_fpn": ModelSpec(
        name="maskrcnn_resnet50_fpn",
        factory=build_maskrcnn_resnet50_fpn,
        description="Torchvision Mask R-CNN with a ResNet-50 FPN backbone.",
        input_notes=("Accepts a list of float32 CHW images in [0, 1].", "Training requires independent bool masks."),
        parameters={"min_size": "Optional shortest side passed to torchvision.", "max_size": "Optional longest side."},
    ),
    "maskrcnn_resnet50_fpn_v2": ModelSpec(
        name="maskrcnn_resnet50_fpn_v2",
        factory=build_maskrcnn_resnet50_fpn_v2,
        description="Torchvision Mask R-CNN ResNet-50 FPN v2 when available.",
        input_notes=("Availability depends on the installed torchvision build.",),
        parameters={"min_size": "Optional shortest side passed to torchvision.", "max_size": "Optional longest side."},
    ),
}


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(list_models())}") from exc


def build_model(name: str, num_classes: int, weights: str, params: dict[str, object] | None = None):
    spec = get_model_spec(name)
    if weights not in spec.supported_weights:
        raise ValueError(f"{name} supports weights: {', '.join(spec.supported_weights)}")
    return spec.factory(num_classes, weights, params or {})
