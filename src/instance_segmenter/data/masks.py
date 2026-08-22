"""Instance-ID mask parsing and geometry helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from instance_segmenter.data.schema import InstanceTarget, LabelSchema, validate_instance_target


class MaskError(ValueError):
    """Raised when a source instance mask is invalid."""


def decode_instance_mask(path: str | Path) -> list[torch.Tensor]:
    """Decode an indexed Penn-Fudan style mask into independent boolean masks."""
    source = Path(path)
    try:
        with Image.open(source) as image:
            array = np.asarray(image)
    except OSError as exc:
        raise MaskError(f"cannot read instance mask {source}: {exc}") from exc
    if array.ndim != 2:
        raise MaskError(f"instance mask {source} must be one-channel indexed data, got shape {array.shape}")
    return masks_from_instance_ids(torch.from_numpy(array.astype(np.int64, copy=True)))


def masks_from_instance_ids(instance_ids: torch.Tensor) -> list[torch.Tensor]:
    """Split positive IDs even when IDs are sparse or non-contiguous."""
    if instance_ids.ndim != 2:
        raise MaskError("instance id mask must have shape [H, W]")
    if instance_ids.dtype == torch.bool or instance_ids.is_floating_point() or instance_ids.is_complex():
        raise MaskError("instance id mask must use an integer dtype")
    unique_ids = torch.unique(instance_ids)
    if torch.any(unique_ids < 0):
        raise MaskError("instance id masks cannot contain negative values")
    return [
        (instance_ids == instance_id).to(dtype=torch.bool) for instance_id in unique_ids.tolist() if instance_id != 0
    ]


def stack_instance_masks(masks: list[torch.Tensor], height: int, width: int) -> torch.Tensor:
    """Create a validated [N, H, W] boolean tensor, including N=0."""
    if height <= 0 or width <= 0:
        raise MaskError("mask dimensions must be positive")
    if not masks:
        return torch.empty((0, height, width), dtype=torch.bool)
    normalized: list[torch.Tensor] = []
    for index, mask in enumerate(masks):
        if mask.dtype != torch.bool or tuple(mask.shape) != (height, width):
            raise MaskError(f"mask {index} must be bool with shape [{height}, {width}]")
        normalized.append(mask)
    return torch.stack(normalized)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Derive half-open xyxy boxes directly from each binary mask."""
    if masks.dtype != torch.bool or masks.ndim != 3:
        raise MaskError("masks must be bool with shape [N, H, W]")
    count = masks.shape[0]
    boxes = torch.zeros((count, 4), dtype=torch.float32, device=masks.device)
    for index, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if xs.numel():
            boxes[index] = torch.tensor(
                [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32, device=masks.device
            )
    return boxes


def remove_empty_instances(target: InstanceTarget) -> InstanceTarget:
    """Filter empty masks and all aligned instance fields without inventing boxes."""
    masks = target["masks"]
    if masks.dtype != torch.bool or masks.ndim != 3:
        raise MaskError("masks must be bool with shape [N, H, W]")
    keep = masks.flatten(1).any(dim=1)
    return {
        "boxes": target["boxes"][keep],
        "labels": target["labels"][keep],
        "masks": masks[keep],
        "image_id": target["image_id"],
        "area": target["area"][keep],
        "iscrowd": target["iscrowd"][keep],
    }


def rebuild_target_geometry(target: InstanceTarget) -> InstanceTarget:
    """Recompute boxes and area after a mask geometry operation."""
    masks = target["masks"].to(dtype=torch.bool)
    rebuilt: InstanceTarget = {
        "boxes": masks_to_boxes(masks),
        "labels": target["labels"].to(dtype=torch.int64),
        "masks": masks,
        "image_id": target["image_id"].to(dtype=torch.int64),
        "area": masks.flatten(1).sum(dim=1).to(dtype=torch.float32),
        "iscrowd": target["iscrowd"].to(dtype=torch.int64),
    }
    return remove_empty_instances(rebuilt)


def validate_instance_masks(masks: torch.Tensor, height: int, width: int, schema: LabelSchema) -> None:
    """Validate mask shape independently from labels; schema establishes shared ownership."""
    del schema
    if masks.dtype != torch.bool or tuple(masks.shape[1:]) != (height, width):
        raise MaskError(f"masks must have dtype bool and shape [N, {height}, {width}]")


def validate_geometry(target: InstanceTarget, height: int, width: int, schema: LabelSchema) -> None:
    """Validate masks, boxes, labels, and area after parsing or transforms."""
    validate_instance_masks(target["masks"], height, width, schema)
    validate_instance_target(target, height=height, width=width, schema=schema)
    boxes = target["boxes"]
    if boxes.numel() and (
        torch.any(boxes[:, 0] < 0)
        or torch.any(boxes[:, 1] < 0)
        or torch.any(boxes[:, 2] > width)
        or torch.any(boxes[:, 3] > height)
    ):
        raise MaskError("boxes are outside image bounds")
