"""Synthetic instance data shared by fast contract tests."""

from __future__ import annotations

import torch

from instance_segmenter.data.masks import masks_to_boxes
from instance_segmenter.data.schema import InstanceTarget


def make_target(
    masks: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    image_id: int = 7,
) -> InstanceTarget:
    if labels is None:
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
    return {
        "boxes": masks_to_boxes(masks),
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": masks.flatten(1).sum(dim=1).to(torch.float32),
        "iscrowd": torch.zeros((masks.shape[0],), dtype=torch.int64),
    }


def sample_image_and_target() -> tuple[torch.Tensor, InstanceTarget]:
    image = torch.arange(90, dtype=torch.float32).reshape(3, 5, 6) / 90.0
    masks = torch.zeros((2, 5, 6), dtype=torch.bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 0, 4:6] = True
    return image, make_target(masks)
