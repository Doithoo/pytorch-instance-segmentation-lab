"""Collation for variable-size images and variable-count instance targets."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from instance_segmenter.data.schema import InstanceTarget


def instance_collate(
    batch: Sequence[tuple[torch.Tensor, InstanceTarget]],
) -> tuple[list[torch.Tensor], list[InstanceTarget]]:
    """Keep each image and target intact for torchvision detection models."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
