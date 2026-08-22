"""Show why torchvision detection batches are lists, not padded tensors."""

from __future__ import annotations

import torch

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.masks import masks_to_boxes
from instance_segmenter.data.schema import InstanceTarget


def target(masks: torch.Tensor, image_id: int) -> InstanceTarget:
    return {
        "boxes": masks_to_boxes(masks),
        "labels": torch.ones(masks.shape[0], dtype=torch.int64),
        "masks": masks,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": masks.flatten(1).sum(1).float(),
        "iscrowd": torch.zeros(masks.shape[0], dtype=torch.int64),
    }


first_masks = torch.zeros((1, 4, 5), dtype=torch.bool)
first_masks[0, 1:3, 1:3] = True
second_masks = torch.zeros((2, 3, 4), dtype=torch.bool)
second_masks[0, :1, :2] = True
second_masks[1, 1:, 2:] = True
images, targets = instance_collate(
    [(torch.zeros((3, 4, 5)), target(first_masks, 1)), (torch.zeros((3, 3, 4)), target(second_masks, 2))]
)
print([tuple(image.shape) for image in images])
print([int(item["masks"].shape[0]) for item in targets])
