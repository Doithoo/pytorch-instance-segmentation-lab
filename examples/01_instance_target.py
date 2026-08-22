"""Create a minimal target with two independent person instances."""

from __future__ import annotations

import torch

from instance_segmenter.data.masks import masks_to_boxes

masks = torch.zeros((2, 5, 6), dtype=torch.bool)
masks[0, 1:3, 1:3] = True
masks[1, 0:2, 4:6] = True
target = {
    "boxes": masks_to_boxes(masks),
    "labels": torch.tensor([1, 1], dtype=torch.int64),
    "masks": masks,
    "image_id": torch.tensor([1], dtype=torch.int64),
    "area": masks.flatten(1).sum(dim=1).float(),
    "iscrowd": torch.zeros(2, dtype=torch.int64),
}
print({key: tuple(value.shape) for key, value in target.items()})
print(target["boxes"])
