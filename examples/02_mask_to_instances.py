"""Split a sparse instance-ID mask into independent masks and boxes."""

from __future__ import annotations

import torch

from instance_segmenter.data.masks import masks_from_instance_ids, masks_to_boxes, stack_instance_masks

instance_ids = torch.tensor([[0, 4, 4, 0], [9, 0, 9, 0]], dtype=torch.int64)
masks = stack_instance_masks(masks_from_instance_ids(instance_ids), 2, 4)
print("masks:", masks.shape)
print("boxes:", masks_to_boxes(masks).tolist())
