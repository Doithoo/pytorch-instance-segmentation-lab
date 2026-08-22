"""Run a tiny contract model through one real optimizer update."""

from __future__ import annotations

import torch
from torch import nn

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.masks import masks_to_boxes
from instance_segmenter.data.schema import InstanceTarget
from instance_segmenter.training.trainer import train_one_epoch


class TinyContractModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: list[torch.Tensor], targets: list[InstanceTarget]) -> dict[str, torch.Tensor]:
        del images, targets
        loss = self.weight.square()
        return {
            "loss_classifier": loss,
            "loss_box_reg": loss,
            "loss_mask": loss,
            "loss_objectness": loss,
            "loss_rpn_box_reg": loss,
        }


masks = torch.zeros((1, 8, 8), dtype=torch.bool)
masks[0, 2:6, 2:6] = True
target: InstanceTarget = {
    "boxes": masks_to_boxes(masks),
    "labels": torch.tensor([1], dtype=torch.int64),
    "masks": masks,
    "image_id": torch.tensor([1], dtype=torch.int64),
    "area": masks.flatten(1).sum(1).float(),
    "iscrowd": torch.zeros(1, dtype=torch.int64),
}
model = TinyContractModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
metrics = train_one_epoch(
    model,
    [instance_collate([(torch.zeros((3, 8, 8)), target)])],
    optimizer,
    torch.device("cpu"),
    amp=False,
    grad_clip_norm=None,
)
print(metrics)
