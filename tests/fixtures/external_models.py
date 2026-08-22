"""Lightweight model fixtures that follow the Mask R-CNN contract."""

from __future__ import annotations

import torch
from torch import nn

from instance_segmenter.data.schema import InstanceTarget


class ContractInstanceModel(nn.Module):
    """Fast test double; it is never registered as a teaching model."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[InstanceTarget] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if targets is not None:
            loss = self.scale.square()
            return {
                "loss_classifier": loss,
                "loss_box_reg": loss * 0.5,
                "loss_mask": loss * 0.25,
                "loss_objectness": loss * 0.125,
                "loss_rpn_box_reg": loss * 0.0625,
            }
        return [
            {
                "boxes": image.new_empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64, device=image.device),
                "scores": image.new_empty((0,)),
                "masks": image.new_empty((0, 1, image.shape[-2], image.shape[-1])),
            }
            for image in images
        ]


def build_contract_model(*, num_classes: int, weights: str, params: dict[str, object]) -> ContractInstanceModel:
    """External-factory signature used by end-to-end plumbing tests."""
    del num_classes, weights, params
    return ContractInstanceModel()
