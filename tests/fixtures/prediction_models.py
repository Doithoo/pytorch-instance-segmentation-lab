"""Evaluation-only fixture that returns the same prepared predictions each call."""

from __future__ import annotations

import torch
from torch import nn


class FixedPredictionModel(nn.Module):
    def __init__(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        if len(images) != len(self.outputs):
            raise ValueError("fixed output count does not match images")
        return [{key: value.clone() for key, value in output.items()} for output in self.outputs]
