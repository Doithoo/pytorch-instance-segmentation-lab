"""Stable model metadata and construction protocols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

ModelFactory = Callable[[int, str, Mapping[str, object]], "nn.Module"]


@dataclass(frozen=True)
class ModelSpec:
    """Construction and learning metadata for a registered instance segmenter."""

    name: str
    factory: ModelFactory
    description: str
    supports_pretrained: bool = True
    supported_weights: tuple[str, ...] = ("none", "coco_v1")
    input_notes: tuple[str, ...] = ()
    parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not callable(self.factory):
            raise ValueError("model specs require a non-empty name and callable factory")
