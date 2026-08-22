"""Synchronized image and instance-mask transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as functional

from instance_segmenter.data.masks import rebuild_target_geometry
from instance_segmenter.data.schema import InstanceTarget


class InstanceTransform(Protocol):
    def __call__(self, image: torch.Tensor, target: InstanceTarget) -> tuple[torch.Tensor, InstanceTarget]: ...


class Compose:
    def __init__(self, transforms: Sequence[InstanceTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: torch.Tensor, target: InstanceTarget) -> tuple[torch.Tensor, InstanceTarget]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Resize:
    """Resize image bilinearly and masks with nearest-neighbor interpolation."""

    def __init__(self, size: tuple[int, int]) -> None:
        if len(size) != 2 or any(not isinstance(item, int) or item <= 0 for item in size):
            raise ValueError("size must be a positive (height, width) tuple")
        self.size = size

    def __call__(self, image: torch.Tensor, target: InstanceTarget) -> tuple[torch.Tensor, InstanceTarget]:
        _validate_image(image)
        image_out = functional.interpolate(
            image.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)
        masks = target["masks"]
        mask_out = functional.interpolate(masks.unsqueeze(1).float(), size=self.size, mode="nearest").squeeze(1).bool()
        transformed: InstanceTarget = {**target, "masks": mask_out}
        return image_out, rebuild_target_geometry(transformed)


class RandomHorizontalFlip:
    def __init__(self, probability: float, *, generator: torch.Generator | None = None) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("flip probability must be between 0 and 1")
        self.probability = probability
        self.generator = generator

    def __call__(self, image: torch.Tensor, target: InstanceTarget) -> tuple[torch.Tensor, InstanceTarget]:
        _validate_image(image)
        if torch.rand((), generator=self.generator).item() >= self.probability:
            return image, target
        flipped: InstanceTarget = {**target, "masks": target["masks"].flip(-1)}
        return image.flip(-1), rebuild_target_geometry(flipped)


class Normalize:
    """Normalize only image channels; masks and geometry remain unchanged."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != len(std) or not mean or any(value <= 0 for value in std):
            raise ValueError("mean and std must be non-empty, same-length sequences with positive std")
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def __call__(self, image: torch.Tensor, target: InstanceTarget) -> tuple[torch.Tensor, InstanceTarget]:
        _validate_image(image)
        if image.shape[0] != len(self.mean):
            raise ValueError("image channels do not match normalization values")
        mean = image.new_tensor(self.mean).reshape(-1, 1, 1)
        std = image.new_tensor(self.std).reshape(-1, 1, 1)
        return (image - mean) / std, target


def _validate_image(image: torch.Tensor) -> None:
    if image.dtype != torch.float32 or image.ndim != 3:
        raise ValueError("image must be float32 with shape [C, H, W]")
