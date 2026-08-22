from __future__ import annotations

import torch

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.transforms import Normalize, RandomHorizontalFlip, Resize
from tests.fixtures import sample_image_and_target


def test_horizontal_flip_keeps_mask_box_geometry_aligned() -> None:
    image, target = sample_image_and_target()
    flipped_image, flipped = RandomHorizontalFlip(1.0)(image, target)
    assert torch.equal(flipped_image, image.flip(-1))
    assert flipped["boxes"].tolist() == [[3.0, 1.0, 5.0, 3.0], [0.0, 0.0, 2.0, 1.0]]
    assert torch.equal(flipped["masks"], target["masks"].flip(-1))
    assert flipped["area"].tolist() == [4.0, 2.0]


def test_resize_uses_nearest_masks_and_rebuilds_boxes() -> None:
    image, target = sample_image_and_target()
    resized_image, resized = Resize((10, 12))(image, target)
    assert resized_image.shape == (3, 10, 12)
    assert resized["masks"].dtype == torch.bool
    assert resized["boxes"].tolist() == [[2.0, 2.0, 6.0, 6.0], [8.0, 0.0, 12.0, 2.0]]
    assert resized["area"].tolist() == [16.0, 8.0]


def test_normalize_does_not_mutate_masks() -> None:
    image, target = sample_image_and_target()
    normalized, same_target = Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))(image, target)
    assert not torch.equal(normalized, image)
    assert same_target is target
    assert torch.equal(same_target["masks"], target["masks"])


def test_collate_preserves_variable_image_sizes_and_instance_counts() -> None:
    image, target = sample_image_and_target()
    empty_target = {
        **target,
        "boxes": target["boxes"][:0],
        "labels": target["labels"][:0],
        "masks": target["masks"][:0],
        "area": target["area"][:0],
        "iscrowd": target["iscrowd"][:0],
    }
    images, targets = instance_collate([(image, target), (image[:, :3, :4], empty_target)])
    assert [tuple(item.shape) for item in images] == [(3, 5, 6), (3, 3, 4)]
    assert [item["masks"].shape[0] for item in targets] == [2, 0]
