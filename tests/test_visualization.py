from __future__ import annotations

import torch

from instance_segmenter.evaluation.visualization import save_overlay
from tests.fixtures.synthetic_instances import sample_image_and_target


def test_overlay_writes_a_rgb_image_without_changing_masks(tmp_path: object) -> None:
    image, target = sample_image_and_target()
    original = target["masks"].clone()
    output = tmp_path / "overlay.png"  # type: ignore[operator]
    path = save_overlay(output, image, target, class_names={0: "background", 1: "person"})
    assert path.is_file()
    assert torch.equal(target["masks"], original)
