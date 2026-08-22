from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from instance_segmenter.data.masks import (
    MaskError,
    decode_instance_mask,
    masks_from_instance_ids,
    masks_to_boxes,
    rebuild_target_geometry,
    stack_instance_masks,
    validate_geometry,
)
from instance_segmenter.data.schema import DEFAULT_LABEL_SCHEMA
from tests.fixtures import make_target


def test_decode_sparse_instance_ids_preserves_independent_instances(tmp_path: object) -> None:
    path = tmp_path / "mask.png"  # type: ignore[operator]
    raw = np.array([[0, 4, 4], [9, 0, 9]], dtype=np.uint8)
    Image.fromarray(raw).save(path)
    masks = decode_instance_mask(path)
    stacked = stack_instance_masks(masks, 2, 3)
    assert stacked.shape == (2, 2, 3)
    assert masks_to_boxes(stacked).tolist() == [[1.0, 0.0, 3.0, 1.0], [0.0, 1.0, 3.0, 2.0]]


def test_empty_instance_mask_has_no_fake_box() -> None:
    masks = stack_instance_masks([], 3, 4)
    assert masks_to_boxes(masks).shape == (0, 4)


def test_rebuild_filters_empty_instances_and_recalculates_area() -> None:
    masks = torch.zeros((2, 3, 4), dtype=torch.bool)
    masks[0, 1:, 2:] = True
    target = make_target(masks)
    rebuilt = rebuild_target_geometry(target)
    assert rebuilt["masks"].shape == (1, 3, 4)
    assert rebuilt["boxes"].tolist() == [[2.0, 1.0, 4.0, 3.0]]
    assert rebuilt["area"].tolist() == [4.0]
    validate_geometry(rebuilt, 3, 4, DEFAULT_LABEL_SCHEMA)


def test_invalid_instance_id_mask_is_rejected() -> None:
    with pytest.raises(MaskError, match="integer"):
        masks_from_instance_ids(torch.ones((2, 2), dtype=torch.float32))
