from __future__ import annotations

import pytest
import torch

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.evaluation.metrics import evaluate_model
from tests.fixtures.prediction_models import FixedPredictionModel
from tests.fixtures.synthetic_instances import sample_image_and_target


def test_exact_prediction_has_perfect_bbox_and_mask_ap() -> None:
    image, target = sample_image_and_target()
    prediction = {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.ones_like(target["area"]),
        "masks": target["masks"].unsqueeze(1).float(),
    }
    summary = evaluate_model(
        FixedPredictionModel([prediction]),
        [instance_collate([(image, target)])],
        torch.device("cpu"),
        score_threshold=0.5,
        mask_threshold=0.5,
    )
    assert summary.metrics["bbox_map"] == pytest.approx(1.0)
    assert summary.metrics["mask_map"] == pytest.approx(1.0)
    assert summary.image_count == 1
    assert summary.target_count == summary.prediction_count == 2


def test_empty_prediction_is_accounted_for() -> None:
    image, target = sample_image_and_target()
    prediction = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
        "masks": torch.empty((0, 1, 5, 6)),
    }
    summary = evaluate_model(
        FixedPredictionModel([prediction]),
        [instance_collate([(image, target)])],
        torch.device("cpu"),
        score_threshold=0.5,
        mask_threshold=0.5,
    )
    assert summary.prediction_count == 0
    assert summary.metrics["bbox_map"] == pytest.approx(0.0)
