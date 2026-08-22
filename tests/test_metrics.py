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
    callback_ids: list[int] = []
    summary = evaluate_model(
        FixedPredictionModel([prediction]),
        [instance_collate([(image, target)])],
        torch.device("cpu"),
        score_floor=0.0,
        mask_threshold=0.5,
        sample_callback=lambda _image, sample_target, _prediction: callback_ids.append(
            int(sample_target["image_id"].item())
        ),
    )
    assert summary.metrics["bbox_map"] == pytest.approx(1.0)
    assert summary.metrics["mask_map"] == pytest.approx(1.0)
    assert summary.image_count == 1
    assert summary.target_count == summary.prediction_count == 2
    assert callback_ids == [int(target["image_id"].item())]


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
        score_floor=0.0,
        mask_threshold=0.5,
    )
    assert summary.prediction_count == 0
    assert summary.metrics["bbox_map"] == pytest.approx(0.0)


def test_multiclass_metrics_report_each_foreground_class() -> None:
    image, target = sample_image_and_target()
    target["labels"] = torch.tensor([1, 2], dtype=torch.int64)
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
        score_floor=0.0,
        mask_threshold=0.5,
    )
    assert {row["class_id"] for row in summary.per_class} == {1, 2}
    assert all(row["mask_map"] == pytest.approx(1.0) for row in summary.per_class)


def test_metric_floor_does_not_change_analysis_callback_predictions() -> None:
    image, target = sample_image_and_target()
    prediction = {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.full_like(target["area"], 0.1),
        "masks": target["masks"].unsqueeze(1).float(),
    }
    callback_counts: list[int] = []
    summary = evaluate_model(
        FixedPredictionModel([prediction]),
        [instance_collate([(image, target)])],
        torch.device("cpu"),
        score_floor=0.5,
        mask_threshold=0.5,
        sample_callback=lambda _image, _target, output: callback_counts.append(output["scores"].shape[0]),
    )
    assert summary.prediction_count == 0
    assert callback_counts == [2]


def test_coco_metric_keeps_low_confidence_true_positives_by_default() -> None:
    image, target = sample_image_and_target()
    prediction = {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.full_like(target["area"], 0.1),
        "masks": target["masks"].unsqueeze(1).float(),
    }
    summary = evaluate_model(
        FixedPredictionModel([prediction]),
        [instance_collate([(image, target)])],
        torch.device("cpu"),
        score_floor=0.0,
        mask_threshold=0.5,
    )
    assert summary.metrics["mask_map"] == pytest.approx(1.0)
    assert summary.prediction_count == 2
