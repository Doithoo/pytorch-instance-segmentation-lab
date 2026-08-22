"""COCO-style bbox and mask metrics backed by torchmetrics and pycocotools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from instance_segmenter.data.schema import InstanceTarget


@dataclass(frozen=True)
class MetricSummary:
    metrics: dict[str, float]
    per_class: tuple[dict[str, float | int], ...]
    image_count: int
    target_count: int
    prediction_count: int


def evaluate_model(
    model: nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[InstanceTarget]]],
    device: torch.device,
    *,
    score_threshold: float,
    mask_threshold: float,
) -> MetricSummary:
    """Evaluate an instance model without exposing targets during model inference."""
    bbox_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    mask_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    was_training = model.training
    model.eval()
    image_count = 0
    target_count = 0
    prediction_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            device_images = [image.to(device) for image in images]
            outputs = model(device_images)
            if not isinstance(outputs, list) or len(outputs) != len(targets):
                raise RuntimeError("model evaluation output must be a list aligned with images")
            predictions = [_prepare_prediction(output, score_threshold, mask_threshold) for output in outputs]
            references = [_prepare_target(target) for target in targets]
            bbox_metric.update(
                [{key: value for key, value in item.items() if key != "masks"} for item in predictions],
                [{key: value for key, value in item.items() if key != "masks"} for item in references],
            )
            mask_metric.update(predictions, references)
            image_count += len(images)
            target_count += sum(int(target["labels"].shape[0]) for target in targets)
            prediction_count += sum(int(prediction["labels"].shape[0]) for prediction in predictions)
    if was_training:
        model.train()
    bbox = bbox_metric.compute()
    masks = mask_metric.compute()
    metrics = {
        "bbox_map": _scalar(bbox["map"]),
        "bbox_map_50": _scalar(bbox["map_50"]),
        "bbox_map_75": _scalar(bbox["map_75"]),
        "bbox_mar_100": _scalar(bbox["mar_100"]),
        "mask_map": _scalar(masks["map"]),
        "mask_map_50": _scalar(masks["map_50"]),
        "mask_map_75": _scalar(masks["map_75"]),
        "mask_mar_100": _scalar(masks["mar_100"]),
    }
    return MetricSummary(metrics, _per_class(bbox, masks), image_count, target_count, prediction_count)


def _prepare_prediction(output: object, score_threshold: float, mask_threshold: float) -> dict[str, torch.Tensor]:
    if not isinstance(output, dict):
        raise RuntimeError("model prediction must be a dictionary")
    required = {"boxes", "labels", "scores", "masks"}
    if required - set(output):
        raise RuntimeError(f"model prediction misses fields: {sorted(required - set(output))}")
    boxes = output["boxes"].detach().cpu().to(torch.float32)
    labels = output["labels"].detach().cpu().to(torch.int64)
    scores = output["scores"].detach().cpu().to(torch.float32)
    masks = output["masks"].detach().cpu()
    if not all(isinstance(value, torch.Tensor) for value in (boxes, labels, scores, masks)):
        raise RuntimeError("model prediction fields must be tensors")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise RuntimeError("prediction masks must have shape [N, 1, H, W] or [N, H, W]")
    keep = scores >= score_threshold
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
        "masks": (masks[keep] >= mask_threshold),
    }


def _prepare_target(target: InstanceTarget) -> dict[str, torch.Tensor]:
    return {
        "boxes": target["boxes"].detach().cpu().to(torch.float32),
        "labels": target["labels"].detach().cpu().to(torch.int64),
        "masks": target["masks"].detach().cpu().to(torch.bool),
        "iscrowd": target["iscrowd"].detach().cpu().to(torch.int64),
    }


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _per_class(bbox: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> tuple[dict[str, float | int], ...]:
    classes = masks.get("classes")
    mask_values = masks.get("map_per_class")
    bbox_values = bbox.get("map_per_class")
    if classes is None or mask_values is None or bbox_values is None or classes.numel() == 0:
        return ()
    return tuple(
        {
            "class_id": int(class_id),
            "mask_map": float(mask_value),
            "bbox_map": float(bbox_value),
        }
        for class_id, mask_value, bbox_value in zip(
            classes.reshape(-1).tolist(),
            mask_values.reshape(-1).tolist(),
            bbox_values.reshape(-1).tolist(),
            strict=True,
        )
    )
