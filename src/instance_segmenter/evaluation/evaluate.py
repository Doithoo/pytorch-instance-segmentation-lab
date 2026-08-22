"""Checkpoint evaluation, machine-readable reports, and ranked overlays."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from instance_segmenter.config import AppConfig, config_from_dict
from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.manifest import DatasetMetadata, load_dataset_metadata, verify_manifest_hashes
from instance_segmenter.data.providers import build_configured_dataset
from instance_segmenter.data.schema import InstanceTarget
from instance_segmenter.evaluation.metrics import MetricSummary, evaluate_model
from instance_segmenter.evaluation.visualization import save_overlay
from instance_segmenter.inference.output import normalize_prediction
from instance_segmenter.models.extensions import load_external_model
from instance_segmenter.models.registry import build_model
from instance_segmenter.training.checkpoint import load_checkpoint, restore_checkpoint
from instance_segmenter.training.precision import resolve_device


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics: dict[str, float]
    per_class: tuple[dict[str, float | int], ...]
    image_count: int
    target_count: int
    prediction_count: int


@dataclass(frozen=True)
class _VisualSample:
    severity: tuple[int, int, int, int]
    image: torch.Tensor
    target: InstanceTarget
    prediction: dict[str, torch.Tensor]


class _SampleCollector:
    def __init__(self, *, score_threshold: float, mask_threshold: float, keep_visuals: int) -> None:
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.keep_visuals = keep_visuals
        self.reports: list[dict[str, int]] = []
        self.visuals: list[_VisualSample] = []

    def __call__(self, image: torch.Tensor, target: InstanceTarget, metric_prediction: dict[str, torch.Tensor]) -> None:
        prediction = normalize_prediction(
            metric_prediction,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
        )
        report = _instance_report(target, prediction)
        self.reports.append(report)
        if self.keep_visuals:
            severity = (
                report["false_negative"],
                report["false_positive"],
                report["low_iou"],
                report["image_id"],
            )
            self.visuals.append(_VisualSample(severity, image, target, prediction))
            self.visuals.sort(key=lambda item: item.severity, reverse=True)
            del self.visuals[self.keep_visuals :]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str,
    output_dir: str | Path | None = None,
    device: str = "auto",
    metric_score_floor: float | None = None,
    score_threshold: float | None = None,
    mask_threshold: float | None = None,
    plot: bool = False,
    overwrite: bool = False,
) -> EvaluationResult:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unknown split {split!r}")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = config_from_dict(checkpoint["resolved_config"])
    metric_score_floor = config.training.evaluation_score_floor if metric_score_floor is None else metric_score_floor
    score_threshold = config.training.score_threshold if score_threshold is None else score_threshold
    mask_threshold = config.training.mask_threshold if mask_threshold is None else mask_threshold
    for name, value in (
        ("metric_score_floor", metric_score_floor),
        ("score_threshold", score_threshold),
        ("mask_threshold", mask_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    resolved_device = resolve_device(device)
    metadata = load_dataset_metadata(config.data.manifest_dir)
    verify_manifest_hashes(config.data.manifest_dir, checkpoint["manifest_hashes"])
    model = _build_model(replace(config, model=replace(config.model, weights="none"))).to(resolved_device)
    restore_checkpoint(
        checkpoint,
        model=model,
        expected_model_name=config.model.name,
        expected_schema=metadata.label_schema,
        expected_manifest_hashes=metadata.split_hashes,
        restore_rng=False,
    )
    limit = {"train": config.data.train_limit, "valid": config.data.valid_limit, "test": config.data.test_limit}[split]
    dataset = build_configured_dataset(config.data, split, training=False, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=instance_collate,
    )
    destination = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent / "evaluation"
    _prepare_output_dir(destination, overwrite)
    collector = _SampleCollector(
        score_threshold=score_threshold, mask_threshold=mask_threshold, keep_visuals=4 if plot else 0
    )
    summary = evaluate_model(
        model,
        loader,
        resolved_device,
        score_floor=metric_score_floor,
        mask_threshold=mask_threshold,
        sample_callback=collector,
    )
    _write_reports(
        destination,
        summary,
        collector.reports,
        split,
        metadata,
        metric_score_floor,
        score_threshold,
        mask_threshold,
    )
    if plot:
        _write_ranked_overlays(destination / "visualizations", collector.visuals, metadata)
    return EvaluationResult(
        destination,
        summary.metrics,
        summary.per_class,
        summary.image_count,
        summary.target_count,
        summary.prediction_count,
    )


def _build_model(config: AppConfig) -> torch.nn.Module:
    if config.model.factory is not None:
        return load_external_model(
            config.model.factory, config.model.num_classes, config.model.weights, config.model.params
        )
    return build_model(config.model.name, config.model.num_classes, config.model.weights, config.model.params)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ValueError(f"evaluation output already exists: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_reports(
    output_dir: Path,
    summary: MetricSummary,
    per_image: list[dict[str, int]],
    split: str,
    metadata: DatasetMetadata,
    metric_score_floor: float,
    score_threshold: float,
    mask_threshold: float,
) -> None:
    class_names = {item.id: item.name for item in metadata.label_schema.classes}
    payload = {
        "split": split,
        "metric_score_floor": metric_score_floor,
        "analysis_score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "metrics": summary.metrics,
        "image_count": summary.image_count,
        "target_count": summary.target_count,
        "prediction_count": summary.prediction_count,
        "class_names": class_names,
        "dataset_identity": metadata.identity,
        "split_hashes": metadata.split_hashes,
        "metric_backend": "torchmetrics.MeanAveragePrecision with pycocotools",
        "metric_protocol": "COCO confidence ranking; no display-threshold filtering",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("class_id", "class_name", "mask_map", "bbox_map"), lineterminator="\n"
        )
        writer.writeheader()
        for row in summary.per_class:
            writer.writerow({**row, "class_name": class_names.get(int(row["class_id"]), str(row["class_id"]))})
    with (output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_id",
                "target_count",
                "prediction_count",
                "matches",
                "false_positive",
                "false_negative",
                "low_iou",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(per_image, key=lambda row: row["image_id"]))


def _write_ranked_overlays(output_dir: Path, samples: list[_VisualSample], metadata: DatasetMetadata) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = {item.id: item.name for item in metadata.label_schema.classes}
    for rank, sample in enumerate(samples, start=1):
        image_id = int(sample.target["image_id"].item())
        prefix = f"worst-{rank:02d}-{image_id}"
        save_overlay(output_dir / f"{prefix}-ground-truth.png", sample.image, sample.target, class_names=class_names)
        save_overlay(output_dir / f"{prefix}-prediction.png", sample.image, sample.prediction, class_names=class_names)


def _instance_report(target: InstanceTarget, prediction: dict[str, torch.Tensor]) -> dict[str, int]:
    target_masks = target["masks"].to(torch.bool)
    target_labels = target["labels"]
    prediction_masks = prediction["masks"].to(torch.bool)
    prediction_labels = prediction["labels"]
    unmatched_targets = set(range(target_masks.shape[0]))
    unmatched_predictions = set(range(prediction_masks.shape[0]))
    candidates: list[tuple[float, int, int]] = []
    for prediction_index in unmatched_predictions:
        for target_index in unmatched_targets:
            if int(prediction_labels[prediction_index]) != int(target_labels[target_index]):
                continue
            intersection = (
                torch.logical_and(prediction_masks[prediction_index], target_masks[target_index]).sum().item()
            )
            union = torch.logical_or(prediction_masks[prediction_index], target_masks[target_index]).sum().item()
            candidates.append((float(intersection / union) if union else 0.0, prediction_index, target_index))
    matches = 0
    for iou, prediction_index, target_index in sorted(candidates, reverse=True):
        if iou < 0.5 or prediction_index not in unmatched_predictions or target_index not in unmatched_targets:
            continue
        unmatched_predictions.remove(prediction_index)
        unmatched_targets.remove(target_index)
        matches += 1
    low_iou = sum(
        1
        for prediction_index in unmatched_predictions
        if any(
            int(prediction_labels[prediction_index]) == int(target_labels[target_index])
            and torch.logical_and(prediction_masks[prediction_index], target_masks[target_index]).any()
            for target_index in unmatched_targets
        )
    )
    return {
        "image_id": int(target["image_id"].item()),
        "target_count": int(target_masks.shape[0]),
        "prediction_count": int(prediction_masks.shape[0]),
        "matches": matches,
        "false_positive": len(unmatched_predictions),
        "false_negative": len(unmatched_targets),
        "low_iou": low_iou,
    }
