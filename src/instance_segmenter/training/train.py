"""Orchestrate reproducible training without ever evaluating test during model selection."""

from __future__ import annotations

import csv
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from instance_segmenter.config import AppConfig, config_to_dict
from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.manifest import load_dataset_metadata
from instance_segmenter.data.providers import build_configured_dataset
from instance_segmenter.evaluation.metrics import evaluate_model
from instance_segmenter.models.extensions import load_external_model
from instance_segmenter.models.registry import build_model
from instance_segmenter.training.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from instance_segmenter.training.precision import amp_enabled, resolve_device
from instance_segmenter.training.trainer import DryRunResult, dry_run, train_one_epoch

METRIC_FIELDS = (
    "epoch",
    "loss_total",
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
    "loss_objectness",
    "loss_rpn_box_reg",
    "valid_mask_map",
    "valid_mask_map_50",
    "valid_mask_map_75",
    "valid_bbox_map",
    "valid_bbox_map_50",
    "valid_bbox_map_75",
    "valid_bbox_mar_100",
    "valid_mask_mar_100",
)


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    completed_epochs: int
    best_epoch: int
    best_metric: float
    dry_run_result: DryRunResult | None = None


def run_training(config: AppConfig, *, resume: str | Path | None = None, dry_run_mode: bool = False) -> TrainingResult:
    """Run a diagnostic update or full train/validation loop with checkpointing."""
    _seed_everything(config.run.seed)
    device = resolve_device(config.device)
    metadata = load_dataset_metadata(config.data.manifest_dir)
    if config.model.num_classes != metadata.label_schema.num_classes:
        raise ValueError(
            f"model.num_classes={config.model.num_classes} does not match dataset schema={metadata.label_schema.num_classes}"
        )
    train_dataset = build_configured_dataset(
        config.data,
        "train",
        training=True,
        limit=config.data.train_limit,
    )
    train_loader = _loader(train_dataset, config, shuffle=True)
    model = _build_model(config).to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    use_amp = amp_enabled(config.training.amp, device)
    if dry_run_mode:
        diagnostics = dry_run(
            model,
            train_loader,
            optimizer,
            device,
            amp=use_amp,
            grad_clip_norm=config.training.grad_clip_norm,
        )
        return TrainingResult(Path(), 0, 0, float("nan"), diagnostics)

    valid_dataset = build_configured_dataset(
        config.data,
        "valid",
        training=False,
        limit=config.data.valid_limit,
    )
    valid_loader = _loader(valid_dataset, config, shuffle=False)
    run_dir = config.run.output_dir / config.run.name
    if run_dir.exists() and resume is None:
        raise ValueError(f"run directory already exists: {run_dir}; choose run.name or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(run_dir, config, metadata.split_hashes, device)

    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = 0
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        restore_checkpoint(
            checkpoint,
            model=model,
            expected_model_name=config.model.name,
            expected_schema=metadata.label_schema,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        if start_epoch >= config.training.epochs:
            raise ValueError("checkpoint already completed all configured epochs")

    metrics_path = run_dir / "metrics.csv"
    if start_epoch == 0:
        _write_metrics_header(metrics_path)
    for epoch in range(start_epoch + 1, config.training.epochs + 1):
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=use_amp,
            grad_clip_norm=config.training.grad_clip_norm,
        )
        if scheduler is not None:
            scheduler.step()
        if epoch % config.training.evaluate_every == 0:
            summary = evaluate_model(
                model,
                valid_loader,
                device,
                score_threshold=config.training.score_threshold,
                mask_threshold=config.training.mask_threshold,
            )
            valid_metrics = summary.metrics
        else:
            valid_metrics = {"mask_map": float("nan")}
        current = valid_metrics[config.training.best_metric]
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                model_name=config.model.name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                label_schema=metadata.label_schema,
                config=config,
                manifest_hashes=metadata.split_hashes,
            )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            model_name=config.model.name,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            label_schema=metadata.label_schema,
            config=config,
            manifest_hashes=metadata.split_hashes,
        )
        _append_metrics(metrics_path, epoch, losses, valid_metrics)
    return TrainingResult(run_dir, config.training.epochs, best_epoch, best_metric)


def _build_model(config: AppConfig) -> torch.nn.Module:
    if config.model.factory is not None:
        return load_external_model(
            config.model.factory, config.model.num_classes, config.model.weights, config.model.params
        )
    return build_model(config.model.name, config.model.num_classes, config.model.weights, config.model.params)


def _loader(dataset: object, config: AppConfig, *, shuffle: bool) -> DataLoader[object]:
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=instance_collate,
    )


def _optimizer(model: torch.nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    if config.training.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )


def _scheduler(optimizer: torch.optim.Optimizer, config: AppConfig) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.training.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.step_size, gamma=config.training.gamma)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_run_metadata(run_dir: Path, config: AppConfig, split_hashes: dict[str, str], device: torch.device) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")
    (run_dir / "manifest-hashes.yaml").write_text(yaml.safe_dump(split_hashes, sort_keys=False), encoding="utf-8")
    environment = [
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"device: {device}",
        f"cuda_device_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
    ]
    (run_dir / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")


def _write_metrics_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()


def _append_metrics(path: Path, epoch: int, losses: dict[str, float], valid: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writerow(
            {
                "epoch": epoch,
                **losses,
                **{f"valid_{name}": value for name, value in valid.items()},
            }
        )
