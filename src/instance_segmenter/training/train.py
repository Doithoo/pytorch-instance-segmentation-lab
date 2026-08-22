"""Orchestrate reproducible training without ever evaluating test during model selection."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from instance_segmenter.config import AppConfig, config_from_dict, config_to_dict
from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.manifest import load_dataset_metadata, verify_manifest_hashes
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
    "learning_rate",
    "epoch_seconds",
    "peak_memory_mb",
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
    verify_manifest_hashes(config.data.manifest_dir, metadata.split_hashes)
    if config.model.num_classes != metadata.label_schema.num_classes:
        raise ValueError(
            f"model.num_classes={config.model.num_classes} does not match dataset schema={metadata.label_schema.num_classes}"
        )
    checkpoint = load_checkpoint(resume, map_location="cpu") if resume is not None else None
    if checkpoint is not None:
        _validate_resume_contract(checkpoint, config, metadata.split_hashes)
    train_dataset = build_configured_dataset(
        config.data,
        "train",
        training=True,
        limit=config.data.train_limit,
    )
    train_loader = _loader(train_dataset, config, shuffle=True)
    model_config = replace(config, model=replace(config.model, weights="none")) if checkpoint is not None else config
    model = _build_model(model_config).to(device)
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

    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = 0
    if checkpoint is not None:
        restore_checkpoint(
            checkpoint,
            model=model,
            expected_model_name=config.model.name,
            expected_schema=metadata.label_schema,
            expected_manifest_hashes=metadata.split_hashes,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        if start_epoch >= config.training.epochs:
            raise ValueError("checkpoint already completed all configured epochs")
    _write_run_metadata(run_dir, config, metadata.split_hashes, device, resume=resume, start_epoch=start_epoch)

    metrics_path = run_dir / "metrics.csv"
    if start_epoch == 0 or not metrics_path.exists():
        _write_metrics_header(metrics_path)
    else:
        _validate_metrics_resume(metrics_path, start_epoch)
    for epoch in range(start_epoch + 1, config.training.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
        if epoch % config.training.evaluate_every == 0 or epoch == config.training.epochs:
            summary = evaluate_model(
                model,
                valid_loader,
                device,
                score_floor=config.training.evaluation_score_floor,
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
        runtime = {
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0,
        }
        _append_metrics(metrics_path, epoch, losses, valid_metrics, runtime)
        _append_run_event(
            run_dir,
            "epoch_completed",
            epoch=epoch,
            losses=losses,
            validation=valid_metrics,
            runtime=runtime,
        )
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


def _validate_resume_contract(
    checkpoint: Mapping[str, object], config: AppConfig, split_hashes: dict[str, str]
) -> None:
    raw_hashes = checkpoint.get("manifest_hashes")
    if not isinstance(raw_hashes, Mapping) or dict(raw_hashes) != split_hashes:
        raise ValueError("resume checkpoint manifest hashes do not match current dataset splits")
    previous = config_to_dict(config_from_dict(checkpoint["resolved_config"]))  # type: ignore[arg-type]
    current = config_to_dict(config)
    allowed = {"run.name", "run.output_dir", "training.epochs", "device", "data.num_workers"}
    differences = [
        key
        for key in sorted(set(_flatten_config(previous)) | set(_flatten_config(current)))
        if key not in allowed and _flatten_config(previous).get(key) != _flatten_config(current).get(key)
    ]
    if differences:
        raise ValueError(f"resume configuration changes immutable fields: {', '.join(differences)}")


def _flatten_config(raw: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def _validate_metrics_resume(path: Path, expected_epoch: int) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError(f"metrics schema is incompatible with this version: {path}")
    if not rows or int(rows[-1]["epoch"]) != expected_epoch:
        raise ValueError(f"metrics last epoch does not match resume checkpoint epoch {expected_epoch}")


def _write_run_metadata(
    run_dir: Path,
    config: AppConfig,
    split_hashes: dict[str, str],
    device: torch.device,
    *,
    resume: str | Path | None,
    start_epoch: int,
) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")
    (run_dir / "manifest-hashes.yaml").write_text(yaml.safe_dump(split_hashes, sort_keys=False), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[3]
    lock_path = project_root / "uv.lock"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "torchmetrics": _package_version("torchmetrics"),
        "pycocotools": _package_version("pycocotools"),
        "numpy": np.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(project_root, "status", "--porcelain")),
        "uv_lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
        "resumed_from": str(resume) if resume is not None else None,
        "resume_epoch": start_epoch,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n", encoding="utf-8"
    )
    _append_run_event(
        run_dir,
        "training_resumed" if resume is not None else "training_started",
        start_epoch=start_epoch,
        checkpoint=str(resume) if resume is not None else None,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_run_event(run_dir: Path, event: str, **details: object) -> None:
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_metrics_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n").writeheader()


def _append_metrics(
    path: Path,
    epoch: int,
    losses: dict[str, float],
    valid: dict[str, float],
    runtime: dict[str, float],
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writerow(
            {
                "epoch": epoch,
                **losses,
                **{f"valid_{name}": value for name, value in valid.items()},
                **runtime,
            }
        )
