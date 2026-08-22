"""Versioned checkpoint creation, validation, and RNG restoration."""

from __future__ import annotations

import os
import random
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from instance_segmenter.config import AppConfig, config_to_dict
from instance_segmenter.data.schema import LabelSchema

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot safely resume this experiment."""


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    try:
        random.setstate(state["python"])  # type: ignore[arg-type]
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid RNG state: {exc}") from exc


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_name: str,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    label_schema: LabelSchema,
    config: AppConfig,
    manifest_hashes: Mapping[str, str],
) -> Path:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "label_schema": label_schema.to_dict(),
        "resolved_config": config_to_dict(config),
        "manifest_hashes": dict(manifest_hashes),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "rng_state": capture_rng_state(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a dictionary")
    required = {
        "format_version",
        "model_name",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "best_metric",
        "best_epoch",
        "label_schema",
        "resolved_config",
        "manifest_hashes",
        "python_version",
        "torch_version",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint misses fields: {sorted(missing)}")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint format {payload['format_version']!r}")
    if not isinstance(payload["model_name"], str) or not isinstance(payload["epoch"], int):
        raise CheckpointError("checkpoint model_name or epoch is invalid")
    return payload


def restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    expected_model_name: str,
    expected_schema: LabelSchema,
    expected_manifest_hashes: Mapping[str, str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    restore_rng: bool = True,
) -> None:
    if payload["model_name"] != expected_model_name:
        raise CheckpointError(f"checkpoint model is {payload['model_name']!r}, expected {expected_model_name!r}")
    if LabelSchema.from_dict(payload["label_schema"]) != expected_schema:
        raise CheckpointError("checkpoint label schema does not match current dataset")
    if expected_manifest_hashes is not None and dict(payload["manifest_hashes"]) != dict(expected_manifest_hashes):
        raise CheckpointError("checkpoint manifest hashes do not match current dataset splits")
    try:
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload["optimizer_state"] is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload["scheduler_state"] is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state"])
    except (RuntimeError, ValueError, TypeError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    if restore_rng:
        restore_rng_state(payload["rng_state"])
