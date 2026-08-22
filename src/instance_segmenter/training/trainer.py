"""One-epoch training and single-batch diagnostics for instance models."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from instance_segmenter.data.schema import InstanceTarget


class NonFiniteLossError(RuntimeError):
    """Raised when a model returns NaN or infinite loss."""


@dataclass(frozen=True)
class DryRunResult:
    batch_size: int
    image_shapes: tuple[tuple[int, ...], ...]
    target_counts: tuple[int, ...]
    losses: dict[str, float]


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[InstanceTarget]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    totals: dict[str, float] = {}
    sample_count = 0
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            losses = model(images, targets)
            total = sum_losses(losses)
        _validate_losses(losses, targets)
        _optimizer_step(total, model, optimizer, scaler, grad_clip_norm)
        sample_count += len(images)
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(total.detach()) * len(images)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(images)
    if sample_count == 0:
        raise ValueError("training loader yielded no batches")
    return {name: value / sample_count for name, value in totals.items()}


def dry_run(
    model: nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[InstanceTarget]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
    grad_clip_norm: float | None,
) -> DryRunResult:
    model.train()
    try:
        images, targets = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("dry run loader yielded no batches") from exc
    image_shapes = tuple(tuple(image.shape) for image in images)
    target_counts = tuple(int(target["masks"].shape[0]) for target in targets)
    images, targets = move_batch(images, targets, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, amp):
        losses = model(images, targets)
        total = sum_losses(losses)
    _validate_losses(losses, targets)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    _optimizer_step(total, model, optimizer, scaler, grad_clip_norm)
    return DryRunResult(
        len(images),
        image_shapes,
        target_counts,
        {"loss_total": float(total.detach()), **{name: float(value.detach()) for name, value in losses.items()}},
    )


def move_batch(
    images: list[torch.Tensor], targets: list[InstanceTarget], device: torch.device
) -> tuple[list[torch.Tensor], list[InstanceTarget]]:
    return [image.to(device) for image in images], [
        {
            "boxes": target["boxes"].to(device),
            "labels": target["labels"].to(device),
            "masks": target["masks"].to(device),
            "image_id": target["image_id"].to(device),
            "area": target["area"].to(device),
            "iscrowd": target["iscrowd"].to(device),
        }
        for target in targets
    ]


def sum_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ValueError("model returned no training losses")
    return torch.stack(tuple(losses.values())).sum()


def _autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager[object]:
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def _validate_losses(losses: Mapping[str, torch.Tensor], targets: list[InstanceTarget]) -> None:
    for name, value in losses.items():
        if value.numel() != 1 or not torch.isfinite(value).item():
            identifiers = [int(target["image_id"].item()) for target in targets]
            raise NonFiniteLossError(f"non-finite {name} for image IDs {identifiers}")


def _optimizer_step(
    total: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float | None,
) -> None:
    if scaler.is_enabled():
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
    else:
        total.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
