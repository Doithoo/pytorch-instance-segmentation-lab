"""Environment preflight checks used by CLI and Kaggle runner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instance_segmenter.training.precision import resolve_device


@dataclass(frozen=True)
class DeviceReport:
    device: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    capability: tuple[int, int] | None


def inspect_device(requested: str = "auto") -> DeviceReport:
    device = resolve_device(requested)
    if device.type != "cuda":
        return DeviceReport(str(device), str(torch.__version__), False, torch.version.cuda, None, None)
    return DeviceReport(
        str(device),
        str(torch.__version__),
        True,
        torch.version.cuda,
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
    )


def require_kaggle_cuda() -> DeviceReport:
    report = inspect_device("cuda")
    if report.capability is None or report.capability[0] < 7:
        capability = "unknown" if report.capability is None else f"sm_{report.capability[0]}{report.capability[1]}"
        raise RuntimeError(f"Kaggle GPU {capability} is unsupported; request a T4 or newer compatible GPU")
    return report
