"""Device resolution and conservative automatic mixed precision."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def amp_enabled(policy: bool | str, device: torch.device) -> bool:
    if policy not in {True, False, "auto"}:
        raise ValueError("AMP policy must be true, false, or 'auto'")
    if device.type != "cuda":
        return False
    return policy is True or policy == "auto"
