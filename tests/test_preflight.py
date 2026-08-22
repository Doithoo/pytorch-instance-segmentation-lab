from __future__ import annotations

import torch

from instance_segmenter.preflight import inspect_device
from instance_segmenter.training.precision import amp_enabled, resolve_device


def test_cpu_preflight_and_amp_policy() -> None:
    report = inspect_device("cpu")
    assert report.device == "cpu"
    assert report.cuda_available is False
    assert resolve_device("cpu") == torch.device("cpu")
    assert amp_enabled("auto", torch.device("cpu")) is False
    assert amp_enabled(True, torch.device("cpu")) is False
