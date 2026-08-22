"""Discover and copy configuration templates from source or installed wheels."""

from __future__ import annotations

import shutil
import sysconfig
from pathlib import Path

CONFIG_TEMPLATES = (
    "custom_dataset_example",
    "custom_model_example",
    "learning_minimal",
    "maskrcnn_mobilenet_v3_large",
    "maskrcnn_resnet50_fpn",
    "reference_maskrcnn",
)


def config_template_path(name: str) -> Path:
    if name not in CONFIG_TEMPLATES:
        raise ValueError(f"unknown config template {name!r}; available: {', '.join(CONFIG_TEMPLATES)}")
    relative = Path(f"{name}.yaml")
    candidates = (
        Path(__file__).resolve().parents[2] / "configs" / relative,
        Path(sysconfig.get_path("data")) / "share" / "pytorch-instance-segmentation-lab" / "configs" / relative,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(f"installed config template is missing: {relative}")


def copy_config_template(name: str, output: str | Path, *, overwrite: bool = False) -> Path:
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise ValueError(f"config output already exists: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_template_path(name), destination)
    return destination
