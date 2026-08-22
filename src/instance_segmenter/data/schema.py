"""Canonical labels and validation for torchvision instance targets."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import yaml


class InstanceTarget(TypedDict):
    """One image's independent instance annotations."""

    boxes: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor
    image_id: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor


@dataclass(frozen=True)
class ClassDefinition:
    """One class shared by providers, models, metrics, and visualization."""

    id: int
    name: str
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0:
            raise ValueError("class id must be a non-negative integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("class name must be a non-empty string")
        if len(self.color) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in self.color):
            raise ValueError("class color must be three integer RGB values")
        if any(item < 0 or item > 255 for item in self.color):
            raise ValueError("class color values must be between 0 and 255")


@dataclass(frozen=True)
class LabelSchema:
    """A contiguous label space with background at index zero."""

    classes: tuple[ClassDefinition, ...]
    ignore_index: int = 255

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes must not be empty")
        expected = list(range(len(self.classes)))
        actual = [item.id for item in self.classes]
        if actual != expected:
            raise ValueError(f"class ids must be contiguous from 0; got {actual}")
        if len({item.name for item in self.classes}) != len(self.classes):
            raise ValueError("class names must be unique")
        if len({item.color for item in self.classes}) != len(self.classes):
            raise ValueError("class colors must be unique")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise ValueError("ignore_index must be an integer")
        if self.ignore_index in actual:
            raise ValueError("ignore_index must not equal a class id")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def foreground_ids(self) -> tuple[int, ...]:
        return tuple(item.id for item in self.classes[1:])

    def class_name(self, class_id: int) -> str:
        if class_id < 0 or class_id >= self.num_classes:
            raise KeyError(f"unknown class id: {class_id}")
        return self.classes[class_id].name

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": [{"id": item.id, "name": item.name, "color": list(item.color)} for item in self.classes],
            "ignore_index": self.ignore_index,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabelSchema:
        if set(raw) - {"classes", "ignore_index"}:
            raise ValueError("label schema has unknown fields")
        classes_raw = raw.get("classes")
        if not isinstance(classes_raw, list):
            raise ValueError("label schema classes must be a list")
        classes: list[ClassDefinition] = []
        for index, item in enumerate(classes_raw):
            if not isinstance(item, dict) or set(item) != {"id", "name", "color"}:
                raise ValueError(f"class {index} requires id, name, and color")
            color = item["color"]
            if not isinstance(color, list | tuple):
                raise ValueError(f"class {index} color must be a sequence")
            classes.append(ClassDefinition(id=item["id"], name=item["name"], color=tuple(color)))
        return cls(tuple(classes), ignore_index=raw.get("ignore_index", 255))

    @classmethod
    def read_yaml(cls, path: str | Path) -> LabelSchema:
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read label schema {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("label schema root must be a mapping")
        return cls.from_dict(raw)

    def write_yaml(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
            ) as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
                temporary_path = Path(handle.name)
            os.replace(temporary_path, output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


DEFAULT_LABEL_SCHEMA = LabelSchema(
    classes=(
        ClassDefinition(0, "background", (32, 32, 32)),
        ClassDefinition(1, "person", (36, 180, 99)),
    )
)


def validate_instance_target(
    target: InstanceTarget,
    *,
    height: int | None = None,
    width: int | None = None,
    schema: LabelSchema = DEFAULT_LABEL_SCHEMA,
) -> None:
    """Validate the shared target contract without changing caller data."""
    required = {"boxes", "labels", "masks", "image_id", "area", "iscrowd"}
    missing = required - set(target)
    extra = set(target) - required
    if missing or extra:
        raise ValueError(f"target fields mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    boxes = target["boxes"]
    labels = target["labels"]
    masks = target["masks"]
    area = target["area"]
    iscrowd = target["iscrowd"]
    image_id = target["image_id"]
    if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("boxes must be float32 with shape [N, 4]")
    if labels.dtype != torch.int64 or labels.ndim != 1:
        raise ValueError("labels must be int64 with shape [N]")
    if masks.dtype != torch.bool or masks.ndim != 3:
        raise ValueError("masks must be bool with shape [N, H, W]")
    if area.dtype != torch.float32 or area.ndim != 1:
        raise ValueError("area must be float32 with shape [N]")
    if iscrowd.dtype != torch.int64 or iscrowd.ndim != 1:
        raise ValueError("iscrowd must be int64 with shape [N]")
    if image_id.dtype != torch.int64 or tuple(image_id.shape) != (1,):
        raise ValueError("image_id must be int64 with shape [1]")
    count = masks.shape[0]
    if any(value.shape[0] != count for value in (boxes, labels, area, iscrowd)):
        raise ValueError("all instance fields must have the same length")
    if height is not None and masks.shape[1] != height:
        raise ValueError(f"mask height {masks.shape[1]} does not match image height {height}")
    if width is not None and masks.shape[2] != width:
        raise ValueError(f"mask width {masks.shape[2]} does not match image width {width}")
    if count:
        if not torch.isin(labels, torch.tensor(schema.foreground_ids, dtype=torch.int64, device=labels.device)).all():
            raise ValueError("labels must contain foreground schema ids")
        if not torch.isfinite(boxes).all() or not torch.isfinite(area).all():
            raise ValueError("boxes and area must be finite")
        if torch.any(boxes[:, 0] >= boxes[:, 2]) or torch.any(boxes[:, 1] >= boxes[:, 3]):
            raise ValueError("boxes must satisfy x1 < x2 and y1 < y2")
        expected_area = masks.flatten(1).sum(dim=1).to(torch.float32)
        if not torch.equal(area, expected_area):
            raise ValueError("area must equal binary mask pixel area")
