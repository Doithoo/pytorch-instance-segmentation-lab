"""Instance overlays that preserve source images and binary masks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PALETTE = ((36, 180, 99), (45, 125, 210), (230, 120, 50), (190, 70, 150), (225, 80, 80), (120, 150, 50))


def render_instances(
    image: torch.Tensor,
    instances: Mapping[str, torch.Tensor],
    *,
    class_names: Mapping[int, str],
    score_threshold: float = 0.0,
) -> Image.Image:
    """Render one stable color per instance without modifying model tensors."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    base = _tensor_image(image).convert("RGBA")
    masks = instances["masks"].detach().cpu()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    boxes = instances["boxes"].detach().cpu()
    labels = instances["labels"].detach().cpu()
    scores = instances.get("scores")
    score_values = scores.detach().cpu() if scores is not None else torch.ones((labels.shape[0],), dtype=torch.float32)
    if masks.ndim != 3 or boxes.shape != (masks.shape[0], 4) or labels.shape != (masks.shape[0],):
        raise ValueError("instances have inconsistent boxes, labels, and masks")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for index, (mask, score) in enumerate(zip(masks, score_values, strict=True)):
        if float(score) < score_threshold:
            continue
        color = PALETTE[index % len(PALETTE)]
        alpha = Image.fromarray((mask.to(torch.bool).numpy() * 120).astype(np.uint8), mode="L")
        fill = Image.new("RGBA", base.size, (*color, 120))
        overlay.alpha_composite(Image.composite(fill, Image.new("RGBA", base.size), alpha))
    rendered = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(rendered)
    for index, (box, label, score) in enumerate(
        zip(boxes.tolist(), labels.tolist(), score_values.tolist(), strict=True)
    ):
        if score < score_threshold:
            continue
        color = PALETTE[index % len(PALETTE)]
        draw.rectangle(box, outline=color, width=3)
        name = class_names.get(int(label), str(label))
        suffix = "" if scores is None else f" {score:.2f}"
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"{name}{suffix}", fill=color)
    return rendered.convert("RGB")


def save_overlay(
    output: str | Path,
    image: torch.Tensor,
    instances: Mapping[str, torch.Tensor],
    *,
    class_names: Mapping[int, str],
    score_threshold: float = 0.0,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_instances(image, instances, class_names=class_names, score_threshold=score_threshold).save(path)
    return path


def _tensor_image(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().to(torch.float32).clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="RGB")
