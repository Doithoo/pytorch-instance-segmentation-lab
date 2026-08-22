from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_fake_penn_fudan(data_dir: Path, *, count: int = 170, source_groups: bool = False) -> Path:
    root = data_dir / "PennFudanPed"
    image_dir = root / "PNGImages"
    mask_dir = root / "PedMasks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(count):
        image_id = f"PennPed{index - 73:05d}" if source_groups and index >= 74 else f"FudanPed{index:05d}"
        width = 6 + index % 3
        height = 5 + index % 2
        pixels = np.full((height, width, 3), index % 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        if index % 2:
            mask[0:1, width - 2 : width] = 4
        Image.fromarray(pixels).save(image_dir / f"{image_id}.png")
        Image.fromarray(mask).save(mask_dir / f"{image_id}_mask.png")
    return root
