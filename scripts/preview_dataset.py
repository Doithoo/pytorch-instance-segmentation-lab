"""Render a compact Penn-Fudan instance-mask preview without changing source labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from instance_segmenter.data.manifest import load_dataset_metadata, read_manifest
from instance_segmenter.data.masks import decode_instance_mask, masks_to_boxes, stack_instance_masks

COLORS = ((36, 180, 99), (45, 125, 210), (230, 120, 50), (190, 70, 150))


def render_preview(data_dir: Path, manifest_dir: Path, output: Path, *, split: str = "train", limit: int = 4) -> Path:
    metadata = load_dataset_metadata(manifest_dir)
    rows = read_manifest(manifest_dir / f"{split}.csv")[:limit]
    if limit <= 0:
        raise ValueError("limit must be positive")
    panels: list[Image.Image] = []
    root = data_dir / metadata.dataset_root
    for row in rows:
        with Image.open(root / row.image_path) as source:
            image = source.convert("RGBA")
        masks = stack_instance_masks(decode_instance_mask(root / row.mask_path), row.height, row.width)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index, mask in enumerate(masks):
            color = COLORS[index % len(COLORS)]
            binary = Image.fromarray((mask.numpy() * 120).astype(np.uint8), mode="L")
            fill = Image.new("RGBA", image.size, (*color, 120))
            overlay.alpha_composite(Image.composite(fill, Image.new("RGBA", image.size), binary))
        result = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(result)
        for index, box in enumerate(masks_to_boxes(masks).tolist()):
            draw.rectangle(box, outline=COLORS[index % len(COLORS)], width=3)
        panels.append(result.convert("RGB"))
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    cursor = 0
    for panel in panels:
        canvas.paste(panel, (0, cursor))
        cursor += panel.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview prepared Penn-Fudan instance masks")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    print(render_preview(args.data_dir, args.manifest_dir, args.output, split=args.split, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
