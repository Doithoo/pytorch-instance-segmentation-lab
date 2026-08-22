"""Single-image checkpoint inference with machine-readable instance outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from instance_segmenter.config import AppConfig, config_from_dict
from instance_segmenter.data.schema import LabelSchema
from instance_segmenter.evaluation.visualization import save_overlay
from instance_segmenter.inference.output import normalize_prediction
from instance_segmenter.models.extensions import load_external_model
from instance_segmenter.models.registry import build_model
from instance_segmenter.training.checkpoint import load_checkpoint, restore_checkpoint
from instance_segmenter.training.precision import resolve_device


@dataclass(frozen=True)
class PredictionResult:
    output_dir: Path
    instance_count: int
    instances_path: Path
    overlay_path: Path


class Predictor:
    def __init__(self, model: torch.nn.Module, schema: LabelSchema, config: AppConfig, device: torch.device) -> None:
        self.model = model.eval()
        self.schema = schema
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "auto") -> Predictor:
        checkpoint = load_checkpoint(path, map_location="cpu")
        config = config_from_dict(checkpoint["resolved_config"])
        resolved_device = resolve_device(device)
        model = _build_model(replace(config, model=replace(config.model, weights="none"))).to(resolved_device)
        schema = LabelSchema.from_dict(checkpoint["label_schema"])
        restore_checkpoint(
            checkpoint, model=model, expected_model_name=config.model.name, expected_schema=schema, restore_rng=False
        )
        return cls(model, schema, config, resolved_device)

    def predict_single(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        score_threshold: float | None = None,
        mask_threshold: float | None = None,
        overwrite: bool = False,
    ) -> PredictionResult:
        score_threshold = self.config.training.score_threshold if score_threshold is None else score_threshold
        mask_threshold = self.config.training.mask_threshold if mask_threshold is None else mask_threshold
        if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("score and mask thresholds must be between 0 and 1")
        image = _read_image(Path(image_path))
        with torch.inference_mode():
            outputs = self.model([image.to(self.device)])
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("model prediction must return one output dictionary")
        prediction = normalize_prediction(outputs[0], score_threshold=score_threshold, mask_threshold=mask_threshold)
        destination = Path(output_dir)
        if destination.exists():
            if not overwrite:
                raise ValueError(f"prediction output already exists: {destination}; use --overwrite")
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        mask_dir = destination / "masks"
        mask_dir.mkdir()
        class_names = {item.id: item.name for item in self.schema.classes}
        records: list[dict[str, object]] = []
        for index, (box, label, score, mask) in enumerate(
            zip(
                prediction["boxes"].tolist(),
                prediction["labels"].tolist(),
                prediction["scores"].tolist(),
                prediction["masks"],
                strict=True,
            ),
            start=1,
        ):
            relative_mask = Path("masks") / f"instance-{index:03d}.png"
            Image.fromarray((mask.numpy() * 255).astype(np.uint8), mode="L").save(destination / relative_mask)
            records.append(
                {
                    "instance_id": index,
                    "label_id": int(label),
                    "label": class_names.get(int(label), str(label)),
                    "score": float(score),
                    "box_xyxy": [float(value) for value in box],
                    "mask_path": relative_mask.as_posix(),
                    "mask_encoding": "binary-png-thresholded",
                }
            )
        instances_path = destination / "instances.json"
        instances_path.write_text(
            json.dumps(
                {
                    "image": str(Path(image_path)),
                    "score_threshold": score_threshold,
                    "mask_threshold": mask_threshold,
                    "instances": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        overlay_path = save_overlay(
            destination / "overlay.png",
            image,
            prediction,
            class_names=class_names,
            score_threshold=0.0,
        )
        return PredictionResult(destination, len(records), instances_path, overlay_path)


def _build_model(config: AppConfig) -> torch.nn.Module:
    if config.model.factory is not None:
        return load_external_model(
            config.model.factory, config.model.num_classes, config.model.weights, config.model.params
        )
    return build_model(config.model.name, config.model.num_classes, config.model.weights, config.model.params)


def _read_image(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as source:
            return pil_to_tensor(source.convert("RGB")).to(torch.float32).div(255.0)
    except OSError as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
