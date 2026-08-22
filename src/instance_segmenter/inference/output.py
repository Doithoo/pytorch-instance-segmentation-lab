"""Validation and thresholding for instance-segmentation model outputs."""

from __future__ import annotations

import torch


def normalize_prediction(
    output: object,
    *,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Move one prediction to CPU, validate its contract, and apply thresholds."""
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("score and mask thresholds must be between 0 and 1")
    if not isinstance(output, dict):
        raise RuntimeError("model prediction must be a dictionary")
    required = {"boxes", "labels", "scores", "masks"}
    missing = required - set(output)
    if missing:
        raise RuntimeError(f"model prediction misses fields: {sorted(missing)}")
    if not all(isinstance(output[name], torch.Tensor) for name in required):
        raise RuntimeError("model prediction fields must be tensors")

    boxes = output["boxes"].detach().cpu().to(torch.float32)
    labels = output["labels"].detach().cpu().to(torch.int64)
    scores = output["scores"].detach().cpu().to(torch.float32)
    masks = output["masks"].detach().cpu()
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise RuntimeError("prediction boxes must have shape [N, 4]")
    if labels.ndim != 1 or scores.ndim != 1:
        raise RuntimeError("prediction labels and scores must have shape [N]")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise RuntimeError("prediction masks must have shape [N, 1, H, W] or [N, H, W]")
    count = boxes.shape[0]
    if labels.shape[0] != count or scores.shape[0] != count or masks.shape[0] != count:
        raise RuntimeError("prediction fields must have the same instance count")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise RuntimeError("prediction boxes and scores must be finite")
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise RuntimeError("prediction scores must be probabilities between 0 and 1")
    if torch.any(labels <= 0):
        raise RuntimeError("prediction labels must contain foreground ids greater than zero")
    if count and (torch.any(boxes[:, 0] >= boxes[:, 2]) or torch.any(boxes[:, 1] >= boxes[:, 3])):
        raise RuntimeError("prediction boxes must satisfy x1 < x2 and y1 < y2")

    keep = scores >= score_threshold
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
        "masks": masks[keep] >= mask_threshold,
    }
