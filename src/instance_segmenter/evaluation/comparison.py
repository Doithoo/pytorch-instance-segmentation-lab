"""Compare compatible metrics across completed run directories."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunMetric:
    run_dir: Path
    metric: str
    value: float
    epoch: int | None
    dataset_identity: str | None


def compare_runs(run_dirs: list[str | Path], metric: str, *, allow_incompatible: bool = False) -> tuple[RunMetric, ...]:
    if len(run_dirs) < 2:
        raise ValueError("compare-runs requires at least two run directories")
    results: list[RunMetric] = []
    signatures: dict[Path, tuple[object, ...]] = {}
    for raw_path in run_dirs:
        path = Path(raw_path)
        result, signature = _read_run_metric(path, metric)
        results.append(result)
        signatures[path] = signature
    if not allow_incompatible and len(set(signatures.values())) != 1:
        details = "; ".join(f"{path}={signature}" for path, signature in signatures.items())
        raise ValueError(f"runs use incompatible dataset or metric protocols: {details}")
    return tuple(sorted(results, key=lambda item: item.value, reverse=True))


def _read_run_metric(path: Path, metric: str) -> tuple[RunMetric, tuple[object, ...]]:
    evaluation_path = path / "evaluation" / "evaluation.json"
    if evaluation_path.is_file():
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        if metric in metrics:
            value = float(metrics[metric])
            identity = payload.get("dataset_identity")
            evaluation_signature: tuple[object, ...] = (
                identity,
                tuple(sorted(payload.get("split_hashes", {}).items())),
                payload.get("metric_backend"),
                payload.get("metric_protocol"),
                payload.get("metric_score_floor"),
                payload.get("mask_threshold"),
            )
            return RunMetric(path, metric, value, None, identity), evaluation_signature

    metrics_path = path / "metrics.csv"
    try:
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read run metrics {metrics_path}: {exc}") from exc
    candidates = [row for row in rows if metric in row and _finite(row[metric])]
    if not candidates:
        raise ValueError(f"run {path} does not have a finite metric {metric!r}")
    best = max(candidates, key=lambda row: float(row[metric]))
    config = _read_yaml(path / "config.yaml")
    hashes = _read_yaml(path / "manifest-hashes.yaml")
    training = config.get("training", {}) if isinstance(config, dict) else {}
    model = config.get("model", {}) if isinstance(config, dict) else {}
    training_signature: tuple[object, ...] = (
        tuple(sorted(hashes.items())) if isinstance(hashes, dict) else (),
        training.get("evaluation_score_floor") if isinstance(training, dict) else None,
        training.get("mask_threshold") if isinstance(training, dict) else None,
        model.get("num_classes") if isinstance(model, dict) else None,
    )
    return RunMetric(path, metric, float(best[metric]), int(best["epoch"]), None), training_signature


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read run metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"run metadata must be a mapping: {path}")
    return raw


def _finite(raw: str | None) -> bool:
    try:
        return raw is not None and math.isfinite(float(raw))
    except ValueError:
        return False
