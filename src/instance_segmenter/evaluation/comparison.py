"""Compare one compatible metric across completed run directories."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunMetric:
    run_dir: Path
    metric: str
    value: float


def compare_runs(run_dirs: list[str | Path], metric: str) -> tuple[RunMetric, ...]:
    results: list[RunMetric] = []
    for raw_path in run_dirs:
        path = Path(raw_path)
        rows = list(csv.DictReader((path / "metrics.csv").open(encoding="utf-8")))
        if not rows or metric not in rows[-1]:
            raise ValueError(f"run {path} does not have metric {metric!r}")
        results.append(RunMetric(path, metric, float(rows[-1][metric])))
    return tuple(sorted(results, key=lambda item: item.value, reverse=True))
