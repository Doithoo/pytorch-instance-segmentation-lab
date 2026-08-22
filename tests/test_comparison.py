from __future__ import annotations

import json
from pathlib import Path

import pytest

from instance_segmenter.evaluation.comparison import compare_runs


def _write_evaluation(run_dir: Path, value: float, identity: str) -> None:
    output = run_dir / "evaluation"
    output.mkdir(parents=True)
    payload = {
        "metrics": {"mask_map": value},
        "dataset_identity": identity,
        "split_hashes": {"train": "a", "valid": "b", "test": "c"},
        "metric_backend": "torchmetrics",
        "metric_protocol": "COCO confidence ranking; no display-threshold filtering",
        "metric_score_floor": 0.0,
        "mask_threshold": 0.5,
    }
    (output / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compare_runs_ranks_compatible_evaluations(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_evaluation(first, 0.4, "same")
    _write_evaluation(second, 0.7, "same")
    results = compare_runs([first, second], "mask_map")
    assert [item.run_dir.name for item in results] == ["second", "first"]


def test_compare_runs_rejects_incompatible_datasets(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_evaluation(first, 0.4, "one")
    _write_evaluation(second, 0.7, "two")
    with pytest.raises(ValueError, match="incompatible"):
        compare_runs([first, second], "mask_map")
