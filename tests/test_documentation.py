from __future__ import annotations

import csv
import hashlib
import json
import re
import runpy
from pathlib import Path

import pytest
import yaml

REQUIRED_PATHS = (
    "README.md",
    "README.zh-CN.md",
    "docs/guides/kaggle.md",
    "docs/guides/kaggle.zh-CN.md",
    "docs/reference/dataset-format.md",
    "docs/architecture/0002-evaluation-and-splits.md",
    "docs/reference/checkpoint-schema.md",
    "docs/tutorial/05-evaluation-and-inference.md",
    "docs/recorded-run/kaggle/kernel-metadata.json",
    "docs/recorded-run/kaggle/run_kaggle.py",
    "docs/recorded-run/kaggle/run_kaggle-v2.py",
    "docs/recorded-run/MODEL_CARD.md",
    "docs/recorded-run/legacy-v1/README.md",
    "configs/reference_maskrcnn.yaml",
    "configs/maskrcnn_mobilenet_v3_large.yaml",
    "examples/05_checkpoint_prediction.py",
)


def test_documented_project_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert all((root / path).is_file() for path in REQUIRED_PATHS)


def test_relative_markdown_links_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    markdown_files = [root / "README.md", root / "README.zh-CN.md", *sorted((root / "docs").rglob("*.md"))]
    broken: list[str] = []
    for source in markdown_files:
        text = source.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (source.parent / target).resolve().exists():
                broken.append(f"{source.relative_to(root)} -> {target}")
    assert not broken, "broken Markdown links:\n" + "\n".join(broken)


def test_protocol_v2_recorded_run_is_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    recorded = root / "docs" / "recorded-run"
    summary = json.loads((recorded / "kaggle-run-summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((recorded / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    run = yaml.safe_load((recorded / "run.yaml").read_text(encoding="utf-8"))
    with (recorded / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "complete"
    assert summary["completed_epochs"] == len(rows) == 20
    assert summary["dataset_identity"] == run["dataset_identity"] == evaluation["dataset_identity"]
    assert summary["metric_score_floor"] == evaluation["metric_score_floor"] == 0.0
    assert summary["metric_protocol"] == evaluation["metric_protocol"]
    assert summary["metrics"] == evaluation["metrics"]
    assert max(float(row["valid_mask_map"]) for row in rows) == pytest.approx(summary["best_valid_mask_map"])
    assert run["best_checkpoint_sha256"] == summary["checkpoint_sha256"]

    submitted_path = recorded / "kaggle" / "run_kaggle-v2.py"
    submitted = runpy.run_path(str(submitted_path))
    assert hashlib.sha256(submitted_path.read_bytes()).hexdigest() == run["submitted_runner_sha256"]
    assert submitted["PROJECT_ARCHIVE_SHA256"] == summary["source_archive_sha256"]
    assert submitted["PROJECT_ARCHIVE_BYTES"] == summary["source_archive_bytes"]


def test_readme_describes_protocol_v2_and_legacy_baseline() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "20-epoch" in readme
    assert "136/17/17" in readme
    assert "protocol v2" in readme
    assert "legacy-v1" in readme
    assert "0.756093" in readme
