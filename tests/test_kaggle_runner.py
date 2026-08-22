from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_kaggle_runner import RUNNER_PATH, archive_members, archive_project, check_runner, write_runner  # noqa: E402


def test_archive_is_deterministic_and_contains_only_runtime_inputs() -> None:
    first = archive_project()
    second = archive_project()
    assert first == second
    members = archive_members(first)
    assert "configs/reference_maskrcnn.yaml" in members
    assert "data/manifests/train.csv" in members
    assert "src/instance_segmenter/training/train.py" in members
    assert "scripts/kaggle_runner.py" in members
    assert "data/manifests/source.yaml" not in members
    assert not any("__pycache__" in member or member.endswith("kaggle.json") for member in members)


def test_committed_runner_is_fresh_and_embeds_a_valid_archive() -> None:
    assert check_runner()
    namespace = runpy.run_path(str(RUNNER_PATH))
    assert namespace["PROJECT_ARCHIVE_BYTES"] > 0
    assert len(namespace["PROJECT_ARCHIVE_SHA256"]) == 64
    assert "PROJECT_ARCHIVE_B64" in RUNNER_PATH.read_text(encoding="utf-8")


def test_check_detects_stale_runner(tmp_path: Path) -> None:
    output = tmp_path / "run_kaggle.py"
    write_runner(output=output)
    assert check_runner(output=output)
    output.write_text("stale\n", encoding="utf-8")
    assert not check_runner(output=output)


def test_kernel_metadata_points_to_generated_script() -> None:
    metadata = json.loads((RUNNER_PATH.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["code_file"] == "run_kaggle.py"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"
    assert metadata["dataset_sources"] == []


def test_snapshot_rejects_secret_like_allowlist_path() -> None:
    from build_kaggle_runner import RunnerBuildError, _validate_runtime_path

    with pytest.raises(RunnerBuildError, match="secret"):
        _validate_runtime_path(Path("src/instance_segmenter/token.txt"))
