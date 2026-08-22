from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    (
        "01_instance_target.py",
        "02_mask_to_instances.py",
        "03_detection_collate.py",
        "04_minimal_training_loop.py",
    ),
)
def test_learning_examples_execute(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "examples" / name))
    assert capsys.readouterr().out.strip()
