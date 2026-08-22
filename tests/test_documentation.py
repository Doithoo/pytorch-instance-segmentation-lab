from __future__ import annotations

from pathlib import Path

REQUIRED_PATHS = (
    "README.md",
    "README.zh-CN.md",
    "docs/guides/kaggle.md",
    "docs/guides/kaggle.zh-CN.md",
    "docs/reference/dataset-format.md",
    "docs/reference/checkpoint-schema.md",
    "docs/tutorial/05-evaluation-and-inference.md",
    "docs/recorded-run/kaggle/kernel-metadata.json",
    "docs/recorded-run/kaggle/run_kaggle.py",
    "configs/reference_maskrcnn.yaml",
    "examples/05_checkpoint_prediction.py",
)


def test_documented_project_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert all((root / path).is_file() for path in REQUIRED_PATHS)


def test_readme_describes_kaggle_complete_training() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "20 epochs" in readme
    assert "136/17/17" in readme
    assert "Kaggle" in readme
