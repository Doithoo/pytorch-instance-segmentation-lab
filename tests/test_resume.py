from __future__ import annotations

from dataclasses import replace

import pytest

from instance_segmenter.config import config_from_dict
from instance_segmenter.data.manifest import prepare_penn_fudan
from instance_segmenter.training.train import run_training
from tests.fixtures import create_fake_penn_fudan


def _config(tmp_path: object, *, epochs: int, evaluate_every: int = 1):
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    output_dir = tmp_path / "artifacts"  # type: ignore[operator]
    if not manifest_dir.exists():
        create_fake_penn_fudan(data_dir)
        prepare_penn_fudan(data_dir, manifest_dir)
    return config_from_dict(
        {
            "run": {"name": "resume-run", "output_dir": str(output_dir)},
            "data": {
                "root": str(data_dir),
                "manifest_dir": str(manifest_dir),
                "image_size": [32, 32],
                "batch_size": 1,
                "train_limit": 1,
                "valid_limit": 1,
                "test_limit": 1,
            },
            "model": {"factory": "tests.fixtures.external_models:build_contract_model"},
            "training": {"epochs": epochs, "evaluate_every": evaluate_every},
            "device": "cpu",
        }
    )


def test_resume_appends_metrics_and_records_lineage(tmp_path: object) -> None:
    first_config = _config(tmp_path, epochs=1)
    first = run_training(first_config)
    second_config = replace(first_config, training=replace(first_config.training, epochs=2))
    second = run_training(second_config, resume=first.run_dir / "last.pt")
    lines = (second.run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[-1].startswith("2,")
    events = (second.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "training_resumed" in events
    assert (second.run_dir / "environment.json").is_file()


def test_resume_rejects_immutable_config_changes(tmp_path: object) -> None:
    first_config = _config(tmp_path, epochs=1)
    first = run_training(first_config)
    changed = replace(
        first_config,
        training=replace(first_config.training, epochs=2),
        data=replace(first_config.data, horizontal_flip=0.5),
    )
    with pytest.raises(ValueError, match="immutable fields"):
        run_training(changed, resume=first.run_dir / "last.pt")


def test_final_epoch_is_evaluated_even_when_not_on_interval(tmp_path: object) -> None:
    config = _config(tmp_path, epochs=1, evaluate_every=5)
    result = run_training(config)
    assert result.best_epoch == 1
    assert (result.run_dir / "best.pt").is_file()
