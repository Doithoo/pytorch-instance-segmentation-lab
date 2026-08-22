from __future__ import annotations

from instance_segmenter.config import config_from_dict
from instance_segmenter.training.train import run_training
from tests.fixtures import create_fake_penn_fudan


def test_run_training_saves_best_and_last_without_using_test_split(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    output_dir = tmp_path / "artifacts"  # type: ignore[operator]
    create_fake_penn_fudan(data_dir)
    from instance_segmenter.data.manifest import prepare_penn_fudan

    prepare_penn_fudan(data_dir, manifest_dir)
    config = config_from_dict(
        {
            "run": {"name": "fixture-run", "output_dir": str(output_dir)},
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
            "training": {"epochs": 1, "evaluate_every": 1},
            "device": "cpu",
        }
    )
    result = run_training(config)
    assert result.completed_epochs == 1
    assert (result.run_dir / "best.pt").is_file()
    assert (result.run_dir / "last.pt").is_file()
    assert (result.run_dir / "metrics.csv").read_text(encoding="utf-8").count("\n") == 2
