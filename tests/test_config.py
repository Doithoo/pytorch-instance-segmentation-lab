from __future__ import annotations

import pytest

from instance_segmenter.config import ConfigError, config_to_dict, load_config


def test_yaml_and_set_override_have_expected_precedence(tmp_path: object) -> None:
    path = tmp_path / "config.yaml"  # type: ignore[operator]
    path.write_text("run:\n  name: from-yaml\ntraining:\n  epochs: 3\n", encoding="utf-8")
    config = load_config(path, [("run.name", "from-cli"), ("model.params.min_size", "128")])
    assert config.run.name == "from-cli"
    assert config.training.epochs == 3
    assert config.model.params == {"min_size": 128}
    assert config_to_dict(config)["data"]["root"] == "data/raw"


def test_unknown_config_field_is_rejected(tmp_path: object) -> None:
    path = tmp_path / "bad.yaml"  # type: ignore[operator]
    path.write_text("model:\n  surprise: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration field"):
        load_config(path)


def test_reference_config_has_kaggle_training_contract() -> None:
    config = load_config("configs/reference_maskrcnn.yaml")
    assert config.training.epochs == 20
    assert config.model.weights == "coco_v1"
    assert config.data.train_limit is None
