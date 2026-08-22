from __future__ import annotations

from instance_segmenter.cli import main


def test_list_datasets_command(capsys: object) -> None:
    assert main(["list-datasets"]) == 0
    assert "pennfudan" in capsys.readouterr().out  # type: ignore[union-attr]


def test_init_config_and_doctor_commands(tmp_path: object, capsys: object) -> None:
    assert main(["init-config", "--list"]) == 0
    assert "reference_maskrcnn" in capsys.readouterr().out  # type: ignore[union-attr]
    output = tmp_path / "config.yaml"  # type: ignore[operator]
    assert main(["init-config", "learning_minimal", "--output", str(output)]) == 0
    assert output.is_file()
    assert main(["doctor", "--device", "cpu"]) == 0
    assert "device: cpu" in capsys.readouterr().out  # type: ignore[union-attr]


def test_list_models_command(capsys: object) -> None:
    assert main(["list-models"]) == 0
    assert "maskrcnn_resnet50_fpn" in capsys.readouterr().out  # type: ignore[union-attr]
