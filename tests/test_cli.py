from __future__ import annotations

from instance_segmenter.cli import main


def test_list_datasets_command(capsys: object) -> None:
    assert main(["list-datasets"]) == 0
    assert "pennfudan" in capsys.readouterr().out  # type: ignore[union-attr]


def test_list_models_command(capsys: object) -> None:
    assert main(["list-models"]) == 0
    assert "maskrcnn_resnet50_fpn" in capsys.readouterr().out  # type: ignore[union-attr]
