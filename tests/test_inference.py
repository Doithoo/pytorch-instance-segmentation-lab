from __future__ import annotations

import json

from instance_segmenter.config import config_from_dict
from instance_segmenter.data.manifest import prepare_penn_fudan
from instance_segmenter.inference.predictor import Predictor
from instance_segmenter.training.checkpoint import save_checkpoint
from tests.fixtures import create_fake_penn_fudan
from tests.fixtures.external_models import ContractInstanceModel


def test_predictor_writes_json_masks_directory_and_overlay(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    root = create_fake_penn_fudan(data_dir)
    metadata = prepare_penn_fudan(data_dir, manifest_dir)
    config = config_from_dict(
        {
            "data": {"root": str(data_dir), "manifest_dir": str(manifest_dir)},
            "model": {"factory": "tests.fixtures.external_models:build_contract_model"},
        }
    )
    checkpoint = tmp_path / "checkpoint.pt"  # type: ignore[operator]
    save_checkpoint(
        checkpoint,
        model=ContractInstanceModel(),
        model_name=config.model.name,
        optimizer=None,
        scheduler=None,
        epoch=1,
        best_metric=0.0,
        best_epoch=1,
        label_schema=metadata.label_schema,
        config=config,
        manifest_hashes=metadata.split_hashes,
    )
    output = tmp_path / "prediction"  # type: ignore[operator]
    result = Predictor.from_checkpoint(checkpoint, device="cpu").predict_single(
        root / "PNGImages/FudanPed00000.png", output
    )
    payload = json.loads(result.instances_path.read_text(encoding="utf-8"))
    assert payload["instances"] == []
    assert result.overlay_path.is_file()
    assert (output / "masks").is_dir()
