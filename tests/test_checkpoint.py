from __future__ import annotations

import pytest
import torch

from instance_segmenter.config import AppConfig
from instance_segmenter.data.schema import DEFAULT_LABEL_SCHEMA
from instance_segmenter.training.checkpoint import CheckpointError, load_checkpoint, restore_checkpoint, save_checkpoint
from tests.fixtures.external_models import ContractInstanceModel


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: object) -> None:
    model = ContractInstanceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.scale.fill_(3.0)
    path = tmp_path / "checkpoint.pt"  # type: ignore[operator]
    save_checkpoint(
        path,
        model=model,
        model_name="contract",
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        best_metric=0.3,
        best_epoch=2,
        label_schema=DEFAULT_LABEL_SCHEMA,
        config=AppConfig(),
        manifest_hashes={"train": "a", "valid": "b", "test": "c"},
    )
    restored = ContractInstanceModel()
    payload = load_checkpoint(path)
    restore_checkpoint(payload, model=restored, expected_model_name="contract", expected_schema=DEFAULT_LABEL_SCHEMA)
    assert float(restored.scale.detach()) == 3.0


def test_checkpoint_rejects_manifest_mismatch(tmp_path: object) -> None:
    model = ContractInstanceModel()
    path = tmp_path / "checkpoint.pt"  # type: ignore[operator]
    save_checkpoint(
        path,
        model=model,
        model_name="contract",
        optimizer=None,
        scheduler=None,
        epoch=1,
        best_metric=0.0,
        best_epoch=1,
        label_schema=DEFAULT_LABEL_SCHEMA,
        config=AppConfig(),
        manifest_hashes={"train": "a", "valid": "b", "test": "c"},
    )
    payload = load_checkpoint(path)
    with pytest.raises(CheckpointError, match="manifest hashes"):
        restore_checkpoint(
            payload,
            model=model,
            expected_model_name="contract",
            expected_schema=DEFAULT_LABEL_SCHEMA,
            expected_manifest_hashes={"train": "changed", "valid": "b", "test": "c"},
        )


def test_checkpoint_rejects_schema_mismatch(tmp_path: object) -> None:
    model = ContractInstanceModel()
    path = tmp_path / "checkpoint.pt"  # type: ignore[operator]
    save_checkpoint(
        path,
        model=model,
        model_name="contract",
        optimizer=None,
        scheduler=None,
        epoch=1,
        best_metric=0.0,
        best_epoch=1,
        label_schema=DEFAULT_LABEL_SCHEMA,
        config=AppConfig(),
        manifest_hashes={"train": "a", "valid": "b", "test": "c"},
    )
    payload = load_checkpoint(path)
    with torch.no_grad():
        pass
    from instance_segmenter.data.schema import ClassDefinition, LabelSchema

    other_schema = LabelSchema((ClassDefinition(0, "background", (1, 2, 3)), ClassDefinition(1, "other", (4, 5, 6))))
    try:
        restore_checkpoint(payload, model=model, expected_model_name="contract", expected_schema=other_schema)
    except CheckpointError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("expected schema mismatch")
