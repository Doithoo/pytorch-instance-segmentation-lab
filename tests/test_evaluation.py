from __future__ import annotations

from instance_segmenter.config import config_from_dict
from instance_segmenter.data.manifest import prepare_penn_fudan
from instance_segmenter.evaluation.evaluate import evaluate_checkpoint
from instance_segmenter.training.checkpoint import save_checkpoint
from tests.fixtures import create_fake_penn_fudan
from tests.fixtures.external_models import ContractInstanceModel


def test_checkpoint_evaluation_writes_machine_readable_reports(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    create_fake_penn_fudan(data_dir)
    metadata = prepare_penn_fudan(data_dir, manifest_dir)
    config = config_from_dict(
        {
            "run": {"output_dir": str(tmp_path / "artifacts")},
            "data": {"root": str(data_dir), "manifest_dir": str(manifest_dir), "batch_size": 1, "test_limit": 1},
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
    output = tmp_path / "evaluation"  # type: ignore[operator]
    result = evaluate_checkpoint(checkpoint, split="test", output_dir=output, device="cpu", plot=True)
    assert result.image_count == 1
    assert (output / "evaluation.json").is_file()
    assert (output / "per_class.csv").is_file()
    assert (output / "per_image.csv").read_text(encoding="utf-8").count("\n") == 2
    assert list((output / "visualizations").glob("*.png"))
