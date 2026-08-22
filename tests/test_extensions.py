from __future__ import annotations

from instance_segmenter.config import config_from_dict
from instance_segmenter.data.manifest import prepare_penn_fudan
from instance_segmenter.data.providers import build_configured_dataset
from tests.fixtures import create_fake_penn_fudan


def test_external_dataset_factory_does_not_modify_registry(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    create_fake_penn_fudan(data_dir)
    prepare_penn_fudan(data_dir, manifest_dir)
    config = config_from_dict(
        {
            "data": {
                "root": str(data_dir),
                "manifest_dir": str(manifest_dir),
                "factory": "examples.extensions.my_dataset:build_dataset",
                "train_limit": 1,
            }
        }
    )
    dataset = build_configured_dataset(config.data, "train", training=True, limit=1)
    assert len(dataset) == 1
