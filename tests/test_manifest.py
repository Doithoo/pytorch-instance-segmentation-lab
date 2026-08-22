from __future__ import annotations

import pytest
import torch

from instance_segmenter.data.dataset import PennFudanDataset
from instance_segmenter.data.manifest import (
    PENN_FUDAN_SPLIT_COUNTS,
    ManifestError,
    load_dataset_metadata,
    prepare_penn_fudan,
    read_manifest,
    verify_prepared_data,
)
from instance_segmenter.data.registry import list_datasets
from tests.fixtures import create_fake_penn_fudan


def test_prepare_writes_stable_136_17_17_manifests(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    create_fake_penn_fudan(data_dir)
    first = prepare_penn_fudan(data_dir, manifest_dir)
    second = prepare_penn_fudan(data_dir, manifest_dir)
    assert first.identity == second.identity
    assert first.split_counts == PENN_FUDAN_SPLIT_COUNTS
    assert [len(read_manifest(manifest_dir / f"{split}.csv")) for split in PENN_FUDAN_SPLIT_COUNTS] == [136, 17, 17]
    assert verify_prepared_data(data_dir, manifest_dir).identity == first.identity


def test_manifest_detects_changed_source_bytes(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    root = create_fake_penn_fudan(data_dir)
    prepare_penn_fudan(data_dir, manifest_dir)
    image = root / "PNGImages/FudanPed00000.png"
    image.write_bytes(b"not a png anymore")
    with pytest.raises(ManifestError, match="hash mismatch"):
        verify_prepared_data(data_dir, manifest_dir)


def test_dataset_rebuilds_instance_target_from_prepared_manifest(tmp_path: object) -> None:
    data_dir = tmp_path / "raw"  # type: ignore[operator]
    manifest_dir = tmp_path / "manifests"  # type: ignore[operator]
    create_fake_penn_fudan(data_dir)
    prepare_penn_fudan(data_dir, manifest_dir)
    dataset = PennFudanDataset.from_manifests(manifest_dir, "train", data_dir=data_dir)
    image, target = dataset[1]
    assert image.dtype.is_floating_point
    assert target["masks"].dtype == torch.bool
    assert target["masks"].shape[0] == 2
    assert target["labels"].tolist() == [1, 1]
    assert load_dataset_metadata(manifest_dir).label_schema.num_classes == 2


def test_pennfudan_is_the_registered_provider() -> None:
    assert list_datasets() == ("pennfudan",)
