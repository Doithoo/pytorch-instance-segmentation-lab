"""Compatibility exports for the built-in Penn-Fudan provider."""

from instance_segmenter.data.dataset import PennFudanDataset, build_pennfudan_dataset
from instance_segmenter.data.manifest import PENN_FUDAN_SPLIT_COUNTS, prepare_penn_fudan

__all__ = ["PENN_FUDAN_SPLIT_COUNTS", "PennFudanDataset", "build_pennfudan_dataset", "prepare_penn_fudan"]
