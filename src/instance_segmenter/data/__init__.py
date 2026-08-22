"""Data contracts and built-in Penn-Fudan provider."""

from instance_segmenter.data.collate import instance_collate
from instance_segmenter.data.schema import DEFAULT_LABEL_SCHEMA, InstanceTarget, LabelSchema

__all__ = ["DEFAULT_LABEL_SCHEMA", "InstanceTarget", "LabelSchema", "instance_collate"]
