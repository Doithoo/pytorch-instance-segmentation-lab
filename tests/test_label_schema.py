from __future__ import annotations

import pytest

from instance_segmenter.data.schema import DEFAULT_LABEL_SCHEMA, ClassDefinition, LabelSchema


def test_default_schema_has_background_and_person() -> None:
    assert DEFAULT_LABEL_SCHEMA.num_classes == 2
    assert DEFAULT_LABEL_SCHEMA.foreground_ids == (1,)
    assert DEFAULT_LABEL_SCHEMA.class_name(1) == "person"


def test_schema_round_trip(tmp_path: object) -> None:
    path = tmp_path / "schema.yaml"  # type: ignore[operator]
    DEFAULT_LABEL_SCHEMA.write_yaml(path)
    assert LabelSchema.read_yaml(path) == DEFAULT_LABEL_SCHEMA


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        ((ClassDefinition(0, "background", (1, 2, 3)), ClassDefinition(2, "person", (4, 5, 6))), "contiguous"),
        ((ClassDefinition(0, "background", (1, 2, 3)), ClassDefinition(1, "person", (1, 2, 3))), "colors"),
    ],
)
def test_schema_rejects_invalid_class_spaces(classes: tuple[ClassDefinition, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LabelSchema(classes)
