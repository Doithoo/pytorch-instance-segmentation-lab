from __future__ import annotations

import pytest
import torch

from instance_segmenter.inference.output import normalize_prediction


def test_prediction_contract_rejects_misaligned_fields() -> None:
    output = {
        "boxes": torch.empty((2, 4)),
        "labels": torch.empty((1,), dtype=torch.int64),
        "scores": torch.empty((2,)),
        "masks": torch.empty((2, 1, 4, 4)),
    }
    with pytest.raises(RuntimeError, match="same instance count"):
        normalize_prediction(output)


def test_prediction_contract_rejects_invalid_probabilities() -> None:
    output = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([1.5]),
        "masks": torch.ones((1, 1, 2, 2)),
    }
    with pytest.raises(RuntimeError, match="probabilities"):
        normalize_prediction(output)


def test_prediction_contract_rejects_non_tensor_fields() -> None:
    with pytest.raises(RuntimeError, match="must be tensors"):
        normalize_prediction({"boxes": [], "labels": [], "scores": [], "masks": []})
