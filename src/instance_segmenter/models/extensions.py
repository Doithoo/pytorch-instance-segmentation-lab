"""Load explicit external model factories without changing the core registry."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from torch import nn


def load_external_model(factory_path: str, num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("external model factory must use module.path:callable_name")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"external model factory {factory_path!r} is not callable")
    model = factory(num_classes=num_classes, weights=weights, params=dict(params))
    if not isinstance(model, nn.Module):
        raise TypeError("external model factory must return torch.nn.Module")
    return model
