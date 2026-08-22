"""Strict YAML configuration with reproducible CLI overrides."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration input cannot describe a safe run."""


@dataclass(frozen=True)
class RunConfig:
    name: str = "learning-minimal"
    seed: int = 42
    output_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class DataConfig:
    provider: str = "pennfudan"
    factory: str | None = None
    root: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    image_size: tuple[int, int] | None = (128, 128)
    batch_size: int = 1
    num_workers: int = 0
    horizontal_flip: float = 0.0
    train_limit: int | None = 2
    valid_limit: int | None = 1
    test_limit: int | None = 1


@dataclass(frozen=True)
class ModelConfig:
    name: str = "maskrcnn_resnet50_fpn"
    factory: str | None = None
    weights: str = "none"
    num_classes: int = 2
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2
    optimizer: str = "sgd"
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    scheduler: str = "none"
    step_size: int = 6
    gamma: float = 0.1
    amp: bool | str = "auto"
    grad_clip_norm: float | None = None
    best_metric: str = "mask_map"
    evaluation_score_floor: float = 0.0
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    evaluate_every: int = 1


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"


def load_config(path: str | Path | None = None, overrides: Sequence[tuple[str, str]] = ()) -> AppConfig:
    values = asdict(AppConfig())
    if path is not None:
        source = Path(path)
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read configuration {source}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ConfigError("configuration root must be a mapping")
        _merge_known(values, raw)
    for key, raw_value in overrides:
        try:
            _set_known(values, key, yaml.safe_load(raw_value))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid override {key}: {exc}") from exc
    return config_from_dict(values)


def load_config_with_sources(
    path: str | Path | None = None, overrides: Sequence[tuple[str, str]] = ()
) -> tuple[AppConfig, dict[str, str]]:
    config = load_config(path, overrides)
    sources = {key: "default" for key in _leaf_paths(asdict(AppConfig()))}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(raw, Mapping):
            sources.update({key: "yaml" for key in _leaf_paths(raw)})
    sources.update({key: "cli" for key, _ in overrides})
    return config, dict(sorted(sources.items()))


def config_from_dict(raw: Mapping[str, object]) -> AppConfig:
    values = asdict(AppConfig())
    _merge_known(values, raw)
    try:
        run = dict(values["run"])
        run["output_dir"] = Path(run["output_dir"])
        data = dict(values["data"])
        data["root"] = Path(data["root"])
        data["manifest_dir"] = Path(data["manifest_dir"])
        image_size = data["image_size"]
        data["image_size"] = None if image_size is None else tuple(image_size)
        config = AppConfig(
            run=RunConfig(**run),
            data=DataConfig(**data),
            model=ModelConfig(**dict(values["model"])),
            training=TrainingConfig(**dict(values["training"])),
            device=values["device"],
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid configuration value: {exc}") from exc
    _validate_config(config)
    return config


def config_to_dict(config: AppConfig) -> dict[str, object]:
    return _serialize(asdict(config))


def _merge_known(target: dict[str, Any], incoming: Mapping[Any, object], prefix: str = "") -> None:
    for raw_key, value in incoming.items():
        if not isinstance(raw_key, str):
            raise ConfigError("configuration keys must be strings")
        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if raw_key not in target:
            raise ConfigError(f"unknown configuration field: {path}")
        current = target[raw_key]
        if isinstance(current, dict) and path != "model.params":
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path} must be a mapping")
            _merge_known(current, value, path)
        else:
            target[raw_key] = value


def _set_known(values: dict[str, Any], key: str, value: object) -> None:
    if not key:
        raise ConfigError("override key must not be empty")
    parts = key.split(".")
    current = values
    for index, part in enumerate(parts):
        path = ".".join(parts[: index + 1])
        if part not in current:
            if ".".join(parts[:index]) == "model.params" and index == len(parts) - 1:
                current[part] = value
                return
            raise ConfigError(f"unknown configuration field: {path}")
        if index == len(parts) - 1:
            current[part] = value
            return
        child = current[part]
        if not isinstance(child, dict):
            raise ConfigError(f"{path} is not a configuration section")
        current = child


def _leaf_paths(raw: Mapping[Any, object], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, value in raw.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            paths.extend(_leaf_paths(value, current))
        else:
            paths.append(current)
    return tuple(paths)


def _validate_config(config: AppConfig) -> None:
    _require_string("run.name", config.run.name)
    _require_integer("run.seed", config.run.seed, minimum=0, maximum=2**32 - 1)
    _require_string("data.provider", config.data.provider)
    if config.data.factory is not None:
        _require_string("data.factory", config.data.factory)
    _require_integer("data.batch_size", config.data.batch_size, minimum=1)
    _require_integer("data.num_workers", config.data.num_workers, minimum=0)
    _require_probability("data.horizontal_flip", config.data.horizontal_flip)
    if config.data.image_size is not None:
        if len(config.data.image_size) != 2:
            raise ConfigError("data.image_size must be [height, width]")
        for item in config.data.image_size:
            _require_integer("data.image_size", item, minimum=1)
    for field_name, value in (
        ("train_limit", config.data.train_limit),
        ("valid_limit", config.data.valid_limit),
        ("test_limit", config.data.test_limit),
    ):
        if value is not None:
            _require_integer(f"data.{field_name}", value, minimum=1)
    _require_string("model.name", config.model.name)
    if config.model.factory is not None:
        _require_string("model.factory", config.model.factory)
    _require_string("model.weights", config.model.weights)
    _require_integer("model.num_classes", config.model.num_classes, minimum=2)
    if not isinstance(config.model.params, dict):
        raise ConfigError("model.params must be a mapping")
    _require_integer("training.epochs", config.training.epochs, minimum=1)
    _require_choice("training.optimizer", config.training.optimizer, {"sgd", "adamw"})
    _require_choice("training.scheduler", config.training.scheduler, {"none", "step"})
    _require_number("training.learning_rate", config.training.learning_rate, minimum=0.0, exclusive=True)
    _require_number("training.momentum", config.training.momentum, minimum=0.0)
    _require_number("training.weight_decay", config.training.weight_decay, minimum=0.0)
    _require_integer("training.step_size", config.training.step_size, minimum=1)
    _require_number("training.gamma", config.training.gamma, minimum=0.0, exclusive=True)
    if config.training.amp not in {True, False, "auto"}:
        raise ConfigError("training.amp must be true, false, or 'auto'")
    if config.training.grad_clip_norm is not None:
        _require_number("training.grad_clip_norm", config.training.grad_clip_norm, minimum=0.0, exclusive=True)
    if config.training.best_metric != "mask_map":
        raise ConfigError("training.best_metric must be 'mask_map'")
    _require_probability("training.evaluation_score_floor", config.training.evaluation_score_floor)
    _require_probability("training.score_threshold", config.training.score_threshold)
    _require_probability("training.mask_threshold", config.training.mask_threshold)
    _require_integer("training.evaluate_every", config.training.evaluate_every, minimum=1)
    _require_choice("device", config.device, {"auto", "cpu", "cuda", "mps"})


def _require_string(path: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")


def _require_integer(path: str, value: object, *, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{path} is outside its allowed range")


def _require_number(path: str, value: object, *, minimum: float, exclusive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{path} must be a finite number")
    if value <= minimum if exclusive else value < minimum:
        raise ConfigError(f"{path} must be {'greater than' if exclusive else 'at least'} {minimum}")


def _require_probability(path: str, value: object) -> None:
    _require_number(path, value, minimum=0.0)
    if value > 1.0:  # type: ignore[operator]
        raise ConfigError(f"{path} must be at most 1.0")


def _require_choice(path: str, value: object, choices: set[str]) -> None:
    _require_string(path, value)
    if value not in choices:
        raise ConfigError(f"{path} must be one of {', '.join(sorted(choices))}")


def _serialize(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    return value
