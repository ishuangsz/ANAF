"""Validated configuration objects for ANAF experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings shared by all supported datasets."""

    seed: int = 2024
    epochs: int = 10
    batch_size: int = 64
    stride: int = 1
    learning_rate: float = 1e-3
    heads: int = 4
    dropout: float = 0.1
    validation_ratio: float = 0.2
    wavelet: str = "db1"
    wavelet_level: int = 3
    alpha: float = 1.0
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    threshold_steps: int = 1000

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.epochs < 1 or self.batch_size < 1 or self.stride < 1:
            raise ValueError("epochs, batch_size, and stride must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio must be between zero and one")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.heads < 1 or self.wavelet_level < 1 or self.threshold_steps < 2:
            raise ValueError("heads, wavelet_level, and threshold_steps are invalid")
        if not isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if self.scheduler_step_size < 1 or self.scheduler_gamma <= 0:
            raise ValueError("scheduler_step_size and scheduler_gamma must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    """Shape and model settings for one benchmark dataset."""

    name: str
    directory_prefix: str
    entity_start: int
    entity_end: int
    channels: int
    window_size: int
    encoder_layers: int
    model_dim: int
    hidden_dim: int
    select_position: str = "mid"
    ignore_dimensions: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entity_start < 0 or self.entity_end < self.entity_start:
            raise ValueError(f"invalid entity range for {self.name}")
        if not self.name or not self.directory_prefix:
            raise ValueError("dataset name and directory_prefix must not be empty")
        if min(
            self.channels,
            self.window_size,
            self.encoder_layers,
            self.model_dim,
            self.hidden_dim,
        ) < 1:
            raise ValueError(f"invalid dimensions for {self.name}")
        if self.select_position not in {"mid", "tail"}:
            raise ValueError("select_position must be 'mid' or 'tail'")
        for entity, dimensions in self.ignore_dimensions.items():
            if not self.entity_start <= entity <= self.entity_end:
                raise ValueError(f"ignored entity {entity} is outside the configured range")
            if any(dimension < 0 or dimension >= self.channels for dimension in dimensions):
                raise ValueError(f"ignored dimension is outside the channel range for {self.name}")

    @property
    def entities(self) -> range:
        return range(self.entity_start, self.entity_end + 1)

    def entity_directory(self, data_root: Path, entity: int) -> Path:
        if entity not in self.entities:
            raise ValueError(f"entity {entity} is not configured for {self.name}")
        return data_root / f"{self.directory_prefix}_{entity}"


@dataclass(frozen=True)
class ProjectConfig:
    """Complete ANAF configuration."""

    training: TrainingConfig
    datasets: dict[str, DatasetConfig]

    def dataset(self, name: str) -> DatasetConfig:
        normalized = name.upper()
        try:
            return self.datasets[normalized]
        except KeyError as exc:
            choices = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset {name!r}; choose one of: {choices}") from exc


def _coerce_ignore_dimensions(value: Any) -> dict[int, tuple[int, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("ignore_dimensions must be a mapping")
    return {
        int(entity): tuple(int(item) for item in dimensions)
        for entity, dimensions in value.items()
    }


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate an ANAF YAML configuration file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")

    training_options = dict(raw.get("training", {}))
    training_options.pop("random_validation", None)  # Accept pre-release YAML files.
    training = TrainingConfig(**training_options)
    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("configuration must define at least one dataset")

    datasets: dict[str, DatasetConfig] = {}
    for name, values in raw_datasets.items():
        if not isinstance(values, dict):
            raise TypeError(f"dataset {name!r} must be a mapping")
        normalized = str(name).upper()
        payload = dict(values)
        payload["name"] = normalized
        payload["ignore_dimensions"] = _coerce_ignore_dimensions(
            payload.get("ignore_dimensions")
        )
        dataset = DatasetConfig(**payload)
        if dataset.model_dim % training.heads:
            raise ValueError(f"model_dim must be divisible by heads for {normalized}")
        datasets[normalized] = dataset
    return ProjectConfig(training=training, datasets=datasets)
