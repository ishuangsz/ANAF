"""Dataset loading, scaling, windowing, and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, TrainingConfig
from .transforms import DWTTransform


@dataclass(frozen=True)
class EntityArrays:
    """Validated arrays loaded for one entity."""

    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ScalerState:
    """Serializable state required to apply a fitted StandardScaler."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("scaler mean and scale must have equal non-zero lengths")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("scaler state contains non-finite values")
        if any(value <= 0 for value in self.scale):
            raise ValueError("scaler scales must be positive")

    @classmethod
    def fit(cls, train: np.ndarray) -> ScalerState:
        scaler = StandardScaler().fit(train)
        return cls(
            mean=tuple(float(value) for value in scaler.mean_),
            scale=tuple(float(value) for value in scaler.scale_),
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ScalerState:
        return cls(
            mean=tuple(float(value) for value in values["mean"]),
            scale=tuple(float(value) for value in values["scale"]),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != mean.shape[0]:
            raise ValueError("array shape does not match the fitted scaler")
        return ((values - mean) / scale).astype(np.float32, copy=False)


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create fixed-length windows from scaled time-series arrays."""

    def __init__(
        self,
        values: np.ndarray,
        window_size: int,
        stride: int,
        *,
        labels: np.ndarray | None = None,
        frequency_transform: DWTTransform | None = None,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, channels]")
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if values.shape[0] < window_size:
            raise ValueError(
                f"series length {values.shape[0]} is shorter than window size {window_size}"
            )
        if labels is not None and labels.shape != (values.shape[0],):
            raise ValueError("labels must be one-dimensional and aligned with values")
        if labels is not None and frequency_transform is not None:
            raise ValueError("evaluation windows do not require a frequency transform")
        if labels is None and frequency_transform is None:
            raise ValueError("training windows require a frequency transform")

        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.labels = None if labels is None else np.ascontiguousarray(labels, dtype=np.int64)
        self.window_size = window_size
        self.stride = stride
        self.frequency_transform = frequency_transform

    def __len__(self) -> int:
        return (self.values.shape[0] - self.window_size) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.stride
        stop = start + self.window_size
        time_window = self.values[start:stop]
        time_tensor = torch.from_numpy(time_window)
        if self.labels is not None:
            return time_tensor, torch.from_numpy(self.labels[start:stop])
        assert self.frequency_transform is not None
        frequency = self.frequency_transform(time_window)
        return time_tensor, torch.from_numpy(frequency)


def _validate_features(values: np.ndarray, channels: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != channels:
        raise ValueError(f"{label} must have shape [time, {channels}], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return array.astype(np.float64, copy=False)


def _validate_labels(values: np.ndarray, expected_length: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or labels.shape[0] != expected_length:
        raise ValueError(
            f"test labels must have shape [{expected_length}] or [{expected_length}, 1]"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("test labels must be binary")
    return labels.astype(np.int64, copy=False)


def load_entity_arrays(entity_directory: str | Path, channels: int) -> EntityArrays:
    """Load and validate ``train.npy`` and ``test.npz`` for one entity."""

    directory = Path(entity_directory)
    train_path = directory / "train.npy"
    test_path = directory / "test.npz"
    if not train_path.is_file():
        raise FileNotFoundError(f"training data not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"test data not found: {test_path}")

    train = _validate_features(np.load(train_path, allow_pickle=False), channels, "train.npy")
    with np.load(test_path, allow_pickle=False) as archive:
        missing = {"x", "y"}.difference(archive.files)
        if missing:
            raise KeyError(f"test.npz is missing keys: {', '.join(sorted(missing))}")
        test = _validate_features(archive["x"], channels, "test.npz['x']")
        labels = _validate_labels(archive["y"], test.shape[0])
    return EntityArrays(train=train, test=test, labels=labels)


def load_test_arrays(entity_directory: str | Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Load only the test inputs and labels for checkpoint evaluation."""

    test_path = Path(entity_directory) / "test.npz"
    if not test_path.is_file():
        raise FileNotFoundError(f"test data not found: {test_path}")
    with np.load(test_path, allow_pickle=False) as archive:
        missing = {"x", "y"}.difference(archive.files)
        if missing:
            raise KeyError(f"test.npz is missing keys: {', '.join(sorted(missing))}")
        test = _validate_features(archive["x"], channels, "test.npz['x']")
        labels = _validate_labels(archive["y"], test.shape[0])
    return test, labels


def split_training_series(
    values: np.ndarray,
    window_size: int,
    validation_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a chronological tail with at least one full window per segment."""

    if len(values) < 2 * window_size:
        raise ValueError(
            f"train.npy needs at least {2 * window_size} points for disjoint "
            f"training and validation windows; got {len(values)}"
        )
    validation_size = max(window_size, int(len(values) * validation_ratio))
    split = max(window_size, len(values) - validation_size)
    return values[:split], values[split:]


def create_training_loaders(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    num_workers: int,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    """Create deterministic train and validation loaders for one entity."""

    transform = DWTTransform(training_config.wavelet, training_config.wavelet_level)
    train_windows = WindowDataset(
        train_values,
        dataset_config.window_size,
        training_config.stride,
        frequency_transform=transform,
    )
    validation_windows = WindowDataset(
        validation_values,
        dataset_config.window_size,
        training_config.stride,
        frequency_transform=transform,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": training_config.batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_windows, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_windows, shuffle=False, **loader_options)
    return train_loader, validation_loader


def create_test_loader(
    test_values: np.ndarray,
    labels: np.ndarray,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    *,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a time-only test loader with stride one."""

    windows = WindowDataset(
        test_values,
        dataset_config.window_size,
        1,
        labels=labels,
    )
    return DataLoader(
        windows,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
