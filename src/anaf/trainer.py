"""Training, checkpointing, and scoring for ANAF."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from . import __version__
from .config import DatasetConfig, ProjectConfig, TrainingConfig
from .data import (
    ScalerState,
    create_test_loader,
    create_training_loaders,
    load_entity_arrays,
    load_test_arrays,
    split_training_series,
)
from .losses import elementwise_gaussian_score, weighted_gaussian_loss
from .metrics import EvaluationReport, aggregate_evaluations, evaluate_entity
from .model import ANAF


def resolve_device(value: str) -> torch.device:
    """Resolve a user device string and validate CUDA availability."""

    normalized = value.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is outside the available device range"
            )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    return device


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(dataset: DatasetConfig, training: TrainingConfig) -> ANAF:
    return ANAF(
        window_size=dataset.window_size,
        channels=dataset.channels,
        model_dim=dataset.model_dim,
        heads=training.heads,
        hidden_dim=dataset.hidden_dim,
        encoder_layers=dataset.encoder_layers,
        dropout=training.dropout,
    )


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"invalid ANAF checkpoint: {path}")
    return payload


def load_checkpoint(path: str | Path, model: ANAF, device: torch.device) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = _torch_load(checkpoint_path, device)
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


class ANAFTrainer:
    """Train ANAF and collect time-only anomaly scores."""

    def __init__(
        self,
        model: ANAF,
        training: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.training = training
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=training.learning_rate)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=training.scheduler_step_size,
            gamma=training.scheduler_gamma,
        )

    def _batch_loss(self, time: torch.Tensor, frequency: torch.Tensor) -> tuple[Any, Any]:
        output = self.model(time, frequency)
        loss = weighted_gaussian_loss(output, time, alpha=self.training.alpha)
        return output, loss

    def train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        self.model.train()
        reconstruction_total = 0.0
        loss_total = 0.0
        for time, frequency in loader:
            time = time.to(self.device, dtype=torch.float32)
            frequency = frequency.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self._batch_loss(time, frequency)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite ANAF training loss")
            loss.total.backward()
            self.optimizer.step()

            reconstruction_total += float(loss.reconstruction.detach())
            loss_total += float(loss.total.detach())
        if not len(loader):
            raise ValueError("training loader is empty")
        return {
            "reconstruction": reconstruction_total / len(loader),
            "total": loss_total / len(loader),
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        reconstruction_total = 0.0
        loss_total = 0.0
        for time, frequency in loader:
            time = time.to(self.device, dtype=torch.float32)
            frequency = frequency.to(self.device, dtype=torch.float32)
            _, loss = self._batch_loss(time, frequency)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite ANAF validation loss")
            reconstruction_total += float(loss.reconstruction)
            loss_total += float(loss.total)
        if not len(loader):
            raise ValueError("validation loader is empty")
        return {
            "reconstruction": reconstruction_total / len(loader),
            "total": loss_total / len(loader),
        }

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        checkpoint_path: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Train for the configured epochs and retain the best validation checkpoint."""

        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        best_validation = float("inf")
        for epoch in range(1, self.training.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            validation_metrics = self.validate(validation_loader)
            if validation_metrics["total"] < best_validation:
                best_validation = validation_metrics["total"]
                payload = {
                    "format_version": 1,
                    "package_version": __version__,
                    "model_state": self.model.state_dict(),
                    "best_validation_loss": best_validation,
                    "epoch": epoch,
                    **metadata,
                }
                temporary = destination.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(destination)
            print(
                f"epoch={epoch:02d}/{self.training.epochs:02d} "
                f"train={train_metrics['total']:.6f} "
                f"validation={validation_metrics['total']:.6f}"
            )
            self.scheduler.step()

    @torch.no_grad()
    def collect_channel_scores(
        self, loader: DataLoader[Any], *, select_position: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect per-channel time-only scores and aligned labels."""

        if select_position not in {"mid", "tail"}:
            raise ValueError("select_position must be 'mid' or 'tail'")
        self.model.eval()
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for time, labels in loader:
            time = time.to(self.device, dtype=torch.float32)
            output = self.model.forward_time_only(time)
            scores = elementwise_gaussian_score(output, time)
            position = time.shape[1] // 2 if select_position == "mid" else time.shape[1] - 1
            all_scores.append(scores[:, position].cpu())
            all_labels.append(labels[:, position].cpu())
        if not all_scores:
            raise ValueError("test loader is empty")
        return (
            torch.cat(all_scores).numpy(),
            torch.cat(all_labels).numpy().astype(np.int64, copy=False),
        )


def train_dataset(
    project: ProjectConfig,
    dataset_name: str,
    data_root: str | Path,
    output_root: str | Path,
    *,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> None:
    """Train a checkpoint for every configured entity."""

    dataset = project.dataset(dataset_name)
    set_global_seed(seed)
    for entity in dataset.entities:
        print(f"training dataset={dataset.name} entity={entity}")
        directory = dataset.entity_directory(Path(data_root), entity)
        arrays = load_entity_arrays(directory, dataset.channels)
        train_values, validation_values = split_training_series(
            arrays.train, dataset.window_size, project.training.validation_ratio
        )
        scaler = ScalerState.fit(train_values)
        train_loader, validation_loader = create_training_loaders(
            scaler.transform(train_values),
            scaler.transform(validation_values),
            dataset,
            project.training,
            seed=seed,
            num_workers=num_workers,
        )
        model = build_model(dataset, project.training)
        trainer = ANAFTrainer(model, project.training, device)
        checkpoint = Path(output_root) / dataset.name / f"entity_{entity:03d}" / "best.pt"
        trainer.fit(
            train_loader,
            validation_loader,
            checkpoint,
            metadata={
                "dataset": dataset.name,
                "entity": entity,
                "dataset_config": asdict(dataset),
                "training_config": asdict(project.training),
                "scaler": scaler.to_dict(),
                "seed": seed,
            },
        )


def evaluate_dataset(
    project: ProjectConfig,
    dataset_name: str,
    data_root: str | Path,
    checkpoint_root: str | Path,
    *,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> EvaluationReport:
    """Evaluate every entity and write a stable dataset-level metrics report."""

    dataset = project.dataset(dataset_name)
    set_global_seed(seed)
    evaluations = []
    for entity in dataset.entities:
        print(f"evaluating dataset={dataset.name} entity={entity}")
        checkpoint = Path(checkpoint_root) / dataset.name / f"entity_{entity:03d}" / "best.pt"
        model = build_model(dataset, project.training).to(device)
        payload = load_checkpoint(checkpoint, model, device)
        if payload.get("dataset") != dataset.name or int(payload.get("entity", -1)) != entity:
            raise ValueError(f"checkpoint metadata does not match {dataset.name} entity {entity}")
        if payload.get("dataset_config") != asdict(dataset):
            raise ValueError(
                f"current dataset configuration does not match checkpoint {checkpoint}"
            )
        checkpoint_training = dict(payload.get("training_config", {}))
        checkpoint_training.pop("random_validation", None)
        if checkpoint_training != asdict(project.training):
            raise ValueError(
                f"current training configuration does not match checkpoint {checkpoint}"
            )
        scaler = ScalerState.from_dict(payload["scaler"])
        directory = dataset.entity_directory(Path(data_root), entity)
        test, labels = load_test_arrays(directory, dataset.channels)
        test_values = scaler.transform(test)
        loader = create_test_loader(
            test_values,
            labels,
            dataset,
            project.training,
            num_workers=num_workers,
        )
        trainer = ANAFTrainer(model, project.training, device)
        channel_scores, aligned_labels = trainer.collect_channel_scores(
            loader, select_position=dataset.select_position
        )
        evaluations.append(
            evaluate_entity(
                entity,
                channel_scores,
                aligned_labels,
                ignore_dimensions=dataset.ignore_dimensions.get(entity, ()),
                threshold_steps=project.training.threshold_steps,
            )
        )

    report = aggregate_evaluations(dataset.name, evaluations)
    destination = Path(checkpoint_root) / dataset.name / "metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    aggregate = report.aggregate
    print(
        f"dataset={dataset.name} f1={aggregate['f1']:.6f} auc={aggregate['auc']} "
        f"f1_pa={aggregate['f1_pa']:.6f} auc_pa={aggregate['auc_pa']}"
    )
    print(f"metrics={destination}")
    return report
