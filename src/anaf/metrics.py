"""Point-wise and point-adjusted ANAF evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter
from typing import Any

import numpy as np
from scipy.stats import iqr
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class EntityEvaluation:
    entity: int
    threshold: float
    threshold_pa: float
    metrics: BinaryMetrics
    metrics_pa: BinaryMetrics
    auc: float | None
    auc_pa: float | None
    scores: np.ndarray
    scores_pa: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    predictions_pa: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "threshold": self.threshold,
            "threshold_pa": self.threshold_pa,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1": self.metrics.f1,
            "f1_pa": self.metrics_pa.f1,
            "auc": self.auc,
            "auc_pa": self.auc_pa,
            "points": int(self.labels.size),
        }


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    aggregate: dict[str, float | int | None]
    entities: tuple[EntityEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "aggregate": self.aggregate,
            "entities": [entity.summary() for entity in self.entities],
        }


def binary_metrics(predictions: np.ndarray, labels: np.ndarray) -> BinaryMetrics:
    """Compute precision, recall, and F1 for aligned binary arrays."""

    predicted = np.asarray(predictions, dtype=np.int64)
    actual = np.asarray(labels, dtype=np.int64)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("predictions and labels must be aligned one-dimensional arrays")
    true_positive = int(np.logical_and(predicted == 1, actual == 1).sum())
    false_positive = int(np.logical_and(predicted == 1, actual == 0).sum())
    false_negative = int(np.logical_and(predicted == 0, actual == 1).sum())
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return BinaryMetrics(float(precision), float(recall), float(f1))


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute ROC AUC, returning ``None`` when only one label class is present."""

    actual = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if actual.shape != values.shape or actual.ndim != 1:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if np.unique(actual).size < 2:
        return None
    return float(roc_auc_score(actual, values))


def candidate_thresholds(scores: np.ndarray, steps: int) -> np.ndarray:
    """Select sorted-position thresholds while retaining both score extremes."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    count = min(values.size, steps)
    positions = np.linspace(0, values.size - 1, num=count, dtype=np.int64)
    return values[np.argsort(values)[np.unique(positions)]]


def best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
) -> tuple[float, np.ndarray, BinaryMetrics]:
    """Find the oracle threshold with the best F1 score."""

    values = np.asarray(scores, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.int64)
    if values.shape != actual.shape or values.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")

    thresholds = candidate_thresholds(values, steps)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    best_threshold = nextafter(float(values.max()), inf)
    best_predictions = np.zeros_like(actual)
    best_metrics = BinaryMetrics(0.0, 0.0, 0.0)
    for threshold in thresholds:
        predictions = (values >= threshold).astype(np.int64)
        metrics = binary_metrics(predictions, actual)
        if metrics.f1 > best_metrics.f1:
            best_threshold = float(threshold)
            best_predictions = predictions
            best_metrics = metrics
    return best_threshold, best_predictions, best_metrics


def point_adjust_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Promote every true anomaly interval to its maximum member score."""

    values = np.asarray(scores, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.int64)
    if values.shape != actual.shape or values.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    adjusted = values.copy()
    transitions = np.diff(actual, prepend=0, append=0)
    starts = np.flatnonzero(transitions > 0)
    stops = np.flatnonzero(transitions < 0)
    for start, stop in zip(starts, stops, strict=True):
        adjusted[start:stop] = values[start:stop].max()
    return adjusted


def aggregate_channel_scores(
    channel_scores: np.ndarray,
    *,
    ignore_dimensions: tuple[int, ...] = (),
) -> np.ndarray:
    """Normalize each channel by median/IQR and retain the largest active score."""

    values = np.asarray(channel_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("channel_scores must have shape [points, channels]")
    median = np.median(values, axis=0)
    spread = iqr(values, axis=0)
    normalized = (values - median) / (spread + 1e-9)
    if ignore_dimensions:
        ignored = np.asarray(ignore_dimensions, dtype=np.int64)
        if np.any(ignored < 0) or np.any(ignored >= values.shape[1]):
            raise ValueError("ignored dimension is outside the channel range")
        if np.unique(ignored).size == values.shape[1]:
            raise ValueError("all channels cannot be ignored")
        normalized[:, ignored] = -np.inf
    return normalized.max(axis=1)


def evaluate_entity(
    entity: int,
    channel_scores: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_dimensions: tuple[int, ...],
    threshold_steps: int,
) -> EntityEvaluation:
    """Calculate all supported metrics for one entity."""

    final_scores = aggregate_channel_scores(
        channel_scores, ignore_dimensions=ignore_dimensions
    )
    adjusted_scores = point_adjust_scores(final_scores, labels)
    threshold, predictions, metrics = best_f1_threshold(
        final_scores, labels, steps=threshold_steps
    )
    threshold_pa, predictions_pa, metrics_pa = best_f1_threshold(
        adjusted_scores, labels, steps=threshold_steps
    )
    return EntityEvaluation(
        entity=entity,
        threshold=threshold,
        threshold_pa=threshold_pa,
        metrics=metrics,
        metrics_pa=metrics_pa,
        auc=safe_roc_auc(labels, final_scores),
        auc_pa=safe_roc_auc(labels, adjusted_scores),
        scores=final_scores,
        scores_pa=adjusted_scores,
        labels=np.asarray(labels, dtype=np.int64),
        predictions=predictions,
        predictions_pa=predictions_pa,
    )


def aggregate_evaluations(dataset: str, entities: list[EntityEvaluation]) -> EvaluationReport:
    """Aggregate per-entity thresholds and continuous scores into dataset metrics."""

    if not entities:
        raise ValueError("at least one entity evaluation is required")
    labels = np.concatenate([entity.labels for entity in entities])
    predictions = np.concatenate([entity.predictions for entity in entities])
    predictions_pa = np.concatenate([entity.predictions_pa for entity in entities])
    scores = np.concatenate([entity.scores for entity in entities])
    scores_pa = np.concatenate([entity.scores_pa for entity in entities])
    raw_metrics = binary_metrics(predictions, labels)
    adjusted_metrics = binary_metrics(predictions_pa, labels)
    aggregate: dict[str, float | int | None] = {
        "precision": raw_metrics.precision,
        "recall": raw_metrics.recall,
        "f1": raw_metrics.f1,
        "f1_pa": adjusted_metrics.f1,
        "auc": safe_roc_auc(labels, scores),
        "auc_pa": safe_roc_auc(labels, scores_pa),
        "points": int(labels.size),
    }
    return EvaluationReport(dataset=dataset, aggregate=aggregate, entities=tuple(entities))
