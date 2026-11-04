"""ANAF loss and anomaly-score functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from .model import ANAFOutput

# Bound the Gaussian exponentials in the float32 training and scoring paths.
_LOG_PRECISION_LIMIT = 20.0


@dataclass(frozen=True)
class LossOutput:
    """Scalar reconstruction and weighted distribution losses."""

    reconstruction: torch.Tensor
    total: torch.Tensor


def elementwise_gaussian_score(output: ANAFOutput, target: torch.Tensor) -> torch.Tensor:
    """Return the factor-independent Gaussian negative log-likelihood score."""

    squared_error = functional.mse_loss(output.mean, target, reduction="none")
    log_precision = output.log_precision.clamp(-_LOG_PRECISION_LIMIT, _LOG_PRECISION_LIMIT)
    return squared_error * log_precision.exp() - log_precision


def weighted_gaussian_loss(
    output: ANAFOutput,
    target: torch.Tensor,
    *,
    alpha: float,
) -> LossOutput:
    """Apply legacy ANAF detached-variance channel weighting."""

    squared_error = functional.mse_loss(output.mean, target, reduction="none")
    log_precision = output.log_precision.clamp(-_LOG_PRECISION_LIMIT, _LOG_PRECISION_LIMIT)
    elementwise = squared_error * log_precision.exp() - log_precision
    variance = (-log_precision.detach()).exp()
    mean_variance = variance.mean(dim=(0, 1)).pow(alpha).clamp_min(torch.finfo(variance.dtype).tiny)
    total = (variance * elementwise / mean_variance).mean()
    return LossOutput(reconstruction=squared_error.mean(), total=total)
