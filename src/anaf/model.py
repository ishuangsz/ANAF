"""ANAF neural network."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ANAFOutput:
    """Distribution parameters predicted for a time-series window."""

    mean: torch.Tensor
    log_precision: torch.Tensor


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        masked_scores = scores.masked_fill(mask, -1e9) if mask is not None else scores
        weights = functional.softmax(masked_scores, dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """Legacy-compatible multi-head attention for batch-first tensors."""

    def __init__(self, model_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.projections = nn.ModuleList(
            nn.Linear(model_dim, model_dim, bias=False) for _ in range(3)
        )
        self.output = nn.Linear(model_dim, model_dim)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        query, key, value = [
            layer(tensor)
            .view(batch_size, -1, self.heads, self.head_dim)
            .transpose(1, 2)
            for layer, tensor in zip(self.projections, (query, key, value), strict=True)
        ]
        attended, weights = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        merged = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.heads * self.head_dim)
        )
        return self.output(merged), weights.mean(dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim, bias=False),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.output_norm = nn.LayerNorm(model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(values, values, values)
        values = self.attention_norm(values + self.attention_dropout(attended))
        return self.output_norm(values + self.output_dropout(self.feed_forward(values)))


class DomainEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        model_dim: int,
        heads: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(channels, model_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(model_dim, heads, hidden_dim, dropout) for _ in range(layers)
        )

    def forward(self, values: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values) + position
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class AdaptiveFusion(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float, layers: int = 2) -> None:
        super().__init__()
        fusion_dim = model_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.gamma = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Sigmoid())
        self.beta = nn.Linear(model_dim, model_dim)
        self.model_dim = model_dim

    def forward(self, time_hidden: torch.Tensor, frequency_hidden: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((time_hidden, frequency_hidden), dim=-1)
        attended = self.encoder(combined)[..., : self.model_dim]
        return (1.0 + self.gamma(attended)) * time_hidden + self.beta(attended)


class ANAF(nn.Module):
    """AntiNoise Adaptive Time-Frequency Fusion network."""

    def __init__(
        self,
        *,
        window_size: int,
        channels: int,
        model_dim: int,
        heads: int,
        hidden_dim: int,
        encoder_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_size < 1 or channels < 1 or encoder_layers < 1:
            raise ValueError("window_size, channels, and encoder_layers must be positive")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.window_size = window_size
        self.channels = channels
        self.position_embedding = nn.Embedding(window_size, model_dim)
        self.time_encoder = DomainEncoder(
            channels, model_dim, heads, hidden_dim, encoder_layers, dropout
        )
        self.frequency_encoder = DomainEncoder(
            channels, model_dim, heads, hidden_dim, encoder_layers, dropout
        )
        self.fusion = AdaptiveFusion(model_dim, heads, dropout)
        self.mean_head = nn.Linear(model_dim, channels)
        self.log_precision_head = nn.Linear(model_dim, channels)

    def _validate(self, values: torch.Tensor, name: str) -> None:
        if values.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, time, channels]")
        if values.shape[1] != self.window_size or values.shape[2] != self.channels:
            raise ValueError(
                f"{name} must have shape [batch, {self.window_size}, {self.channels}], "
                f"got {tuple(values.shape)}"
            )

    def _position(self, values: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(self.window_size, device=values.device)
        return self.position_embedding(indices)

    def _output(self, hidden: torch.Tensor) -> ANAFOutput:
        return ANAFOutput(
            mean=self.mean_head(hidden),
            log_precision=self.log_precision_head(hidden),
        )

    def forward(self, time: torch.Tensor, frequency: torch.Tensor) -> ANAFOutput:
        """Run the dual-domain training path."""

        self._validate(time, "time")
        self._validate(frequency, "frequency")
        if time.shape != frequency.shape:
            raise ValueError("time and frequency inputs must have identical shapes")
        position = self._position(time)
        time_hidden = self.time_encoder(time, position)
        frequency_hidden = self.frequency_encoder(frequency, position)
        return self._output(self.fusion(time_hidden, frequency_hidden))

    def forward_time_only(self, time: torch.Tensor) -> ANAFOutput:
        """Run the temporal-only scoring path used by the published implementation."""

        self._validate(time, "time")
        hidden = self.time_encoder(time, self._position(time))
        return self._output(hidden)
