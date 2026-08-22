"""Primary SSL encoder for DS-005 bout-level temporal representation learning.

Frozen design direction for the first experiment
------------------------------------------------
Primary SSL objective:
    Temporal contrastive learning.

Primary encoder family:
    Small 1D temporal convolutional neural network.

Expected input:
    Tensor of shape (batch, 175, 3)

Channels:
    0 = sin(orientation_smooth)
    1 = cos(orientation_smooth)
    2 = normalized speed_head

Primary embedding:
    64-dimensional vector per bout.

Architecture:
    Conv1d(3 -> 32, kernel=7)
    Conv1d(32 -> 64, kernel=5)
    Conv1d(64 -> 128, kernel=3)
    Global average pooling
    Linear(128 -> 64)

A small projection head is also provided for contrastive training. Clustering
and downstream analysis should use the encoder embedding, not the projection
head output.

The module intentionally avoids:
    - Transformers;
    - recurrent networks;
    - attention;
    - large architecture searches;
    - metadata inputs;
    - test-set-dependent design choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


EXPECTED_TIMESTEPS = 175
EXPECTED_INPUT_CHANNELS = 3
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_PROJECTION_DIM = 64


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the primary DS-005 temporal CNN encoder."""

    input_channels: int = EXPECTED_INPUT_CHANNELS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    projection_dim: int = DEFAULT_PROJECTION_DIM
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.input_channels != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"Primary Input B expects {EXPECTED_INPUT_CHANNELS} channels."
            )

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class ConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> GELU -> Dropout block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalConvEncoder(nn.Module):
    """Small 1D CNN producing one embedding per behavioral bout.

    Input shape
    -----------
    (batch, 175, 3)

    Output shape
    ------------
    (batch, embedding_dim)

    Notes
    -----
    PyTorch Conv1d expects (batch, channels, time), so the input is transposed
    internally from (B, T, C) to (B, C, T).
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or EncoderConfig()

        self.features = nn.Sequential(
            ConvBlock(
                self.config.input_channels,
                32,
                kernel_size=7,
                dropout=self.config.dropout,
            ),
            ConvBlock(
                32,
                64,
                kernel_size=5,
                dropout=self.config.dropout,
            ),
            ConvBlock(
                64,
                128,
                kernel_size=3,
                dropout=self.config.dropout,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Linear(
            128,
            self.config.embedding_dim,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with 3 dimensions (B, T, C), got {x.shape}."
            )

        if x.shape[1] != EXPECTED_TIMESTEPS:
            raise ValueError(
                f"Expected {EXPECTED_TIMESTEPS} timesteps, got {x.shape[1]}."
            )

        if x.shape[2] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} channels, "
                f"got {x.shape[2]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError("Encoder input contains NaN or Inf.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        z = self.embedding(x)

        if not torch.isfinite(z).all():
            raise RuntimeError("Encoder produced NaN or Inf.")

        return z


class ProjectionHead(nn.Module):
    """Projection head used only for contrastive training.

    The encoder embedding should be retained for downstream clustering.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0 or projection_dim <= 0:
            raise ValueError(
                "embedding_dim and projection_dim must be positive."
            )

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings (B, D), got {z.shape}."
            )

        projected = self.net(z)

        if not torch.isfinite(projected).all():
            raise RuntimeError("Projection head produced NaN or Inf.")

        return projected


class ContrastiveModel(nn.Module):
    """Convenience wrapper combining encoder and projection head."""

    def __init__(
        self,
        config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or EncoderConfig()
        self.encoder = TemporalConvEncoder(self.config)
        self.projector = ProjectionHead(
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return encoder embedding and projection-head output."""

        embedding = self.encoder(x)
        projection = self.projector(embedding)

        return embedding, projection


__all__ = [
    "ContrastiveModel",
    "EncoderConfig",
    "ProjectionHead",
    "TemporalConvEncoder",
]
