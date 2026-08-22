"""Contrastive losses for DS-005 self-supervised bout representation learning.

Primary objective
-----------------
The first experiment uses temporal contrastive learning with an NT-Xent /
InfoNCE-style objective.

For each behavioral bout, two conservative augmented views are produced.
The encoder and projection head map each view to a representation. The loss
encourages the two views from the same bout to be close while treating other
bouts in the batch as negatives.

Expected inputs
---------------
z_a: Tensor of shape (batch, projection_dim)
z_b: Tensor of shape (batch, projection_dim)

The primary projection dimension is currently 64.

Important
---------
Downstream clustering should use the encoder embeddings, not the projection
head outputs used by this loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_TEMPERATURE = 0.10


def _validate_embeddings(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> None:
    """Validate paired contrastive embeddings."""

    if z_a.ndim != 2 or z_b.ndim != 2:
        raise ValueError(
            "Contrastive embeddings must be 2D tensors with shape (B, D)."
        )

    if z_a.shape != z_b.shape:
        raise ValueError(
            f"Paired embedding shapes must match; "
            f"got {tuple(z_a.shape)} and {tuple(z_b.shape)}."
        )

    if z_a.shape[0] < 2:
        raise ValueError(
            "Contrastive loss requires batch size >= 2 so negatives exist."
        )

    if z_a.shape[1] < 1:
        raise ValueError("Embedding dimension must be positive.")

    if not torch.isfinite(z_a).all():
        raise ValueError("z_a contains NaN or Inf.")

    if not torch.isfinite(z_b).all():
        raise ValueError("z_b contains NaN or Inf.")


def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Compute symmetric NT-Xent / InfoNCE loss.

    Parameters
    ----------
    z_a:
        Projection-head output for augmented view A, shape (B, D).
    z_b:
        Projection-head output for augmented view B, shape (B, D).
    temperature:
        Positive temperature scaling parameter.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Notes
    -----
    The implementation uses in-batch negatives. For batch size B, each anchor
    has exactly one positive counterpart and 2B - 2 negatives.

    All vectors are L2-normalized before cosine-similarity computation.
    """

    _validate_embeddings(z_a, z_b)

    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be numeric.")

    temperature = float(temperature)

    if not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite.")

    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    batch_size = z_a.shape[0]

    # Normalize so dot product is cosine similarity.
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    # Stack the two views:
    # [a_0 ... a_B-1, b_0 ... b_B-1]
    representations = torch.cat([z_a, z_b], dim=0)

    # Pairwise cosine similarities.
    logits = representations @ representations.T
    logits = logits / temperature

    # Remove self-similarity from the denominator.
    identity = torch.eye(
        2 * batch_size,
        device=logits.device,
        dtype=torch.bool,
    )
    logits = logits.masked_fill(identity, float("-inf"))

    # Positive partner index:
    # a_i -> b_i
    # b_i -> a_i
    targets = torch.arange(
        2 * batch_size,
        device=logits.device,
    )
    targets = (targets + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

    if not torch.isfinite(loss):
        raise RuntimeError("NT-Xent loss produced NaN or Inf.")

    return loss


class NTXentLoss(nn.Module):
    """nn.Module wrapper for the primary temporal contrastive objective."""

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()

        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be numeric.")

        temperature = float(temperature)

        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        return nt_xent_loss(
            z_a,
            z_b,
            temperature=self.temperature,
        )


__all__ = [
    "DEFAULT_TEMPERATURE",
    "NTXentLoss",
    "nt_xent_loss",
]
