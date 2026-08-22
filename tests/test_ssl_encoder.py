"""Tests for the primary DS-005 SSL temporal CNN encoder."""

from __future__ import annotations

import pytest
import torch

from src.ssl.encoder import (
    ContrastiveModel,
    EncoderConfig,
    ProjectionHead,
    TemporalConvEncoder,
)


def make_batch(
    batch_size: int = 8,
    timesteps: int = 175,
    channels: int = 3,
) -> torch.Tensor:
    """Create a finite synthetic batch matching Input B shape."""
    torch.manual_seed(20260822)
    return torch.randn(
        batch_size,
        timesteps,
        channels,
        dtype=torch.float32,
    )


def test_encoder_output_shape() -> None:
    x = make_batch(batch_size=8)

    model = TemporalConvEncoder(
        EncoderConfig(
            embedding_dim=64,
            projection_dim=64,
            dropout=0.0,
        )
    )
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert z.shape == (8, 64)


def test_encoder_output_is_finite() -> None:
    x = make_batch(batch_size=8)

    model = TemporalConvEncoder(
        EncoderConfig(dropout=0.0)
    )
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert torch.isfinite(z).all()


def test_encoder_accepts_single_bout_batch() -> None:
    x = make_batch(batch_size=1)

    model = TemporalConvEncoder(
        EncoderConfig(dropout=0.0)
    )
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert z.shape == (1, 64)
    assert torch.isfinite(z).all()


def test_encoder_rejects_wrong_timestep_count() -> None:
    x = make_batch(
        batch_size=4,
        timesteps=174,
        channels=3,
    )

    model = TemporalConvEncoder()

    with pytest.raises(ValueError):
        model(x)


def test_encoder_rejects_wrong_channel_count() -> None:
    x = make_batch(
        batch_size=4,
        timesteps=175,
        channels=4,
    )

    model = TemporalConvEncoder()

    with pytest.raises(ValueError):
        model(x)


def test_encoder_rejects_wrong_rank() -> None:
    x = torch.randn(175, 3)

    model = TemporalConvEncoder()

    with pytest.raises(ValueError):
        model(x)


def test_encoder_rejects_nonfinite_input() -> None:
    x = make_batch(batch_size=4)
    x[0, 0, 0] = float("nan")

    model = TemporalConvEncoder()

    with pytest.raises(ValueError):
        model(x)


def test_projection_head_output_shape() -> None:
    z = torch.randn(8, 64)

    head = ProjectionHead(
        embedding_dim=64,
        projection_dim=64,
    )
    head.eval()

    with torch.no_grad():
        p = head(z)

    assert p.shape == (8, 64)
    assert torch.isfinite(p).all()


def test_projection_head_rejects_wrong_rank() -> None:
    z = torch.randn(8, 1, 64)

    head = ProjectionHead()

    with pytest.raises(ValueError):
        head(z)


def test_contrastive_model_returns_embedding_and_projection() -> None:
    x = make_batch(batch_size=8)

    model = ContrastiveModel(
        EncoderConfig(
            embedding_dim=64,
            projection_dim=64,
            dropout=0.0,
        )
    )
    model.eval()

    with torch.no_grad():
        embedding, projection = model(x)

    assert embedding.shape == (8, 64)
    assert projection.shape == (8, 64)

    assert torch.isfinite(embedding).all()
    assert torch.isfinite(projection).all()


def test_eval_mode_is_deterministic() -> None:
    x = make_batch(batch_size=8)

    model = TemporalConvEncoder(
        EncoderConfig(dropout=0.10)
    )
    model.eval()

    with torch.no_grad():
        z1 = model(x)
        z2 = model(x)

    assert torch.equal(z1, z2)


def test_custom_embedding_dimension() -> None:
    x = make_batch(batch_size=5)

    model = TemporalConvEncoder(
        EncoderConfig(
            embedding_dim=32,
            projection_dim=64,
            dropout=0.0,
        )
    )
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert z.shape == (5, 32)


def test_encoder_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        EncoderConfig(dropout=1.0)


def test_encoder_config_rejects_invalid_input_channels() -> None:
    with pytest.raises(ValueError):
        EncoderConfig(input_channels=4)


def test_backward_pass_produces_gradients() -> None:
    x = make_batch(batch_size=8)

    model = TemporalConvEncoder(
        EncoderConfig(dropout=0.0)
    )
    model.train()

    z = model(x)
    loss = (z ** 2).mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients if grad is not None)
