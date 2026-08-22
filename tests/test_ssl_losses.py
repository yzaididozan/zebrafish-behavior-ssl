"""Tests for DS-005 contrastive losses."""

from __future__ import annotations

import pytest
import torch

from src.ssl.losses import (
    DEFAULT_TEMPERATURE,
    NTXentLoss,
    nt_xent_loss,
)


def make_embeddings(
    batch_size: int = 8,
    dim: int = 64,
    *,
    seed: int = 20260822,
) -> torch.Tensor:
    """Create deterministic finite embeddings."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, dim, dtype=torch.float32)


def test_nt_xent_returns_scalar_finite_loss() -> None:
    z_a = make_embeddings()
    z_b = make_embeddings(seed=20260823)

    loss = nt_xent_loss(z_a, z_b)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_nt_xent_identical_pairs_have_lower_loss_than_shuffled_pairs() -> None:
    z_a = make_embeddings(batch_size=16)
    z_b = z_a.clone()

    identical_loss = nt_xent_loss(z_a, z_b)

    permutation = torch.randperm(z_b.shape[0])
    shuffled_loss = nt_xent_loss(z_a, z_b[permutation])

    assert identical_loss < shuffled_loss


def test_nt_xent_is_symmetric() -> None:
    z_a = make_embeddings(seed=1)
    z_b = make_embeddings(seed=2)

    loss_ab = nt_xent_loss(z_a, z_b)
    loss_ba = nt_xent_loss(z_b, z_a)

    assert torch.allclose(loss_ab, loss_ba, atol=1e-6)


def test_nt_xent_temperature_changes_loss() -> None:
    z_a = make_embeddings(seed=3)
    z_b = make_embeddings(seed=4)

    low_temp = nt_xent_loss(z_a, z_b, temperature=0.05)
    high_temp = nt_xent_loss(z_a, z_b, temperature=0.50)

    assert torch.isfinite(low_temp)
    assert torch.isfinite(high_temp)
    assert not torch.allclose(low_temp, high_temp)


def test_nt_xent_rejects_mismatched_shapes() -> None:
    z_a = make_embeddings(batch_size=8, dim=64)
    z_b = make_embeddings(batch_size=7, dim=64)

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b)


def test_nt_xent_rejects_wrong_rank() -> None:
    z_a = torch.randn(8, 1, 64)
    z_b = torch.randn(8, 1, 64)

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b)


def test_nt_xent_rejects_batch_size_one() -> None:
    z_a = make_embeddings(batch_size=1)
    z_b = make_embeddings(batch_size=1, seed=9)

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b)


def test_nt_xent_rejects_nonfinite_input() -> None:
    z_a = make_embeddings()
    z_b = make_embeddings(seed=10)
    z_a[0, 0] = float("nan")

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b)


def test_nt_xent_rejects_zero_temperature() -> None:
    z_a = make_embeddings()
    z_b = make_embeddings(seed=11)

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b, temperature=0.0)


def test_nt_xent_rejects_negative_temperature() -> None:
    z_a = make_embeddings()
    z_b = make_embeddings(seed=12)

    with pytest.raises(ValueError):
        nt_xent_loss(z_a, z_b, temperature=-0.1)


def test_module_wrapper_matches_function() -> None:
    z_a = make_embeddings(seed=13)
    z_b = make_embeddings(seed=14)

    module = NTXentLoss(
        temperature=DEFAULT_TEMPERATURE
    )

    loss_module = module(z_a, z_b)
    loss_function = nt_xent_loss(
        z_a,
        z_b,
        temperature=DEFAULT_TEMPERATURE,
    )

    assert torch.allclose(
        loss_module,
        loss_function,
        atol=1e-7,
    )


def test_module_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        NTXentLoss(temperature=0.0)


def test_backward_pass_produces_finite_gradients() -> None:
    z_a = make_embeddings(seed=15).requires_grad_(True)
    z_b = make_embeddings(seed=16).requires_grad_(True)

    loss = nt_xent_loss(z_a, z_b)
    loss.backward()

    assert z_a.grad is not None
    assert z_b.grad is not None

    assert torch.isfinite(z_a.grad).all()
    assert torch.isfinite(z_b.grad).all()


def test_identical_pairs_do_not_produce_nan_gradients() -> None:
    z_a = make_embeddings(seed=17).requires_grad_(True)
    z_b = z_a.clone().detach().requires_grad_(True)

    loss = nt_xent_loss(z_a, z_b)
    loss.backward()

    assert torch.isfinite(loss)
    assert z_a.grad is not None
    assert z_b.grad is not None
    assert torch.isfinite(z_a.grad).all()
    assert torch.isfinite(z_b.grad).all()


def test_projection_dimension_one_is_allowed() -> None:
    z_a = make_embeddings(batch_size=4, dim=1)
    z_b = make_embeddings(batch_size=4, dim=1, seed=18)

    loss = nt_xent_loss(z_a, z_b)

    assert torch.isfinite(loss)
