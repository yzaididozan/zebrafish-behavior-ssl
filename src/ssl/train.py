"""Reusable SSL training utilities for DS-005.

This module contains model-training mechanics only. Dataset loading and CLI
orchestration live in ``scripts/train_ssl.py``.

The held-out TEST partition must never be passed to these functions during
model selection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn


Batch = Tuple[torch.Tensor, torch.Tensor]


def _ensure_finite_scalar(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value}")


def assert_finite_gradients(model: nn.Module) -> None:
    """Raise if gradients are missing entirely or contain NaN/Inf."""
    found = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        found = True
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient detected in {name}.")
    if not found:
        raise RuntimeError("No gradients were produced.")


def train_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip_max_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """Train for one epoch over paired augmented-view batches."""
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    for view_a, view_b in batches:
        if view_a.ndim != 3 or view_b.ndim != 3:
            raise ValueError("Expected batched SSL views with shape (B, T, C).")
        if view_a.shape != view_b.shape:
            raise ValueError(
                f"View shapes differ: {tuple(view_a.shape)} vs {tuple(view_b.shape)}"
            )
        if view_a.shape[0] < 2:
            # NT-Xent requires negatives. Skip incomplete singleton batches.
            continue

        view_a = view_a.to(device=device, dtype=torch.float32)
        view_b = view_b.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        _, projection_a = model(view_a)
        _, projection_b = model(view_b)

        loss = loss_fn(projection_a, projection_b)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became NaN or Inf.")

        loss.backward()
        assert_finite_gradients(model)

        if gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_max_norm),
            )
            assert_finite_gradients(model)

        optimizer.step()

        batch_n = int(view_a.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_n
        total_examples += batch_n
        total_batches += 1

    if total_batches == 0 or total_examples == 0:
        raise RuntimeError("No valid training batches were produced.")

    mean_loss = total_loss / total_examples
    _ensure_finite_scalar(mean_loss, "mean training loss")

    return {
        "loss": mean_loss,
        "examples": float(total_examples),
        "batches": float(total_batches),
    }


@torch.no_grad()
def validate_one_epoch(
    *,
    model: nn.Module,
    batches: Iterable[Batch],
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate contrastive validation loss without parameter updates."""
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    for view_a, view_b in batches:
        if view_a.ndim != 3 or view_b.ndim != 3:
            raise ValueError("Expected batched SSL views with shape (B, T, C).")
        if view_a.shape != view_b.shape:
            raise ValueError(
                f"View shapes differ: {tuple(view_a.shape)} vs {tuple(view_b.shape)}"
            )
        if view_a.shape[0] < 2:
            continue

        view_a = view_a.to(device=device, dtype=torch.float32)
        view_b = view_b.to(device=device, dtype=torch.float32)

        _, projection_a = model(view_a)
        _, projection_b = model(view_b)

        loss = loss_fn(projection_a, projection_b)
        if not torch.isfinite(loss):
            raise RuntimeError("Validation loss became NaN or Inf.")

        batch_n = int(view_a.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_n
        total_examples += batch_n
        total_batches += 1

    if total_batches == 0 or total_examples == 0:
        raise RuntimeError("No valid validation batches were produced.")

    mean_loss = total_loss / total_examples
    _ensure_finite_scalar(mean_loss, "mean validation loss")

    return {
        "loss": mean_loss,
        "examples": float(total_examples),
        "batches": float(total_batches),
    }


def _config_to_dict(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return repr(config)


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_seed: int,
    train_loss: float,
    validation_loss: float,
    model_config: Any = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a reproducible SSL checkpoint."""
    _ensure_finite_scalar(train_loss, "train_loss")
    _ensure_finite_scalar(validation_loss, "validation_loss")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "training_seed": int(training_seed),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "model_config": _config_to_dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)

    torch.save(payload, path)

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint was not written correctly: {path}")


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location=map_location)

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint is missing model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise RuntimeError("Checkpoint is missing optimizer_state_dict.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"Non-finite parameter after checkpoint load: {name}"
            )

    return checkpoint
