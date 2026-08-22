#!/usr/bin/env python3
"""TRAIN-only smoke test for DS-005 temporal contrastive SSL.

Purpose
-------
Verify that the SSL training pipeline is wired correctly before full training.

This script:
- loads TRAIN bouts only,
- converts each bout to the primary SSL input representation,
- applies frozen TRAIN-only normalization,
- creates two independent augmented views,
- trains the 1D CNN contrastive model on a small subset,
- checks finite loss and finite gradients,
- verifies model parameters update,
- saves a checkpoint,
- reloads the checkpoint successfully.

The held-out TEST partition is never loaded.

Run from the repository root:

    PYTHONPATH=. python3 scripts/smoke_train_ssl.py

Optional arguments:

    PYTHONPATH=. python3 scripts/smoke_train_ssl.py \
        --max-bouts 2048 \
        --epochs 3 \
        --batch-size 256 \
        --seed 11
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.ds005 import DS005
from src.ssl.augmentations import make_two_views
from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.input import bout_to_ssl_input
from src.ssl.losses import NTXentLoss


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_PATH = REPO_ROOT / "configs" / "ssl" / "normalization.json"
CHECKPOINT_DIR = REPO_ROOT / "results" / "ssl" / "smoke_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRAIN-only smoke test for DS-005 SSL."
    )
    parser.add_argument(
        "--max-bouts",
        type=int,
        default=2048,
        help="Maximum number of TRAIN bouts to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of smoke-test epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.10,
        help="NT-Xent temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Training seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep at 0 for maximum portability.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_speed_normalization(
    path: Path = NORMALIZATION_PATH,
) -> Tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    candidates: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        candidates.append(data)

        for key in (
            "speed",
            "speed_head",
            "normalization",
            "statistics",
            "stats",
        ):
            value = data.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    mean = None
    std = None

    # Support both the flat file previously generated for this project
    # and common nested variants.
    for block in candidates:
        if mean is None:
            for key in ("speed_mean", "mean"):
                if key in block and isinstance(block[key], (int, float)):
                    mean = float(block[key])
                    break

        if std is None:
            for key in ("speed_std", "std"):
                if key in block and isinstance(block[key], (int, float)):
                    std = float(block[key])
                    break

        if mean is not None and std is not None:
            break

    if mean is None or std is None:
        # Search one additional nested level.
        for block in candidates:
            for value in block.values():
                if not isinstance(value, dict):
                    continue

                if mean is None:
                    for key in ("speed_mean", "mean"):
                        if key in value and isinstance(
                            value[key], (int, float)
                        ):
                            mean = float(value[key])
                            break

                if std is None:
                    for key in ("speed_std", "std"):
                        if key in value and isinstance(
                            value[key], (int, float)
                        ):
                            std = float(value[key])
                            break

                if mean is not None and std is not None:
                    break

    if mean is None or std is None:
        raise ValueError(
            "Could not find speed mean/std in normalization.json."
        )

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Invalid speed normalization values: mean={mean}, std={std}"
        )

    return mean, std


def normalize_ssl_input(
    x: np.ndarray,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    """Apply frozen TRAIN-only normalization to primary SSL input."""

    x = np.asarray(x, dtype=np.float32).copy()

    if x.shape != (175, 3):
        raise ValueError(
            f"Expected SSL input shape (175, 3), got {x.shape}."
        )

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN or Inf before normalization.")

    # Orientation sin/cos channels remain unchanged.
    # Speed channel receives frozen TRAIN-only z-score normalization.
    x[:, 2] = (x[:, 2] - speed_mean) / speed_std

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN or Inf after normalization.")

    return x


class SmokeBoutDataset(Dataset):
    """In-memory TRAIN-only subset used for the smoke test."""

    def __init__(
        self,
        samples: List[np.ndarray],
        *,
        base_seed: int,
    ) -> None:
        if len(samples) < 2:
            raise ValueError("Smoke test needs at least 2 bouts.")

        self.samples = samples
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.samples[index]

        # Deterministic but epoch-varying augmentation seeds.
        # This makes reruns reproducible while allowing new views each epoch.
        seed = (
            self.base_seed
            + self.epoch * 1_000_003
            + index * 97
        )

        view_a, view_b = make_two_views(
            x,
            seed=seed,
        )

        view_a = np.asarray(view_a, dtype=np.float32)
        view_b = np.asarray(view_b, dtype=np.float32)

        if view_a.shape != (175, 3):
            raise ValueError(
                f"Augmented view A has wrong shape: {view_a.shape}"
            )

        if view_b.shape != (175, 3):
            raise ValueError(
                f"Augmented view B has wrong shape: {view_b.shape}"
            )

        if not np.isfinite(view_a).all():
            raise ValueError("Augmented view A contains NaN or Inf.")

        if not np.isfinite(view_b).all():
            raise ValueError("Augmented view B contains NaN or Inf.")

        return (
            torch.from_numpy(view_a),
            torch.from_numpy(view_b),
        )


def collect_train_samples(
    max_bouts: int,
    speed_mean: float,
    speed_std: float,
) -> List[np.ndarray]:
    """Collect a small subset of primary-QC TRAIN bouts only."""

    if max_bouts < 2:
        raise ValueError("--max-bouts must be >= 2.")

    samples: List[np.ndarray] = []

    print("Loading TRAIN bouts only...")
    print("TEST partition status: NOT LOADED")

    with DS005(
        repo_root=REPO_ROOT,
        validate=True,
        verify_split_hash=True,
    ) as dataset:
        for bout in dataset.iter_bouts(
            partition="train",
            primary_qc_only=True,
            include_optional=False,
        ):
            x = bout_to_ssl_input(bout)
            x = normalize_ssl_input(
                x,
                speed_mean=speed_mean,
                speed_std=speed_std,
            )

            samples.append(x)

            if len(samples) >= max_bouts:
                break

    if len(samples) < 2:
        raise RuntimeError(
            f"Only collected {len(samples)} TRAIN bouts."
        )

    return samples


def clone_trainable_parameters(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def parameters_changed(
    before: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> bool:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        old = before[name]
        new = param.detach().cpu()

        if not torch.equal(old, new):
            return True

    return False


def assert_finite_gradients(
    model: torch.nn.Module,
) -> None:
    found_gradient = False

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        found_gradient = True

        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected in parameter: {name}"
            )

    if not found_gradient:
        raise RuntimeError("No gradients were produced.")


def train_one_epoch(
    *,
    model: ContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NTXentLoss,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    batches = 0

    for view_a, view_b in loader:
        view_a = view_a.to(
            device=device,
            dtype=torch.float32,
        )
        view_b = view_b.to(
            device=device,
            dtype=torch.float32,
        )

        optimizer.zero_grad(set_to_none=True)

        _, projection_a = model(view_a)
        _, projection_b = model(view_b)

        loss = loss_fn(
            projection_a,
            projection_b,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became NaN or Inf.")

        loss.backward()

        assert_finite_gradients(model)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        assert_finite_gradients(model)

        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1

    if batches == 0:
        raise RuntimeError("No training batches were produced.")

    return total_loss / batches


def save_checkpoint(
    *,
    path: Path,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "smoke_test": True,
        "dataset_id": "DS-005",
        "partition": "train",
        "test_partition_loaded": False,
        "epoch": int(epoch),
        "loss": float(loss),
        "training_seed": int(args.seed),
        "max_bouts": int(args.max_bouts),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        path,
    )

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Checkpoint was not written correctly: {path}"
        )


def verify_checkpoint_reload(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "loss",
        "partition",
        "test_partition_loaded",
    }

    missing = required_keys.difference(
        checkpoint.keys()
    )

    if missing:
        raise RuntimeError(
            f"Checkpoint missing keys: {sorted(missing)}"
        )

    if checkpoint["partition"] != "train":
        raise RuntimeError(
            "Smoke-test checkpoint partition is not TRAIN."
        )

    if checkpoint["test_partition_loaded"]:
        raise RuntimeError(
            "Checkpoint incorrectly indicates TEST was loaded."
        )

    reloaded_model = ContrastiveModel(
        config=EncoderConfig(
            input_channels=3,
            embedding_dim=64,
            projection_dim=64,
            dropout=0.10,
        )
    ).to(device)

    reloaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    reloaded_model.eval()

    # If loading state dict succeeds, checkpoint serialization is wired
    # correctly for this architecture.
    for param in reloaded_model.parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(
                "Reloaded checkpoint contains non-finite parameters."
            )


def main() -> None:
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")

    seed_everything(args.seed)

    print("=" * 72)
    print("DS-005 SSL SMOKE TEST")
    print("=" * 72)
    print("Purpose: pipeline verification only")
    print("Training partition: TRAIN")
    print("Validation partition: NOT LOADED")
    print("TEST partition: NOT LOADED")
    print(f"Seed: {args.seed}")
    print(f"Requested bouts: {args.max_bouts}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    speed_mean, speed_std = load_speed_normalization()

    print("Frozen TRAIN-only normalization:")
    print(f"  speed_mean = {speed_mean:.12f}")
    print(f"  speed_std  = {speed_std:.12f}")
    print()

    samples = collect_train_samples(
        max_bouts=args.max_bouts,
        speed_mean=speed_mean,
        speed_std=speed_std,
    )

    print(f"Collected TRAIN bouts: {len(samples):,}")
    print()

    dataset = SmokeBoutDataset(
        samples,
        base_seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=generator,
    )

    if len(loader) == 0:
        raise RuntimeError(
            "No batches available. Reduce --batch-size or increase --max-bouts."
        )

    # Prefer Apple Metal when available, otherwise CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Batches per epoch: {len(loader)}")
    print()

    model = ContrastiveModel(
        config=EncoderConfig(
            input_channels=3,
            embedding_dim=64,
            projection_dim=64,
            dropout=0.10,
        )
    ).to(device)

    loss_fn = NTXentLoss(
        temperature=args.temperature
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    before = clone_trainable_parameters(model)

    epoch_losses: List[float] = []

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)

        epoch_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        if not np.isfinite(epoch_loss):
            raise RuntimeError(
                f"Epoch {epoch} produced non-finite mean loss."
            )

        epoch_losses.append(epoch_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"mean_train_loss={epoch_loss:.6f}"
        )

    elapsed = time.time() - start

    changed = parameters_changed(
        before,
        model,
    )

    if not changed:
        raise RuntimeError(
            "Model parameters did not change during training."
        )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"smoke_ssl_seed{args.seed}.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=epoch_losses[-1],
        args=args,
    )

    verify_checkpoint_reload(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Smoke test is intentionally conservative:
    # we require no catastrophic divergence, not strict monotonic decrease.
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]

    divergence_limit = initial_loss * 2.0

    if final_loss > divergence_limit:
        raise RuntimeError(
            "Smoke-test loss appears to diverge: "
            f"initial={initial_loss:.6f}, final={final_loss:.6f}"
        )

    print()
    print("=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)
    print(f"TRAIN bouts used: {len(samples):,}")
    print("VALIDATION used: NO")
    print("TEST used: NO")
    print(f"Initial epoch loss: {initial_loss:.6f}")
    print(f"Final epoch loss:   {final_loss:.6f}")
    print(f"Parameters updated: YES")
    print(f"Finite gradients:   YES")
    print(f"Checkpoint written: {checkpoint_path}")
    print("Checkpoint reload:  PASSED")
    print(f"Elapsed seconds:    {elapsed:.2f}")
    print()
    print(
        "Note: a smoke test verifies training mechanics; "
        "it is not model-selection evidence."
    )


if __name__ == "__main__":
    main()
