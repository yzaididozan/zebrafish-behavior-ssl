#!/usr/bin/env python3
"""Full TRAIN/VALIDATION SSL training launcher for DS-005.

This script deliberately never requests the TEST partition.

Run from the repository root:

    PYTHONPATH=. python3 scripts/train_ssl.py

Run one seed only:

    PYTHONPATH=. python3 scripts/train_ssl.py --seed 11

For a short preflight run:

    PYTHONPATH=. python3 scripts/train_ssl.py \
        --seed 11 --epochs 2 --max-train-bouts 10000 --max-validation-bouts 2000

Requirements:
    PyYAML is used to read configs/ssl/training.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required to read configs/ssl/training.yaml.\n"
        "Install it inside the active .venv with:\n"
        "  python3 -m pip install pyyaml"
    ) from exc

from src.data.ds005 import DS005
from src.ssl.augmentations import make_two_views
from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.input import bout_to_ssl_input
from src.ssl.losses import NTXentLoss
from src.ssl.train import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_NORMALIZATION = REPO_ROOT / "configs" / "ssl" / "normalization.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DS-005 temporal contrastive SSL using TRAIN/VALIDATION only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to training YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run only one predefined seed instead of all configured seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional temporary epoch override.",
    )
    parser.add_argument(
        "--max-train-bouts",
        type=int,
        default=None,
        help="Optional TRAIN-bout cap for preflight/debug runs.",
    )
    parser.add_argument(
        "--max-validation-bouts",
        type=int,
        default=None,
        help="Optional VALIDATION-bout cap for preflight/debug runs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Training config must contain a YAML mapping: {path}")
    if "training" not in data:
        raise ValueError("Expected top-level 'training:' key.")
    return data["training"]


def load_speed_normalization(path: Path) -> Tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    def search(obj: Any) -> Tuple[Optional[float], Optional[float]]:
        if isinstance(obj, dict):
            mean = None
            std = None
            for key in ("speed_mean", "mean"):
                if isinstance(obj.get(key), (int, float)):
                    mean = float(obj[key])
                    break
            for key in ("speed_std", "std"):
                if isinstance(obj.get(key), (int, float)):
                    std = float(obj[key])
                    break
            if mean is not None and std is not None:
                return mean, std
            for value in obj.values():
                found_mean, found_std = search(value)
                if found_mean is not None and found_std is not None:
                    return found_mean, found_std
        return None, None

    mean, std = search(data)
    if mean is None or std is None:
        raise ValueError("Could not locate speed mean/std in normalization.json.")
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid speed normalization: mean={mean}, std={std}")

    return float(mean), float(std)


def normalize_ssl_input(
    x: np.ndarray,
    *,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    if x.shape != (175, 3):
        raise ValueError(f"Expected SSL input (175, 3), got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN/Inf before normalization.")

    x[:, 2] = (x[:, 2] - speed_mean) / speed_std

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN/Inf after normalization.")
    return x


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def paired_batches(
    *,
    dataset: DS005,
    partition: str,
    batch_size: int,
    speed_mean: float,
    speed_std: float,
    epoch: int,
    seed: int,
    max_bouts: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield deterministic augmented batches lazily from one partition.

    DS005.iter_bouts is intentionally called with an explicit TRAIN or
    VALIDATION partition. TEST is forbidden here.
    """
    if partition not in ("train", "validation"):
        raise ValueError(
            "Full training may only request train or validation; TEST is protected."
        )
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2.")

    view_a_batch: List[np.ndarray] = []
    view_b_batch: List[np.ndarray] = []

    count = 0

    for bout in dataset.iter_bouts(
        partition=partition,
        primary_qc_only=True,
        include_optional=False,
    ):
        x = bout_to_ssl_input(bout)
        x = normalize_ssl_input(
            x,
            speed_mean=speed_mean,
            speed_std=speed_std,
        )

        # Stable per-bout seed. Validation views are deterministic across runs;
        # training views vary by epoch.
        epoch_component = epoch if partition == "train" else 0
        aug_seed = (
            int(seed)
            + epoch_component * 1_000_003
            + count * 97
            + (0 if partition == "train" else 50_000_000)
        )

        view_a, view_b = make_two_views(x, seed=aug_seed)
        view_a_batch.append(np.asarray(view_a, dtype=np.float32))
        view_b_batch.append(np.asarray(view_b, dtype=np.float32))
        count += 1

        if len(view_a_batch) == batch_size:
            yield (
                torch.from_numpy(np.stack(view_a_batch)),
                torch.from_numpy(np.stack(view_b_batch)),
            )
            view_a_batch.clear()
            view_b_batch.clear()

        if max_bouts is not None and count >= max_bouts:
            break

    # Keep the final partial batch if it has at least 2 examples.
    if len(view_a_batch) >= 2:
        yield (
            torch.from_numpy(np.stack(view_a_batch)),
            torch.from_numpy(np.stack(view_b_batch)),
        )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_model(training: Dict[str, Any]) -> Tuple[ContrastiveModel, EncoderConfig]:
    encoder = training["encoder"]
    projection = training["projection_head"]

    config = EncoderConfig(
        input_channels=int(encoder["architecture"]["input_channels"]),
        embedding_dim=int(encoder["embedding_dim"]),
        projection_dim=int(projection["projection_dim"]),
        dropout=float(encoder["architecture"]["dropout"]),
    )
    return ContrastiveModel(config=config), config


def run_seed(
    *,
    training: Dict[str, Any],
    seed: int,
    epochs: int,
    max_train_bouts: Optional[int],
    max_validation_bouts: Optional[int],
    device: torch.device,
    speed_mean: float,
    speed_std: float,
) -> None:
    seed_everything(seed)

    optimization = training["optimization"]
    validation_cfg = training["validation"]
    checkpoint_cfg = training["checkpointing"]
    logging_cfg = training["logging"]

    batch_size = int(optimization["batch_size"])
    learning_rate = float(optimization["learning_rate"])
    weight_decay = float(optimization["weight_decay"])
    temperature = float(training["objective"]["temperature"])

    model, model_config = build_model(training)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = NTXentLoss(temperature=temperature)

    clip_cfg = optimization.get("gradient_clipping", {})
    clip_enabled = bool(clip_cfg.get("enabled", False))
    clip_max_norm = (
        float(clip_cfg["max_norm"]) if clip_enabled else None
    )

    checkpoint_dir = REPO_ROOT / checkpoint_cfg["directory"]
    log_dir = REPO_ROOT / logging_cfg["directory"]
    log_path = log_dir / f"ssl_seed{seed}.jsonl"

    # Start a fresh log for an explicit new run.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    best_validation_loss = float("inf")
    best_checkpoint: Optional[Path] = None
    epochs_without_improvement = 0

    early = validation_cfg.get("early_stopping", {})
    early_enabled = bool(early.get("enabled", False))
    patience = int(early.get("patience_epochs", 0))
    minimum_delta = float(early.get("minimum_delta", 0.0))

    commit = git_commit()

    print()
    print("=" * 72)
    print(f"SSL TRAINING — SEED {seed}")
    print("=" * 72)
    print("TRAIN partition: LOADED FOR FITTING")
    print("VALIDATION partition: LOADED FOR MODEL SELECTION")
    print("TEST partition: NOT LOADED")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    started = time.time()

    with DS005(
        repo_root=REPO_ROOT,
        validate=True,
        verify_split_hash=True,
    ) as dataset:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model=model,
                batches=paired_batches(
                    dataset=dataset,
                    partition="train",
                    batch_size=batch_size,
                    speed_mean=speed_mean,
                    speed_std=speed_std,
                    epoch=epoch,
                    seed=seed,
                    max_bouts=max_train_bouts,
                ),
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                gradient_clip_max_norm=clip_max_norm,
            )

            validation_metrics = validate_one_epoch(
                model=model,
                batches=paired_batches(
                    dataset=dataset,
                    partition="validation",
                    batch_size=batch_size,
                    speed_mean=speed_mean,
                    speed_std=speed_std,
                    epoch=epoch,
                    seed=seed,
                    max_bouts=max_validation_bouts,
                ),
                loss_fn=loss_fn,
                device=device,
            )

            train_loss = float(train_metrics["loss"])
            validation_loss = float(validation_metrics["loss"])

            record = {
                "epoch": epoch,
                "seed": seed,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_examples": int(train_metrics["examples"]),
                "validation_examples": int(validation_metrics["examples"]),
                "epoch_seconds": time.time() - epoch_start,
                "test_partition_loaded": False,
            }
            append_jsonl(log_path, record)

            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"train={train_loss:.6f} "
                f"validation={validation_loss:.6f} "
                f"({record['epoch_seconds']:.1f}s)"
            )

            improved = validation_loss < (
                best_validation_loss - minimum_delta
            )

            if improved:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0

                best_checkpoint = (
                    checkpoint_dir / f"ssl_seed{seed}_best.pt"
                )

                save_checkpoint(
                    path=best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    training_seed=seed,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    model_config=model_config,
                    extra_metadata={
                        "git_commit": commit,
                        "dataset_id": training["dataset"]["id"],
                        "split_seed": training["dataset"]["split_seed"],
                        "training_config_version": training["version"],
                        "test_partition_loaded": False,
                    },
                )
            else:
                epochs_without_improvement += 1

            if early_enabled and epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {epoch} epochs; "
                    f"no validation improvement for {patience} epochs."
                )
                break

        last_checkpoint = checkpoint_dir / f"ssl_seed{seed}_last.pt"
        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            training_seed=seed,
            train_loss=train_loss,
            validation_loss=validation_loss,
            model_config=model_config,
            extra_metadata={
                "git_commit": commit,
                "dataset_id": training["dataset"]["id"],
                "split_seed": training["dataset"]["split_seed"],
                "training_config_version": training["version"],
                "test_partition_loaded": False,
            },
        )

    if best_checkpoint is None:
        raise RuntimeError("No best checkpoint was selected.")

    # Mechanical verification that the selected checkpoint can be restored.
    reloaded_model, _ = build_model(training)
    reloaded_model = reloaded_model.to(device)
    load_checkpoint(
        path=best_checkpoint,
        model=reloaded_model,
        optimizer=None,
        map_location=device,
    )

    print()
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Last checkpoint: {last_checkpoint}")
    print(f"Log: {log_path}")
    print(f"Checkpoint reload: PASSED")
    print(f"TEST partition used: NO")
    print(f"Seed elapsed seconds: {time.time() - started:.1f}")


def main() -> None:
    args = parse_args()
    training = load_yaml(args.config)

    speed_mean, speed_std = load_speed_normalization(
        DEFAULT_NORMALIZATION
    )

    configured_seeds: Sequence[int] = training["seeds"]["values"]
    seeds = [int(s) for s in configured_seeds]

    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(
                f"Seed {args.seed} is not in predefined seed policy: {seeds}"
            )
        seeds = [args.seed]

    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(training["optimization"]["epochs"])
    )

    if epochs < 1:
        raise ValueError("epochs must be >= 1.")

    device = choose_device(args.device)

    print("=" * 72)
    print("DS-005 FULL SSL TRAINING")
    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Config status: {training.get('status', 'UNKNOWN')}")
    print(f"Seeds: {seeds}")
    print(f"Speed mean: {speed_mean:.12f}")
    print(f"Speed std:  {speed_std:.12f}")
    print("TEST partition: PROTECTED / NOT LOADED")

    if args.max_train_bouts is not None or args.max_validation_bouts is not None:
        print(
            "WARNING: bout caps are active; this is a preflight/debug run, "
            "not the final full-data training run."
        )

    for seed in seeds:
        run_seed(
            training=training,
            seed=seed,
            epochs=epochs,
            max_train_bouts=args.max_train_bouts,
            max_validation_bouts=args.max_validation_bouts,
            device=device,
            speed_mean=speed_mean,
            speed_std=speed_std,
        )


if __name__ == "__main__":
    main()
