#!/usr/bin/env python3
"""Export DS-005 SSL encoder embeddings from frozen best checkpoints.

Default behavior
----------------
Exports TRAIN and VALIDATION embeddings for all configured SSL seeds.

The held-out TEST partition is NOT exported by default.

Examples
--------
Export all TRAIN/VALIDATION embeddings:

    PYTHONPATH=. python3 scripts/export_ssl_embeddings.py

Export one seed:

    PYTHONPATH=. python3 scripts/export_ssl_embeddings.py --seed 11

Preflight export:

    PYTHONPATH=. python3 scripts/export_ssl_embeddings.py \
        --seed 11 \
        --max-bouts 2000

Explicit TEST export, only after final design freeze:

    PYTHONPATH=. python3 scripts/export_ssl_embeddings.py \
        --seed 11 \
        --partitions test \
        --allow-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required.\n"
        "Install it inside the active .venv with:\n"
        "  python3 -m pip install pyyaml"
    ) from exc

from src.data.ds005 import DS005
from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.export_embeddings import (
    encode_batch,
    normalize_ssl_input,
    save_embedding_partition,
)
from src.ssl.input import bout_to_ssl_input
from src.ssl.train import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
NORMALIZATION_CONFIG = REPO_ROOT / "configs" / "ssl" / "normalization.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DS-005 encoder embeddings from best SSL checkpoints."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Export one predefined seed instead of all configured seeds.",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="Partitions to export. Default: train validation.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required to export TEST. Use only after analysis design is frozen.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Encoder inference batch size.",
    )
    parser.add_argument(
        "--max-bouts",
        type=int,
        default=None,
        help="Optional per-partition cap for preflight/debug export.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "training" not in data:
        raise ValueError("Expected top-level training: mapping.")

    return data["training"]


def load_speed_normalization(path: Path) -> Tuple[float, float]:
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
        raise ValueError("Could not find speed mean/std in normalization config.")

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Invalid speed normalization values: mean={mean}, std={std}"
        )

    return float(mean), float(std)


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


def checkpoint_for_seed(training: Dict[str, Any], seed: int) -> Path:
    checkpoint_dir = REPO_ROOT / training["checkpointing"]["directory"]
    return checkpoint_dir / f"ssl_seed{seed}_best.pt"


def safe_getattr(obj: Any, name: str, default: Any = "") -> Any:
    value = getattr(obj, name, default)
    return default if value is None else value


def bout_metadata_row(
    bout: Any,
    *,
    partition: str,
    seed: int,
    row_index: int,
) -> Dict[str, Any]:
    """Extract non-input metadata for downstream auditing/validation."""
    key = safe_getattr(bout, "key", None)

    fish_id = safe_getattr(bout, "fish_id", "")
    bout_index = safe_getattr(bout, "bout_index", "")
    session_id = safe_getattr(bout, "session_id", "")

    if key is not None:
        fish_id = safe_getattr(key, "fish_id", fish_id)
        bout_index = safe_getattr(key, "bout_index", bout_index)
        session_id = safe_getattr(key, "session_id", session_id)

    return {
        "row_index": int(row_index),
        "partition": partition,
        "training_seed": int(seed),
        "fish_id": fish_id,
        "session_id": session_id,
        "bout_index": bout_index,
        "context_id": safe_getattr(bout, "context_id", ""),
        "context_name": safe_getattr(bout, "context_name", ""),
        "bout_type": safe_getattr(bout, "bout_type", ""),
        "stimulus_code": safe_getattr(bout, "stimulus_code", ""),
    }


def expected_partition_rows(training: Dict[str, Any], partition: str) -> Optional[int]:
    # Known frozen DS-005 bout counts for current processed baseline artifacts.
    known = {
        "train": 842841,
        "validation": 168464,
        "test": None,
    }
    return known.get(partition)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def export_partition(
    *,
    dataset: DS005,
    model: ContrastiveModel,
    device: torch.device,
    partition: str,
    seed: int,
    batch_size: int,
    max_bouts: Optional[int],
    speed_mean: float,
    speed_std: float,
    output_dir: Path,
    checkpoint_path: Path,
    split_seed: int,
    expected_rows: Optional[int],
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    rows: List[Dict[str, Any]] = []
    embedding_chunks: List[np.ndarray] = []
    input_batch: List[np.ndarray] = []

    count = 0
    started = time.time()

    if max_bouts is not None and expected_rows is not None:
        total_rows = min(max_bouts, expected_rows)
    elif max_bouts is not None:
        total_rows = max_bouts
    else:
        total_rows = expected_rows

    def flush_batch() -> None:
        nonlocal input_batch

        if not input_batch:
            return

        tensor = torch.from_numpy(
            np.stack(input_batch).astype(np.float32, copy=False)
        )

        embeddings = encode_batch(
            model=model,
            batch=tensor,
            device=device,
        )

        embedding_chunks.append(
            embeddings.numpy().astype(np.float32, copy=False)
        )
        input_batch = []

    print()
    print(f"Exporting {partition.upper()} embeddings...")
    print(f"TEST partition active: {'YES' if partition == 'test' else 'NO'}")

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

        input_batch.append(x)

        rows.append(
            bout_metadata_row(
                bout,
                partition=partition,
                seed=seed,
                row_index=count,
            )
        )

        count += 1

        if len(input_batch) >= batch_size:
            flush_batch()

            elapsed = max(time.time() - started, 1e-9)
            rate = count / elapsed

            if total_rows:
                fraction = min(count / total_rows, 1.0)
                remaining = max(total_rows - count, 0)
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"  {count:,}/{total_rows:,} "
                    f"({100*fraction:5.1f}%) | "
                    f"{rate:,.0f} bouts/s | "
                    f"ETA {eta/60:.1f} min",
                    flush=True,
                )
            else:
                print(
                    f"  {count:,} bouts | {rate:,.0f} bouts/s",
                    flush=True,
                )

        if max_bouts is not None and count >= max_bouts:
            break

    flush_batch()

    if not embedding_chunks:
        raise RuntimeError(f"No embeddings produced for partition={partition}.")

    embeddings = np.concatenate(embedding_chunks, axis=0)

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            "Embedding count does not match metadata row count."
        )

    embedding_path, metadata_path, manifest_path = save_embedding_partition(
        output_dir=output_dir,
        partition=partition,
        seed=seed,
        embeddings=embeddings,
        metadata_rows=rows,
        checkpoint_path=checkpoint_path,
        split_seed=split_seed,
    )

    checksum_path = output_dir / f"ssl_seed{seed}_{partition}_SHA256SUMS"

    checksum_lines = []
    for artifact in (embedding_path, metadata_path, manifest_path):
        checksum_lines.append(
            f"{sha256_file(artifact)}  {artifact.name}"
        )

    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Export complete: {partition}")
    print(f"  rows:          {embeddings.shape[0]:,}")
    print(f"  embedding dim: {embeddings.shape[1]}")
    print(f"  embeddings:    {embedding_path}")
    print(f"  metadata:      {metadata_path}")
    print(f"  manifest:      {manifest_path}")
    print(f"  checksums:     {checksum_path}")


def main() -> None:
    args = parse_args()

    requested_partitions = list(args.partitions)

    if "test" in requested_partitions and not args.allow_test:
        raise SystemExit(
            "Refusing to export TEST without --allow-test.\n"
            "Keep TEST untouched until all discovery/comparison rules are frozen."
        )

    training = load_yaml(TRAINING_CONFIG)
    configured_seeds = [int(x) for x in training["seeds"]["values"]]

    if args.seed is None:
        seeds = configured_seeds
    else:
        if args.seed not in configured_seeds:
            raise ValueError(
                f"Seed {args.seed} is not in predefined policy: {configured_seeds}"
            )
        seeds = [args.seed]

    speed_mean, speed_std = load_speed_normalization(
        NORMALIZATION_CONFIG
    )
    device = choose_device(args.device)

    print("=" * 72)
    print("DS-005 SSL EMBEDDING EXPORT")
    print("=" * 72)
    print(f"Config status: {training.get('status', 'UNKNOWN')}")
    print(f"Seeds: {seeds}")
    print(f"Partitions: {requested_partitions}")
    print(f"Device: {device}")
    print(f"Speed mean: {speed_mean:.12f}")
    print(f"Speed std:  {speed_std:.12f}")
    print(
        "TEST partition status: "
        + ("EXPLICITLY REQUESTED" if "test" in requested_partitions else "NOT LOADED")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        checkpoint_path = checkpoint_for_seed(training, seed)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found for seed {seed}: {checkpoint_path}\n"
                "Wait for that seed's training run to finish before exporting."
            )

        model, _model_config = build_model(training)
        model = model.to(device)

        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            map_location=device,
        )

        checkpoint_metadata = checkpoint.get("metadata", {})
        if checkpoint_metadata.get("test_partition_loaded") is True:
            raise RuntimeError(
                f"Checkpoint for seed {seed} indicates TEST was used."
            )

        print()
        print("-" * 72)
        print(f"Seed {seed}")
        print("-" * 72)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Selected epoch: {checkpoint.get('epoch', 'UNKNOWN')}")
        print(
            f"Validation loss: "
            f"{checkpoint.get('validation_loss', 'UNKNOWN')}"
        )

        with DS005(
            repo_root=REPO_ROOT,
            validate=True,
            verify_split_hash=True,
        ) as dataset:
            for partition in requested_partitions:
                export_partition(
                    dataset=dataset,
                    model=model,
                    device=device,
                    partition=partition,
                    seed=seed,
                    batch_size=args.batch_size,
                    max_bouts=args.max_bouts,
                    speed_mean=speed_mean,
                    speed_std=speed_std,
                    output_dir=args.output_dir,
                    checkpoint_path=checkpoint_path,
                    split_seed=int(training["dataset"]["split_seed"]),
                    expected_rows=expected_partition_rows(
                        training,
                        partition,
                    ),
                )

    print()
    print("=" * 72)
    print("EMBEDDING EXPORT COMPLETE")
    print("=" * 72)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
