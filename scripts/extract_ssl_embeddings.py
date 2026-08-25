#!/usr/bin/env python3
"""Extract DS-005 encoder embeddings from the frozen best SSL checkpoints.

Purpose
-------
Create one 64-dimensional encoder embedding per primary-QC-valid DS-005 bout
for TRAIN and VALIDATION only.

Safety rules
------------
- Loads only TRAIN and VALIDATION.
- There is no CLI option for TEST.
- Explicitly rejects any partition other than TRAIN/VALIDATION.
- Loads only ``ssl_seed*_best.pt`` checkpoints.
- Calls ``model.encoder(...)`` directly.
- Never saves projection-head outputs.
- Preserves stable bout/fish/session/context and timing/speed metadata.
- Verifies the frozen fish split through the canonical DS005 loader.
- Refuses checkpoints whose metadata indicate TEST was used.

Default output
--------------
data/processed/DS-005/ssl/
    seed11/
        train_embeddings.npz
        train_metadata.csv
        train_manifest.json
        train_SHA256SUMS
        validation_embeddings.npz
        validation_metadata.csv
        validation_manifest.json
        validation_SHA256SUMS
    seed23/
        ...
    seed37/
        ...
    seed51/
        ...
    seed79/
        ...

Usage
-----
From repository root:

    PYTHONPATH=. python3 scripts/extract_ssl_embeddings.py

One seed only:

    PYTHONPATH=. python3 scripts/extract_ssl_embeddings.py --seed 11

Small TRAIN/VALIDATION smoke export:

    PYTHONPATH=. python3 scripts/extract_ssl_embeddings.py \
        --seed 11 \
        --max-bouts 2000

The TEST partition is intentionally inaccessible from this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from src.ssl.input import bout_to_ssl_input
from src.ssl.train import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
NORMALIZATION_CONFIG = REPO_ROOT / "configs" / "ssl" / "normalization.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"

# Hard-coded by design. TEST must not become a runtime option here.
ALLOWED_PARTITIONS: Tuple[str, str] = ("train", "validation")

# Frozen primary-QC bout counts observed in the completed DS-005 pipeline.
EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DS-005 TRAIN/VALIDATION encoder embeddings from "
            "frozen best SSL checkpoints. TEST is prohibited."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAINING_CONFIG,
        help="Frozen SSL training YAML.",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=NORMALIZATION_CONFIG,
        help="TRAIN-fitted SSL normalization JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Extract one predefined seed instead of all configured seeds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Encoder inference batch size. Reduce if MPS memory is limited.",
    )
    parser.add_argument(
        "--max-bouts",
        type=int,
        default=None,
        help=(
            "Optional per-partition cap for a smoke/debug extraction. "
            "Do not use capped artifacts for final clustering."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "training" not in data:
        raise ValueError("Expected top-level 'training:' mapping.")

    training = data["training"]
    if not isinstance(training, dict):
        raise ValueError("'training' must be a mapping.")

    return training


def load_speed_normalization(path: Path) -> Tuple[float, float]:
    """Read the existing TRAIN-only speed mean/std without refitting anything."""
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    def search(obj: Any) -> Tuple[Optional[float], Optional[float]]:
        if isinstance(obj, dict):
            mean = None
            std = None

            for key in ("speed_mean", "mean"):
                value = obj.get(key)
                if isinstance(value, (int, float)):
                    mean = float(value)
                    break

            for key in ("speed_std", "std"):
                value = obj.get(key)
                if isinstance(value, (int, float)):
                    std = float(value)
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
        raise ValueError("Could not locate speed mean/std in normalization JSON.")

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Invalid TRAIN-only speed normalization: mean={mean}, std={std}"
        )

    return float(mean), float(std)


def normalize_ssl_input(
    x: np.ndarray,
    *,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    """Apply exactly the same speed-only normalization used during training."""
    x = np.asarray(x, dtype=np.float32).copy()

    if x.shape != (175, 3):
        raise ValueError(f"Expected SSL input shape (175, 3), got {x.shape}.")

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN/Inf before normalization.")

    x[:, 2] = (x[:, 2] - speed_mean) / speed_std

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN/Inf after normalization.")

    return x


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available.")
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(training: Dict[str, Any]) -> Tuple[ContrastiveModel, EncoderConfig]:
    encoder_cfg = training["encoder"]
    projection_cfg = training["projection_head"]

    config = EncoderConfig(
        input_channels=int(
            encoder_cfg["architecture"]["input_channels"]
        ),
        embedding_dim=int(encoder_cfg["embedding_dim"]),
        projection_dim=int(projection_cfg["projection_dim"]),
        dropout=float(encoder_cfg["architecture"]["dropout"]),
    )

    return ContrastiveModel(config=config), config


def checkpoint_for_seed(training: Dict[str, Any], seed: int) -> Path:
    checkpoint_dir = REPO_ROOT / training["checkpointing"]["directory"]
    return checkpoint_dir / f"ssl_seed{seed}_best.pt"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def scalar_from_bout(bout: Any, name: str) -> float:
    """Read scalar metadata fields if the canonical BoutData exposes them."""
    value = getattr(bout, name, None)
    if value is None:
        return float("nan")

    arr = np.asarray(value)
    if arr.size != 1:
        return float("nan")

    return safe_float(arr.reshape(-1)[0])


def stable_bout_id(fish_id: str, bout_index: int) -> str:
    """Create an explicit, stable analysis-row identifier."""
    return f"{fish_id}::bout{int(bout_index):06d}"


def metadata_row(
    *,
    dataset: DS005,
    bout: Any,
    partition: str,
    training_seed: int,
    row_index: int,
) -> Dict[str, Any]:
    """Build audit metadata without using metadata as encoder input."""
    if partition not in ALLOWED_PARTITIONS:
        raise RuntimeError("TEST/unknown partition reached metadata extraction.")

    key = bout.key
    fish = dataset.get_fish(key.fish_id)

    if key.partition != partition:
        raise RuntimeError(
            f"Partition mismatch for {key.fish_id} bout {key.bout_index}: "
            f"expected {partition}, observed {key.partition}"
        )

    speed = np.asarray(bout.speed_head, dtype=np.float64).reshape(-1)
    times = np.asarray(bout.times_bouts, dtype=np.float64).reshape(-1)

    if speed.size == 0:
        speed_mean = speed_std = speed_max = speed_rms = float("nan")
    else:
        speed_mean = float(np.mean(speed))
        speed_std = float(np.std(speed))
        speed_max = float(np.max(speed))
        speed_rms = float(np.sqrt(np.mean(np.square(speed))))

    start_frame = float("nan")
    end_frame = float("nan")
    if times.size >= 2:
        start_frame = safe_float(times[0])
        end_frame = safe_float(times[1])

    frame_rate_hz = safe_float(dataset.frame_rate_hz)

    start_time_s = (
        start_frame / frame_rate_hz
        if math.isfinite(start_frame) and frame_rate_hz > 0
        else float("nan")
    )
    end_time_s = (
        end_frame / frame_rate_hz
        if math.isfinite(end_frame) and frame_rate_hz > 0
        else float("nan")
    )

    return {
        "row_index": int(row_index),
        "dataset_id": "DS-005",
        "partition": partition,
        "training_seed": int(training_seed),
        "fish_id": str(key.fish_id),
        "fish_index": int(key.fish_index),
        "session_id": str(fish.canonical_session_id),
        "bout_id": stable_bout_id(key.fish_id, key.bout_index),
        "bout_index": int(key.bout_index),

        # DS-005 analysis unit is a complete 175-sample bout, not a sliding window.
        "window_index": int(key.bout_index),
        "samples_per_bout": 175,

        "context_id": str(key.context_id),
        "context_name": str(key.context_name),

        # Source timing/frame metadata when available.
        "source_start_frame": start_frame,
        "source_end_frame": end_frame,
        "source_start_time_s": start_time_s,
        "source_end_time_s": end_time_s,

        # Speed metadata for nuisance/speed-dependence analyses.
        # These values are metadata only and are never passed to the encoder
        # except through the frozen temporal speed channel in the SSL input.
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_max": speed_max,
        "speed_rms": speed_rms,

        # Existing optional author metadata, if exposed by BoutData.
        "stimulus_code": scalar_from_bout(bout, "stimulus_code"),
        "bout_type": scalar_from_bout(bout, "bout_type"),

        # Existing frozen QC flags.
        "all_zero_speed": bool(getattr(bout.qc, "all_zero_speed", False)),
        "extreme_speed_gt_100": bool(
            getattr(bout.qc, "extreme_speed_gt_100", False)
        ),
    }


@torch.inference_mode()
def encode_batch(
    *,
    model: ContrastiveModel,
    batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return ENCODER embeddings only.

    Critically, this calls ``model.encoder`` directly. The projection head is
    never executed and therefore cannot be exported accidentally.
    """
    if batch.ndim != 3 or batch.shape[1:] != (175, 3):
        raise ValueError(
            f"Expected inference batch (B, 175, 3), got {batch.shape}."
        )

    tensor = torch.from_numpy(
        np.asarray(batch, dtype=np.float32)
    ).to(device=device, dtype=torch.float32)

    model.eval()

    # DO NOT replace with model(tensor)[1].
    # DO NOT save model.projector(...).
    embeddings = model.encoder(tensor)

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Encoder returned unexpected shape {tuple(embeddings.shape)}."
        )

    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Encoder produced NaN/Inf embeddings.")

    return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)


def write_metadata_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("Cannot write empty metadata CSV.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


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
    seed_output_dir: Path,
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    split_seed: int,
    embedding_dim: int,
) -> None:
    """Extract one TRAIN or VALIDATION partition for one SSL seed."""
    if partition not in ALLOWED_PARTITIONS:
        raise RuntimeError(
            f"Partition {partition!r} is prohibited. "
            "This script only permits TRAIN and VALIDATION."
        )

    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if max_bouts is not None and max_bouts < 1:
        raise ValueError("--max-bouts must be >= 1 when supplied.")

    expected_rows = EXPECTED_ROWS[partition]
    target_rows = (
        min(expected_rows, max_bouts)
        if max_bouts is not None
        else expected_rows
    )

    metadata: List[Dict[str, Any]] = []
    embedding_chunks: List[np.ndarray] = []
    input_batch: List[np.ndarray] = []

    count = 0
    started = time.time()

    def flush() -> None:
        nonlocal input_batch

        if not input_batch:
            return

        X = np.stack(input_batch, axis=0).astype(np.float32, copy=False)

        Z = encode_batch(
            model=model,
            batch=X,
            device=device,
        )

        if Z.shape[1] != embedding_dim:
            raise RuntimeError(
                f"Expected embedding dim {embedding_dim}, got {Z.shape[1]}."
            )

        embedding_chunks.append(Z)
        input_batch = []

    print()
    print(f"{partition.upper()} extraction")
    print("-" * 72)
    print("TEST partition: NOT LOADED")

    for bout in dataset.iter_bouts(
        partition=partition,
        primary_qc_only=True,
        include_optional=False,
    ):
        # Belt-and-suspenders check in case loader behavior ever changes.
        if bout.key.partition not in ALLOWED_PARTITIONS:
            raise RuntimeError(
                f"Protected partition reached iterator: {bout.key.partition}"
            )

        x_raw = bout_to_ssl_input(bout)
        x = normalize_ssl_input(
            x_raw,
            speed_mean=speed_mean,
            speed_std=speed_std,
        )

        input_batch.append(x)

        metadata.append(
            metadata_row(
                dataset=dataset,
                bout=bout,
                partition=partition,
                training_seed=seed,
                row_index=count,
            )
        )

        count += 1

        if len(input_batch) >= batch_size:
            flush()

            elapsed = max(time.time() - started, 1e-9)
            rate = count / elapsed
            remaining = max(target_rows - count, 0)
            eta_s = remaining / rate if rate > 0 else float("inf")

            print(
                f"  {count:,}/{target_rows:,} "
                f"({100.0 * min(count / target_rows, 1.0):5.1f}%) | "
                f"{rate:,.0f} bouts/s | "
                f"ETA {eta_s / 60.0:.1f} min",
                flush=True,
            )

        if max_bouts is not None and count >= max_bouts:
            break

    flush()

    if not embedding_chunks:
        raise RuntimeError(f"No embeddings generated for {partition}.")

    embeddings = np.concatenate(embedding_chunks, axis=0)

    if embeddings.shape != (count, embedding_dim):
        raise RuntimeError(
            "Unexpected embedding matrix shape: "
            f"{embeddings.shape}; expected {(count, embedding_dim)}."
        )

    if len(metadata) != count:
        raise RuntimeError(
            f"Metadata rows ({len(metadata)}) != embedding rows ({count})."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Saved embedding matrix would contain NaN/Inf.")

    if max_bouts is None and count != expected_rows:
        raise RuntimeError(
            f"{partition} row count mismatch. "
            f"Expected {expected_rows:,}, observed {count:,}. "
            "Refusing to write final artifacts."
        )

    seed_output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = seed_output_dir / f"{partition}_embeddings.npz"
    metadata_path = seed_output_dir / f"{partition}_metadata.csv"
    manifest_path = seed_output_dir / f"{partition}_manifest.json"
    checksum_path = seed_output_dir / f"{partition}_SHA256SUMS"

    # Keep the embedding artifact compact and numerical.
    # Metadata lives in a row-aligned CSV and is keyed by row_index/bout_id.
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        row_index=np.arange(count, dtype=np.int64),
    )

    write_metadata_csv(metadata_path, metadata)

    manifest = {
        "dataset_id": "DS-005",
        "partition": partition,
        "training_seed": int(seed),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": checkpoint.get("epoch"),
        "validation_loss": checkpoint.get("validation_loss"),
        "split_seed": int(split_seed),
        "rows": int(count),
        "embedding_dim": int(embedding_dim),
        "dtype": "float32",
        "representation": "encoder_embedding",
        "projection_head_output_saved": False,
        "inference_call": "model.encoder(x)",
        "input_shape_per_bout": [175, 3],
        "normalization": {
            "orientation_channels": "none",
            "speed_channel": "TRAIN-only z-score",
            "speed_mean": float(speed_mean),
            "speed_std": float(speed_std),
        },
        "primary_qc_only": True,
        "test_partition_loaded": False,
        "capped_debug_export": max_bouts is not None,
        "max_bouts": max_bouts,
        "metadata_columns": list(metadata[0].keys()),
        "created_unix_time": time.time(),
    }

    write_json(manifest_path, manifest)

    artifacts = [embedding_path, metadata_path, manifest_path]
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in artifacts
        ),
        encoding="utf-8",
    )

    print(f"Completed {partition.upper()}:")
    print(f"  rows:          {count:,}")
    print(f"  embedding dim: {embedding_dim}")
    print(f"  embeddings:    {embedding_path}")
    print(f"  metadata:      {metadata_path}")
    print(f"  manifest:      {manifest_path}")
    print(f"  checksums:     {checksum_path}")


def main() -> None:
    args = parse_args()

    training = load_yaml(args.config)

    dataset_id = str(training["dataset"]["id"])
    if dataset_id != "DS-005":
        raise RuntimeError(
            f"This extractor is frozen for DS-005, got dataset={dataset_id!r}."
        )

    # A protected TEST declaration is required in the frozen config.
    test_cfg = training["dataset"]["partitions"]["test"]
    if not bool(test_cfg.get("protected", False)):
        raise RuntimeError(
            "training.yaml does not mark TEST as protected. "
            "Refusing extraction until the config is corrected."
        )

    configured_seeds: Sequence[int] = training["seeds"]["values"]
    configured_seeds = [int(seed) for seed in configured_seeds]

    if args.seed is None:
        seeds = list(configured_seeds)
    else:
        if args.seed not in configured_seeds:
            raise ValueError(
                f"Seed {args.seed} is not in the frozen seed policy: "
                f"{configured_seeds}"
            )
        seeds = [int(args.seed)]

    speed_mean, speed_std = load_speed_normalization(args.normalization)
    device = choose_device(args.device)

    embedding_dim = int(training["encoder"]["embedding_dim"])
    split_seed = int(training["dataset"]["split_seed"])

    print("=" * 72)
    print("DS-005 SSL ENCODER EMBEDDING EXTRACTION")
    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Config status: {training.get('status', 'UNKNOWN')}")
    print(f"Seeds: {seeds}")
    print("Partitions: ['train', 'validation']")
    print("TEST partition: PROTECTED / NOT ACCESSIBLE BY THIS SCRIPT")
    print(f"Device: {device}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Speed mean: {speed_mean:.12f}")
    print(f"Speed std:  {speed_std:.12f}")

    if args.max_bouts is not None:
        print(
            "WARNING: --max-bouts is active. "
            "These are debug artifacts, not final clustering inputs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        checkpoint_path = checkpoint_for_seed(training, seed)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint missing for seed {seed}: {checkpoint_path}"
            )

        model, model_config = build_model(training)
        model = model.to(device)

        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=None,
            map_location=device,
        )

        checkpoint_metadata = checkpoint.get("metadata", {}) or {}

        if checkpoint_metadata.get("test_partition_loaded") is True:
            raise RuntimeError(
                f"Checkpoint seed {seed} records TEST usage. "
                "Refusing to extract confirmatory embeddings."
            )

        # Verify checkpoint seed when metadata is available.
        checkpoint_seed = checkpoint.get(
            "training_seed",
            checkpoint_metadata.get("training_seed"),
        )
        if checkpoint_seed is not None and int(checkpoint_seed) != seed:
            raise RuntimeError(
                f"Checkpoint seed mismatch: requested {seed}, "
                f"checkpoint records {checkpoint_seed}."
            )

        print()
        print("=" * 72)
        print(f"SSL SEED {seed}")
        print("=" * 72)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Checkpoint SHA-256: {sha256_file(checkpoint_path)}")
        print(f"Selected epoch: {checkpoint.get('epoch', 'UNKNOWN')}")
        print(
            "Validation loss: "
            f"{checkpoint.get('validation_loss', 'UNKNOWN')}"
        )
        print("Representation to save: ENCODER EMBEDDING")
        print("Projection-head output: NOT SAVED")

        seed_output_dir = args.output_dir / f"seed{seed}"

        # Open the canonical dataset once per seed. The only iter_bouts calls
        # below use hard-coded TRAIN/VALIDATION strings.
        with DS005(
            repo_root=REPO_ROOT,
            validate=True,
            verify_split_hash=True,
        ) as dataset:
            dataset.assert_no_fish_overlap()

            for partition in ALLOWED_PARTITIONS:
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
                    seed_output_dir=seed_output_dir,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    split_seed=split_seed,
                    embedding_dim=int(model_config.embedding_dim),
                )

    print()
    print("=" * 72)
    print("SSL EMBEDDING EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Output root: {args.output_dir}")
    print("TRAIN extracted: YES")
    print("VALIDATION extracted: YES")
    print("TEST partition used: NO")
    print("Projection-head outputs saved: NO")


if __name__ == "__main__":
    main()
