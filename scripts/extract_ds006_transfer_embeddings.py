#!/usr/bin/env python3
"""Extract DS-006 TRAIN/VALIDATION transfer embeddings with frozen DS-005 encoders.

Safety:
- DS-006 TEST is inaccessible: only train.npz and validation.npz are loaded.
- DS-005 checkpoints are read-only and hash-checked before/after inference.
- model.encoder(...) is called directly; projection head is never executed.
- No retraining, fine-tuning, or normalization fitting occurs.
- Outputs are restricted to data/processed/DS-006/.

Expected DS-006 NPZ schema:
    X       : (N, 175, 3)
    bout_id : (N,)

Expected frozen counts:
    train       118100
    validation   18835
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.train import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DS006_SSL_ROOT = REPO_ROOT / "data" / "processed" / "DS-006" / "ssl"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-006" / "transfer_embeddings"
)

PARTITIONS: Tuple[str, str] = ("train", "validation")
EXPECTED_ROWS: Mapping[str, int] = {
    "train": 118_100,
    "validation": 18_835,
}
EXPECTED_INPUT_SHAPE = (175, 3)
EXPECTED_DIM = 64
EXPECTED_SEEDS = (11, 23, 37, 51, 79)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=TRAINING_CONFIG)
    ap.add_argument("--input-root", type=Path, default=DS006_SSL_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-bouts", type=int, default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        f.write(payload)
        tmp = f.name
    os.replace(tmp, path)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_training(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "training" not in obj:
        raise RuntimeError("Expected top-level training mapping.")
    return obj["training"]


def verify_training(training: Dict[str, Any]) -> Sequence[int]:
    if str(training["dataset"]["id"]) != "DS-005":
        raise RuntimeError("Config must be the frozen DS-005 training config.")
    if not bool(training["dataset"]["partitions"]["test"].get("protected", False)):
        raise RuntimeError("DS-005 TEST is not marked protected.")
    seeds = tuple(int(x) for x in training["seeds"]["values"])
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"Frozen seed set changed: {seeds}")
    shape = (
        int(training["input"]["shape"]["timesteps"]),
        int(training["input"]["shape"]["channels"]),
    )
    if shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Frozen input shape changed: {shape}")
    if int(training["encoder"]["embedding_dim"]) != EXPECTED_DIM:
        raise RuntimeError("Frozen embedding dimension is no longer 64.")
    return seeds


def build_model(training: Dict[str, Any]) -> Tuple[ContrastiveModel, EncoderConfig]:
    enc = training["encoder"]
    proj = training["projection_head"]
    cfg = EncoderConfig(
        input_channels=int(enc["architecture"]["input_channels"]),
        embedding_dim=int(enc["embedding_dim"]),
        projection_dim=int(proj["projection_dim"]),
        dropout=float(enc["architecture"]["dropout"]),
    )
    return ContrastiveModel(config=cfg), cfg


def checkpoint_for_seed(training: Dict[str, Any], seed: int) -> Path:
    return (
        REPO_ROOT
        / training["checkpointing"]["directory"]
        / f"ssl_seed{seed}_best.pt"
    )


def assert_safe_paths(input_root: Path, output_dir: Path) -> None:
    expected_input = DS006_SSL_ROOT.resolve()
    if input_root.resolve() != expected_input:
        raise RuntimeError(
            f"--input-root must be exactly {expected_input}; got {input_root.resolve()}"
        )

    out = output_dir.resolve()
    ds006 = (REPO_ROOT / "data" / "processed" / "DS-006").resolve()
    ds005 = (REPO_ROOT / "data" / "processed" / "DS-005").resolve()

    try:
        out.relative_to(ds006)
    except ValueError as exc:
        raise RuntimeError("Outputs must stay under data/processed/DS-006.") from exc

    try:
        out.relative_to(ds005)
        raise RuntimeError("Refusing to write anywhere under DS-005.")
    except ValueError:
        pass


def load_partition(
    root: Path, partition: str, max_bouts: Optional[int]
) -> Dict[str, Any]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    # Hard-coded mapping: TEST can never be selected here.
    path = root / ("train.npz" if partition == "train" else "validation.npz")
    if "test" in path.name.lower():
        raise RuntimeError("Protected TEST path reached loader.")
    if not path.exists():
        raise FileNotFoundError(path)

    source_hash = sha256_file(path)

    with np.load(path, allow_pickle=False) as npz:
        if "X" not in npz.files or "bout_id" not in npz.files:
            raise RuntimeError(f"{path} must contain X and bout_id.")
        X = np.asarray(npz["X"], dtype=np.float32)
        bout_ids = np.asarray(npz["bout_id"]).astype(str)

    if X.ndim != 3 or tuple(X.shape[1:]) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"{path}: expected (N,175,3), got {X.shape}.")
    if bout_ids.ndim != 1 or len(bout_ids) != len(X):
        raise RuntimeError(f"{path}: bout_id row alignment failed.")
    if len(X) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, got {len(X):,}."
        )
    if not np.isfinite(X).all():
        raise RuntimeError(f"{path}: non-finite input values.")
    if len(np.unique(bout_ids)) != len(bout_ids):
        raise RuntimeError(f"{path}: duplicate bout IDs.")
    if np.any(np.char.str_len(bout_ids) == 0):
        raise RuntimeError(f"{path}: empty bout IDs.")

    full_rows = len(X)
    if max_bouts is not None:
        if max_bouts < 1:
            raise ValueError("--max-bouts must be >= 1.")
        n = min(full_rows, max_bouts)
        X = X[:n]
        bout_ids = bout_ids[:n]

    return {
        "path": path,
        "sha256": source_hash,
        "full_rows": full_rows,
        "X": X,
        "bout_ids": bout_ids,
    }


@torch.inference_mode()
def encode(
    model: ContrastiveModel,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    label: str,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    out = np.empty((len(X), EXPECTED_DIM), dtype=np.float32)
    model.eval()
    started = time.time()

    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = torch.from_numpy(X[start:stop]).to(
            device=device, dtype=torch.float32
        )

        # Critical: direct encoder inference only.
        z = model.encoder(batch)

        if tuple(z.shape) != (stop - start, EXPECTED_DIM):
            raise RuntimeError(
                f"Unexpected encoder output {tuple(z.shape)} at rows {start}:{stop}."
            )
        if not torch.isfinite(z).all():
            raise RuntimeError("Encoder produced NaN/Inf.")

        out[start:stop] = z.detach().cpu().numpy().astype(np.float32, copy=False)

        batch_i = start // batch_size
        if batch_i % 10 == 0 or stop == len(X):
            elapsed = max(time.time() - started, 1e-9)
            rate = stop / elapsed
            eta = (len(X) - stop) / rate if rate > 0 else float("inf")
            print(
                f"  {label}: {stop:,}/{len(X):,} "
                f"({100*stop/len(X):5.1f}%) | {rate:,.0f} bouts/s | "
                f"ETA {eta/60:.1f} min",
                flush=True,
            )

    if not np.isfinite(out).all():
        raise RuntimeError("Final embedding matrix contains NaN/Inf.")
    return out


def write_metadata_csv(
    path: Path, bout_ids: np.ndarray, partition: str, seed: int
) -> None:
    fieldnames = [
        "row_index",
        "dataset_id",
        "partition",
        "source_encoder_dataset",
        "training_seed",
        "bout_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, bout_id in enumerate(bout_ids):
            writer.writerow(
                {
                    "row_index": i,
                    "dataset_id": "DS-006",
                    "partition": partition,
                    "source_encoder_dataset": "DS-005",
                    "training_seed": seed,
                    "bout_id": str(bout_id),
                }
            )


def export_partition(
    *,
    model: ContrastiveModel,
    device: torch.device,
    seed: int,
    partition: str,
    data: Dict[str, Any],
    batch_size: int,
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    output_dir: Path,
    overwrite: bool,
    debug: bool,
) -> Sequence[Path]:
    seed_dir = output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    emb_path = seed_dir / f"{partition}_embeddings.npz"
    meta_path = seed_dir / f"{partition}_metadata.csv"
    manifest_path = seed_dir / f"{partition}_manifest.json"
    checksum_path = seed_dir / f"{partition}_SHA256SUMS"

    for path in (emb_path, meta_path, manifest_path, checksum_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists; use --overwrite for intentional rerun."
            )

    X = data["X"]
    bout_ids = data["bout_ids"]

    print(f"\n{partition.upper()} TRANSFER")
    print("-" * 72)
    print(f"Input:         {data['path']}")
    print(f"Input SHA256:  {data['sha256']}")
    print(f"Rows:          {len(X):,}")
    print("TEST loaded:   NO")

    embeddings = encode(
        model=model,
        X=X,
        device=device,
        batch_size=batch_size,
        label=partition.upper(),
    )

    if len(embeddings) != len(bout_ids):
        raise RuntimeError("Embedding/bout_id row count mismatch.")

    row_index = np.arange(len(embeddings), dtype=np.int64)
    np.savez_compressed(
        emb_path,
        embeddings=embeddings,
        row_index=row_index,
        bout_id=bout_ids,
    )

    write_metadata_csv(meta_path, bout_ids, partition, seed)

    checkpoint_metadata = checkpoint.get("metadata", {}) or {}
    atomic_json(
        manifest_path,
        {
            "dataset_id": "DS-006",
            "analysis_role": "external_transfer_replication",
            "source_encoder_dataset": "DS-005",
            "partition": partition,
            "training_seed": seed,
            "source_input": str(data["path"].relative_to(REPO_ROOT)),
            "source_input_sha256": data["sha256"],
            "source_input_full_rows": int(data["full_rows"]),
            "rows_exported": int(len(embeddings)),
            "input_shape_per_bout": [175, 3],
            "input_preprocessing": (
                "Frozen DS-006 preprocessing; X consumed as already prepared. "
                "No normalization fit or changed here."
            ),
            "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_epoch": checkpoint.get("epoch"),
            "ds005_validation_loss": checkpoint.get("validation_loss"),
            "checkpoint_training_seed": checkpoint.get(
                "training_seed",
                checkpoint_metadata.get("training_seed"),
            ),
            "embedding_dim": 64,
            "dtype": "float32",
            "representation": "encoder_embedding",
            "inference_call": "model.encoder(x)",
            "projection_head_executed": False,
            "projection_head_output_saved": False,
            "encoder_fine_tuned_on_ds006": False,
            "encoder_parameters_updated": False,
            "ds006_normalization_refit": False,
            "bout_id_preserved": True,
            "row_alignment_verified": True,
            "finite_values_verified": True,
            "capped_debug_export": debug,
            "test_partition_loaded": False,
            "ds005_files_modified": False,
            "created_unix_time": time.time(),
        },
    )

    artifacts = (emb_path, meta_path, manifest_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(x)}  {x.name}\n" for x in artifacts),
        encoding="utf-8",
    )

    # Post-write integrity check.
    with np.load(emb_path, allow_pickle=False) as npz:
        z = np.asarray(npz["embeddings"])
        rows = np.asarray(npz["row_index"])
        saved_ids = np.asarray(npz["bout_id"]).astype(str)

    if z.shape != (len(bout_ids), EXPECTED_DIM):
        raise RuntimeError("Post-write embedding shape check failed.")
    if not np.array_equal(rows, row_index):
        raise RuntimeError("Post-write row_index check failed.")
    if not np.array_equal(saved_ids, bout_ids):
        raise RuntimeError("Post-write bout_id alignment check failed.")
    if not np.isfinite(z).all():
        raise RuntimeError("Post-write finite-value check failed.")

    print(f"Completed {partition.upper()}: {z.shape}")
    print("  row alignment: PASS")
    print("  finite values: PASS")
    print("  TEST loaded:   NO")

    return (emb_path, meta_path, manifest_path, checksum_path)


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()

    assert_safe_paths(input_root, output_dir)

    training = load_training(config)
    configured_seeds = verify_training(training)

    if args.seed is None:
        seeds = list(configured_seeds)
    else:
        if args.seed not in configured_seeds:
            raise ValueError(
                f"Seed {args.seed} is not frozen; allowed: {tuple(configured_seeds)}"
            )
        seeds = [args.seed]

    if args.max_bouts is not None and args.max_bouts < 1:
        raise ValueError("--max-bouts must be >= 1.")

    device = choose_device(args.device)

    print("=" * 72)
    print("DS-006 TRANSFER EMBEDDING EXTRACTION")
    print("=" * 72)
    print(f"Frozen DS-005 seeds: {seeds}")
    print(f"DS-006 input root:   {input_root}")
    print("Partitions:          TRAIN + VALIDATION only")
    print("TEST partition:      PROTECTED / INACCESSIBLE")
    print(f"Device:              {device}")
    print("Embedding dim:       64")
    print("Projection head:     NOT EXECUTED")
    print("Fine-tuning:         NO")
    print("DS-005 writes:       PROHIBITED")

    if args.max_bouts is not None:
        print("WARNING: capped smoke/debug extraction active.")

    # The only DS-006 arrays loaded by this script.
    data = {
        partition: load_partition(input_root, partition, args.max_bouts)
        for partition in PARTITIONS
    }

    for partition in PARTITIONS:
        print(
            f"Loaded {partition.upper():<10}: "
            f"{data[partition]['X'].shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for seed in seeds:
        checkpoint_path = checkpoint_for_seed(training, seed)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint_hash_before = sha256_file(checkpoint_path)

        model, cfg = build_model(training)
        if int(cfg.embedding_dim) != EXPECTED_DIM:
            raise RuntimeError("Model embedding dim is not 64.")
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
                f"Seed {seed} checkpoint records DS-005 TEST usage."
            )

        recorded_seed = checkpoint.get(
            "training_seed",
            checkpoint_metadata.get("training_seed"),
        )
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise RuntimeError(
                f"Checkpoint seed mismatch: expected {seed}, got {recorded_seed}."
            )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

        print("\n" + "=" * 72)
        print(f"FROZEN DS-005 ENCODER — SEED {seed}")
        print("=" * 72)
        print(f"Checkpoint:      {checkpoint_path}")
        print(f"Checkpoint hash: {checkpoint_hash_before}")
        print("Inference call:  model.encoder(x)")
        print("Fine-tuning:     NO")
        print("Projection head: NOT EXECUTED")

        for partition in PARTITIONS:
            written.extend(
                export_partition(
                    model=model,
                    device=device,
                    seed=seed,
                    partition=partition,
                    data=data[partition],
                    batch_size=args.batch_size,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    debug=args.max_bouts is not None,
                )
            )

        checkpoint_hash_after = sha256_file(checkpoint_path)
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RuntimeError(
                f"DS-005 checkpoint changed during seed {seed} inference."
            )
        print(f"DS-005 checkpoint immutability seed {seed}: PASS")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    global_hashes = output_dir / "TRANSFER_EMBEDDINGS_SHA256SUMS"
    if global_hashes.exists() and not args.overwrite:
        raise FileExistsError(
            f"{global_hashes} exists; use --overwrite for intentional rerun."
        )
    global_hashes.write_text(
        "".join(
            f"{sha256_file(x)}  {x.relative_to(output_dir)}\n"
            for x in sorted(written, key=lambda y: str(y))
            if x.exists()
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("DS-006 TRANSFER EMBEDDING EXTRACTION COMPLETE")
    print("=" * 72)
    print("TRAIN extracted:          YES")
    print("VALIDATION extracted:     YES")
    print("TEST partition used:      NO")
    print("DS-005 encoder retrained: NO")
    print("DS-005 files modified:    NO")
    print("Projection head executed: NO")
    print("Embedding dimension:      64")
    print("Bout-ID alignment:        VERIFIED")
    print("Finite-value checks:      PASSED")
    print(f"Output root:              {output_dir}")
    print(f"Global checksums:         {global_hashes}")


if __name__ == "__main__":
    main()
