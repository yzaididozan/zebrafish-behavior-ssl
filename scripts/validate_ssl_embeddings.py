#!/usr/bin/env python3
"""QC gate for DS-005 SSL encoder embeddings.

Validates TRAIN and VALIDATION exports for all frozen SSL seeds.
This script never loads the TEST partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_SSL_DIR = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_DIM = 64
PARTITIONS = ("train", "validation")
VARIANCE_EPS = 1e-12

REQUIRED_METADATA = {
    "row_index",
    "dataset_id",
    "partition",
    "training_seed",
    "fish_id",
    "session_id",
    "bout_id",
    "bout_index",
    "window_index",
    "speed_mean",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate DS-005 TRAIN/VALIDATION SSL embedding exports."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--ssl-dir", type=Path, default=DEFAULT_SSL_DIR)
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_EPS,
    )
    p.add_argument(
        "--skip-checksums",
        action="store_true",
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_training(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "training" not in obj:
        raise ValueError("Expected top-level training: mapping")
    return obj["training"]


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if "embeddings" not in z.files:
            raise ValueError("missing embeddings array")
        embeddings = np.asarray(z["embeddings"])
        row_index = (
            np.asarray(z["row_index"])
            if "row_index" in z.files
            else np.array([], dtype=np.int64)
        )
    return embeddings, row_index


def read_metadata(path: Path):
    fish_ids: List[str] = []
    bout_ids: List[str] = []
    session_ids: List[str] = []
    row_indices: List[int] = []

    partitions: Set[str] = set()
    seeds: Set[int] = set()
    dataset_ids: Set[str] = set()
    columns: Set[str] = set()

    duplicate_bouts = 0
    seen_bouts: Set[str] = set()
    empty_required = 0
    malformed_rows = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("metadata CSV has no header")
        columns = set(reader.fieldnames)

        for row in reader:
            fish = (row.get("fish_id") or "").strip()
            bout = (row.get("bout_id") or "").strip()
            session = (row.get("session_id") or "").strip()

            fish_ids.append(fish)
            bout_ids.append(bout)
            session_ids.append(session)

            if not fish or not bout or not session:
                empty_required += 1

            if bout in seen_bouts:
                duplicate_bouts += 1
            seen_bouts.add(bout)

            part = (row.get("partition") or "").strip()
            if part:
                partitions.add(part)

            ds = (row.get("dataset_id") or "").strip()
            if ds:
                dataset_ids.add(ds)

            try:
                seeds.add(int(row.get("training_seed", "")))
            except Exception:
                pass

            try:
                row_indices.append(int(row.get("row_index", "")))
            except Exception:
                row_indices.append(-1)
                malformed_rows += 1

    return {
        "fish_ids": fish_ids,
        "bout_ids": bout_ids,
        "session_ids": session_ids,
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "columns": columns,
        "partitions": partitions,
        "seeds": seeds,
        "dataset_ids": dataset_ids,
        "duplicate_bouts": duplicate_bouts,
        "empty_required": empty_required,
        "malformed_rows": malformed_rows,
    }


def load_manifest(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("manifest is not a JSON object")
    return obj


def parse_checksums(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            result[parts[1].lstrip("*").strip()] = parts[0].lower()
    return result


def check(name: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{suffix}")
    return passed


def find_test_artifacts(root: Path) -> List[Path]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if (
            name.startswith("test_")
            or "_test_" in name
            or name.endswith("_test.npz")
            or name.endswith("_test.csv")
            or name.endswith("_test.json")
        ):
            hits.append(p)
    return sorted(hits)


def validate_partition(
    seed: int,
    partition: str,
    seed_dir: Path,
    variance_threshold: float,
    skip_checksums: bool,
):
    ok = True
    expected_rows = EXPECTED_ROWS[partition]

    emb_path = seed_dir / f"{partition}_embeddings.npz"
    meta_path = seed_dir / f"{partition}_metadata.csv"
    manifest_path = seed_dir / f"{partition}_manifest.json"
    sums_path = seed_dir / f"{partition}_SHA256SUMS"

    print(f"{partition.upper()}")

    exists = all(p.exists() for p in (emb_path, meta_path, manifest_path))
    ok &= check("artifacts exist", exists)

    if not exists:
        return False, None, set()

    try:
        embeddings, npz_row_index = load_npz(emb_path)

        ok &= check(
            "embedding shape",
            embeddings.shape == (expected_rows, EXPECTED_DIM),
            f"observed {embeddings.shape}; expected ({expected_rows}, {EXPECTED_DIM})",
        )

        finite = bool(np.isfinite(embeddings).all())
        ok &= check("finite embeddings", finite)

        expected_index = np.arange(embeddings.shape[0], dtype=np.int64)
        ok &= check(
            "NPZ row_index",
            npz_row_index.shape == expected_index.shape
            and np.array_equal(npz_row_index.astype(np.int64), expected_index),
        )

        if finite and embeddings.ndim == 2 and embeddings.shape[1] == EXPECTED_DIM:
            x = embeddings.astype(np.float64, copy=False)
            variances = np.var(x, axis=0)
            collapsed = np.flatnonzero(variances <= variance_threshold)

            ok &= check(
                "no collapsed dimensions",
                len(collapsed) == 0,
                (
                    f"min variance={variances.min():.6e}; "
                    f"collapsed={collapsed.tolist()}"
                ),
            )

            norms = np.linalg.norm(x, axis=1)
            ok &= check(
                "embedding norms",
                np.isfinite(norms).all()
                and float(np.median(norms)) > 1e-8
                and float(np.max(norms)) > float(np.min(norms)),
                (
                    f"min={np.min(norms):.6e}, "
                    f"median={np.median(norms):.6e}, "
                    f"max={np.max(norms):.6e}"
                ),
            )

    except Exception as exc:
        ok &= check("embedding load", False, f"{type(exc).__name__}: {exc}")
        embeddings = np.empty((0, EXPECTED_DIM), dtype=np.float32)

    try:
        meta = read_metadata(meta_path)
        missing = REQUIRED_METADATA - meta["columns"]

        ok &= check(
            "required metadata columns",
            not missing,
            "all present" if not missing else f"missing {sorted(missing)}",
        )

        ok &= check(
            "metadata row count",
            len(meta["bout_ids"]) == expected_rows,
            f"{len(meta['bout_ids']):,} rows",
        )

        ok &= check(
            "metadata/embedding alignment",
            len(meta["bout_ids"]) == embeddings.shape[0],
        )

        expected_meta_index = np.arange(len(meta["bout_ids"]), dtype=np.int64)
        ok &= check(
            "metadata row_index",
            np.array_equal(meta["row_indices"], expected_meta_index),
        )

        ok &= check(
            "metadata partition",
            meta["partitions"] == {partition},
            repr(sorted(meta["partitions"])),
        )

        ok &= check(
            "metadata seed",
            meta["seeds"] == {seed},
            repr(sorted(meta["seeds"])),
        )

        ok &= check(
            "metadata dataset",
            meta["dataset_ids"] == {"DS-005"},
            repr(sorted(meta["dataset_ids"])),
        )

        ok &= check(
            "bout IDs unique",
            meta["duplicate_bouts"] == 0,
            f"duplicates={meta['duplicate_bouts']}",
        )

        ok &= check(
            "required identifiers populated",
            meta["empty_required"] == 0,
            f"empty rows={meta['empty_required']}",
        )

        ok &= check(
            "metadata row_index parse",
            meta["malformed_rows"] == 0,
            f"malformed={meta['malformed_rows']}",
        )

    except Exception as exc:
        ok &= check("metadata load", False, f"{type(exc).__name__}: {exc}")
        meta = None

    try:
        manifest = load_manifest(manifest_path)

        ok &= check(
            "manifest representation",
            manifest.get("representation") == "encoder_embedding",
            repr(manifest.get("representation")),
        )

        ok &= check(
            "projection head excluded",
            manifest.get("projection_head_output_saved") is False,
            repr(manifest.get("projection_head_output_saved")),
        )

        ok &= check(
            "TEST not loaded",
            manifest.get("test_partition_loaded") is False,
            repr(manifest.get("test_partition_loaded")),
        )

        ok &= check(
            "final export is uncapped",
            manifest.get("capped_debug_export") is False,
            repr(manifest.get("capped_debug_export")),
        )

        ok &= check(
            "manifest rows",
            int(manifest.get("rows", -1)) == expected_rows,
            repr(manifest.get("rows")),
        )

        ok &= check(
            "manifest embedding dim",
            int(manifest.get("embedding_dim", -1)) == EXPECTED_DIM,
            repr(manifest.get("embedding_dim")),
        )

        ok &= check(
            "manifest inference call",
            manifest.get("inference_call") == "model.encoder(x)",
            repr(manifest.get("inference_call")),
        )

        ok &= check(
            "manifest seed",
            int(manifest.get("training_seed", -1)) == seed,
            repr(manifest.get("training_seed")),
        )

        ok &= check(
            "manifest partition",
            manifest.get("partition") == partition,
            repr(manifest.get("partition")),
        )

    except Exception as exc:
        ok &= check("manifest load", False, f"{type(exc).__name__}: {exc}")

    if skip_checksums:
        ok &= check("checksums", True, "skipped")
    else:
        try:
            expected_hashes = parse_checksums(sums_path)
            hash_ok = True
            details = []
            for artifact in (emb_path, meta_path, manifest_path):
                recorded = expected_hashes.get(artifact.name)
                observed = sha256_file(artifact)
                if recorded != observed:
                    hash_ok = False
                    details.append(artifact.name)

            ok &= check(
                "checksums",
                hash_ok,
                "all match" if hash_ok else f"mismatch: {details}",
            )
        except Exception as exc:
            ok &= check("checksums", False, f"{type(exc).__name__}: {exc}")

    fish_set = set(meta["fish_ids"]) if meta is not None else set()
    return ok, meta, fish_set


def same_order(a, b) -> Tuple[bool, str]:
    if a is None or b is None:
        return False, "metadata unavailable"

    if len(a["bout_ids"]) != len(b["bout_ids"]):
        return False, "row counts differ"

    if a["bout_ids"] != b["bout_ids"]:
        for i, (x, y) in enumerate(zip(a["bout_ids"], b["bout_ids"])):
            if x != y:
                return False, f"first bout mismatch at row {i}"
        return False, "bout order differs"

    if a["fish_ids"] != b["fish_ids"]:
        for i, (x, y) in enumerate(zip(a["fish_ids"], b["fish_ids"])):
            if x != y:
                return False, f"first fish mismatch at row {i}"
        return False, "fish order differs"

    if a["session_ids"] != b["session_ids"]:
        return False, "session order differs"

    return True, "identical bout/fish/session ordering"


def main() -> int:
    args = parse_args()

    training = load_training(args.config)

    if training["dataset"]["id"] != "DS-005":
        print("ERROR: expected DS-005.", file=sys.stderr)
        return 1

    if training["dataset"]["partitions"]["test"].get("protected") is not True:
        print("ERROR: TEST is not marked protected in training.yaml.", file=sys.stderr)
        return 1

    seeds = [int(s) for s in training["seeds"]["values"]]
    config_dim = int(training["encoder"]["embedding_dim"])

    print("=" * 80)
    print("DS-005 SSL EMBEDDING QC")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Config status: {training.get('status', 'UNKNOWN')}")
    print(f"SSL root: {args.ssl_dir}")
    print(f"Seeds: {seeds}")
    print(f"Expected TRAIN: {EXPECTED_ROWS['train']:,} x {EXPECTED_DIM}")
    print(f"Expected VALIDATION: {EXPECTED_ROWS['validation']:,} x {EXPECTED_DIM}")
    print("TEST partition: PROTECTED / NOT LOADED")
    print()

    overall_ok = True

    if config_dim != EXPECTED_DIM:
        overall_ok &= check(
            "frozen embedding dimension",
            False,
            f"config={config_dim}, expected={EXPECTED_DIM}",
        )

    metadata_by_partition: Dict[str, Dict[int, dict]] = {
        "train": {},
        "validation": {},
    }
    fish_by_partition: Dict[str, Dict[int, Set[str]]] = {
        "train": {},
        "validation": {},
    }

    for seed in seeds:
        print("=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)

        seed_dir = args.ssl_dir / f"seed{seed}"

        if not seed_dir.exists():
            overall_ok &= check("seed directory", False, str(seed_dir))
            print()
            continue

        for partition in PARTITIONS:
            part_ok, meta, fish_set = validate_partition(
                seed=seed,
                partition=partition,
                seed_dir=seed_dir,
                variance_threshold=args.variance_threshold,
                skip_checksums=args.skip_checksums,
            )

            overall_ok &= part_ok
            metadata_by_partition[partition][seed] = meta
            fish_by_partition[partition][seed] = fish_set
            print()

    print("=" * 80)
    print("CROSS-SEED / SPLIT QC")
    print("=" * 80)

    reference_seed = seeds[0]

    for partition in PARTITIONS:
        ref = metadata_by_partition[partition].get(reference_seed)

        for seed in seeds[1:]:
            candidate = metadata_by_partition[partition].get(seed)
            aligned, detail = same_order(ref, candidate)

            overall_ok &= check(
                f"{partition} seed{reference_seed} vs seed{seed}",
                aligned,
                detail,
            )

    train_fish = fish_by_partition["train"].get(reference_seed, set())
    val_fish = fish_by_partition["validation"].get(reference_seed, set())
    overlap = train_fish & val_fish

    overall_ok &= check(
        "TRAIN/VALIDATION fish overlap",
        len(overlap) == 0,
        "0 fish" if not overlap else f"{len(overlap)} overlapping fish",
    )

    for seed in seeds[1:]:
        overall_ok &= check(
            f"seed{seed} TRAIN fish set",
            fish_by_partition["train"].get(seed, set()) == train_fish,
        )
        overall_ok &= check(
            f"seed{seed} VALIDATION fish set",
            fish_by_partition["validation"].get(seed, set()) == val_fish,
        )

    test_artifacts = find_test_artifacts(args.ssl_dir)
    overall_ok &= check(
        "TEST artifacts absent",
        len(test_artifacts) == 0,
        "none found"
        if not test_artifacts
        else "; ".join(str(p) for p in test_artifacts[:10]),
    )

    print()
    print("=" * 80)
    print("FINAL STATUS")
    print("=" * 80)

    if overall_ok:
        print("PASS")
        print("SSL embeddings are structurally valid for TRAIN/VALIDATION clustering.")
        print("TEST partition used: NO")
        print("Next permitted stage: SSL clustering/model selection.")
        return 0

    print("FAIL")
    print("Do NOT begin SSL clustering until failed QC checks are resolved.")
    print("TEST partition used: NO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
