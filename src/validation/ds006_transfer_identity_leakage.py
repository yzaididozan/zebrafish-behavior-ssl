#!/usr/bin/env python3
"""Fish-well identity leakage analysis for DS-006 transfer clusters.

Purpose
-------
Test whether frozen DS-006 transfer clusters disproportionately encode the
canonical fish-well identity rather than recurring behavior.

DS-006 canonical identity:
    DS006::<recording_id>::wellXX

Important caveat
----------------
This is a canonical fish-well unit identifier. Biological uniqueness of fish
across separate recordings is not independently verified. Results must be
described as fish-well identity leakage, not definitive individual-animal
identity leakage across recordings.

TRAIN and VALIDATION only. TEST is never loaded.

For each SSL seed and partition, this script computes:
- NMI between fish_id and cluster
- adjusted mutual information (AMI)
- Cramer's V
- mean/max fraction of each cluster supplied by one fish-well
- effective number of fish-wells represented per cluster
- per-fish-well normalized cluster entropy
- per-fish-well dominant-cluster fraction

Weak identity leakage is supported when:
- NMI / AMI are low;
- no cluster is dominated by a few fish-wells;
- effective fish-well count per cluster is broad;
- fish-wells themselves occupy multiple clusters (high entropy).

Inputs
------
data/processed/DS-006/metadata/bout_metadata.csv
data/processed/DS-006/transfer_clustering/seedXX/{train,validation}_labels.npy

Outputs
-------
data/processed/DS-006/transfer_identity_leakage/
    seed11/
        train_identity_metrics.json
        validation_identity_metrics.json
    ...
    aggregate_summary.json
    DS006_TRANSFER_IDENTITY_LEAKAGE_SHA256SUMS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "data" / "processed" / "DS-006" / "metadata" / "bout_metadata.csv"
LABEL_ROOT = REPO_ROOT / "data" / "processed" / "DS-006" / "transfer_clustering"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "DS-006" / "transfer_identity_leakage"

SEEDS = (11, 23, 37, 51, 79)
PARTITIONS = ("train", "validation")
EXPECTED_ROWS = {"train": 118_100, "validation": 18_835}
K = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=METADATA_PATH)
    p.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(payload)
        tmp = f.name
    os.replace(tmp, path)


def prohibit_test(path: Path) -> None:
    if "test" in path.name.lower():
        raise RuntimeError(f"Protected TEST path reached: {path}")


def load_metadata(path: Path, partition: str) -> Dict[str, np.ndarray]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")
    if not path.exists():
        raise FileNotFoundError(path)

    required = {
        "bout_id", "fish_id", "recording_id", "family", "well",
        "condition_code", "condition_label", "partition",
    }
    rows: List[Dict[str, str]] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing CSV header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}.")
        for row in reader:
            if row["partition"] == partition:
                rows.append(row)

    expected = EXPECTED_ROWS[partition]
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected:,} {partition} rows, got {len(rows):,}.")

    out = {
        field: np.asarray([row[field].strip() or "__MISSING__" for row in rows], dtype=str)
        for field in required
        if field != "partition"
    }
    out["partition"] = np.asarray([partition] * expected, dtype=str)

    if len(np.unique(out["bout_id"])) != expected:
        raise RuntimeError(f"{path}: duplicate bout IDs in {partition}.")
    return out


def load_cluster_bout_ids(label_root: Path, seed: int, partition: str) -> np.ndarray:
    manifest_path = label_root / f"seed{seed}" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "source_train_embedding" if partition == "train" else "source_validation_embedding"
    rel = manifest.get(key)
    if not rel:
        raise RuntimeError(f"{manifest_path}: missing {key}.")
    source = (REPO_ROOT / rel).resolve()
    prohibit_test(source)
    with np.load(source, allow_pickle=False) as npz:
        return np.asarray(npz["bout_id"]).astype(str)


def load_labels(label_root: Path, seed: int, partition: str) -> np.ndarray:
    path = label_root / f"seed{seed}" / f"{partition}_labels.npy"
    prohibit_test(path)
    if not path.exists():
        raise FileNotFoundError(path)
    labels = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(f"{path}: unexpected shape {labels.shape}.")
    if not np.array_equal(np.unique(labels), np.arange(K)):
        raise RuntimeError(f"{path}: expected cluster IDs 0..{K-1}.")
    return labels


def align_metadata(meta: Dict[str, np.ndarray], cluster_ids: np.ndarray) -> Dict[str, np.ndarray]:
    if np.array_equal(meta["bout_id"], cluster_ids):
        return meta

    lookup = {bout_id: i for i, bout_id in enumerate(meta["bout_id"])}
    missing = [x for x in cluster_ids if x not in lookup]
    if missing:
        raise RuntimeError(f"Metadata alignment failed; missing bout ID {missing[0]!r}.")
    idx = np.asarray([lookup[x] for x in cluster_ids], dtype=np.int64)
    if len(np.unique(idx)) != len(idx):
        raise RuntimeError("Metadata alignment reused rows.")
    return {key: values[idx] for key, values in meta.items()}


def encode(values: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    categories = sorted(set(str(x) for x in values))
    mapping = {x: i for i, x in enumerate(categories)}
    return np.asarray([mapping[str(x)] for x in values], dtype=np.int64), categories


def contingency(values: Sequence[str], labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    encoded, categories = encode(values)
    table = np.zeros((len(categories), K), dtype=np.int64)
    np.add.at(table, (encoded, labels), 1)
    return table, categories


def cramers_v(table: np.ndarray) -> float:
    n = int(table.sum())
    if n == 0:
        return 0.0
    rs = table.sum(axis=1, keepdims=True)
    cs = table.sum(axis=0, keepdims=True)
    expected = rs @ cs / n
    mask = expected > 0
    chi2 = float(np.sum((table[mask] - expected[mask]) ** 2 / expected[mask]))
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(math.sqrt((chi2 / n) / denom)) if denom > 0 else 0.0


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(K))


def analyze_identity(fish_ids: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    encoded, categories = encode(fish_ids)
    table, categories2 = contingency(fish_ids, labels)
    if categories != categories2:
        raise RuntimeError("Category encoding mismatch.")

    per_cluster: Dict[str, Any] = {}
    max_shares: List[float] = []
    effective_counts: List[float] = []

    for cluster in range(K):
        counts = table[:, cluster].astype(float)
        total = counts.sum()
        if total <= 0:
            raise RuntimeError(f"Cluster {cluster} is empty.")
        p = counts / total
        effective = float(1.0 / np.sum(p[p > 0] ** 2))
        max_share = float(np.max(p))
        max_shares.append(max_share)
        effective_counts.append(effective)
        per_cluster[str(cluster)] = {
            "bout_count": int(total),
            "fish_wells_with_bouts": int(np.sum(counts > 0)),
            "max_single_fish_well_fraction": max_share,
            "effective_fish_wells": effective,
            "dominant_fish_well": categories[int(np.argmax(p))],
        }

    entropies: List[float] = []
    dominant: List[float] = []
    per_fish: Dict[str, Any] = {}

    for i, fish_id in enumerate(categories):
        counts = table[i].astype(float)
        total = counts.sum()
        if total <= 0:
            continue
        p = counts / total
        ent = normalized_entropy(p)
        dom = float(np.max(p))
        entropies.append(ent)
        dominant.append(dom)
        per_fish[fish_id] = {
            "bout_count": int(total),
            "cluster_counts": counts.astype(int).tolist(),
            "normalized_cluster_entropy": ent,
            "dominant_cluster_fraction": dom,
            "dominant_cluster": int(np.argmax(p)),
        }

    return {
        "fish_well_count": len(categories),
        "association": {
            "normalized_mutual_information": float(normalized_mutual_info_score(encoded, labels)),
            "adjusted_mutual_information": float(adjusted_mutual_info_score(encoded, labels)),
            "cramers_v": cramers_v(table),
        },
        "cluster_concentration_summary": {
            "mean_max_single_fish_well_fraction": float(np.mean(max_shares)),
            "max_max_single_fish_well_fraction": float(np.max(max_shares)),
            "mean_effective_fish_wells_per_cluster": float(np.mean(effective_counts)),
            "min_effective_fish_wells_per_cluster": float(np.min(effective_counts)),
        },
        "fish_well_distribution_summary": {
            "mean_normalized_cluster_entropy": float(np.mean(entropies)),
            "std_normalized_cluster_entropy": float(np.std(entropies)),
            "min_normalized_cluster_entropy": float(np.min(entropies)),
            "mean_dominant_cluster_fraction": float(np.mean(dominant)),
            "max_dominant_cluster_fraction": float(np.max(dominant)),
        },
        "per_cluster": per_cluster,
        "per_fish_well": per_fish,
    }


def summarize(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    label_root = args.label_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if metadata_path != METADATA_PATH.resolve():
        raise RuntimeError(f"--metadata must resolve exactly to {METADATA_PATH.resolve()}")
    if label_root != LABEL_ROOT.resolve():
        raise RuntimeError(f"--label-root must resolve exactly to {LABEL_ROOT.resolve()}")

    ds006 = (REPO_ROOT / "data" / "processed" / "DS-006").resolve()
    try:
        output_dir.relative_to(ds006)
    except ValueError as exc:
        raise RuntimeError("Outputs must remain under DS-006.") from exc

    aggregate_path = output_dir / "aggregate_summary.json"
    if aggregate_path.exists() and not args.overwrite:
        raise FileExistsError(f"{aggregate_path} exists; use --overwrite for intentional rerun.")

    output_dir.mkdir(parents=True, exist_ok=True)

    partition_metadata = {
        p: load_metadata(metadata_path, p)
        for p in PARTITIONS
    }

    print("=" * 80)
    print("DS-006 TRANSFER FISH-WELL IDENTITY LEAKAGE")
    print("=" * 80)
    print(f"Seeds:            {list(SEEDS)}")
    print("Identity unit:    canonical fish-well ID")
    print("Evaluation:       TRAIN + VALIDATION")
    print("TEST partition:   PROTECTED / NOT LOADED")
    print("Caveat:           cross-recording biological uniqueness is unverified")
    print()

    results: Dict[int, Dict[str, Any]] = {}
    written: List[Path] = []

    for seed in SEEDS:
        results[seed] = {}
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)

        for partition in PARTITIONS:
            labels = load_labels(label_root, seed, partition)
            cluster_ids = load_cluster_bout_ids(label_root, seed, partition)
            meta = align_metadata(partition_metadata[partition], cluster_ids)
            result = analyze_identity(meta["fish_id"], labels)
            result.update({
                "dataset_id": "DS-006",
                "partition": partition,
                "ssl_encoder_seed": seed,
                "identity_unit": "canonical_fish_well",
                "bout_id_alignment_verified": True,
                "biological_uniqueness_across_recordings_verified": False,
                "test_partition_used": False,
            })
            results[seed][partition] = result

            out = seed_dir / f"{partition}_identity_metrics.json"
            atomic_json(out, result)
            written.append(out)

            a = result["association"]
            c = result["cluster_concentration_summary"]
            d = result["fish_well_distribution_summary"]

            print(partition.upper())
            print(f"  Fish-well categories:                 {result['fish_well_count']}")
            print(f"  NMI:                                  {a['normalized_mutual_information']:.6f}")
            print(f"  AMI:                                  {a['adjusted_mutual_information']:.6f}")
            print(f"  Cramer's V:                           {a['cramers_v']:.6f}")
            print(f"  Mean fish-well cluster entropy:       {d['mean_normalized_cluster_entropy']:.6f}")
            print(f"  Mean dominant cluster fraction:       {d['mean_dominant_cluster_fraction']:.6f}")
            print(f"  Mean max fish-well share / cluster:   {c['mean_max_single_fish_well_fraction']:.6f}")
            print(f"  Mean effective fish-wells / cluster:  {c['mean_effective_fish_wells_per_cluster']:.2f}")
        print("TEST partition used: NO")
        print()

    aggregate: Dict[str, Any] = {
        "dataset_id": "DS-006",
        "analysis": "transfer_fish_well_identity_leakage",
        "identity_unit": "canonical_fish_well",
        "biological_uniqueness_across_recordings_verified": False,
        "seeds": list(SEEDS),
        "test_partition_used": False,
        "metrics": {},
    }

    for partition in PARTITIONS:
        aggregate["metrics"][partition] = {
            "nmi": summarize([
                results[s][partition]["association"]["normalized_mutual_information"]
                for s in SEEDS
            ]),
            "ami": summarize([
                results[s][partition]["association"]["adjusted_mutual_information"]
                for s in SEEDS
            ]),
            "cramers_v": summarize([
                results[s][partition]["association"]["cramers_v"]
                for s in SEEDS
            ]),
            "mean_normalized_cluster_entropy": summarize([
                results[s][partition]["fish_well_distribution_summary"]["mean_normalized_cluster_entropy"]
                for s in SEEDS
            ]),
            "mean_max_single_fish_well_fraction": summarize([
                results[s][partition]["cluster_concentration_summary"]["mean_max_single_fish_well_fraction"]
                for s in SEEDS
            ]),
            "mean_effective_fish_wells_per_cluster": summarize([
                results[s][partition]["cluster_concentration_summary"]["mean_effective_fish_wells_per_cluster"]
                for s in SEEDS
            ]),
        }

    atomic_json(aggregate_path, aggregate)
    written.append(aggregate_path)

    checksum = output_dir / "DS006_TRANSFER_IDENTITY_LEAKAGE_SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{sha256_file(p)}  {p.relative_to(output_dir)}\n"
            for p in sorted(written, key=lambda x: str(x))
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("DS-006 IDENTITY-LEAKAGE SUMMARY")
    print("=" * 80)
    for partition in PARTITIONS:
        m = aggregate["metrics"][partition]
        print(partition.upper())
        print(f"  Mean NMI:                        {m['nmi']['mean']:.6f}")
        print(f"  Mean AMI:                        {m['ami']['mean']:.6f}")
        print(f"  Mean Cramer's V:                 {m['cramers_v']['mean']:.6f}")
        print(f"  Mean fish-well entropy:          {m['mean_normalized_cluster_entropy']['mean']:.6f}")
        print(f"  Mean max fish-well share:        {m['mean_max_single_fish_well_fraction']['mean']:.6f}")
        print(f"  Mean effective fish-wells:       {m['mean_effective_fish_wells_per_cluster']['mean']:.2f}")
    print("TEST partition used: NO")
    print(f"Aggregate: {aggregate_path}")
    print(f"Checksums: {checksum}")


if __name__ == "__main__":
    main()
