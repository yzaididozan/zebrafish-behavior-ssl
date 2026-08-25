#!/usr/bin/env python3
"""Fish-identity leakage analysis for selected DS-005 SSL clusters.

Purpose
-------
Test the preregistered identity-leakage threat:

    Are SSL clusters partly grouping bouts by individual fish rather than behavior?

This script uses the selected k=8 cluster assignments from
``src/discovery/ssl_cluster_stability.py`` and row-aligned metadata from
``scripts/extract_ssl_embeddings.py``.

TRAIN and VALIDATION only. TEST is never loaded.

Analyses
--------
For each SSL seed and partition:

1. Verify label/metadata row alignment.
2. Build fish x cluster contingency tables.
3. Quantify fish-cluster association with:
   - normalized mutual information (NMI)
   - adjusted mutual information (AMI)
   - Cramer's V
4. Quantify concentration of each cluster across fish with:
   - effective number of fish (Hill number / inverse Simpson)
   - maximum single-fish fraction
   - top-5-fish fraction
5. Quantify how stereotyped each fish is with:
   - per-fish cluster entropy
   - normalized entropy
   - dominant-cluster fraction
6. Predict fish identity using cluster composition:
   - one vector per fish: normalized k=8 cluster histogram
   - leave-one-bout-aggregation is not used; each fish is represented by its
     full partition-level cluster composition
   - nearest-centroid self-consistency diagnostic within each partition

Important interpretation
------------------------
Because TRAIN and VALIDATION contain disjoint fish, direct classification of
validation fish identities from TRAIN fish labels is not meaningful. Therefore,
the primary identity-leakage evidence is distributional association:
NMI/AMI/Cramer's V, cluster concentration by fish, and per-fish entropy.

Strong leakage would look like:
- high fish-cluster association,
- clusters dominated by a small number of fish,
- low per-fish entropy / one dominant cluster per fish.

Weak leakage would look like:
- low association,
- clusters broadly distributed across many fish,
- fish exhibiting diverse cluster mixtures.

TEST safety
-----------
- Only TRAIN and VALIDATION metadata/labels are permitted.
- No TEST CLI option exists.
- TEST artifacts beneath the SSL metadata root cause an immediate refusal.

Outputs
-------
data/processed/DS-005/ssl_identity_leakage/
    seed11/
        train_identity_metrics.json
        validation_identity_metrics.json
    seed23/
        ...
    aggregate_summary.json
    IDENTITY_LEAKAGE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ssl_identity_leakage.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ssl_identity_leakage.py --overwrite
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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_LABEL_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_identity_leakage"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_K = 8
PARTITIONS = ("train", "validation")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp = handle.name
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_checksums(path: Path, artifacts: Sequence[Path]) -> None:
    path.write_text(
        "".join(
            f"{sha256_file(artifact)}  {artifact.name}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)

    training = obj.get("training", {})
    seeds = training.get("seeds", {}).get("values")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No frozen SSL seeds found in training.yaml.")

    return tuple(int(seed) for seed in seeds)


def assert_no_test_artifacts(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(root)

    hits: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if (
            name.startswith("test_")
            or "_test_" in name
            or name in {"test.npz", "test.npy", "test.csv", "test.json"}
        ):
            hits.append(path)

    if hits:
        raise RuntimeError(
            "TEST artifacts detected under SSL root; refusing identity analysis:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def load_metadata_fish(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Tuple[List[str], List[str]]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    fish_ids: List[str] = []
    bout_ids: List[str] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")

        required = {
            "row_index",
            "fish_id",
            "bout_id",
            "partition",
            "training_seed",
        }
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"{path} missing columns: {sorted(missing)}"
            )

        for expected_row, row in enumerate(reader):
            row_index = int(row["row_index"])

            if row_index != expected_row:
                raise RuntimeError(
                    f"{path}: row_index mismatch at row {expected_row}: "
                    f"observed {row_index}"
                )

            if row["partition"] != partition:
                raise RuntimeError(
                    f"{path}: partition mismatch {row['partition']!r}"
                )

            if int(row["training_seed"]) != ssl_seed:
                raise RuntimeError(
                    f"{path}: training_seed mismatch "
                    f"{row['training_seed']!r}"
                )

            fish_id = row["fish_id"].strip()
            bout_id = row["bout_id"].strip()

            if not fish_id or not bout_id:
                raise RuntimeError(
                    f"{path}: empty fish_id/bout_id at row {expected_row}"
                )

            fish_ids.append(fish_id)
            bout_ids.append(bout_id)

    if len(fish_ids) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"observed {len(fish_ids):,}"
        )

    return fish_ids, bout_ids


def load_labels(
    label_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        label_root
        / f"seed{ssl_seed}"
        / f"{partition}_labels.npy"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.int64,
    )

    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected shape "
            f"({EXPECTED_ROWS[partition]},), got {labels.shape}"
        )

    unique = np.unique(labels)
    expected = np.arange(EXPECTED_K)

    if not np.array_equal(unique, expected):
        raise RuntimeError(
            f"{path}: expected cluster IDs {expected.tolist()}, "
            f"observed {unique.tolist()}"
        )

    return labels


def encode_fish_ids(fish_ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    unique = sorted(set(fish_ids))
    index = {fish_id: i for i, fish_id in enumerate(unique)}
    encoded = np.asarray(
        [index[fish_id] for fish_id in fish_ids],
        dtype=np.int64,
    )
    return encoded, unique


def contingency_table(
    fish_ids: Sequence[str],
    labels: np.ndarray,
    *,
    k: int,
) -> Tuple[np.ndarray, List[str]]:
    encoded, fish_names = encode_fish_ids(fish_ids)

    table = np.zeros(
        (len(fish_names), k),
        dtype=np.int64,
    )

    np.add.at(table, (encoded, labels), 1)

    return table, fish_names


def cramers_v(table: np.ndarray) -> float:
    n = int(np.sum(table))

    if n <= 0:
        return 0.0

    row_sums = np.sum(table, axis=1, keepdims=True)
    col_sums = np.sum(table, axis=0, keepdims=True)

    expected = (row_sums @ col_sums) / n

    mask = expected > 0

    chi2 = float(
        np.sum(
            ((table[mask] - expected[mask]) ** 2)
            / expected[mask]
        )
    )

    phi2 = chi2 / n
    r, c = table.shape
    denom = min(c - 1, r - 1)

    if denom <= 0:
        return 0.0

    return float(math.sqrt(phi2 / denom))


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]

    if probabilities.size <= 1:
        return 0.0

    entropy = float(
        -np.sum(probabilities * np.log(probabilities))
    )

    max_entropy = math.log(EXPECTED_K)

    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def cluster_concentration_metrics(
    table: np.ndarray,
    fish_names: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cluster in range(table.shape[1]):
        counts = table[:, cluster].astype(np.float64)
        total = float(np.sum(counts))

        if total <= 0:
            raise RuntimeError(f"Cluster {cluster} is empty.")

        p = counts / total
        p_nonzero = p[p > 0]

        inverse_simpson = float(
            1.0 / np.sum(p_nonzero ** 2)
        )

        max_single = float(np.max(p))
        top5 = float(np.sum(np.sort(p)[-5:]))

        dominant_fish_idx = int(np.argmax(p))

        out[str(cluster)] = {
            "total_bouts": int(total),
            "fish_with_any_bouts": int(np.sum(counts > 0)),
            "effective_number_of_fish": inverse_simpson,
            "max_single_fish_fraction": max_single,
            "top5_fish_fraction": top5,
            "dominant_fish_id": fish_names[dominant_fish_idx],
        }

    return out


def fish_distribution_metrics(
    table: np.ndarray,
    fish_names: Sequence[str],
) -> Dict[str, Any]:
    normalized_entropies: List[float] = []
    dominant_fractions: List[float] = []
    bouts_per_fish: List[int] = []

    per_fish: Dict[str, Any] = {}

    for i, fish_id in enumerate(fish_names):
        counts = table[i].astype(np.float64)
        total = float(np.sum(counts))

        if total <= 0:
            continue

        probs = counts / total
        entropy = normalized_entropy(probs)
        dominant = float(np.max(probs))

        normalized_entropies.append(entropy)
        dominant_fractions.append(dominant)
        bouts_per_fish.append(int(total))

        per_fish[fish_id] = {
            "bout_count": int(total),
            "cluster_counts": counts.astype(int).tolist(),
            "cluster_fractions": probs.astype(float).tolist(),
            "normalized_entropy": entropy,
            "dominant_cluster_fraction": dominant,
            "dominant_cluster": int(np.argmax(probs)),
        }

    entropy_arr = np.asarray(
        normalized_entropies,
        dtype=np.float64,
    )
    dominant_arr = np.asarray(
        dominant_fractions,
        dtype=np.float64,
    )
    bouts_arr = np.asarray(
        bouts_per_fish,
        dtype=np.float64,
    )

    return {
        "summary": {
            "fish_count": int(len(per_fish)),
            "mean_normalized_entropy": float(
                np.mean(entropy_arr)
            ),
            "std_normalized_entropy": float(
                np.std(entropy_arr)
            ),
            "min_normalized_entropy": float(
                np.min(entropy_arr)
            ),
            "max_normalized_entropy": float(
                np.max(entropy_arr)
            ),
            "mean_dominant_cluster_fraction": float(
                np.mean(dominant_arr)
            ),
            "max_dominant_cluster_fraction": float(
                np.max(dominant_arr)
            ),
            "median_bouts_per_fish": float(
                np.median(bouts_arr)
            ),
        },
        "per_fish": per_fish,
    }


def nearest_centroid_self_consistency(
    table: np.ndarray,
) -> Dict[str, float]:
    """Diagnostic using each fish's normalized cluster histogram.

    This is NOT a held-out identity classifier. It measures how distinct fish
    composition vectors are within a partition by asking whether each fish is
    closest to its own histogram versus all other fish histograms after a small
    deterministic leave-fraction-out perturbation.
    """
    counts = table.astype(np.float64)

    totals = np.sum(counts, axis=1, keepdims=True)
    hist = counts / np.maximum(totals, 1.0)

    # Deterministic perturbation: remove 10% of each cluster count (floored)
    # to create a pseudo-query histogram from the same fish.
    query_counts = np.maximum(
        counts - np.floor(counts * 0.10),
        0.0,
    )
    query_totals = np.sum(
        query_counts,
        axis=1,
        keepdims=True,
    )
    query_hist = query_counts / np.maximum(query_totals, 1.0)

    correct = 0

    for i in range(hist.shape[0]):
        distances = np.linalg.norm(
            hist - query_hist[i],
            axis=1,
        )
        predicted = int(np.argmin(distances))
        correct += int(predicted == i)

    return {
        "self_consistency_accuracy": float(
            correct / hist.shape[0]
        ),
        "chance_level": float(
            1.0 / hist.shape[0]
        ),
    }


def analyze_partition(
    fish_ids: Sequence[str],
    labels: np.ndarray,
) -> Dict[str, Any]:
    fish_encoded, fish_names = encode_fish_ids(fish_ids)

    nmi = float(
        normalized_mutual_info_score(
            fish_encoded,
            labels,
        )
    )

    ami = float(
        adjusted_mutual_info_score(
            fish_encoded,
            labels,
        )
    )

    table, fish_names_2 = contingency_table(
        fish_ids,
        labels,
        k=EXPECTED_K,
    )

    if fish_names != fish_names_2:
        raise RuntimeError("Fish encoding mismatch.")

    concentration = cluster_concentration_metrics(
        table,
        fish_names,
    )

    fish_metrics = fish_distribution_metrics(
        table,
        fish_names,
    )

    self_consistency = nearest_centroid_self_consistency(
        table
    )

    max_single_values = np.asarray(
        [
            concentration[str(c)]["max_single_fish_fraction"]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    top5_values = np.asarray(
        [
            concentration[str(c)]["top5_fish_fraction"]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    effective_fish_values = np.asarray(
        [
            concentration[str(c)]["effective_number_of_fish"]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    return {
        "fish_count": int(len(fish_names)),
        "bout_count": int(len(fish_ids)),
        "association": {
            "normalized_mutual_information": nmi,
            "adjusted_mutual_information": ami,
            "cramers_v": cramers_v(table),
        },
        "cluster_concentration": concentration,
        "cluster_concentration_summary": {
            "mean_max_single_fish_fraction": float(
                np.mean(max_single_values)
            ),
            "max_max_single_fish_fraction": float(
                np.max(max_single_values)
            ),
            "mean_top5_fish_fraction": float(
                np.mean(top5_values)
            ),
            "max_top5_fish_fraction": float(
                np.max(top5_values)
            ),
            "mean_effective_number_of_fish": float(
                np.mean(effective_fish_values)
            ),
            "min_effective_number_of_fish": float(
                np.min(effective_fish_values)
            ),
        },
        "fish_distribution": fish_metrics,
        "fish_histogram_self_consistency": self_consistency,
    }


def aggregate_results(
    results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = defaultdict(list)

    for result in results.values():
        for partition in PARTITIONS:
            part = result[partition]

            metrics[
                f"{partition}_nmi"
            ].append(
                part["association"][
                    "normalized_mutual_information"
                ]
            )

            metrics[
                f"{partition}_ami"
            ].append(
                part["association"][
                    "adjusted_mutual_information"
                ]
            )

            metrics[
                f"{partition}_cramers_v"
            ].append(
                part["association"]["cramers_v"]
            )

            metrics[
                f"{partition}_mean_entropy"
            ].append(
                part["fish_distribution"]["summary"][
                    "mean_normalized_entropy"
                ]
            )

            metrics[
                f"{partition}_mean_dominant_fraction"
            ].append(
                part["fish_distribution"]["summary"][
                    "mean_dominant_cluster_fraction"
                ]
            )

            metrics[
                f"{partition}_mean_max_single_fish_fraction"
            ].append(
                part["cluster_concentration_summary"][
                    "mean_max_single_fish_fraction"
                ]
            )

            metrics[
                f"{partition}_mean_effective_fish"
            ].append(
                part["cluster_concentration_summary"][
                    "mean_effective_number_of_fish"
                ]
            )

    out: Dict[str, Any] = {}

    for name, values in metrics.items():
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        out[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether selected SSL clusters leak individual fish identity."
        )
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_root = args.metadata_root.resolve()
    label_root = args.label_root.resolve()
    training_config = args.training_config.resolve()
    output_dir = args.output_dir.resolve()

    assert_no_test_artifacts(metadata_root)

    seeds = configured_ssl_seeds(
        training_config
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        output_dir / "aggregate_summary.json"
    )
    checksum_path = (
        output_dir
        / "IDENTITY_LEAKAGE_SHA256SUMS"
    )

    if not args.overwrite and aggregate_path.exists():
        raise FileExistsError(
            f"{aggregate_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL FISH-IDENTITY LEAKAGE ANALYSIS")
    print("=" * 80)
    print(f"SSL seeds:          {list(seeds)}")
    print(f"Selected clusters:  k={EXPECTED_K}")
    print("Evaluation:         TRAIN + VALIDATION")
    print("TEST partition:     PROTECTED / NOT LOADED")
    print()

    seed_results: Dict[int, Dict[str, Any]] = {}
    written_artifacts: List[Path] = []

    for seed in seeds:
        print("=" * 80)
        print(f"SSL SEED {seed}")
        print("=" * 80)

        seed_result: Dict[str, Any] = {
            "ssl_seed": int(seed),
            "k": EXPECTED_K,
            "test_partition_used": False,
        }

        seed_dir = (
            output_dir / f"seed{seed}"
        )
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for partition in PARTITIONS:
            fish_ids, bout_ids = load_metadata_fish(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
            )

            labels = load_labels(
                label_root,
                ssl_seed=seed,
                partition=partition,
            )

            if len(bout_ids) != labels.shape[0]:
                raise RuntimeError(
                    f"{partition}: metadata/label length mismatch."
                )

            metrics = analyze_partition(
                fish_ids,
                labels,
            )

            seed_result[partition] = metrics

            output_path = (
                seed_dir
                / f"{partition}_identity_metrics.json"
            )

            atomic_write_json(
                output_path,
                {
                    "ssl_seed": int(seed),
                    "partition": partition,
                    "k": EXPECTED_K,
                    **metrics,
                    "test_partition_used": False,
                },
            )

            written_artifacts.append(
                output_path
            )

            assoc = metrics["association"]
            fish_summary = (
                metrics["fish_distribution"]["summary"]
            )
            conc_summary = (
                metrics["cluster_concentration_summary"]
            )

            print(partition.upper())
            print(
                f"  Fish count:                    "
                f"{metrics['fish_count']}"
            )
            print(
                f"  Fish-cluster NMI:              "
                f"{assoc['normalized_mutual_information']:.6f}"
            )
            print(
                f"  Fish-cluster AMI:              "
                f"{assoc['adjusted_mutual_information']:.6f}"
            )
            print(
                f"  Cramer's V:                    "
                f"{assoc['cramers_v']:.6f}"
            )
            print(
                f"  Mean fish cluster entropy:     "
                f"{fish_summary['mean_normalized_entropy']:.6f}"
            )
            print(
                f"  Mean dominant cluster fraction:"
                f" {fish_summary['mean_dominant_cluster_fraction']:.6f}"
            )
            print(
                f"  Mean max single-fish share:    "
                f"{conc_summary['mean_max_single_fish_fraction']:.6f}"
            )
            print(
                f"  Mean effective fish / cluster: "
                f"{conc_summary['mean_effective_number_of_fish']:.2f}"
            )
            print()

        seed_results[seed] = seed_result
        print("TEST partition used: NO")
        print()

    aggregate_stats = aggregate_results(
        seed_results
    )

    aggregate_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "selected_clustering": {
            "method": "kmeans",
            "k": EXPECTED_K,
        },
        "ssl_training_seeds": list(seeds),
        "aggregate_metrics": aggregate_stats,
        "per_seed": {
            str(seed): result
            for seed, result in seed_results.items()
        },
        "interpretation_guardrails": {
            "primary_metrics": [
                "normalized_mutual_information",
                "adjusted_mutual_information",
                "cramers_v",
                "cluster_concentration",
                "per_fish_cluster_entropy",
            ],
            "note": (
                "Because TRAIN and VALIDATION contain disjoint fish, "
                "direct cross-partition fish-ID classification is not "
                "well-defined. Identity leakage is therefore assessed "
                "primarily via fish-cluster association and concentration."
            ),
            "no_fixed_pass_fail_threshold": True,
        },
        "test_partition_used": False,
    }

    atomic_write_json(
        aggregate_path,
        aggregate_payload,
    )

    written_artifacts.append(
        aggregate_path
    )

    write_checksums(
        checksum_path,
        written_artifacts,
    )

    print("=" * 80)
    print("IDENTITY-LEAKAGE SUMMARY")
    print("=" * 80)

    print(
        "Mean TRAIN fish-cluster NMI:       "
        f"{aggregate_stats['train_nmi']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION fish-cluster NMI:  "
        f"{aggregate_stats['validation_nmi']['mean']:.6f}"
    )
    print(
        "Mean TRAIN fish-cluster AMI:       "
        f"{aggregate_stats['train_ami']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION fish-cluster AMI:  "
        f"{aggregate_stats['validation_ami']['mean']:.6f}"
    )
    print(
        "Mean TRAIN Cramer's V:             "
        f"{aggregate_stats['train_cramers_v']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION Cramer's V:        "
        f"{aggregate_stats['validation_cramers_v']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION fish entropy:      "
        f"{aggregate_stats['validation_mean_entropy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION dominant fraction: "
        f"{aggregate_stats['validation_mean_dominant_fraction']['mean']:.6f}"
    )
    print()
    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
