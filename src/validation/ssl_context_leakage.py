#!/usr/bin/env python3
"""Session/context leakage analysis for selected DS-005 SSL clusters.

Purpose
-------
Test the preregistered context/session leakage threat:

    Are SSL clusters partly grouping bouts by recording/session/context
    metadata rather than behavior?

This script uses the selected k=8 cluster assignments from
``src/discovery/ssl_cluster_stability.py`` and the row-aligned metadata from
``scripts/extract_ssl_embeddings.py``.

TRAIN and VALIDATION only. TEST is never loaded.

Metadata tested
---------------
- session_id
- context_id
- context_name

Analyses
--------
For each SSL seed and partition, and for each metadata variable:
1. Verify label/metadata row alignment.
2. Compute association between metadata category and cluster:
   - normalized mutual information (NMI)
   - adjusted mutual information (AMI)
   - Cramer's V
3. Quantify cluster concentration by metadata category:
   - maximum single-category fraction
   - top-3-category fraction
   - effective number of categories (inverse Simpson / Hill number)
4. Quantify per-category cluster diversity:
   - normalized cluster entropy
   - dominant-cluster fraction

Interpretation
--------------
Strong context leakage would look like:
- high NMI/AMI/Cramer's V,
- clusters dominated by a few sessions/contexts,
- metadata categories strongly concentrated in one cluster.

Weak context leakage would look like:
- low association,
- clusters broadly distributed across sessions/contexts,
- high per-category cluster entropy.

TEST safety
-----------
- Only TRAIN and VALIDATION metadata/labels are permitted.
- No TEST CLI option exists.
- TEST artifacts beneath the SSL metadata root cause immediate refusal.

Outputs
-------
data/processed/DS-005/ssl_context_leakage/
    seed11/
        train_context_metrics.json
        validation_context_metrics.json
    seed23/
        ...
    aggregate_summary.json
    CONTEXT_LEAKAGE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ssl_context_leakage.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ssl_context_leakage.py --overwrite
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
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_context_leakage"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_K = 8
PARTITIONS = ("train", "validation")
METADATA_FIELDS = ("session_id", "context_id", "context_name")


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
            "TEST artifacts detected under SSL root; refusing context analysis:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def load_metadata(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Dict[str, List[str]]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    values: Dict[str, List[str]] = {
        "bout_id": [],
        "session_id": [],
        "context_id": [],
        "context_name": [],
    }

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")

        required = {
            "row_index",
            "bout_id",
            "partition",
            "training_seed",
            "session_id",
            "context_id",
            "context_name",
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

            bout_id = row["bout_id"].strip()

            if not bout_id:
                raise RuntimeError(
                    f"{path}: empty bout_id at row {expected_row}"
                )

            values["bout_id"].append(bout_id)

            for field in METADATA_FIELDS:
                value = row[field].strip()

                # Keep missing metadata explicit rather than dropping rows.
                if not value:
                    value = "__MISSING__"

                values[field].append(value)

    expected_rows = EXPECTED_ROWS[partition]

    if len(values["bout_id"]) != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows:,} rows, "
            f"observed {len(values['bout_id']):,}"
        )

    return values


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


def encode_categories(
    values: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    categories = sorted(set(values))
    index = {
        category: i
        for i, category in enumerate(categories)
    }

    encoded = np.asarray(
        [index[value] for value in values],
        dtype=np.int64,
    )

    return encoded, categories


def contingency_table(
    values: Sequence[str],
    labels: np.ndarray,
    *,
    k: int,
) -> Tuple[np.ndarray, List[str]]:
    encoded, categories = encode_categories(values)

    table = np.zeros(
        (len(categories), k),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (encoded, labels),
        1,
    )

    return table, categories


def cramers_v(table: np.ndarray) -> float:
    n = int(np.sum(table))

    if n <= 0:
        return 0.0

    row_sums = np.sum(
        table,
        axis=1,
        keepdims=True,
    )
    col_sums = np.sum(
        table,
        axis=0,
        keepdims=True,
    )

    expected = (
        row_sums @ col_sums
    ) / n

    mask = expected > 0

    chi2 = float(
        np.sum(
            ((table[mask] - expected[mask]) ** 2)
            / expected[mask]
        )
    )

    phi2 = chi2 / n
    r, c = table.shape
    denom = min(r - 1, c - 1)

    if denom <= 0:
        return 0.0

    return float(
        math.sqrt(phi2 / denom)
    )


def normalized_cluster_entropy(
    probabilities: np.ndarray,
) -> float:
    probabilities = probabilities[
        probabilities > 0
    ]

    if probabilities.size <= 1:
        return 0.0

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    max_entropy = math.log(EXPECTED_K)

    if max_entropy <= 0:
        return 0.0

    return float(
        entropy / max_entropy
    )


def cluster_concentration_metrics(
    table: np.ndarray,
    categories: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cluster in range(table.shape[1]):
        counts = table[:, cluster].astype(
            np.float64
        )
        total = float(np.sum(counts))

        if total <= 0:
            raise RuntimeError(
                f"Cluster {cluster} is empty."
            )

        p = counts / total
        p_nonzero = p[p > 0]

        effective_categories = float(
            1.0 / np.sum(p_nonzero ** 2)
        )

        max_single = float(
            np.max(p)
        )

        top3 = float(
            np.sum(
                np.sort(p)[-3:]
            )
        )

        dominant_idx = int(
            np.argmax(p)
        )

        out[str(cluster)] = {
            "total_bouts": int(total),
            "categories_with_any_bouts": int(
                np.sum(counts > 0)
            ),
            "effective_number_of_categories": (
                effective_categories
            ),
            "max_single_category_fraction": (
                max_single
            ),
            "top3_category_fraction": (
                top3
            ),
            "dominant_category": (
                categories[dominant_idx]
            ),
        }

    return out


def category_distribution_metrics(
    table: np.ndarray,
    categories: Sequence[str],
) -> Dict[str, Any]:
    entropies: List[float] = []
    dominant_fractions: List[float] = []
    bouts_per_category: List[int] = []

    per_category: Dict[str, Any] = {}

    for i, category in enumerate(categories):
        counts = table[i].astype(
            np.float64
        )
        total = float(
            np.sum(counts)
        )

        if total <= 0:
            continue

        probs = counts / total
        entropy = normalized_cluster_entropy(
            probs
        )
        dominant = float(
            np.max(probs)
        )

        entropies.append(
            entropy
        )
        dominant_fractions.append(
            dominant
        )
        bouts_per_category.append(
            int(total)
        )

        per_category[category] = {
            "bout_count": int(total),
            "cluster_counts": (
                counts.astype(int).tolist()
            ),
            "cluster_fractions": (
                probs.astype(float).tolist()
            ),
            "normalized_entropy": (
                entropy
            ),
            "dominant_cluster_fraction": (
                dominant
            ),
            "dominant_cluster": int(
                np.argmax(probs)
            ),
        }

    entropy_arr = np.asarray(
        entropies,
        dtype=np.float64,
    )
    dominant_arr = np.asarray(
        dominant_fractions,
        dtype=np.float64,
    )
    bouts_arr = np.asarray(
        bouts_per_category,
        dtype=np.float64,
    )

    return {
        "summary": {
            "category_count": int(
                len(per_category)
            ),
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
            "median_bouts_per_category": float(
                np.median(bouts_arr)
            ),
        },
        "per_category": per_category,
    }


def analyze_field(
    values: Sequence[str],
    labels: np.ndarray,
) -> Dict[str, Any]:
    encoded, categories = encode_categories(
        values
    )

    nmi = float(
        normalized_mutual_info_score(
            encoded,
            labels,
        )
    )

    ami = float(
        adjusted_mutual_info_score(
            encoded,
            labels,
        )
    )

    table, categories_2 = contingency_table(
        values,
        labels,
        k=EXPECTED_K,
    )

    if categories != categories_2:
        raise RuntimeError(
            "Category encoding mismatch."
        )

    concentration = (
        cluster_concentration_metrics(
            table,
            categories,
        )
    )

    category_metrics = (
        category_distribution_metrics(
            table,
            categories,
        )
    )

    max_single_values = np.asarray(
        [
            concentration[str(c)][
                "max_single_category_fraction"
            ]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    top3_values = np.asarray(
        [
            concentration[str(c)][
                "top3_category_fraction"
            ]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    effective_values = np.asarray(
        [
            concentration[str(c)][
                "effective_number_of_categories"
            ]
            for c in range(EXPECTED_K)
        ],
        dtype=np.float64,
    )

    return {
        "category_count": int(
            len(categories)
        ),
        "association": {
            "normalized_mutual_information": (
                nmi
            ),
            "adjusted_mutual_information": (
                ami
            ),
            "cramers_v": (
                cramers_v(table)
            ),
        },
        "cluster_concentration": (
            concentration
        ),
        "cluster_concentration_summary": {
            "mean_max_single_category_fraction": float(
                np.mean(max_single_values)
            ),
            "max_max_single_category_fraction": float(
                np.max(max_single_values)
            ),
            "mean_top3_category_fraction": float(
                np.mean(top3_values)
            ),
            "max_top3_category_fraction": float(
                np.max(top3_values)
            ),
            "mean_effective_number_of_categories": float(
                np.mean(effective_values)
            ),
            "min_effective_number_of_categories": float(
                np.min(effective_values)
            ),
        },
        "category_distribution": (
            category_metrics
        ),
    }


def analyze_partition(
    metadata: Dict[str, List[str]],
    labels: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for field in METADATA_FIELDS:
        result[field] = analyze_field(
            metadata[field],
            labels,
        )

    return result


def aggregate_results(
    results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = defaultdict(list)

    for result in results.values():
        for partition in PARTITIONS:
            for field in METADATA_FIELDS:
                field_result = result[partition][field]

                metrics[
                    f"{partition}_{field}_nmi"
                ].append(
                    field_result[
                        "association"
                    ][
                        "normalized_mutual_information"
                    ]
                )

                metrics[
                    f"{partition}_{field}_ami"
                ].append(
                    field_result[
                        "association"
                    ][
                        "adjusted_mutual_information"
                    ]
                )

                metrics[
                    f"{partition}_{field}_cramers_v"
                ].append(
                    field_result[
                        "association"
                    ][
                        "cramers_v"
                    ]
                )

                metrics[
                    f"{partition}_{field}_mean_entropy"
                ].append(
                    field_result[
                        "category_distribution"
                    ][
                        "summary"
                    ][
                        "mean_normalized_entropy"
                    ]
                )

                metrics[
                    f"{partition}_{field}_mean_dominant_fraction"
                ].append(
                    field_result[
                        "category_distribution"
                    ][
                        "summary"
                    ][
                        "mean_dominant_cluster_fraction"
                    ]
                )

    out: Dict[str, Any] = {}

    for name, values in metrics.items():
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        out[name] = {
            "mean": float(
                np.mean(arr)
            ),
            "std": float(
                np.std(arr)
            ),
            "min": float(
                np.min(arr)
            ),
            "max": float(
                np.max(arr)
            ),
        }

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether selected SSL clusters leak session/context metadata."
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

    metadata_root = (
        args.metadata_root.resolve()
    )
    label_root = (
        args.label_root.resolve()
    )
    training_config = (
        args.training_config.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    assert_no_test_artifacts(
        metadata_root
    )

    seeds = configured_ssl_seeds(
        training_config
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        output_dir
        / "aggregate_summary.json"
    )

    checksum_path = (
        output_dir
        / "CONTEXT_LEAKAGE_SHA256SUMS"
    )

    if (
        not args.overwrite
        and aggregate_path.exists()
    ):
        raise FileExistsError(
            f"{aggregate_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL SESSION/CONTEXT LEAKAGE ANALYSIS")
    print("=" * 80)
    print(f"SSL seeds:          {list(seeds)}")
    print(f"Selected clusters:  k={EXPECTED_K}")
    print(
        "Metadata fields:    "
        + ", ".join(METADATA_FIELDS)
    )
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
            output_dir
            / f"seed{seed}"
        )
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for partition in PARTITIONS:
            metadata = load_metadata(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
            )

            labels = load_labels(
                label_root,
                ssl_seed=seed,
                partition=partition,
            )

            if (
                len(metadata["bout_id"])
                != labels.shape[0]
            ):
                raise RuntimeError(
                    f"{partition}: metadata/label length mismatch."
                )

            metrics = analyze_partition(
                metadata,
                labels,
            )

            seed_result[
                partition
            ] = metrics

            output_path = (
                seed_dir
                / f"{partition}_context_metrics.json"
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

            print(partition.upper())

            for field in METADATA_FIELDS:
                field_metrics = (
                    metrics[field]
                )
                assoc = (
                    field_metrics["association"]
                )
                category_summary = (
                    field_metrics[
                        "category_distribution"
                    ]["summary"]
                )
                conc_summary = (
                    field_metrics[
                        "cluster_concentration_summary"
                    ]
                )

                print(f"  {field}")
                print(
                    f"    Categories:                    "
                    f"{field_metrics['category_count']}"
                )
                print(
                    f"    NMI:                           "
                    f"{assoc['normalized_mutual_information']:.6f}"
                )
                print(
                    f"    AMI:                           "
                    f"{assoc['adjusted_mutual_information']:.6f}"
                )
                print(
                    f"    Cramer's V:                    "
                    f"{assoc['cramers_v']:.6f}"
                )
                print(
                    f"    Mean category cluster entropy: "
                    f"{category_summary['mean_normalized_entropy']:.6f}"
                )
                print(
                    f"    Mean dominant cluster fraction:"
                    f" {category_summary['mean_dominant_cluster_fraction']:.6f}"
                )
                print(
                    f"    Mean max category share/cluster:"
                    f" {conc_summary['mean_max_single_category_fraction']:.6f}"
                )
                print(
                    f"    Mean effective categories/cluster:"
                    f" {conc_summary['mean_effective_number_of_categories']:.2f}"
                )

            print()

        seed_results[
            seed
        ] = seed_result

        print(
            "TEST partition used: NO"
        )
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
        "metadata_fields": list(
            METADATA_FIELDS
        ),
        "ssl_training_seeds": list(
            seeds
        ),
        "aggregate_metrics": (
            aggregate_stats
        ),
        "per_seed": {
            str(seed): result
            for seed, result
            in seed_results.items()
        },
        "interpretation_guardrails": {
            "primary_metrics": [
                "normalized_mutual_information",
                "adjusted_mutual_information",
                "cramers_v",
                "cluster_concentration",
                "per_category_cluster_entropy",
            ],
            "note": (
                "Session/context association may reflect true behavior-context "
                "relationships as well as nuisance leakage. Interpret together "
                "with experimental design and baseline behavior distributions."
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
    print("CONTEXT-LEAKAGE SUMMARY")
    print("=" * 80)

    for field in METADATA_FIELDS:
        print(field.upper())
        print(
            "  Mean TRAIN NMI:       "
            f"{aggregate_stats[f'train_{field}_nmi']['mean']:.6f}"
        )
        print(
            "  Mean VALIDATION NMI:  "
            f"{aggregate_stats[f'validation_{field}_nmi']['mean']:.6f}"
        )
        print(
            "  Mean TRAIN AMI:       "
            f"{aggregate_stats[f'train_{field}_ami']['mean']:.6f}"
        )
        print(
            "  Mean VALIDATION AMI:  "
            f"{aggregate_stats[f'validation_{field}_ami']['mean']:.6f}"
        )
        print(
            "  Mean TRAIN Cramer's V:"
            f" {aggregate_stats[f'train_{field}_cramers_v']['mean']:.6f}"
        )
        print(
            "  Mean VALIDATION Cramer's V:"
            f" {aggregate_stats[f'validation_{field}_cramers_v']['mean']:.6f}"
        )
        print(
            "  Mean VALIDATION entropy:"
            f" {aggregate_stats[f'validation_{field}_mean_entropy']['mean']:.6f}"
        )
        print()

    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
