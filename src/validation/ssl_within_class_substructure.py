#!/usr/bin/env python3
"""Within-known-class SSL substructure analysis for DS-005.

Purpose
-------
Test whether SSL consistently subdivides a single conventional zebrafish bout
class into reproducible finer-grained structure.

This is a targeted follow-up to known-behavior alignment.

For each known behavior class (e.g. LLC, Slow2, SAT, HAT), the script:
1. selects only bouts belonging to that known class;
2. examines the distribution of selected SSL k=8 labels;
3. measures how many SSL clusters meaningfully contribute;
4. quantifies cross-seed agreement of within-class SSL assignments;
5. aligns seed-specific SSL cluster IDs to reference seed 11 using TRAIN only;
6. applies the same TRAIN-derived mappings to VALIDATION;
7. reports whether the within-class subdivision is reproducible on held-out fish.

TRAIN and VALIDATION only. TEST is never loaded.

Primary questions
-----------------
- Does a known class occupy several SSL clusters rather than one?
- Is that multi-cluster structure reproducible across SSL seeds?
- Does it persist on held-out VALIDATION fish?
- Which known classes show the strongest candidate substructure?

Metrics
-------
Per known class / partition:
- count
- SSL cluster fractions
- dominant SSL cluster fraction
- normalized SSL-cluster entropy
- effective number of SSL clusters

Cross-seed within-class reproducibility:
- pairwise Adjusted Rand Index (ARI)
- pairwise Normalized Mutual Information (NMI)
- pairwise Hungarian-aligned raw agreement
- pairwise Jensen-Shannon similarity of cluster-fraction distributions

Important guardrail
-------------------
This script does NOT claim that multiple SSL clusters inside a known class are
new biological behaviors. Reproducible within-class subdivision is only a
candidate signal that must later be characterized using movement features,
speed, stimulus/context, and ideally external validation.

Outputs
-------
data/processed/DS-005/ssl_within_class_substructure/
    per_class/
        LLC.json
        Slow2.json
        SAT.json
        HAT.json
        ...
    summary.json
    WITHIN_CLASS_SUBSTRUCTURE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ssl_within_class_substructure.py

Optional class subset:

    PYTHONPATH=. python3 src/validation/ssl_within_class_substructure.py \
        --classes LLC Slow2 SAT HAT

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ssl_within_class_substructure.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_LABEL_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_within_class_substructure"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_SSL_K = 8
REFERENCE_SEED = 11
PARTITIONS = ("train", "validation")

DEFAULT_CLASS_NAMES = [
    "Short_CS",
    "Long_CS",
    "BS",
    "O_bend",
    "J_turn",
    "SLC",
    "Slow1",
    "RT",
    "Slow2",
    "LLC",
    "AS",
    "SAT",
    "HAT",
]

LABEL_COLUMN_CANDIDATES = (
    "bout_type",
    "bout_type_name",
    "known_bout_type",
    "known_behavior",
    "behavior_label",
    "bout_label",
)


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
            f"{sha256_file(artifact)}  {artifact.relative_to(path.parent)}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )


def prohibit_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(
            f"TEST access prohibited during within-class analysis: {path}"
        )


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
            or name in {"test.npy", "test.npz", "test.csv", "test.json"}
        ):
            hits.append(path)

    if hits:
        raise RuntimeError(
            "Protected TEST artifacts found beneath an SSL input root; "
            "refusing to continue:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)

    training = obj.get("training", {})
    seeds = training.get("seeds", {}).get("values")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No frozen SSL seeds found in training.yaml.")

    return tuple(int(seed) for seed in seeds)


def detect_label_column(
    fieldnames: Sequence[str],
    requested: Optional[str],
) -> str:
    available = set(fieldnames)

    if requested is not None:
        if requested not in available:
            raise RuntimeError(
                f"Requested label column {requested!r} not present. "
                f"Available columns: {sorted(available)}"
            )
        return requested

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate

    raise RuntimeError(
        "Could not find a known-behavior label column in metadata. "
        f"Tried {list(LABEL_COLUMN_CANDIDATES)}. "
        "Use --label-column if your metadata uses a different name."
    )


def normalize_known_label(raw: str) -> str:
    value = raw.strip()

    if value == "":
        return "__MISSING__"

    if value in DEFAULT_CLASS_NAMES:
        return value

    try:
        numeric = float(value)
        integer = int(numeric)

        if math.isclose(numeric, integer):
            if 0 <= integer < len(DEFAULT_CLASS_NAMES):
                return DEFAULT_CLASS_NAMES[integer]

            if 1 <= integer <= len(DEFAULT_CLASS_NAMES):
                return DEFAULT_CLASS_NAMES[integer - 1]
    except ValueError:
        pass

    return value


def load_known_labels(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
    requested_label_column: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    known_labels: List[str] = []
    bout_ids: List[str] = []
    detected_column: Optional[str] = None

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header.")

        required = {
            "row_index",
            "bout_id",
            "partition",
            "training_seed",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"{path} missing columns: {sorted(missing)}"
            )

        detected_column = detect_label_column(
            reader.fieldnames,
            requested_label_column,
        )

        for expected_row, row in enumerate(reader):
            if int(row["row_index"]) != expected_row:
                raise RuntimeError(
                    f"{path}: row_index mismatch at row {expected_row}."
                )

            if row["partition"] != partition:
                raise RuntimeError(
                    f"{path}: partition mismatch at row {expected_row}."
                )

            if int(row["training_seed"]) != ssl_seed:
                raise RuntimeError(
                    f"{path}: training_seed mismatch at row {expected_row}."
                )

            bout_id = row["bout_id"].strip()

            if not bout_id:
                raise RuntimeError(
                    f"{path}: empty bout_id at row {expected_row}."
                )

            known_labels.append(
                normalize_known_label(
                    row[detected_column]
                )
            )
            bout_ids.append(bout_id)

    if len(known_labels) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"observed {len(known_labels):,}"
        )

    return (
        np.asarray(known_labels, dtype=str),
        np.asarray(bout_ids, dtype=str),
        detected_column,
    )


def load_ssl_labels(
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
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.int64,
    )

    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected ({EXPECTED_ROWS[partition]},), "
            f"got {labels.shape}"
        )

    if not np.array_equal(
        np.unique(labels),
        np.arange(EXPECTED_SSL_K),
    ):
        raise RuntimeError(
            f"{path}: expected SSL labels 0..{EXPECTED_SSL_K - 1}."
        )

    return labels


def hungarian_map_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[int, int]:
    table = np.zeros(
        (EXPECTED_SSL_K, EXPECTED_SSL_K),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (
            reference_labels.astype(int),
            candidate_labels.astype(int),
        ),
        1,
    )

    row_ind, col_ind = linear_sum_assignment(-table)

    return {
        int(candidate): int(reference)
        for reference, candidate
        in zip(row_ind, col_ind)
    }


def apply_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray(
        [
            mapping[int(label)]
            for label in labels
        ],
        dtype=np.int16,
    )


def entropy_from_counts(
    counts: np.ndarray,
) -> float:
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    total = float(
        np.sum(counts)
    )

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total

    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def normalized_entropy(
    counts: np.ndarray,
) -> float:
    if len(counts) <= 1:
        return 0.0

    entropy = entropy_from_counts(
        counts
    )

    maximum = math.log(
        len(counts)
    )

    if maximum <= 0:
        return 0.0

    return float(
        entropy / maximum
    )


def effective_number(
    counts: np.ndarray,
) -> float:
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    total = float(
        np.sum(counts)
    )

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total

    return float(
        math.exp(
            -np.sum(
                p * np.log(p)
            )
        )
    )


def cluster_distribution(
    labels: np.ndarray,
) -> Dict[str, Any]:
    counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_SSL_K,
    ).astype(np.int64)

    total = int(
        np.sum(counts)
    )

    fractions = counts / max(
        total,
        1,
    )

    return {
        "count": total,
        "cluster_counts": (
            counts.astype(int).tolist()
        ),
        "cluster_fractions": (
            fractions.astype(float).tolist()
        ),
        "dominant_cluster": int(
            np.argmax(counts)
        ),
        "dominant_cluster_fraction": float(
            np.max(fractions)
        ),
        "normalized_cluster_entropy": (
            normalized_entropy(counts)
        ),
        "effective_number_of_clusters": (
            effective_number(counts)
        ),
        "clusters_with_any_bouts": int(
            np.sum(counts > 0)
        ),
        "clusters_with_at_least_5pct": int(
            np.sum(fractions >= 0.05)
        ),
        "clusters_with_at_least_10pct": int(
            np.sum(fractions >= 0.10)
        ),
    }


def pairwise_label_metrics(
    labels_by_seed: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    seeds = sorted(
        labels_by_seed
    )

    pairs: List[Dict[str, Any]] = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            a = labels_by_seed[seed_a]
            b = labels_by_seed[seed_b]

            if a.shape != b.shape:
                raise RuntimeError(
                    f"Within-class pair shape mismatch: "
                    f"seed {seed_a} {a.shape}, seed {seed_b} {b.shape}"
                )

            mapping = hungarian_map_to_reference(
                a,
                b,
            )

            b_aligned = apply_mapping(
                b,
                mapping,
            )

            pairs.append(
                {
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "ari": float(
                        adjusted_rand_score(
                            a,
                            b,
                        )
                    ),
                    "nmi": float(
                        normalized_mutual_info_score(
                            a,
                            b,
                        )
                    ),
                    "aligned_agreement": float(
                        np.mean(
                            a == b_aligned
                        )
                    ),
                }
            )

    if not pairs:
        return {
            "pair_count": 0,
            "pairs": [],
        }

    def summarize(name: str) -> Dict[str, float]:
        arr = np.asarray(
            [
                pair[name]
                for pair in pairs
            ],
            dtype=np.float64,
        )

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "pair_count": int(
            len(pairs)
        ),
        "ari": summarize("ari"),
        "nmi": summarize("nmi"),
        "aligned_agreement": (
            summarize(
                "aligned_agreement"
            )
        ),
        "pairs": pairs,
    }


def pairwise_js_similarity(
    distributions: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    seeds = sorted(distributions)
    pairs: List[Dict[str, Any]] = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            p = distributions[seed_a]
            q = distributions[seed_b]

            distance = float(
                jensenshannon(
                    p,
                    q,
                    base=2,
                )
            )

            pairs.append(
                {
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "similarity": float(
                        1.0 - distance
                    ),
                }
            )

    if not pairs:
        return {
            "pair_count": 0,
            "pairs": [],
        }

    values = np.asarray(
        [
            pair["similarity"]
            for pair in pairs
        ],
        dtype=np.float64,
    )

    return {
        "pair_count": int(
            len(pairs)
        ),
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
        "pairs": pairs,
    }


def candidate_score(
    validation_distribution_mean_effective_clusters: float,
    validation_pairwise_ari: float,
    validation_pairwise_js_similarity: float,
) -> float:
    """Descriptive prioritization score, NOT a preregistered inferential metric.

    Rewards:
    - more than one effective SSL cluster inside the known class;
    - stronger label-level cross-seed ARI;
    - stronger distribution-level JS similarity.

    This score is used only to rank candidate classes for manual follow-up.
    """
    subdivision_strength = min(
        max(
            validation_distribution_mean_effective_clusters - 1.0,
            0.0,
        ) / 3.0,
        1.0,
    )

    ari_component = min(
        max(validation_pairwise_ari, 0.0),
        1.0,
    )

    js_component = min(
        max(
            validation_pairwise_js_similarity,
            0.0,
        ),
        1.0,
    )

    return float(
        0.40 * subdivision_strength
        + 0.35 * ari_component
        + 0.25 * js_component
    )


def analyze_class(
    class_name: str,
    *,
    train_known: np.ndarray,
    validation_known: np.ndarray,
    train_ssl_by_seed: Mapping[int, np.ndarray],
    validation_ssl_by_seed: Mapping[int, np.ndarray],
    mappings: Mapping[int, Mapping[int, int]],
) -> Dict[str, Any]:
    train_mask = (
        train_known == class_name
    )
    validation_mask = (
        validation_known == class_name
    )

    train_count = int(
        np.sum(train_mask)
    )
    validation_count = int(
        np.sum(validation_mask)
    )

    if train_count == 0 or validation_count == 0:
        return {
            "class_name": class_name,
            "train_count": train_count,
            "validation_count": validation_count,
            "status": "insufficient_data",
        }

    partition_results: Dict[str, Any] = {}

    for partition, mask, labels_by_seed in (
        (
            "train",
            train_mask,
            train_ssl_by_seed,
        ),
        (
            "validation",
            validation_mask,
            validation_ssl_by_seed,
        ),
    ):
        aligned_within_class: Dict[
            int,
            np.ndarray,
        ] = {}

        per_seed_distribution: Dict[
            str,
            Any,
        ] = {}

        distribution_vectors: Dict[
            int,
            np.ndarray,
        ] = {}

        for seed, full_labels in labels_by_seed.items():
            subset = full_labels[
                mask
            ]

            aligned_subset = apply_mapping(
                subset,
                mappings[seed],
            )

            aligned_within_class[
                seed
            ] = aligned_subset

            distribution = cluster_distribution(
                aligned_subset
            )

            per_seed_distribution[
                str(seed)
            ] = distribution

            distribution_vectors[
                seed
            ] = np.asarray(
                distribution[
                    "cluster_fractions"
                ],
                dtype=np.float64,
            )

        pairwise = pairwise_label_metrics(
            aligned_within_class
        )

        js = pairwise_js_similarity(
            distribution_vectors
        )

        effective_values = np.asarray(
            [
                value[
                    "effective_number_of_clusters"
                ]
                for value in per_seed_distribution.values()
            ],
            dtype=np.float64,
        )

        entropy_values = np.asarray(
            [
                value[
                    "normalized_cluster_entropy"
                ]
                for value in per_seed_distribution.values()
            ],
            dtype=np.float64,
        )

        dominant_values = np.asarray(
            [
                value[
                    "dominant_cluster_fraction"
                ]
                for value in per_seed_distribution.values()
            ],
            dtype=np.float64,
        )

        five_pct_values = np.asarray(
            [
                value[
                    "clusters_with_at_least_5pct"
                ]
                for value in per_seed_distribution.values()
            ],
            dtype=np.float64,
        )

        partition_results[
            partition
        ] = {
            "count": int(
                np.sum(mask)
            ),
            "per_seed_distribution": (
                per_seed_distribution
            ),
            "distribution_summary": {
                "mean_effective_number_of_clusters": float(
                    np.mean(
                        effective_values
                    )
                ),
                "mean_normalized_cluster_entropy": float(
                    np.mean(
                        entropy_values
                    )
                ),
                "mean_dominant_cluster_fraction": float(
                    np.mean(
                        dominant_values
                    )
                ),
                "mean_clusters_with_at_least_5pct": float(
                    np.mean(
                        five_pct_values
                    )
                ),
            },
            "cross_seed_label_reproducibility": (
                pairwise
            ),
            "cross_seed_distribution_js_similarity": (
                js
            ),
        }

    val_effective = partition_results[
        "validation"
    ][
        "distribution_summary"
    ][
        "mean_effective_number_of_clusters"
    ]

    val_ari = partition_results[
        "validation"
    ][
        "cross_seed_label_reproducibility"
    ].get(
        "ari",
        {},
    ).get(
        "mean",
        0.0,
    )

    val_js = partition_results[
        "validation"
    ][
        "cross_seed_distribution_js_similarity"
    ].get(
        "mean",
        0.0,
    )

    score = candidate_score(
        val_effective,
        val_ari,
        val_js,
    )

    return {
        "class_name": class_name,
        "train_count": train_count,
        "validation_count": validation_count,
        "status": "analyzed",
        "train": partition_results["train"],
        "validation": (
            partition_results[
                "validation"
            ]
        ),
        "candidate_substructure_score": (
            score
        ),
        "candidate_score_note": (
            "Descriptive prioritization only; not a preregistered inferential metric."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test reproducible SSL substructure within conventional DS-005 "
            "known behavior classes."
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
        "--label-column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=(
            "Optional known-class subset. Default: analyze every observed "
            "known behavior class."
        ),
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

    for path in (
        metadata_root,
        label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    assert_no_test_artifacts(
        metadata_root
    )
    assert_no_test_artifacts(
        label_root
    )

    seeds = configured_ssl_seeds(
        training_config
    )

    if REFERENCE_SEED not in seeds:
        raise RuntimeError(
            f"Reference seed {REFERENCE_SEED} missing from frozen seeds {list(seeds)}."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_class_dir = (
        output_dir / "per_class"
    )
    per_class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir / "summary.json"
    )
    checksum_path = (
        output_dir
        / "WITHIN_CLASS_SUBSTRUCTURE_SHA256SUMS"
    )

    if (
        not args.overwrite
        and summary_path.exists()
    ):
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL WITHIN-KNOWN-CLASS SUBSTRUCTURE")
    print("=" * 80)
    print(f"SSL seeds:        {list(seeds)}")
    print(f"SSL clusters:     k={EXPECTED_SSL_K}")
    print(f"Reference seed:   {REFERENCE_SEED}")
    print("Alignment:        TRAIN-derived Hungarian mapping")
    print("Evaluation:       TRAIN + VALIDATION")
    print("TEST partition:   PROTECTED / NOT LOADED")
    print()

    train_ssl_by_seed: Dict[
        int,
        np.ndarray,
    ] = {}
    validation_ssl_by_seed: Dict[
        int,
        np.ndarray,
    ] = {}

    reference_known: Dict[
        str,
        np.ndarray,
    ] = {}
    reference_bouts: Dict[
        str,
        np.ndarray,
    ] = {}
    detected_columns: Dict[
        str,
        str,
    ] = {}

    for seed in seeds:
        for partition in PARTITIONS:
            known, bouts, detected = load_known_labels(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
                requested_label_column=(
                    args.label_column
                ),
            )

            ssl_labels = load_ssl_labels(
                label_root,
                ssl_seed=seed,
                partition=partition,
            )

            if partition not in reference_known:
                reference_known[
                    partition
                ] = known.copy()

                reference_bouts[
                    partition
                ] = bouts.copy()

                detected_columns[
                    partition
                ] = detected
            else:
                if not np.array_equal(
                    reference_known[
                        partition
                    ],
                    known,
                ):
                    raise RuntimeError(
                        f"Known labels differ across seeds for {partition}."
                    )

                if not np.array_equal(
                    reference_bouts[
                        partition
                    ],
                    bouts,
                ):
                    raise RuntimeError(
                        f"Bout ordering differs across seeds for {partition}."
                    )

            if partition == "train":
                train_ssl_by_seed[
                    seed
                ] = ssl_labels
            else:
                validation_ssl_by_seed[
                    seed
                ] = ssl_labels

    # Global SSL-cluster label alignment estimated once from TRAIN.
    reference_train_labels = (
        train_ssl_by_seed[
            REFERENCE_SEED
        ]
    )

    mappings: Dict[
        int,
        Dict[int, int],
    ] = {
        REFERENCE_SEED: {
            i: i
            for i in range(
                EXPECTED_SSL_K
            )
        }
    }

    for seed in seeds:
        if seed == REFERENCE_SEED:
            continue

        mappings[
            seed
        ] = hungarian_map_to_reference(
            reference_train_labels,
            train_ssl_by_seed[
                seed
            ],
        )

    observed_classes = sorted(
        set(
            reference_known[
                "train"
            ].tolist()
        )
        | set(
            reference_known[
                "validation"
            ].tolist()
        )
    )

    if args.classes is None:
        classes_to_analyze = (
            observed_classes
        )
    else:
        missing = [
            value
            for value in args.classes
            if value not in observed_classes
        ]

        if missing:
            raise RuntimeError(
                "Requested classes not observed: "
                + ", ".join(missing)
            )

        classes_to_analyze = (
            list(args.classes)
        )

    print(
        "Known classes analyzed: "
        + ", ".join(
            classes_to_analyze
        )
    )
    print()

    class_results: Dict[
        str,
        Dict[str, Any],
    ] = {}
    written_artifacts: List[
        Path
    ] = []

    for class_name in classes_to_analyze:
        result = analyze_class(
            class_name,
            train_known=reference_known[
                "train"
            ],
            validation_known=reference_known[
                "validation"
            ],
            train_ssl_by_seed=(
                train_ssl_by_seed
            ),
            validation_ssl_by_seed=(
                validation_ssl_by_seed
            ),
            mappings=mappings,
        )

        class_results[
            class_name
        ] = result

        output_path = (
            per_class_dir
            / f"{class_name}.json"
        )

        atomic_write_json(
            output_path,
            result,
        )

        written_artifacts.append(
            output_path
        )

        print("=" * 80)
        print(class_name)
        print("=" * 80)
        print(
            f"TRAIN count:      "
            f"{result['train_count']:,}"
        )
        print(
            f"VALIDATION count: "
            f"{result['validation_count']:,}"
        )

        if result[
            "status"
        ] != "analyzed":
            print(
                "Status: insufficient data"
            )
            print()
            continue

        validation = result[
            "validation"
        ]

        print(
            "Mean effective SSL clusters: "
            f"{validation['distribution_summary']['mean_effective_number_of_clusters']:.3f}"
        )
        print(
            "Mean dominant cluster fraction: "
            f"{validation['distribution_summary']['mean_dominant_cluster_fraction']:.3f}"
        )
        print(
            "Mean cross-seed ARI: "
            f"{validation['cross_seed_label_reproducibility']['ari']['mean']:.3f}"
        )
        print(
            "Mean cross-seed NMI: "
            f"{validation['cross_seed_label_reproducibility']['nmi']['mean']:.3f}"
        )
        print(
            "Mean aligned agreement: "
            f"{validation['cross_seed_label_reproducibility']['aligned_agreement']['mean']:.3f}"
        )
        print(
            "Mean JS similarity: "
            f"{validation['cross_seed_distribution_js_similarity']['mean']:.3f}"
        )
        print(
            "Candidate substructure score: "
            f"{result['candidate_substructure_score']:.3f}"
        )
        print()

    ranked = sorted(
        [
            {
                "class_name": name,
                "candidate_substructure_score": result[
                    "candidate_substructure_score"
                ],
                "validation_count": result[
                    "validation_count"
                ],
                "validation_mean_effective_clusters": result[
                    "validation"
                ][
                    "distribution_summary"
                ][
                    "mean_effective_number_of_clusters"
                ],
                "validation_mean_cross_seed_ari": result[
                    "validation"
                ][
                    "cross_seed_label_reproducibility"
                ][
                    "ari"
                ][
                    "mean"
                ],
                "validation_mean_cross_seed_nmi": result[
                    "validation"
                ][
                    "cross_seed_label_reproducibility"
                ][
                    "nmi"
                ][
                    "mean"
                ],
                "validation_mean_js_similarity": result[
                    "validation"
                ][
                    "cross_seed_distribution_js_similarity"
                ][
                    "mean"
                ],
            }
            for name, result
            in class_results.items()
            if result.get(
                "status"
            ) == "analyzed"
        ],
        key=lambda item: (
            -item[
                "candidate_substructure_score"
            ],
            -item[
                "validation_count"
            ],
            item[
                "class_name"
            ],
        ),
    )

    summary = {
        "dataset_id": "DS-005",
        "representation": (
            "ssl_encoder_embedding"
        ),
        "selected_clustering": {
            "method": "kmeans",
            "k": EXPECTED_SSL_K,
        },
        "ssl_training_seeds": list(
            seeds
        ),
        "reference_seed": (
            REFERENCE_SEED
        ),
        "known_label_columns": (
            detected_columns
        ),
        "classes_analyzed": (
            classes_to_analyze
        ),
        "train_derived_cluster_mappings_to_reference": {
            str(seed): {
                str(src): int(dst)
                for src, dst
                in mapping.items()
            }
            for seed, mapping
            in mappings.items()
        },
        "candidate_ranking": (
            ranked
        ),
        "per_class": (
            class_results
        ),
        "interpretation_guardrails": {
            "candidate_score_is_descriptive_only": True,
            "reproducible_subdivision_is_not_proof_of_new_behavior": True,
            "follow_up_characterization_required": [
                "speed",
                "turning",
                "orientation",
                "stimulus_context",
                "handcrafted_features",
            ],
            "test_partition_used": False,
        },
        "test_partition_used": False,
    }

    atomic_write_json(
        summary_path,
        summary,
    )

    written_artifacts.append(
        summary_path
    )

    write_checksums(
        checksum_path,
        written_artifacts,
    )

    print("=" * 80)
    print("WITHIN-CLASS SUBSTRUCTURE SUMMARY")
    print("=" * 80)

    for rank, item in enumerate(
        ranked[:10],
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{item['class_name']:<12} "
            f"score={item['candidate_substructure_score']:.3f}  "
            f"eff_clusters={item['validation_mean_effective_clusters']:.2f}  "
            f"ARI={item['validation_mean_cross_seed_ari']:.3f}  "
            f"NMI={item['validation_mean_cross_seed_nmi']:.3f}  "
            f"JS={item['validation_mean_js_similarity']:.3f}  "
            f"n={item['validation_count']:,}"
        )

    print()
    print("TEST partition used: NO")
    print(f"Summary:    {summary_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
