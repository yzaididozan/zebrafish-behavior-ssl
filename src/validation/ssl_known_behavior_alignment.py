#!/usr/bin/env python3
"""Known-behavior alignment for selected DS-005 SSL clusters.

Purpose
-------
Interpret the selected SSL k=8 clusters using existing DS-005 bout labels
WITHOUT having used those labels for SSL training or clustering.

Questions
---------
1. Do SSL clusters align with known zebrafish bout categories?
2. Which known behaviors dominate each SSL cluster?
3. Are known behaviors subdivided across multiple SSL clusters?
4. Are cluster-to-behavior relationships reproducible across SSL seeds?
5. Do the same relationships generalize from TRAIN to held-out VALIDATION fish?

Known DS-005 JM bout classes
----------------------------
If a numeric ``bout_type`` column is present, the script can map indices to:

    Short_CS, Long_CS, BS, O_bend, J_turn, SLC, Slow1,
    RT, Slow2, LLC, AS, SAT, HAT

The script first tries to load ``classnames_jm.npy`` from the dataset tree.
If unavailable, it falls back to the class list above.

TRAIN and VALIDATION only. TEST is never loaded.

Primary metrics
---------------
For each SSL seed and partition:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Adjusted Mutual Information (AMI)
- homogeneity
- completeness
- V-measure
- cluster purity
- known-class purity
- normalized conditional entropy H(known | SSL)
- normalized conditional entropy H(SSL | known)

Interpretation tables
---------------------
For every SSL cluster:
- known-class counts/fractions
- dominant known class
- dominant-class fraction
- normalized known-class entropy
- effective number of known classes

For every known bout class:
- SSL-cluster counts/fractions
- dominant SSL cluster
- dominant-cluster fraction
- normalized SSL-cluster entropy
- effective number of SSL clusters

Cross-seed biological consistency
---------------------------------
Cluster IDs are arbitrary across independently trained encoders. Therefore:
- seed 11 is used only as a deterministic reference label numbering.
- each other seed is aligned to seed 11 using TRAIN cluster assignments and
  Hungarian matching.
- the TRAIN-derived mapping is then applied unchanged to VALIDATION.
- for each aligned SSL cluster, the script compares known-class distributions
  between seeds using Jensen-Shannon similarity (1 - JS distance).

This allows statements such as:
    "Aligned cluster 3 has a similar known-behavior composition across seeds."

Important guardrail
-------------------
Alignment with existing labels does not make an SSL cluster a biological
behavior automatically. Conversely, low alignment can reflect:
- meaningful substructure inside conventional classes,
- boundaries different from the historical labels,
- continuous behavior,
- or unstable/noisy clustering.

Existing labels are used here only for post-hoc interpretation/validation.

Outputs
-------
data/processed/DS-005/ssl_known_behavior_alignment/
    seed11/
        train_alignment.json
        validation_alignment.json
    seed23/
        ...
    cross_seed_biological_consistency.json
    aggregate_summary.json
    KNOWN_BEHAVIOR_ALIGNMENT_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ssl_known_behavior_alignment.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ssl_known_behavior_alignment.py --overwrite

Optional explicit metadata label column:

    PYTHONPATH=. python3 src/validation/ssl_known_behavior_alignment.py \
        --label-column bout_type

Optional explicit class-name file:

    PYTHONPATH=. python3 src/validation/ssl_known_behavior_alignment.py \
        --classnames path/to/classnames_jm.npy
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
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_LABEL_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_known_behavior_alignment"
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

CLASSNAME_CANDIDATES = (
    REPO_ROOT / "data" / "raw" / "DS-005-v1" / "Datasets" / "JM_data" / "classnames_jm.npy",
    REPO_ROOT / "data" / "DS-005-v1" / "Datasets" / "JM_data" / "classnames_jm.npy",
    REPO_ROOT / "DS-005-v1" / "Datasets" / "JM_data" / "classnames_jm.npy",
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
            f"TEST access prohibited during known-behavior alignment: {path}"
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


def load_class_names(explicit: Optional[Path]) -> Tuple[List[str], str]:
    if explicit is not None:
        path = explicit.resolve()
        prohibit_test_path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        names = np.asarray(
            np.load(path, allow_pickle=False)
        ).astype(str).tolist()

        return names, str(path)

    for path in CLASSNAME_CANDIDATES:
        if path.exists():
            names = np.asarray(
                np.load(path, allow_pickle=False)
            ).astype(str).tolist()

            return names, str(path)

    return list(DEFAULT_CLASS_NAMES), "built_in_DS005_JM_classnames"


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


def normalize_known_label(
    raw: str,
    class_names: Sequence[str],
) -> str:
    value = raw.strip()

    if value == "":
        return "__MISSING__"

    # Direct textual class name.
    if value in class_names:
        return value

    # Numeric bout type. Accept strings such as "4", "4.0", etc.
    try:
        numeric = float(value)
        integer = int(numeric)

        if math.isclose(numeric, integer):
            if 0 <= integer < len(class_names):
                return class_names[integer]

            # Some external tables are one-indexed. Only use this fallback
            # if zero-indexing is impossible for that value.
            if 1 <= integer <= len(class_names):
                return class_names[integer - 1]
    except ValueError:
        pass

    # Preserve unexpected labels explicitly rather than dropping them.
    return value


def load_known_labels(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
    class_names: Sequence[str],
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

    labels: List[str] = []
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
                f"{path} missing required metadata columns: {sorted(missing)}"
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

            labels.append(
                normalize_known_label(
                    row[detected_column],
                    class_names,
                )
            )
            bout_ids.append(bout_id)

    if len(labels) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"observed {len(labels):,}"
        )

    return (
        np.asarray(labels, dtype=str),
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
            f"{path}: expected SSL cluster labels 0..{EXPECTED_SSL_K - 1}."
        )

    return labels


def encode_strings(
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    categories = sorted(set(labels.astype(str).tolist()))
    index = {
        category: i
        for i, category in enumerate(categories)
    }

    encoded = np.asarray(
        [index[value] for value in labels.astype(str)],
        dtype=np.int64,
    )

    return encoded, categories


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total

    return float(
        -np.sum(p * np.log(p))
    )


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    nonzero_categories = int(np.sum(counts > 0))

    if nonzero_categories <= 1:
        return 0.0

    entropy = entropy_from_counts(counts)
    maximum = math.log(len(counts))

    if maximum <= 0:
        return 0.0

    return float(entropy / maximum)


def effective_number(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total

    return float(
        math.exp(
            -np.sum(p * np.log(p))
        )
    )


def normalized_conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Return H(Y | X) / H(Y)."""
    x_values = sorted(set(x.tolist()))
    y_values = sorted(set(y.tolist()))

    x_index = {value: i for i, value in enumerate(x_values)}
    y_index = {value: i for i, value in enumerate(y_values)}

    table = np.zeros(
        (len(x_values), len(y_values)),
        dtype=np.int64,
    )

    for xv, yv in zip(x, y):
        table[x_index[xv], y_index[yv]] += 1

    h_y = entropy_from_counts(
        np.sum(table, axis=0)
    )

    if h_y <= 0:
        return 0.0

    total = float(np.sum(table))
    conditional = 0.0

    for row in table:
        row_total = float(np.sum(row))

        if row_total <= 0:
            continue

        conditional += (
            row_total / total
        ) * entropy_from_counts(row)

    return float(conditional / h_y)


def contingency_ssl_x_known(
    ssl_labels: np.ndarray,
    known_labels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    known_encoded, known_names = encode_strings(known_labels)

    table = np.zeros(
        (EXPECTED_SSL_K, len(known_names)),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (
            ssl_labels.astype(int),
            known_encoded.astype(int),
        ),
        1,
    )

    return table, known_names


def cluster_purity(table: np.ndarray) -> float:
    total = float(np.sum(table))

    if total <= 0:
        return 0.0

    return float(
        np.sum(np.max(table, axis=1))
        / total
    )


def known_class_purity(table: np.ndarray) -> float:
    total = float(np.sum(table))

    if total <= 0:
        return 0.0

    return float(
        np.sum(np.max(table, axis=0))
        / total
    )


def interpret_ssl_clusters(
    table: np.ndarray,
    known_names: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cluster in range(EXPECTED_SSL_K):
        counts = table[cluster].astype(np.float64)
        total = float(np.sum(counts))

        if total <= 0:
            raise RuntimeError(f"SSL cluster {cluster} is empty.")

        fractions = counts / total
        dominant_idx = int(np.argmax(counts))

        out[str(cluster)] = {
            "count": int(total),
            "known_class_counts": {
                known_names[i]: int(counts[i])
                for i in range(len(known_names))
            },
            "known_class_fractions": {
                known_names[i]: float(fractions[i])
                for i in range(len(known_names))
            },
            "dominant_known_class": known_names[dominant_idx],
            "dominant_known_class_fraction": float(
                fractions[dominant_idx]
            ),
            "normalized_known_class_entropy": normalized_entropy(counts),
            "effective_number_of_known_classes": effective_number(counts),
        }

    return out


def interpret_known_classes(
    table: np.ndarray,
    known_names: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for known_idx, known_name in enumerate(known_names):
        counts = table[:, known_idx].astype(np.float64)
        total = float(np.sum(counts))

        if total <= 0:
            continue

        fractions = counts / total
        dominant_cluster = int(np.argmax(counts))

        out[known_name] = {
            "count": int(total),
            "ssl_cluster_counts": counts.astype(int).tolist(),
            "ssl_cluster_fractions": fractions.astype(float).tolist(),
            "dominant_ssl_cluster": dominant_cluster,
            "dominant_ssl_cluster_fraction": float(
                fractions[dominant_cluster]
            ),
            "normalized_ssl_cluster_entropy": normalized_entropy(counts),
            "effective_number_of_ssl_clusters": effective_number(counts),
        }

    return out


def alignment_metrics(
    ssl_labels: np.ndarray,
    known_labels: np.ndarray,
) -> Dict[str, Any]:
    known_encoded, known_names = encode_strings(known_labels)

    table, table_known_names = contingency_ssl_x_known(
        ssl_labels,
        known_labels,
    )

    if known_names != table_known_names:
        raise RuntimeError("Known-label encoding mismatch.")

    return {
        "known_class_count": int(len(known_names)),
        "known_class_names": known_names,
        "adjusted_rand_index": float(
            adjusted_rand_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "homogeneity": float(
            homogeneity_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "completeness": float(
            completeness_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "v_measure": float(
            v_measure_score(
                known_encoded,
                ssl_labels,
            )
        ),
        "cluster_purity": cluster_purity(table),
        "known_class_purity": known_class_purity(table),
        "normalized_H_known_given_ssl": normalized_conditional_entropy(
            ssl_labels.astype(int),
            known_encoded.astype(int),
        ),
        "normalized_H_ssl_given_known": normalized_conditional_entropy(
            known_encoded.astype(int),
            ssl_labels.astype(int),
        ),
        "contingency_ssl_x_known": table.astype(int).tolist(),
        "ssl_cluster_interpretation": interpret_ssl_clusters(
            table,
            known_names,
        ),
        "known_class_substructure": interpret_known_classes(
            table,
            known_names,
        ),
    }


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
        for reference, candidate in zip(row_ind, col_ind)
    }


def apply_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray(
        [mapping[int(label)] for label in labels],
        dtype=np.int16,
    )


def distribution_for_aligned_cluster(
    labels: np.ndarray,
    known_labels: np.ndarray,
    *,
    cluster: int,
    global_known_names: Sequence[str],
) -> np.ndarray:
    selected = known_labels[labels == cluster]

    counts = np.zeros(
        len(global_known_names),
        dtype=np.float64,
    )

    index = {
        name: i
        for i, name in enumerate(global_known_names)
    }

    for value in selected.astype(str):
        if value in index:
            counts[index[value]] += 1.0

    total = float(np.sum(counts))

    if total <= 0:
        return counts

    return counts / total


def pairwise_js_similarity(
    distributions: Sequence[np.ndarray],
) -> List[float]:
    similarities: List[float] = []

    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            p = distributions[i]
            q = distributions[j]

            if np.sum(p) <= 0 or np.sum(q) <= 0:
                continue

            distance = float(
                jensenshannon(
                    p,
                    q,
                    base=2,
                )
            )

            similarities.append(
                float(1.0 - distance)
            )

    return similarities


def cross_seed_biological_consistency(
    train_ssl_by_seed: Mapping[int, np.ndarray],
    validation_ssl_by_seed: Mapping[int, np.ndarray],
    train_known: np.ndarray,
    validation_known: np.ndarray,
) -> Dict[str, Any]:
    if REFERENCE_SEED not in train_ssl_by_seed:
        raise RuntimeError(
            f"Reference seed {REFERENCE_SEED} not available."
        )

    train_reference = train_ssl_by_seed[
        REFERENCE_SEED
    ]

    mappings: Dict[int, Dict[int, int]] = {
        REFERENCE_SEED: {
            i: i
            for i in range(EXPECTED_SSL_K)
        }
    }

    for seed, labels in train_ssl_by_seed.items():
        if seed == REFERENCE_SEED:
            continue

        mappings[seed] = hungarian_map_to_reference(
            train_reference,
            labels,
        )

    aligned_train = {
        seed: apply_mapping(
            labels,
            mappings[seed],
        )
        for seed, labels
        in train_ssl_by_seed.items()
    }

    # Critical: reuse TRAIN-derived mappings on VALIDATION.
    aligned_validation = {
        seed: apply_mapping(
            labels,
            mappings[seed],
        )
        for seed, labels
        in validation_ssl_by_seed.items()
    }

    global_known_names = sorted(
        set(train_known.astype(str).tolist())
        | set(validation_known.astype(str).tolist())
    )

    partition_results: Dict[str, Any] = {}

    for partition, labels_by_seed, known in (
        ("train", aligned_train, train_known),
        ("validation", aligned_validation, validation_known),
    ):
        per_cluster: Dict[str, Any] = {}
        all_similarities: List[float] = []

        for cluster in range(EXPECTED_SSL_K):
            distributions = [
                distribution_for_aligned_cluster(
                    labels_by_seed[seed],
                    known,
                    cluster=cluster,
                    global_known_names=global_known_names,
                )
                for seed in sorted(labels_by_seed)
            ]

            similarities = pairwise_js_similarity(
                distributions
            )

            all_similarities.extend(similarities)

            per_cluster[str(cluster)] = {
                "mean_pairwise_js_similarity": (
                    float(np.mean(similarities))
                    if similarities
                    else None
                ),
                "min_pairwise_js_similarity": (
                    float(np.min(similarities))
                    if similarities
                    else None
                ),
                "pair_count": int(
                    len(similarities)
                ),
            }

        partition_results[partition] = {
            "mean_pairwise_js_similarity_all_clusters": (
                float(np.mean(all_similarities))
                if all_similarities
                else None
            ),
            "min_pairwise_js_similarity_all_clusters": (
                float(np.min(all_similarities))
                if all_similarities
                else None
            ),
            "per_aligned_cluster": per_cluster,
        }

    return {
        "reference_seed": REFERENCE_SEED,
        "alignment_rule": (
            "Hungarian mapping estimated on TRAIN same-bout cluster assignments; "
            "same mapping applied unchanged to VALIDATION."
        ),
        "train_derived_cluster_mappings_to_reference": {
            str(seed): {
                str(src): int(dst)
                for src, dst in mapping.items()
            }
            for seed, mapping in mappings.items()
        },
        "known_class_names": global_known_names,
        **partition_results,
        "test_partition_used": False,
    }


def aggregate_seed_metrics(
    per_seed: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    metric_names = (
        "adjusted_rand_index",
        "normalized_mutual_information",
        "adjusted_mutual_information",
        "homogeneity",
        "completeness",
        "v_measure",
        "cluster_purity",
        "known_class_purity",
        "normalized_H_known_given_ssl",
        "normalized_H_ssl_given_known",
    )

    output: Dict[str, Any] = {}

    for partition in PARTITIONS:
        output[partition] = {}

        for metric in metric_names:
            values = np.asarray(
                [
                    result[partition][metric]
                    for result
                    in per_seed.values()
                ],
                dtype=np.float64,
            )

            output[partition][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpret DS-005 SSL k=8 clusters using known bout labels, "
            "TRAIN/VALIDATION only."
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
        help=(
            "Known-behavior metadata column. If omitted, tries common "
            "DS-005 names such as bout_type."
        ),
    )

    parser.add_argument(
        "--classnames",
        type=Path,
        default=None,
        help=(
            "Optional classnames_jm.npy path. If omitted, common DS-005 "
            "locations are searched and then a built-in class list is used."
        ),
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

    for path in (
        metadata_root,
        label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    if args.classnames is not None:
        prohibit_test_path(args.classnames.resolve())

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
            f"Expected reference seed {REFERENCE_SEED} among {list(seeds)}."
        )

    class_names, class_name_source = load_class_names(
        args.classnames
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        output_dir
        / "aggregate_summary.json"
    )

    cross_seed_path = (
        output_dir
        / "cross_seed_biological_consistency.json"
    )

    checksum_path = (
        output_dir
        / "KNOWN_BEHAVIOR_ALIGNMENT_SHA256SUMS"
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
    print("DS-005 SSL KNOWN-BEHAVIOR ALIGNMENT")
    print("=" * 80)
    print(f"SSL seeds:          {list(seeds)}")
    print(f"SSL clusters:       k={EXPECTED_SSL_K}")
    print(f"Reference seed:     {REFERENCE_SEED}")
    print(f"Class-name source:  {class_name_source}")
    print(f"Known class names:  {class_names}")
    print("Evaluation:         TRAIN + VALIDATION")
    print("TEST partition:     PROTECTED / NOT LOADED")
    print()

    per_seed: Dict[int, Dict[str, Any]] = {}
    train_ssl_by_seed: Dict[int, np.ndarray] = {}
    validation_ssl_by_seed: Dict[int, np.ndarray] = {}
    written_artifacts: List[Path] = []

    reference_known_by_partition: Dict[str, np.ndarray] = {}
    reference_bouts_by_partition: Dict[str, np.ndarray] = {}
    detected_label_columns: Dict[str, str] = {}

    for seed in seeds:
        print("=" * 80)
        print(f"SSL SEED {seed}")
        print("=" * 80)

        seed_result: Dict[str, Any] = {
            "ssl_seed": int(seed),
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
            known_labels, bout_ids, detected_column = load_known_labels(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
                class_names=class_names,
                requested_label_column=args.label_column,
            )

            ssl_labels = load_ssl_labels(
                label_root,
                ssl_seed=seed,
                partition=partition,
            )

            if known_labels.shape[0] != ssl_labels.shape[0]:
                raise RuntimeError(
                    f"seed={seed} {partition}: known-label/SSL-label "
                    "length mismatch."
                )

            if partition not in reference_known_by_partition:
                reference_known_by_partition[partition] = (
                    known_labels.copy()
                )
                reference_bouts_by_partition[partition] = (
                    bout_ids.copy()
                )
                detected_label_columns[partition] = (
                    detected_column
                )
            else:
                if not np.array_equal(
                    reference_bouts_by_partition[partition],
                    bout_ids,
                ):
                    raise RuntimeError(
                        f"seed={seed} {partition}: bout ordering differs "
                        "across SSL seeds."
                    )

                if not np.array_equal(
                    reference_known_by_partition[partition],
                    known_labels,
                ):
                    raise RuntimeError(
                        f"seed={seed} {partition}: known behavior labels "
                        "differ across SSL seeds."
                    )

            metrics = alignment_metrics(
                ssl_labels,
                known_labels,
            )

            seed_result[partition] = metrics

            if partition == "train":
                train_ssl_by_seed[seed] = (
                    ssl_labels
                )
            else:
                validation_ssl_by_seed[seed] = (
                    ssl_labels
                )

            output_path = (
                seed_dir
                / f"{partition}_alignment.json"
            )

            atomic_write_json(
                output_path,
                {
                    "dataset_id": "DS-005",
                    "ssl_seed": int(seed),
                    "partition": partition,
                    "ssl_k": EXPECTED_SSL_K,
                    "known_label_column": detected_column,
                    "class_name_source": class_name_source,
                    **metrics,
                    "test_partition_used": False,
                },
            )

            written_artifacts.append(
                output_path
            )

            print(partition.upper())
            print(
                f"  Known classes:            "
                f"{metrics['known_class_count']}"
            )
            print(
                f"  ARI:                      "
                f"{metrics['adjusted_rand_index']:.6f}"
            )
            print(
                f"  NMI:                      "
                f"{metrics['normalized_mutual_information']:.6f}"
            )
            print(
                f"  AMI:                      "
                f"{metrics['adjusted_mutual_information']:.6f}"
            )
            print(
                f"  V-measure:                "
                f"{metrics['v_measure']:.6f}"
            )
            print(
                f"  Cluster purity:           "
                f"{metrics['cluster_purity']:.6f}"
            )
            print(
                f"  Known-class purity:       "
                f"{metrics['known_class_purity']:.6f}"
            )
            print(
                f"  H(known|SSL)/H(known):    "
                f"{metrics['normalized_H_known_given_ssl']:.6f}"
            )
            print(
                f"  H(SSL|known)/H(SSL):      "
                f"{metrics['normalized_H_ssl_given_known']:.6f}"
            )

            dominant = {
                cluster: data["dominant_known_class"]
                for cluster, data
                in metrics[
                    "ssl_cluster_interpretation"
                ].items()
            }

            print(
                f"  Dominant known classes:   "
                f"{dominant}"
            )
            print()

        per_seed[seed] = seed_result
        print("TEST partition used: NO")
        print()

    cross_seed = cross_seed_biological_consistency(
        train_ssl_by_seed,
        validation_ssl_by_seed,
        reference_known_by_partition["train"],
        reference_known_by_partition["validation"],
    )

    atomic_write_json(
        cross_seed_path,
        cross_seed,
    )
    written_artifacts.append(
        cross_seed_path
    )

    aggregate_metrics = aggregate_seed_metrics(
        per_seed
    )

    aggregate_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "selected_clustering": {
            "method": "kmeans",
            "k": EXPECTED_SSL_K,
        },
        "ssl_training_seeds": list(
            seeds
        ),
        "known_label_columns": detected_label_columns,
        "class_name_source": class_name_source,
        "class_names": class_names,
        "aggregate_metrics": aggregate_metrics,
        "cross_seed_biological_consistency": {
            "reference_seed": REFERENCE_SEED,
            "train_mean_pairwise_js_similarity": (
                cross_seed["train"][
                    "mean_pairwise_js_similarity_all_clusters"
                ]
            ),
            "validation_mean_pairwise_js_similarity": (
                cross_seed["validation"][
                    "mean_pairwise_js_similarity_all_clusters"
                ]
            ),
            "alignment_rule": cross_seed[
                "alignment_rule"
            ],
        },
        "per_seed": {
            str(seed): result
            for seed, result
            in per_seed.items()
        },
        "interpretation_guardrails": {
            "labels_not_used_for_training_or_clustering": True,
            "cluster_is_not_automatically_a_behavior": True,
            "low_alignment_can_indicate_substructure": True,
            "substructure_requires_reproducibility_and_characterization": True,
            "test_partition_used": False,
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

    validation = aggregate_metrics[
        "validation"
    ]

    print("=" * 80)
    print("KNOWN-BEHAVIOR ALIGNMENT SUMMARY")
    print("=" * 80)
    print(
        "Mean VALIDATION ARI:                 "
        f"{validation['adjusted_rand_index']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION NMI:                 "
        f"{validation['normalized_mutual_information']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION AMI:                 "
        f"{validation['adjusted_mutual_information']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION V-measure:           "
        f"{validation['v_measure']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION cluster purity:      "
        f"{validation['cluster_purity']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION known-class purity:  "
        f"{validation['known_class_purity']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION H(known|SSL)/H(known): "
        f"{validation['normalized_H_known_given_ssl']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION H(SSL|known)/H(SSL): "
        f"{validation['normalized_H_ssl_given_known']['mean']:.6f}"
    )
    print(
        "Cross-seed VALIDATION biological JS similarity: "
        f"{cross_seed['validation']['mean_pairwise_js_similarity_all_clusters']:.6f}"
    )
    print()
    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Cross-seed: {cross_seed_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
