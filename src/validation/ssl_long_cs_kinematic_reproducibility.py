#!/usr/bin/env python3
"""Reproducibility of Long_CS kinematic SSL substructure across seeds.

Purpose
-------
Test whether the visually observed Long_CS differences in bout duration and
acceleration are reproducible across independently trained SSL encoders and
held-out VALIDATION fish.

Primary question
----------------
Within the conventional Long_CS bout class, do aligned SSL subclusters show
consistent differences in:

    bout_duration_s
    accel_rms
    accel_abs_std

across SSL seeds 11, 23, 37, 51, and 79?

Design
------
- Restrict analysis to known class Long_CS only.
- Use the selected SSL KMeans k=8 labels.
- Use seed 11 as the deterministic reference cluster numbering.
- Estimate Hungarian cluster mappings from TRAIN assignments only.
- Apply those TRAIN-derived mappings unchanged to VALIDATION.
- Evaluate TRAIN + VALIDATION only.
- TEST is never loaded.

Metrics
-------
For each feature, SSL seed, and partition:
- Long_CS bout count
- per-aligned-subcluster:
    count
    fraction
    mean
    median
    standard deviation
    p25
    p75
    IQR
- eta-squared:
    feature ~ aligned SSL subcluster
- standardized range of subcluster means

Across seeds:
- pairwise Spearman correlation of aligned 8-subcluster mean profiles
- pairwise Spearman correlation of aligned 8-subcluster median profiles
- TRAIN-to-VALIDATION profile Spearman for every seed
- aggregate eta-squared across seeds
- aggregate profile reproducibility across seeds

Guardrails
----------
This is a targeted post-hoc reproducibility analysis.

Strong results support reproducible kinematic heterogeneity within Long_CS,
but do not by themselves establish distinct biological behaviors.

Outputs
-------
data/processed/DS-005/ssl_long_cs_kinematic_reproducibility/
    seed11/
        train.json
        validation.json
    seed23/
        ...
    cross_seed_summary.json
    summary.json
    LONG_CS_KINEMATIC_REPRODUCIBILITY_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 \
        src/validation/ssl_long_cs_kinematic_reproducibility.py

Intentional rerun:

    PYTHONPATH=. python3 \
        src/validation/ssl_long_cs_kinematic_reproducibility.py \
        --overwrite
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
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
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
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT / "configs" / "ssl" / "training.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_long_cs_kinematic_reproducibility"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_FEATURES = 18
EXPECTED_SSL_K = 8
REFERENCE_SEED = 11
KNOWN_CLASS = "Long_CS"
TARGET_FEATURES = (
    "bout_duration_s",
    "accel_rms",
    "accel_abs_std",
)
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

    payload = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

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


def write_checksums(
    path: Path,
    artifacts: Sequence[Path],
) -> None:
    path.write_text(
        "".join(
            f"{sha256_file(artifact)}  "
            f"{artifact.relative_to(path.parent)}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )


def prohibit_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(
            "TEST access prohibited during Long_CS "
            f"kinematic reproducibility: {path}"
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
            or name
            in {
                "test.npy",
                "test.npz",
                "test.csv",
                "test.json",
            }
        ):
            hits.append(path)

    if hits:
        raise RuntimeError(
            "Protected TEST artifacts found beneath an SSL "
            "input root; refusing to continue:\n"
            + "\n".join(
                str(path)
                for path in hits[:20]
            )
        )


def configured_ssl_seeds(
    path: Path,
) -> Tuple[int, ...]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        obj = yaml.safe_load(handle)

    seeds = (
        obj.get("training", {})
        .get("seeds", {})
        .get("values")
    )

    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError(
            "No frozen SSL seeds found in training.yaml."
        )

    return tuple(
        int(seed)
        for seed in seeds
    )


def detect_label_column(
    fieldnames: Sequence[str],
    requested: Optional[str],
) -> str:
    available = set(fieldnames)

    if requested is not None:
        if requested not in available:
            raise RuntimeError(
                f"Requested label column {requested!r} "
                "is absent. Available columns: "
                f"{sorted(available)}"
            )

        return requested

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate

    raise RuntimeError(
        "Could not locate known-behavior label column. "
        f"Tried {list(LABEL_COLUMN_CANDIDATES)}."
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

        if math.isclose(
            numeric,
            integer,
        ):
            if (
                0
                <= integer
                < len(DEFAULT_CLASS_NAMES)
            ):
                return DEFAULT_CLASS_NAMES[
                    integer
                ]

            if (
                1
                <= integer
                <= len(DEFAULT_CLASS_NAMES)
            ):
                return DEFAULT_CLASS_NAMES[
                    integer - 1
                ]
    except ValueError:
        pass

    return value


def load_baseline_raw(
    baseline_dir: Path,
    *,
    partition: str,
) -> Tuple[
    np.ndarray,
    List[str],
    Dict[str, np.ndarray],
]:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    path = (
        baseline_dir
        / f"{partition}_core_raw.npz"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        required = {
            "X",
            "feature_names",
            "fish_id",
            "session_id",
            "bout_index",
            "partition",
            "context_id",
        }

        missing = (
            required
            - set(npz.files)
        )

        if missing:
            raise RuntimeError(
                f"{path} missing arrays: "
                f"{sorted(missing)}"
            )

        X = np.asarray(
            npz["X"],
            dtype=np.float64,
        )

        feature_names = np.asarray(
            npz["feature_names"]
        ).astype(str).tolist()

        alignment = {
            "fish_id": np.asarray(
                npz["fish_id"]
            ).astype(str),
            "session_id": np.asarray(
                npz["session_id"]
            ).astype(str),
            "bout_index": np.asarray(
                npz["bout_index"]
            ).astype(str),
            "partition": np.asarray(
                npz["partition"]
            ).astype(str),
            "context_id": np.asarray(
                npz["context_id"]
            ).astype(str),
        }

    expected_shape = (
        EXPECTED_ROWS[partition],
        EXPECTED_FEATURES,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"{path}: expected {expected_shape}, "
            f"got {X.shape}."
        )

    missing_features = [
        feature
        for feature in TARGET_FEATURES
        if feature
        not in feature_names
    ]

    if missing_features:
        raise RuntimeError(
            "Required Long_CS features are absent: "
            f"{missing_features}. "
            f"Available features: {feature_names}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{path}: non-finite values in "
            "handcrafted features."
        )

    return (
        X,
        feature_names,
        alignment,
    )


def load_metadata(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
    requested_label_column: Optional[str],
) -> Tuple[
    Dict[str, np.ndarray],
    str,
]:
    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    fields: Dict[
        str,
        List[str],
    ] = {
        "fish_id": [],
        "session_id": [],
        "bout_index": [],
        "partition": [],
        "context_id": [],
        "bout_id": [],
        "known_label": [],
    }

    detected_label_column: Optional[
        str
    ] = None

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"{path}: CSV header missing."
            )

        detected_label_column = (
            detect_label_column(
                reader.fieldnames,
                requested_label_column,
            )
        )

        required = {
            "row_index",
            "fish_id",
            "session_id",
            "bout_index",
            "partition",
            "context_id",
            "bout_id",
            "training_seed",
        }

        missing = (
            required
            - set(reader.fieldnames)
        )

        if missing:
            raise RuntimeError(
                f"{path} missing required fields: "
                f"{sorted(missing)}"
            )

        for expected_row, row in enumerate(
            reader
        ):
            if (
                int(row["row_index"])
                != expected_row
            ):
                raise RuntimeError(
                    f"{path}: row_index mismatch "
                    f"at row {expected_row}."
                )

            if (
                row["partition"]
                != partition
            ):
                raise RuntimeError(
                    f"{path}: partition mismatch "
                    f"at row {expected_row}."
                )

            if (
                int(row["training_seed"])
                != ssl_seed
            ):
                raise RuntimeError(
                    f"{path}: training_seed mismatch "
                    f"at row {expected_row}."
                )

            fields[
                "fish_id"
            ].append(
                row["fish_id"]
            )

            fields[
                "session_id"
            ].append(
                row["session_id"]
            )

            fields[
                "bout_index"
            ].append(
                str(
                    int(
                        row["bout_index"]
                    )
                )
            )

            fields[
                "partition"
            ].append(
                row["partition"]
            )

            fields[
                "context_id"
            ].append(
                row["context_id"]
            )

            fields[
                "bout_id"
            ].append(
                row["bout_id"]
            )

            fields[
                "known_label"
            ].append(
                normalize_known_label(
                    row[
                        detected_label_column
                    ]
                )
            )

    if (
        len(fields["fish_id"])
        != EXPECTED_ROWS[
            partition
        ]
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"{EXPECTED_ROWS[partition]:,} rows, "
            f"got {len(fields['fish_id']):,}."
        )

    return (
        {
            key: np.asarray(
                value,
                dtype=str,
            )
            for key, value
            in fields.items()
        },
        detected_label_column,
    )


def verify_alignment(
    baseline_alignment: Mapping[
        str,
        np.ndarray,
    ],
    metadata: Mapping[
        str,
        np.ndarray,
    ],
    *,
    seed: int,
    partition: str,
) -> None:
    for field in (
        "fish_id",
        "session_id",
        "bout_index",
        "partition",
        "context_id",
    ):
        a = np.asarray(
            baseline_alignment[
                field
            ]
        ).astype(str)

        b = np.asarray(
            metadata[
                field
            ]
        ).astype(str)

        if a.shape != b.shape:
            raise RuntimeError(
                "Alignment shape mismatch "
                f"seed={seed}, "
                f"partition={partition}, "
                f"field={field}."
            )

        mismatch = np.flatnonzero(
            a != b
        )

        if mismatch.size:
            idx = int(
                mismatch[0]
            )

            raise RuntimeError(
                "Alignment failed "
                f"seed={seed}, "
                f"partition={partition}, "
                f"field={field}, "
                f"row={idx}: "
                f"{a[idx]!r} != {b[idx]!r}"
            )


def load_ssl_labels(
    label_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    path = (
        label_root
        / f"seed{seed}"
        / f"{partition}_labels.npy"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        ),
        dtype=np.int64,
    )

    if labels.shape != (
        EXPECTED_ROWS[
            partition
        ],
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"({EXPECTED_ROWS[partition]},), "
            f"got {labels.shape}."
        )

    if not np.array_equal(
        np.unique(labels),
        np.arange(
            EXPECTED_SSL_K
        ),
    ):
        raise RuntimeError(
            f"{path}: expected labels "
            f"0..{EXPECTED_SSL_K - 1}."
        )

    return labels


def hungarian_map_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[int, int]:
    table = np.zeros(
        (
            EXPECTED_SSL_K,
            EXPECTED_SSL_K,
        ),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (
            reference_labels.astype(
                int
            ),
            candidate_labels.astype(
                int
            ),
        ),
        1,
    )

    row_ind, col_ind = (
        linear_sum_assignment(
            -table
        )
    )

    return {
        int(candidate): int(
            reference
        )
        for reference, candidate
        in zip(
            row_ind,
            col_ind,
        )
    }


def apply_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray(
        [
            mapping[
                int(label)
            ]
            for label
            in labels
        ],
        dtype=np.int16,
    )


def eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    labels = np.asarray(
        labels
    )

    finite = np.isfinite(
        values
    )
    values = values[
        finite
    ]
    labels = labels[
        finite
    ]

    if values.size == 0:
        return 0.0

    grand_mean = float(
        np.mean(values)
    )

    ss_total = float(
        np.sum(
            (
                values
                - grand_mean
            )
            ** 2
        )
    )

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for cluster in np.unique(
        labels
    ):
        subset = values[
            labels
            == cluster
        ]

        if subset.size == 0:
            continue

        cluster_mean = float(
            np.mean(
                subset
            )
        )

        ss_between += (
            float(
                subset.size
            )
            * (
                cluster_mean
                - grand_mean
            )
            ** 2
        )

    return float(
        ss_between
        / ss_total
    )


def standardized_mean_range(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(
        values
    )
    values = values[
        finite
    ]
    labels = labels[
        finite
    ]

    if values.size == 0:
        return 0.0

    global_std = float(
        np.std(values)
    )

    if global_std <= 0:
        return 0.0

    means = [
        float(
            np.mean(
                values[
                    labels
                    == cluster
                ]
            )
        )
        for cluster
        in range(
            EXPECTED_SSL_K
        )
        if np.any(
            labels
            == cluster
        )
    ]

    if len(means) <= 1:
        return 0.0

    return float(
        (
            max(means)
            - min(means)
        )
        / global_std
    )


def characterize_feature(
    *,
    values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    if (
        values.shape
        != labels.shape
    ):
        raise RuntimeError(
            "Feature/label shape mismatch: "
            f"{values.shape} vs "
            f"{labels.shape}"
        )

    per_cluster: Dict[
        str,
        Any,
    ] = {}

    mean_profile = np.full(
        EXPECTED_SSL_K,
        np.nan,
        dtype=np.float64,
    )

    median_profile = np.full(
        EXPECTED_SSL_K,
        np.nan,
        dtype=np.float64,
    )

    counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_SSL_K,
    )

    total = int(
        np.sum(counts)
    )

    for cluster in range(
        EXPECTED_SSL_K
    ):
        subset = values[
            labels
            == cluster
        ]

        subset = subset[
            np.isfinite(
                subset
            )
        ]

        if subset.size == 0:
            per_cluster[
                str(cluster)
            ] = {
                "count": 0,
                "fraction": 0.0,
                "mean": None,
                "median": None,
                "std": None,
                "p25": None,
                "p75": None,
                "iqr": None,
            }
            continue

        mean = float(
            np.mean(
                subset
            )
        )

        median = float(
            np.median(
                subset
            )
        )

        p25 = float(
            np.percentile(
                subset,
                25,
            )
        )

        p75 = float(
            np.percentile(
                subset,
                75,
            )
        )

        mean_profile[
            cluster
        ] = mean

        median_profile[
            cluster
        ] = median

        per_cluster[
            str(cluster)
        ] = {
            "count": int(
                subset.size
            ),
            "fraction": float(
                subset.size
                / total
            ),
            "mean": mean,
            "median": median,
            "std": float(
                np.std(
                    subset
                )
            ),
            "p25": p25,
            "p75": p75,
            "iqr": float(
                p75
                - p25
            ),
        }

    return {
        "count": total,
        "eta_squared": eta_squared(
            values,
            labels,
        ),
        "standardized_mean_range": (
            standardized_mean_range(
                values,
                labels,
            )
        ),
        "mean_profile": (
            mean_profile.tolist()
        ),
        "median_profile": (
            median_profile.tolist()
        ),
        "per_cluster": per_cluster,
    }


def safe_spearman(
    a: Sequence[float],
    b: Sequence[float],
) -> Tuple[
    Optional[float],
    Optional[float],
    int,
]:
    x = np.asarray(
        a,
        dtype=np.float64,
    )

    y = np.asarray(
        b,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if x.size < 3:
        return (
            None,
            None,
            int(x.size),
        )

    if (
        np.allclose(
            x,
            x[0],
        )
        or np.allclose(
            y,
            y[0],
        )
    ):
        return (
            None,
            None,
            int(x.size),
        )

    result = spearmanr(
        x,
        y,
    )

    rho = float(
        result.statistic
    )

    pvalue = float(
        result.pvalue
    )

    if not np.isfinite(
        rho
    ):
        rho = None

    if not np.isfinite(
        pvalue
    ):
        pvalue = None

    return (
        rho,
        pvalue,
        int(x.size),
    )


def pairwise_profile_correlations(
    per_seed_feature_results: Mapping[
        int,
        Dict[str, Any],
    ],
    *,
    profile_key: str,
) -> Dict[str, Any]:
    seeds = sorted(
        per_seed_feature_results
    )

    pairs: List[
        Dict[str, Any]
    ] = []

    for i, seed_a in enumerate(
        seeds
    ):
        for seed_b in seeds[
            i + 1:
        ]:
            (
                rho,
                pvalue,
                n,
            ) = safe_spearman(
                per_seed_feature_results[
                    seed_a
                ][
                    profile_key
                ],
                per_seed_feature_results[
                    seed_b
                ][
                    profile_key
                ],
            )

            pairs.append(
                {
                    "seed_a": int(
                        seed_a
                    ),
                    "seed_b": int(
                        seed_b
                    ),
                    "rho": rho,
                    "pvalue": pvalue,
                    "clusters_compared": n,
                }
            )

    finite = np.asarray(
        [
            pair["rho"]
            for pair in pairs
            if pair[
                "rho"
            ]
            is not None
        ],
        dtype=np.float64,
    )

    return {
        "pair_count": int(
            len(pairs)
        ),
        "finite_pair_count": int(
            finite.size
        ),
        "mean_rho": (
            float(
                np.mean(
                    finite
                )
            )
            if finite.size
            else None
        ),
        "median_rho": (
            float(
                np.median(
                    finite
                )
            )
            if finite.size
            else None
        ),
        "min_rho": (
            float(
                np.min(
                    finite
                )
            )
            if finite.size
            else None
        ),
        "max_rho": (
            float(
                np.max(
                    finite
                )
            )
            if finite.size
            else None
        ),
        "pairs": pairs,
    }


def summarize_scalar_metric(
    per_seed_feature_results: Mapping[
        int,
        Dict[str, Any],
    ],
    metric: str,
) -> Dict[str, float]:
    values = np.asarray(
        [
            float(
                result[
                    metric
                ]
            )
            for result
            in per_seed_feature_results.values()
        ],
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": float(
            np.std(
                values
            )
        ),
        "min": float(
            np.min(
                values
            )
        ),
        "max": float(
            np.max(
                values
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify reproducibility of "
            "Long_CS duration/acceleration "
            "differences across frozen SSL seeds."
        )
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
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
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_dir = (
        args.baseline_dir.resolve()
    )

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
        baseline_dir,
        metadata_root,
        label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(
            path
        )

    assert_no_test_artifacts(
        metadata_root
    )

    assert_no_test_artifacts(
        label_root
    )

    seeds = configured_ssl_seeds(
        training_config
    )

    if (
        REFERENCE_SEED
        not in seeds
    ):
        raise RuntimeError(
            f"Reference seed {REFERENCE_SEED} "
            "is absent from frozen seeds."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    cross_seed_path = (
        output_dir
        / "cross_seed_summary.json"
    )

    checksum_path = (
        output_dir
        / (
            "LONG_CS_KINEMATIC_"
            "REPRODUCIBILITY_SHA256SUMS"
        )
    )

    if (
        summary_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print(
        "=" * 80
    )

    print(
        "DS-005 LONG_CS KINEMATIC REPRODUCIBILITY"
    )

    print(
        "=" * 80
    )

    print(
        f"Known class:       {KNOWN_CLASS}"
    )

    print(
        f"Features:          {list(TARGET_FEATURES)}"
    )

    print(
        f"SSL seeds:         {list(seeds)}"
    )

    print(
        f"Reference seed:    {REFERENCE_SEED}"
    )

    print(
        "Alignment:         TRAIN-derived Hungarian mapping"
    )

    print(
        "Evaluation:        TRAIN + VALIDATION"
    )

    print(
        "TEST partition:    PROTECTED / NOT LOADED"
    )

    print()

    baseline_x: Dict[
        str,
        np.ndarray,
    ] = {}

    baseline_alignment: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    feature_names: Optional[
        List[str]
    ] = None

    for partition in PARTITIONS:
        (
            X,
            names,
            alignment,
        ) = load_baseline_raw(
            baseline_dir,
            partition=partition,
        )

        baseline_x[
            partition
        ] = X

        baseline_alignment[
            partition
        ] = alignment

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError(
                "TRAIN/VALIDATION baseline "
                "feature names differ."
            )

    assert feature_names is not None

    feature_indices = {
        feature: feature_names.index(
            feature
        )
        for feature
        in TARGET_FEATURES
    }

    metadata_by_seed: Dict[
        int,
        Dict[
            str,
            Dict[str, np.ndarray],
        ],
    ] = {}

    labels_by_seed: Dict[
        int,
        Dict[str, np.ndarray],
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
        metadata_by_seed[
            seed
        ] = {}

        labels_by_seed[
            seed
        ] = {}

        for partition in PARTITIONS:
            (
                metadata,
                detected_column,
            ) = load_metadata(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
                requested_label_column=(
                    args.label_column
                ),
            )

            verify_alignment(
                baseline_alignment[
                    partition
                ],
                metadata,
                seed=seed,
                partition=partition,
            )

            labels = load_ssl_labels(
                label_root,
                seed=seed,
                partition=partition,
            )

            metadata_by_seed[
                seed
            ][
                partition
            ] = metadata

            labels_by_seed[
                seed
            ][
                partition
            ] = labels

            known = metadata[
                "known_label"
            ]

            bouts = metadata[
                "bout_id"
            ]

            if (
                partition
                not in reference_known
            ):
                reference_known[
                    partition
                ] = known.copy()

                reference_bouts[
                    partition
                ] = bouts.copy()

                detected_columns[
                    partition
                ] = detected_column
            else:
                if not np.array_equal(
                    reference_known[
                        partition
                    ],
                    known,
                ):
                    raise RuntimeError(
                        "Known labels differ "
                        f"across seeds for {partition}."
                    )

                if not np.array_equal(
                    reference_bouts[
                        partition
                    ],
                    bouts,
                ):
                    raise RuntimeError(
                        "Bout ordering differs "
                        f"across seeds for {partition}."
                    )

    train_reference = (
        labels_by_seed[
            REFERENCE_SEED
        ][
            "train"
        ]
    )

    mappings: Dict[
        int,
        Dict[int, int],
    ] = {
        REFERENCE_SEED: {
            cluster: cluster
            for cluster
            in range(
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
            train_reference,
            labels_by_seed[
                seed
            ][
                "train"
            ],
        )

    aligned_labels = {
        seed: {
            partition: apply_mapping(
                labels_by_seed[
                    seed
                ][
                    partition
                ],
                mappings[
                    seed
                ],
            )
            for partition
            in PARTITIONS
        }
        for seed
        in seeds
    }

    class_masks = {
        partition: (
            reference_known[
                partition
            ]
            == KNOWN_CLASS
        )
        for partition
        in PARTITIONS
    }

    for partition in PARTITIONS:
        if not np.any(
            class_masks[
                partition
            ]
        ):
            raise RuntimeError(
                f"No {KNOWN_CLASS} bouts "
                f"found in {partition}."
            )

    per_seed: Dict[
        int,
        Dict[str, Any],
    ] = {}

    written_artifacts: List[
        Path
    ] = []

    for seed in seeds:
        print(
            "=" * 80
        )

        print(
            f"SSL SEED {seed}"
        )

        print(
            "=" * 80
        )

        per_seed[
            seed
        ] = {}

        seed_dir = (
            output_dir
            / f"seed{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for partition in PARTITIONS:
            mask = class_masks[
                partition
            ]

            class_labels = (
                aligned_labels[
                    seed
                ][
                    partition
                ][
                    mask
                ]
            )

            partition_result: Dict[
                str,
                Any,
            ] = {
                "count": int(
                    np.sum(
                        mask
                    )
                ),
                "features": {},
            }

            print(
                f"{partition.upper():<11} "
                f"n={partition_result['count']:,}"
            )

            for feature in TARGET_FEATURES:
                values = (
                    baseline_x[
                        partition
                    ][
                        mask,
                        feature_indices[
                            feature
                        ],
                    ]
                )

                result = (
                    characterize_feature(
                        values=values,
                        labels=class_labels,
                    )
                )

                partition_result[
                    "features"
                ][
                    feature
                ] = result

                print(
                    f"  {feature:<22} "
                    f"eta^2={result['eta_squared']:.6f}  "
                    f"mean-range-z="
                    f"{result['standardized_mean_range']:.3f}"
                )

            per_seed[
                seed
            ][
                partition
            ] = partition_result

            output_path = (
                seed_dir
                / f"{partition}.json"
            )

            atomic_write_json(
                output_path,
                {
                    "dataset_id": "DS-005",
                    "known_class": KNOWN_CLASS,
                    "ssl_seed": int(
                        seed
                    ),
                    "partition": partition,
                    "features": partition_result[
                        "features"
                    ],
                    "count": partition_result[
                        "count"
                    ],
                    "cluster_mapping_to_reference": {
                        str(src): int(
                            dst
                        )
                        for src, dst
                        in mappings[
                            seed
                        ].items()
                    },
                    "test_partition_used": False,
                },
            )

            written_artifacts.append(
                output_path
            )

        per_seed[
            seed
        ][
            "train_validation_profile_spearman"
        ] = {}

        print(
            "  TRAIN -> VALIDATION profile Spearman:"
        )

        for feature in TARGET_FEATURES:
            (
                mean_rho,
                mean_p,
                mean_n,
            ) = safe_spearman(
                per_seed[
                    seed
                ][
                    "train"
                ][
                    "features"
                ][
                    feature
                ][
                    "mean_profile"
                ],
                per_seed[
                    seed
                ][
                    "validation"
                ][
                    "features"
                ][
                    feature
                ][
                    "mean_profile"
                ],
            )

            (
                median_rho,
                median_p,
                median_n,
            ) = safe_spearman(
                per_seed[
                    seed
                ][
                    "train"
                ][
                    "features"
                ][
                    feature
                ][
                    "median_profile"
                ],
                per_seed[
                    seed
                ][
                    "validation"
                ][
                    "features"
                ][
                    feature
                ][
                    "median_profile"
                ],
            )

            per_seed[
                seed
            ][
                "train_validation_profile_spearman"
            ][
                feature
            ] = {
                "mean_profile": {
                    "rho": mean_rho,
                    "pvalue": mean_p,
                    "clusters_compared": mean_n,
                },
                "median_profile": {
                    "rho": median_rho,
                    "pvalue": median_p,
                    "clusters_compared": median_n,
                },
            }

            print(
                f"    {feature:<22} "
                f"mean rho="
                f"{mean_rho if mean_rho is not None else 'NA'}  "
                f"median rho="
                f"{median_rho if median_rho is not None else 'NA'}"
            )

        print(
            "TEST partition used: NO"
        )

        print()

    cross_seed: Dict[
        str,
        Any,
    ] = {}

    for partition in PARTITIONS:
        cross_seed[
            partition
        ] = {}

        for feature in TARGET_FEATURES:
            feature_results = {
                seed: per_seed[
                    seed
                ][
                    partition
                ][
                    "features"
                ][
                    feature
                ]
                for seed
                in seeds
            }

            cross_seed[
                partition
            ][
                feature
            ] = {
                "eta_squared": (
                    summarize_scalar_metric(
                        feature_results,
                        "eta_squared",
                    )
                ),
                "standardized_mean_range": (
                    summarize_scalar_metric(
                        feature_results,
                        "standardized_mean_range",
                    )
                ),
                "mean_profile_spearman": (
                    pairwise_profile_correlations(
                        feature_results,
                        profile_key="mean_profile",
                    )
                ),
                "median_profile_spearman": (
                    pairwise_profile_correlations(
                        feature_results,
                        profile_key="median_profile",
                    )
                ),
            }

    train_validation_summary: Dict[
        str,
        Any,
    ] = {}

    for feature in TARGET_FEATURES:
        mean_rhos = np.asarray(
            [
                per_seed[
                    seed
                ][
                    "train_validation_profile_spearman"
                ][
                    feature
                ][
                    "mean_profile"
                ][
                    "rho"
                ]
                for seed
                in seeds
                if per_seed[
                    seed
                ][
                    "train_validation_profile_spearman"
                ][
                    feature
                ][
                    "mean_profile"
                ][
                    "rho"
                ]
                is not None
            ],
            dtype=np.float64,
        )

        median_rhos = np.asarray(
            [
                per_seed[
                    seed
                ][
                    "train_validation_profile_spearman"
                ][
                    feature
                ][
                    "median_profile"
                ][
                    "rho"
                ]
                for seed
                in seeds
                if per_seed[
                    seed
                ][
                    "train_validation_profile_spearman"
                ][
                    feature
                ][
                    "median_profile"
                ][
                    "rho"
                ]
                is not None
            ],
            dtype=np.float64,
        )

        train_validation_summary[
            feature
        ] = {
            "mean_profile": {
                "mean_rho": (
                    float(
                        np.mean(
                            mean_rhos
                        )
                    )
                    if mean_rhos.size
                    else None
                ),
                "min_rho": (
                    float(
                        np.min(
                            mean_rhos
                        )
                    )
                    if mean_rhos.size
                    else None
                ),
                "max_rho": (
                    float(
                        np.max(
                            mean_rhos
                        )
                    )
                    if mean_rhos.size
                    else None
                ),
            },
            "median_profile": {
                "mean_rho": (
                    float(
                        np.mean(
                            median_rhos
                        )
                    )
                    if median_rhos.size
                    else None
                ),
                "min_rho": (
                    float(
                        np.min(
                            median_rhos
                        )
                    )
                    if median_rhos.size
                    else None
                ),
                "max_rho": (
                    float(
                        np.max(
                            median_rhos
                        )
                    )
                    if median_rhos.size
                    else None
                ),
            },
            "by_seed": {
                str(seed): per_seed[
                    seed
                ][
                    "train_validation_profile_spearman"
                ][
                    feature
                ]
                for seed
                in seeds
            },
        }

    cross_seed[
        "train_validation_generalization"
    ] = train_validation_summary

    cross_seed_payload = {
        "dataset_id": "DS-005",
        "known_class": KNOWN_CLASS,
        "features": list(
            TARGET_FEATURES
        ),
        "ssl_training_seeds": list(
            seeds
        ),
        "reference_seed": REFERENCE_SEED,
        "train_derived_cluster_mappings_to_reference": {
            str(seed): {
                str(src): int(
                    dst
                )
                for src, dst
                in mapping.items()
            }
            for seed, mapping
            in mappings.items()
        },
        **cross_seed,
        "test_partition_used": False,
    }

    atomic_write_json(
        cross_seed_path,
        cross_seed_payload,
    )

    written_artifacts.append(
        cross_seed_path
    )

    summary_features: Dict[
        str,
        Any,
    ] = {}

    for feature in TARGET_FEATURES:
        validation = (
            cross_seed[
                "validation"
            ][
                feature
            ]
        )

        generalization = (
            train_validation_summary[
                feature
            ]
        )

        summary_features[
            feature
        ] = {
            "validation_eta_squared": (
                validation[
                    "eta_squared"
                ]
            ),
            "validation_mean_profile_spearman": (
                validation[
                    "mean_profile_spearman"
                ]
            ),
            "validation_median_profile_spearman": (
                validation[
                    "median_profile_spearman"
                ]
            ),
            "train_validation_generalization": (
                generalization
            ),
        }

    summary = {
        "dataset_id": "DS-005",
        "analysis": (
            "ssl_long_cs_kinematic_reproducibility"
        ),
        "known_class": KNOWN_CLASS,
        "features": list(
            TARGET_FEATURES
        ),
        "ssl_training_seeds": list(
            seeds
        ),
        "reference_seed": REFERENCE_SEED,
        "known_label_columns": (
            detected_columns
        ),
        "summary_by_feature": (
            summary_features
        ),
        "per_seed": {
            str(seed): per_seed[
                seed
            ]
            for seed
            in seeds
        },
        "interpretation_guardrails": {
            "post_hoc_targeted_reproducibility_analysis": True,
            "cluster_alignment_estimated_on_train_only": True,
            "validation_fish_held_out": True,
            "kinematic_association_not_proof_of_new_behavior": True,
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

    print(
        "=" * 80
    )

    print(
        "LONG_CS KINEMATIC REPRODUCIBILITY SUMMARY"
    )

    print(
        "=" * 80
    )

    for feature in TARGET_FEATURES:
        validation = cross_seed[
            "validation"
        ][
            feature
        ]

        generalization = (
            train_validation_summary[
                feature
            ]
        )

        print(
            feature
        )

        print(
            "  Mean VALIDATION eta^2: "
            f"{validation['eta_squared']['mean']:.6f}"
        )

        print(
            "  VALIDATION eta^2 range: "
            f"{validation['eta_squared']['min']:.6f} - "
            f"{validation['eta_squared']['max']:.6f}"
        )

        print(
            "  Mean cross-seed VALIDATION Spearman "
            "(mean profiles): "
            f"{validation['mean_profile_spearman']['mean_rho']}"
        )

        print(
            "  Mean cross-seed VALIDATION Spearman "
            "(median profiles): "
            f"{validation['median_profile_spearman']['mean_rho']}"
        )

        print(
            "  Mean TRAIN->VALIDATION Spearman "
            "(mean profiles): "
            f"{generalization['mean_profile']['mean_rho']}"
        )

        print(
            "  Mean TRAIN->VALIDATION Spearman "
            "(median profiles): "
            f"{generalization['median_profile']['mean_rho']}"
        )

        print()

    print(
        "TEST partition used: NO"
    )

    print(
        f"Summary:    {summary_path}"
    )

    print(
        f"Cross-seed: {cross_seed_path}"
    )

    print(
        f"Checksums:  {checksum_path}"
    )


if __name__ == "__main__":
    main()
