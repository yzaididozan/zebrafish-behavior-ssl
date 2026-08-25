#!/usr/bin/env python3
"""Characterize DS-006 transfer-cluster behavioral/kinematic substructure.

Scientific purpose
------------------
DS-006 does not provide the same conventional bout-class labels used in DS-005
(e.g. Long_CS, LLC). Therefore this script does NOT claim direct within-class
replication of those labels.

Instead it asks whether the frozen transferred k=8 organization reproducibly
captures analogous kinematic axes in DS-006:

Long_CS-like axes:
    - bout_duration
    - speed_change_abs_mean
    - speed_change_std
    - speed_change_max
    - speed_change_rms

LLC-like axes:
    - turn_net
    - turn_total_abs
    - turn_abs_mean
    - turn_std
    - turn_max
    - turn_rms

All 18 frozen handcrafted features are analyzed, so the result is not restricted
to the targeted axes above.

For each SSL seed (11, 23, 37, 51, 79) and partition (TRAIN, VALIDATION):
- align frozen baseline rows to frozen transfer-cluster labels by bout_id;
- compute eta^2 between cluster membership and every handcrafted feature;
- compute per-cluster mean/median/std/count profiles.

Reproducibility:
- use the TRAIN-derived seed->seed11 Hungarian cluster alignment already frozen
  by ds006_transfer_clustering.py;
- compare TRAIN vs VALIDATION cluster profiles within each seed using Spearman rho;
- compare VALIDATION profiles across encoder seeds after TRAIN-derived alignment;
- summarize targeted duration/speed-change and turning axes.

This script never:
- loads DS-006 TEST;
- selects k;
- refits clusters;
- changes feature definitions;
- uses DS-006 results to redesign the primary DS-005 method.

Inputs
------
data/processed/DS-006/baseline/
    train_core_raw.npz
    validation_core_raw.npz
    feature_manifest.json

data/processed/DS-006/transfer_clustering/
    seedXX/train_labels.npy
    seedXX/validation_labels.npy
    seedXX/train_labels_aligned.npy
    seedXX/validation_labels_aligned.npy
    cross_seed_stability.json

Outputs
-------
data/processed/DS-006/transfer_substructure/
    seed11/
        train_feature_characterization.json
        validation_feature_characterization.json
        train_validation_reproducibility.json
    ...
    cross_seed_validation_profiles.json
    targeted_axes_summary.json
    summary.json
    DS006_TRANSFER_SUBSTRUCTURE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ds006_transfer_subcluster_characterization.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ds006_transfer_subcluster_characterization.py \
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "baseline"
)

CLUSTER_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "transfer_clustering"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "transfer_substructure"
)

FEATURE_MANIFEST = (
    BASELINE_ROOT
    / "feature_manifest.json"
)

CROSS_SEED_STABILITY = (
    CLUSTER_ROOT
    / "cross_seed_stability.json"
)

SEEDS = (11, 23, 37, 51, 79)
REFERENCE_SEED = 11
PARTITIONS = ("train", "validation")

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

K = 8

EXPECTED_FEATURE_NAMES = [
    "bout_duration",
    "inter_bout_interval",
    "speed_mean",
    "speed_std",
    "speed_median",
    "speed_max",
    "speed_p95",
    "speed_rms",
    "speed_change_abs_mean",
    "speed_change_std",
    "speed_change_max",
    "speed_change_rms",
    "turn_total_abs",
    "turn_net",
    "turn_abs_mean",
    "turn_std",
    "turn_max",
    "turn_rms",
]

LONG_CS_LIKE_AXES = [
    "bout_duration",
    "speed_change_abs_mean",
    "speed_change_std",
    "speed_change_max",
    "speed_change_rms",
]

LLC_LIKE_AXES = [
    "turn_net",
    "turn_total_abs",
    "turn_abs_mean",
    "turn_std",
    "turn_max",
    "turn_rms",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Characterize reproducible DS-006 kinematic substructure "
            "in frozen transfer clusters. TRAIN/VALIDATION only."
        )
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=BASELINE_ROOT,
    )
    p.add_argument(
        "--cluster-root",
        type=Path,
        default=CLUSTER_ROOT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
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
    ) as f:
        f.write(payload)
        tmp = f.name

    os.replace(tmp, path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_paths(
    baseline_root: Path,
    cluster_root: Path,
    output_root: Path,
) -> None:
    baseline_root = baseline_root.resolve()
    cluster_root = cluster_root.resolve()
    output_root = output_root.resolve()

    if baseline_root != BASELINE_ROOT.resolve():
        raise RuntimeError(
            "--baseline-root must resolve exactly to the frozen DS-006 "
            f"baseline directory: {BASELINE_ROOT.resolve()}"
        )

    if cluster_root != CLUSTER_ROOT.resolve():
        raise RuntimeError(
            "--cluster-root must resolve exactly to the frozen DS-006 "
            f"transfer clustering directory: {CLUSTER_ROOT.resolve()}"
        )

    ds006 = (
        REPO_ROOT
        / "data"
        / "processed"
        / "DS-006"
    ).resolve()

    if not is_relative_to(
        output_root,
        ds006,
    ):
        raise RuntimeError(
            "Outputs must remain under data/processed/DS-006."
        )

    if "test" in str(output_root).lower():
        raise RuntimeError(
            "Output path unexpectedly contains TEST."
        )


def load_feature_manifest() -> Dict[str, Any]:
    if not FEATURE_MANIFEST.exists():
        raise FileNotFoundError(
            FEATURE_MANIFEST
        )

    obj = json.loads(
        FEATURE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    names = obj.get(
        "feature_names"
    )

    if names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Frozen DS-006 feature order changed.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Observed: {names}"
        )

    return obj


def baseline_path(
    baseline_root: Path,
    partition: str,
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    path = (
        baseline_root
        / f"{partition}_core_raw.npz"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST path reached."
        )

    return path


def load_baseline(
    baseline_root: Path,
    partition: str,
) -> Dict[str, Any]:
    path = baseline_path(
        baseline_root,
        partition,
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        required = {
            "X",
            "bout_id",
        }

        missing = required - set(
            npz.files
        )

        if missing:
            raise RuntimeError(
                f"{path}: missing {sorted(missing)}"
            )

        X = np.asarray(
            npz["X"],
            dtype=np.float64,
        )

        bout_id = np.asarray(
            npz["bout_id"]
        ).astype(str)

    expected = EXPECTED_ROWS[
        partition
    ]

    if X.shape != (
        expected,
        len(
            EXPECTED_FEATURE_NAMES
        ),
    ):
        raise RuntimeError(
            f"{path}: unexpected X shape {X.shape}."
        )

    if bout_id.shape != (
        expected,
    ):
        raise RuntimeError(
            f"{path}: unexpected bout_id shape."
        )

    # Raw baseline intentionally permits NaN only for IBI.
    non_ibi = np.delete(
        X,
        1,
        axis=1,
    )

    if not np.isfinite(
        non_ibi
    ).all():
        raise RuntimeError(
            f"{path}: non-finite values outside inter_bout_interval."
        )

    return {
        "path": path,
        "sha256": sha256_file(
            path
        ),
        "X": X,
        "bout_id": bout_id,
    }


def cluster_label_path(
    cluster_root: Path,
    *,
    seed: int,
    partition: str,
    aligned: bool,
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    suffix = (
        "_labels_aligned.npy"
        if aligned
        else "_labels.npy"
    )

    path = (
        cluster_root
        / f"seed{seed}"
        / f"{partition}{suffix}"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST path reached."
        )

    return path


def load_labels(
    cluster_root: Path,
    *,
    seed: int,
    partition: str,
    aligned: bool,
) -> np.ndarray:
    path = cluster_label_path(
        cluster_root,
        seed=seed,
        partition=partition,
        aligned=aligned,
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    labels = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        ),
        dtype=np.int64,
    )

    expected = EXPECTED_ROWS[
        partition
    ]

    if labels.shape != (
        expected,
    ):
        raise RuntimeError(
            f"{path}: unexpected label shape {labels.shape}."
        )

    if not np.array_equal(
        np.unique(
            labels
        ),
        np.arange(
            K
        ),
    ):
        raise RuntimeError(
            f"{path}: expected all cluster IDs 0..{K-1}."
        )

    return labels


def cluster_source_bout_ids(
    cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    manifest_path = (
        cluster_root
        / f"seed{seed}"
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    key = (
        "source_train_embedding"
        if partition == "train"
        else "source_validation_embedding"
    )

    rel = manifest.get(
        key
    )

    if not rel:
        raise RuntimeError(
            f"{manifest_path}: missing {key}"
        )

    source = (
        REPO_ROOT
        / rel
    ).resolve()

    if "test" in source.name.lower():
        raise RuntimeError(
            "Protected TEST source reached."
        )

    with np.load(
        source,
        allow_pickle=False,
    ) as npz:
        ids = np.asarray(
            npz["bout_id"]
        ).astype(str)

    return ids


def align_baseline(
    baseline: Dict[str, Any],
    cluster_ids: np.ndarray,
) -> np.ndarray:
    ids = baseline[
        "bout_id"
    ]

    X = baseline[
        "X"
    ]

    if np.array_equal(
        ids,
        cluster_ids,
    ):
        return X

    lookup = {
        bout_id: i
        for i, bout_id
        in enumerate(
            ids
        )
    }

    missing = [
        bout_id
        for bout_id
        in cluster_ids
        if bout_id not in lookup
    ]

    if missing:
        raise RuntimeError(
            "Baseline/cluster alignment failed; "
            f"missing bout ID {missing[0]!r}."
        )

    index = np.asarray(
        [
            lookup[
                bout_id
            ]
            for bout_id
            in cluster_ids
        ],
        dtype=np.int64,
    )

    if len(
        np.unique(
            index
        )
    ) != len(
        index
    ):
        raise RuntimeError(
            "Baseline alignment reused rows."
        )

    return X[
        index
    ]


def eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    valid = np.isfinite(
        values
    )

    values = values[
        valid
    ]

    labels = labels[
        valid
    ]

    if values.size == 0:
        return 0.0

    grand = float(
        np.mean(
            values
        )
    )

    ss_total = float(
        np.sum(
            (
                values
                - grand
            )
            ** 2
        )
    )

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for cluster in range(
        K
    ):
        mask = (
            labels
            == cluster
        )

        if not np.any(
            mask
        ):
            continue

        group = values[
            mask
        ]

        ss_between += (
            len(
                group
            )
            * (
                float(
                    np.mean(
                        group
                    )
                )
                - grand
            )
            ** 2
        )

    return float(
        ss_between
        / ss_total
    )


def cluster_profile(
    values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {}

    for cluster in range(
        K
    ):
        mask = (
            labels
            == cluster
        )

        x = values[
            mask
        ]

        x = x[
            np.isfinite(
                x
            )
        ]

        if x.size == 0:
            output[
                str(
                    cluster
                )
            ] = {
                "n": 0,
                "mean": None,
                "median": None,
                "std": None,
            }
        else:
            output[
                str(
                    cluster
                )
            ] = {
                "n": int(
                    x.size
                ),
                "mean": float(
                    np.mean(
                        x
                    )
                ),
                "median": float(
                    np.median(
                        x
                    )
                ),
                "std": float(
                    np.std(
                        x
                    )
                ),
            }

    return output


def profile_vector(
    profile: Mapping[
        str,
        Any,
    ],
    statistic: str,
) -> np.ndarray:
    return np.asarray(
        [
            float(
                profile[
                    str(
                        cluster
                    )
                ][
                    statistic
                ]
            )
            for cluster
            in range(
                K
            )
        ],
        dtype=np.float64,
    )


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if (
        x.size < 3
        or np.std(
            x
        ) == 0
        or np.std(
            y
        ) == 0
    ):
        return None

    result = spearmanr(
        x,
        y,
    )

    rho = float(
        result.statistic
    )

    return (
        rho
        if math.isfinite(
            rho
        )
        else None
    )


def characterize_partition(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    features: Dict[
        str,
        Any,
    ] = {}

    for index, name in enumerate(
        EXPECTED_FEATURE_NAMES
    ):
        values = X[
            :,
            index
        ]

        features[
            name
        ] = {
            "feature_index": int(
                index
            ),
            "eta_squared": eta_squared(
                values,
                labels,
            ),
            "cluster_profile": cluster_profile(
                values,
                labels,
            ),
        }

    ranked = sorted(
        EXPECTED_FEATURE_NAMES,
        key=lambda name: features[
            name
        ][
            "eta_squared"
        ],
        reverse=True,
    )

    return {
        "feature_count": len(
            EXPECTED_FEATURE_NAMES
        ),
        "features": features,
        "ranking_by_eta_squared": [
            {
                "rank": rank + 1,
                "feature": name,
                "eta_squared": float(
                    features[
                        name
                    ][
                        "eta_squared"
                    ]
                ),
            }
            for rank, name
            in enumerate(
                ranked
            )
        ],
    }


def train_validation_reproducibility(
    train_characterization: Mapping[
        str,
        Any,
    ],
    validation_characterization: Mapping[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {}

    for feature in EXPECTED_FEATURE_NAMES:
        train_profile = train_characterization[
            "features"
        ][
            feature
        ][
            "cluster_profile"
        ]

        validation_profile = validation_characterization[
            "features"
        ][
            feature
        ][
            "cluster_profile"
        ]

        output[
            feature
        ] = {
            "train_eta_squared": float(
                train_characterization[
                    "features"
                ][
                    feature
                ][
                    "eta_squared"
                ]
            ),
            "validation_eta_squared": float(
                validation_characterization[
                    "features"
                ][
                    feature
                ][
                    "eta_squared"
                ]
            ),
            "mean_profile_spearman": safe_spearman(
                profile_vector(
                    train_profile,
                    "mean",
                ),
                profile_vector(
                    validation_profile,
                    "mean",
                ),
            ),
            "median_profile_spearman": safe_spearman(
                profile_vector(
                    train_profile,
                    "median",
                ),
                profile_vector(
                    validation_profile,
                    "median",
                ),
            ),
        }

    return output


def cross_seed_validation_reproducibility(
    validation_by_seed: Mapping[
        int,
        Mapping[
            str,
            Any,
        ],
    ],
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {}

    for feature in EXPECTED_FEATURE_NAMES:
        mean_pairwise: List[
            float
        ] = []

        median_pairwise: List[
            float
        ] = []

        seeds = list(
            SEEDS
        )

        for i, seed_a in enumerate(
            seeds
        ):
            profile_a = validation_by_seed[
                seed_a
            ][
                "features"
            ][
                feature
            ][
                "cluster_profile"
            ]

            for seed_b in seeds[
                i + 1:
            ]:
                profile_b = validation_by_seed[
                    seed_b
                ][
                    "features"
                ][
                    feature
                ][
                    "cluster_profile"
                ]

                rho_mean = safe_spearman(
                    profile_vector(
                        profile_a,
                        "mean",
                    ),
                    profile_vector(
                        profile_b,
                        "mean",
                    ),
                )

                rho_median = safe_spearman(
                    profile_vector(
                        profile_a,
                        "median",
                    ),
                    profile_vector(
                        profile_b,
                        "median",
                    ),
                )

                if rho_mean is not None:
                    mean_pairwise.append(
                        rho_mean
                    )

                if rho_median is not None:
                    median_pairwise.append(
                        rho_median
                    )

        eta_values = np.asarray(
            [
                validation_by_seed[
                    seed
                ][
                    "features"
                ][
                    feature
                ][
                    "eta_squared"
                ]
                for seed
                in SEEDS
            ],
            dtype=np.float64,
        )

        output[
            feature
        ] = {
            "validation_eta_squared": {
                "mean": float(
                    np.mean(
                        eta_values
                    )
                ),
                "std": float(
                    np.std(
                        eta_values
                    )
                ),
                "min": float(
                    np.min(
                        eta_values
                    )
                ),
                "max": float(
                    np.max(
                        eta_values
                    )
                ),
                "by_seed": {
                    str(
                        seed
                    ): float(
                        validation_by_seed[
                            seed
                        ][
                            "features"
                        ][
                            feature
                        ][
                            "eta_squared"
                        ]
                    )
                    for seed
                    in SEEDS
                },
            },
            "cross_seed_validation_mean_profile_spearman": {
                "mean": (
                    float(
                        np.mean(
                            mean_pairwise
                        )
                    )
                    if mean_pairwise
                    else None
                ),
                "median": (
                    float(
                        np.median(
                            mean_pairwise
                        )
                    )
                    if mean_pairwise
                    else None
                ),
                "min": (
                    float(
                        np.min(
                            mean_pairwise
                        )
                    )
                    if mean_pairwise
                    else None
                ),
                "max": (
                    float(
                        np.max(
                            mean_pairwise
                        )
                    )
                    if mean_pairwise
                    else None
                ),
            },
            "cross_seed_validation_median_profile_spearman": {
                "mean": (
                    float(
                        np.mean(
                            median_pairwise
                        )
                    )
                    if median_pairwise
                    else None
                ),
                "median": (
                    float(
                        np.median(
                            median_pairwise
                        )
                    )
                    if median_pairwise
                    else None
                ),
                "min": (
                    float(
                        np.min(
                            median_pairwise
                        )
                    )
                    if median_pairwise
                    else None
                ),
                "max": (
                    float(
                        np.max(
                            median_pairwise
                        )
                    )
                    if median_pairwise
                    else None
                ),
            },
        }

    return output


def targeted_axis_summary(
    cross_seed: Mapping[
        str,
        Any,
    ],
    train_val_by_seed: Mapping[
        int,
        Mapping[
            str,
            Any,
        ],
    ],
) -> Dict[str, Any]:
    groups = {
        "long_cs_like_duration_speed_change_axes": (
            LONG_CS_LIKE_AXES
        ),
        "llc_like_turning_axes": (
            LLC_LIKE_AXES
        ),
    }

    result: Dict[
        str,
        Any,
    ] = {}

    for group_name, features in groups.items():
        result[
            group_name
        ] = {
            "important_limitation": (
                "These are kinematic-axis analogues only. DS-006 does not "
                "provide the DS-005 Long_CS/LLC class labels, so this is not "
                "a direct within-class replication."
            ),
            "features": {},
        }

        for feature in features:
            train_val_mean_rhos = [
                train_val_by_seed[
                    seed
                ][
                    feature
                ][
                    "mean_profile_spearman"
                ]
                for seed
                in SEEDS
                if train_val_by_seed[
                    seed
                ][
                    feature
                ][
                    "mean_profile_spearman"
                ]
                is not None
            ]

            train_val_median_rhos = [
                train_val_by_seed[
                    seed
                ][
                    feature
                ][
                    "median_profile_spearman"
                ]
                for seed
                in SEEDS
                if train_val_by_seed[
                    seed
                ][
                    feature
                ][
                    "median_profile_spearman"
                ]
                is not None
            ]

            result[
                group_name
            ][
                "features"
            ][
                feature
            ] = {
                **cross_seed[
                    feature
                ],
                "train_to_validation_profile_reproducibility": {
                    "mean_profile_spearman_mean_across_seeds": (
                        float(
                            np.mean(
                                train_val_mean_rhos
                            )
                        )
                        if train_val_mean_rhos
                        else None
                    ),
                    "median_profile_spearman_mean_across_seeds": (
                        float(
                            np.mean(
                                train_val_median_rhos
                            )
                        )
                        if train_val_median_rhos
                        else None
                    ),
                    "by_seed": {
                        str(
                            seed
                        ): train_val_by_seed[
                            seed
                        ][
                            feature
                        ]
                        for seed
                        in SEEDS
                    },
                },
            }

    return result


def write_checksums(
    output_root: Path,
    files: Sequence[
        Path
    ],
) -> Path:
    path = (
        output_root
        / "DS006_TRANSFER_SUBSTRUCTURE_SHA256SUMS"
    )

    path.write_text(
        "".join(
            f"{sha256_file(file)}  "
            f"{file.relative_to(output_root)}\n"
            for file
            in sorted(
                files,
                key=lambda p: str(
                    p
                ),
            )
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    args = parse_args()

    baseline_root = (
        args.baseline_root
        .expanduser()
        .resolve()
    )

    cluster_root = (
        args.cluster_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    assert_safe_paths(
        baseline_root,
        cluster_root,
        output_root,
    )

    summary_path = (
        output_root
        / "summary.json"
    )

    if (
        summary_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    manifest = load_feature_manifest()

    baseline = {
        partition: load_baseline(
            baseline_root,
            partition,
        )
        for partition
        in PARTITIONS
    }

    if not CROSS_SEED_STABILITY.exists():
        raise FileNotFoundError(
            CROSS_SEED_STABILITY
        )

    cross_seed_stability_hash = (
        sha256_file(
            CROSS_SEED_STABILITY
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 TRANSFER BEHAVIORAL/KINEMATIC SUBSTRUCTURE"
    )
    print(
        "=" * 80
    )
    print(
        "Scientific mode:   feature characterization, not class-label replication"
    )
    print(
        "DS-005 classes:    Long_CS / LLC equivalents NOT available in DS-006"
    )
    print(
        "Features:          18 frozen DS-006 handcrafted features"
    )
    print(
        f"Seeds:             {list(SEEDS)}"
    )
    print(
        "Cluster alignment: TRAIN-derived seed->seed11 mapping"
    )
    print(
        "TEST partition:    PROTECTED / NOT LOADED"
    )
    print()

    per_seed: Dict[
        int,
        Dict[
            str,
            Any,
        ],
    ] = {}

    train_val_by_seed: Dict[
        int,
        Dict[
            str,
            Any,
        ],
    ] = {}

    validation_by_seed: Dict[
        int,
        Dict[
            str,
            Any,
        ],
    ] = {}

    written: List[
        Path
    ] = []

    for seed in SEEDS:
        seed_dir = (
            output_root
            / f"seed{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        per_seed[
            seed
        ] = {}

        print(
            "=" * 80
        )
        print(
            f"SEED {seed}"
        )
        print(
            "=" * 80
        )

        for partition in PARTITIONS:
            cluster_ids = cluster_source_bout_ids(
                cluster_root,
                seed=seed,
                partition=partition,
            )

            X = align_baseline(
                baseline[
                    partition
                ],
                cluster_ids,
            )

            labels = load_labels(
                cluster_root,
                seed=seed,
                partition=partition,
                aligned=True,
            )

            characterization = characterize_partition(
                X,
                labels,
            )

            result = {
                "dataset_id": "DS-006",
                "analysis": (
                    "transfer_cluster_kinematic_characterization"
                ),
                "partition": (
                    partition
                ),
                "ssl_encoder_seed": int(
                    seed
                ),
                "cluster_count": int(
                    K
                ),
                "cluster_labels": (
                    "TRAIN-derived alignment to seed11"
                ),
                "feature_manifest": str(
                    FEATURE_MANIFEST.relative_to(
                        REPO_ROOT
                    )
                ),
                "feature_manifest_sha256": sha256_file(
                    FEATURE_MANIFEST
                ),
                "baseline_source": str(
                    baseline[
                        partition
                    ][
                        "path"
                    ].relative_to(
                        REPO_ROOT
                    )
                ),
                "baseline_source_sha256": baseline[
                    partition
                ][
                    "sha256"
                ],
                "bout_id_alignment_verified": True,
                "class_label_replication": False,
                "reason_class_label_replication_unavailable": (
                    "Frozen DS-006 processed schema contains experimental "
                    "conditions but no DS-005-equivalent conventional "
                    "bout-class labels such as Long_CS or LLC."
                ),
                "test_partition_used": False,
                **characterization,
            }

            per_seed[
                seed
            ][
                partition
            ] = result

            if partition == "validation":
                validation_by_seed[
                    seed
                ] = result

            out_path = (
                seed_dir
                / f"{partition}_feature_characterization.json"
            )

            atomic_json(
                out_path,
                result,
            )

            written.append(
                out_path
            )

            print(
                partition.upper()
            )

            for row in characterization[
                "ranking_by_eta_squared"
            ][
                :8
            ]:
                print(
                    f"  #{row['rank']:<2} "
                    f"{row['feature']:<24} "
                    f"eta^2={row['eta_squared']:.6f}"
                )

        train_val = train_validation_reproducibility(
            per_seed[
                seed
            ][
                "train"
            ],
            per_seed[
                seed
            ][
                "validation"
            ],
        )

        train_val_by_seed[
            seed
        ] = train_val

        train_val_path = (
            seed_dir
            / "train_validation_reproducibility.json"
        )

        atomic_json(
            train_val_path,
            {
                "dataset_id": "DS-006",
                "ssl_encoder_seed": int(
                    seed
                ),
                "analysis": (
                    "train_validation_cluster_feature_profile_reproducibility"
                ),
                "features": train_val,
                "test_partition_used": False,
            },
        )

        written.append(
            train_val_path
        )

        print(
            "Targeted TRAIN->VALIDATION profile reproducibility:"
        )

        for feature in (
            "bout_duration",
            "speed_change_rms",
            "speed_change_std",
            "turn_net",
            "turn_total_abs",
            "turn_rms",
        ):
            values = train_val[
                feature
            ]

            print(
                f"  {feature:<24} "
                f"VAL eta^2={values['validation_eta_squared']:.6f}  "
                f"mean rho={values['mean_profile_spearman']}  "
                f"median rho={values['median_profile_spearman']}"
            )

        print(
            "TEST partition used: NO"
        )
        print()

    cross_seed = cross_seed_validation_reproducibility(
        validation_by_seed
    )

    cross_seed_path = (
        output_root
        / "cross_seed_validation_profiles.json"
    )

    atomic_json(
        cross_seed_path,
        {
            "dataset_id": "DS-006",
            "analysis": (
                "cross_seed_validation_feature_profile_reproducibility"
            ),
            "reference_alignment": (
                "TRAIN-derived alignment to seed11"
            ),
            "features": cross_seed,
            "test_partition_used": False,
        },
    )

    written.append(
        cross_seed_path
    )

    targeted = targeted_axis_summary(
        cross_seed,
        train_val_by_seed,
    )

    targeted_path = (
        output_root
        / "targeted_axes_summary.json"
    )

    atomic_json(
        targeted_path,
        {
            "dataset_id": "DS-006",
            "analysis": (
                "targeted_kinematic_axis_replication"
            ),
            "important_limitation": (
                "DS-006 lacks DS-005-equivalent conventional bout-class "
                "labels; these are analogous kinematic axes across the "
                "frozen transfer clusters, not direct Long_CS/LLC "
                "within-class replications."
            ),
            "groups": targeted,
            "test_partition_used": False,
        },
    )

    written.append(
        targeted_path
    )

    # Aggregate ranking across encoder seeds using mean validation eta^2.
    mean_eta = {
        feature: float(
            np.mean(
                [
                    validation_by_seed[
                        seed
                    ][
                        "features"
                    ][
                        feature
                    ][
                        "eta_squared"
                    ]
                    for seed
                    in SEEDS
                ]
            )
        )
        for feature
        in EXPECTED_FEATURE_NAMES
    }

    ranked_features = sorted(
        EXPECTED_FEATURE_NAMES,
        key=lambda name: mean_eta[
            name
        ],
        reverse=True,
    )

    summary = {
        "dataset_id": "DS-006",
        "analysis": (
            "transfer_behavioral_kinematic_substructure"
        ),
        "scientific_scope": (
            "reproducible kinematic-axis characterization of frozen "
            "transfer clusters"
        ),
        "direct_long_cs_or_llc_class_replication_available": False,
        "direct_class_replication_limitation": (
            "The frozen DS-006 processed schema contains no conventional "
            "bout-class labels equivalent to DS-005 Long_CS or LLC."
        ),
        "feature_manifest_sha256": sha256_file(
            FEATURE_MANIFEST
        ),
        "cross_seed_stability_source_sha256": (
            cross_seed_stability_hash
        ),
        "seeds": list(
            SEEDS
        ),
        "validation_feature_ranking_by_mean_eta_squared": [
            {
                "rank": index + 1,
                "feature": feature,
                "mean_validation_eta_squared": float(
                    mean_eta[
                        feature
                    ]
                ),
                "cross_seed_mean_profile_spearman": (
                    cross_seed[
                        feature
                    ][
                        "cross_seed_validation_mean_profile_spearman"
                    ][
                        "mean"
                    ]
                ),
                "cross_seed_median_profile_spearman": (
                    cross_seed[
                        feature
                    ][
                        "cross_seed_validation_median_profile_spearman"
                    ][
                        "mean"
                    ]
                ),
                "mean_train_to_validation_mean_profile_spearman": (
                    float(
                        np.mean(
                            [
                                train_val_by_seed[
                                    seed
                                ][
                                    feature
                                ][
                                    "mean_profile_spearman"
                                ]
                                for seed
                                in SEEDS
                                if train_val_by_seed[
                                    seed
                                ][
                                    feature
                                ][
                                    "mean_profile_spearman"
                                ]
                                is not None
                            ]
                        )
                    )
                    if any(
                        train_val_by_seed[
                            seed
                        ][
                            feature
                        ][
                            "mean_profile_spearman"
                        ]
                        is not None
                        for seed
                        in SEEDS
                    )
                    else None
                ),
            }
            for index, feature
            in enumerate(
                ranked_features
            )
        ],
        "targeted_axes": {
            "duration_speed_change": (
                LONG_CS_LIKE_AXES
            ),
            "turning": (
                LLC_LIKE_AXES
            ),
        },
        "no_feature_selection_performed": True,
        "no_cluster_refitting_performed": True,
        "test_partition_used": False,
    }

    atomic_json(
        summary_path,
        summary,
    )

    written.append(
        summary_path
    )

    checksum_path = write_checksums(
        output_root,
        written,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 SUBSTRUCTURE SUMMARY"
    )
    print(
        "=" * 80
    )
    print(
        "Top VALIDATION features by mean eta^2 across seeds:"
    )

    for index, feature in enumerate(
        ranked_features[
            :10
        ],
        start=1,
    ):
        stats = cross_seed[
            feature
        ]

        print(
            f"  #{index:<2} "
            f"{feature:<24} "
            f"mean eta^2={mean_eta[feature]:.6f}  "
            f"cross-seed mean-profile rho="
            f"{stats['cross_seed_validation_mean_profile_spearman']['mean']}"
        )

    print()
    print(
        "Direct Long_CS/LLC class-label replication: NO "
        "(labels unavailable in DS-006)"
    )
    print(
        "Analogous duration/speed-change characterization: YES"
    )
    print(
        "Analogous turning characterization:              YES"
    )
    print(
        "TEST partition used:                              NO"
    )
    print(
        f"Summary:       {summary_path}"
    )
    print(
        f"Targeted axes: {targeted_path}"
    )
    print(
        f"Cross-seed:    {cross_seed_path}"
    )
    print(
        f"Checksums:     {checksum_path}"
    )


if __name__ == "__main__":
    main()
