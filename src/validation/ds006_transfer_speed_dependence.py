#!/usr/bin/env python3
"""DS-006 speed-dependence control for frozen transfer clustering.

Purpose
-------
Test whether the frozen k=8 DS-006 transfer clusters are associated with
locomotor speed, and whether cluster membership can be explained by speed alone.

This script mirrors the logic of the DS-005 speed-dependence control:

1. quantify speed dependence using eta-squared;
2. fit a speed-only multinomial classifier on DS-006 TRAIN;
3. evaluate on held-out DS-006 VALIDATION;
4. aggregate results across frozen SSL encoder seeds 11, 23, 37, 51, 79.

Scientific interpretation
-------------------------
A strong eta-squared means cluster membership is related to locomotor speed.
A low held-out speed-only classification score means the cluster structure is
not reducible to mean speed alone.

This script does NOT:
- access DS-006 TEST;
- refit clustering;
- change k;
- retrain/fine-tune the SSL encoder;
- modify DS-005;
- choose speed thresholds from validation results.

Inputs
------
Frozen DS-006 handcrafted/metadata source:
    data/processed/DS-006/baseline/
or equivalent NPZ/CSV containing row-aligned bout IDs and mean-speed feature.

Frozen DS-006 transfer clustering:
    data/processed/DS-006/transfer_clustering/
        seed11/train_labels.npy
        seed11/validation_labels.npy
        ...
        seed79/

The script tries several frozen DS-006 processed artifact locations and
feature names so it can work with the existing preparation pipeline without
manual editing.

Outputs
-------
data/processed/DS-006/transfer_speed_dependence/
    seed11.json
    seed23.json
    seed37.json
    seed51.json
    seed79.json
    summary.json
    DS006_TRANSFER_SPEED_DEPENDENCE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ds006_transfer_speed_dependence.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ds006_transfer_speed_dependence.py \
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

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
    / "transfer_speed_dependence"
)

FEATURE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "baseline"
    / "feature_manifest.json"
)

SEEDS = (11, 23, 37, 51, 79)
PARTITIONS = ("train", "validation")

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

FROZEN_K = 8
CLASSIFICATION_RANDOM_SEED = 20260822

# Preferred frozen feature naming used in the project.
SPEED_FEATURE_CANDIDATES = (
    "speed_mean",
    "mean_speed",
    "speed_mean_mm_s",
    "mean_speed_mm_s",
    "bout_speed_mean",
    "head_speed_mean",
)

# Candidate processed sources created by DS-006 preparation.
NPZ_CANDIDATES = {
    "train": (
        "data/processed/DS-006/baseline/train.npz",
        "data/processed/DS-006/baseline/train_core_raw.npz",
        "data/processed/DS-006/train_core_raw.npz",
        "data/processed/DS-006/train.npz",
    ),
    "validation": (
        "data/processed/DS-006/baseline/validation.npz",
        "data/processed/DS-006/baseline/validation_core_raw.npz",
        "data/processed/DS-006/validation_core_raw.npz",
        "data/processed/DS-006/validation.npz",
    ),
}

CSV_CANDIDATES = {
    "train": (
        "data/processed/DS-006/baseline/train_metadata.csv",
        "data/processed/DS-006/train_metadata.csv",
    ),
    "validation": (
        "data/processed/DS-006/baseline/validation_metadata.csv",
        "data/processed/DS-006/validation_metadata.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retest DS-006 transfer-cluster dependence on mean speed. "
            "TRAIN/VALIDATION only; TEST prohibited."
        )
    )

    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=CLUSTER_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--speed-feature",
        type=str,
        default=None,
        help=(
            "Optional explicit speed feature name. If omitted, the script "
            "searches the frozen candidate names."
        ),
    )

    parser.add_argument(
        "--train-source",
        type=Path,
        default=None,
        help=(
            "Optional explicit TRAIN NPZ/CSV source containing speed and bout IDs."
        ),
    )

    parser.add_argument(
        "--validation-source",
        type=Path,
        default=None,
        help=(
            "Optional explicit VALIDATION NPZ/CSV source containing speed and bout IDs."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_paths(
    cluster_root: Path,
    output_root: Path,
) -> None:
    cluster_root = cluster_root.resolve()
    output_root = output_root.resolve()

    expected_cluster_root = CLUSTER_ROOT.resolve()

    if cluster_root != expected_cluster_root:
        raise RuntimeError(
            "For replication safety, --cluster-root must resolve exactly to "
            f"{expected_cluster_root}; got {cluster_root}"
        )

    ds006_processed = (
        REPO_ROOT
        / "data"
        / "processed"
        / "DS-006"
    ).resolve()

    ds005_processed = (
        REPO_ROOT
        / "data"
        / "processed"
        / "DS-005"
    ).resolve()

    if not is_relative_to(
        output_root,
        ds006_processed,
    ):
        raise RuntimeError(
            "Outputs must remain under data/processed/DS-006."
        )

    if is_relative_to(
        output_root,
        ds005_processed,
    ):
        raise RuntimeError(
            "Refusing to write anything under DS-005."
        )


def resolve_source(
    partition: str,
    explicit: Optional[Path],
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    if explicit is not None:
        path = explicit.expanduser().resolve()

        if "test" in path.name.lower():
            raise RuntimeError(
                f"Protected TEST source supplied: {path}"
            )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        return path

    candidates = (
        NPZ_CANDIDATES[
            partition
        ]
        + CSV_CANDIDATES[
            partition
        ]
    )

    existing = [
        (
            REPO_ROOT
            / candidate
        ).resolve()
        for candidate
        in candidates
        if (
            REPO_ROOT
            / candidate
        ).exists()
    ]

    if not existing:
        raise FileNotFoundError(
            "Could not find a frozen DS-006 speed-feature source for "
            f"{partition.upper()}.\n"
            "Tried:\n"
            + "\n".join(
                "  "
                + str(
                    REPO_ROOT
                    / candidate
                )
                for candidate
                in candidates
            )
            + "\n\nUse --train-source/--validation-source to specify the "
              "prepared DS-006 feature artifact explicitly."
        )

    return existing[
        0
    ]


def _pick_bout_id_key(
    keys: Iterable[str],
) -> Optional[str]:
    candidates = (
        "bout_id",
        "bout_ids",
        "id",
        "ids",
    )

    keyset = set(
        keys
    )

    for candidate in candidates:
        if candidate in keyset:
            return candidate

    return None


def _pick_speed_key(
    keys: Iterable[str],
    explicit_name: Optional[str],
) -> Optional[str]:
    keyset = set(
        keys
    )

    if explicit_name is not None:
        if explicit_name in keyset:
            return explicit_name
        return None

    for candidate in SPEED_FEATURE_CANDIDATES:
        if candidate in keyset:
            return candidate

    return None


def load_frozen_feature_names() -> List[str]:
    """Read the frozen DS-006 baseline feature order from feature_manifest.json."""
    path = FEATURE_MANIFEST.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen DS-006 feature manifest not found: {path}"
        )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    names = obj.get(
        "feature_names"
    )

    if not isinstance(
        names,
        list,
    ):
        raise RuntimeError(
            f"{path}: missing list field 'feature_names'."
        )

    names = [
        str(
            x
        )
        for x
        in names
    ]

    if len(
        names
    ) != 18:
        raise RuntimeError(
            f"{path}: expected 18 frozen baseline features, got {len(names)}."
        )

    if "speed_mean" not in names:
        raise RuntimeError(
            f"{path}: frozen feature 'speed_mean' is missing."
        )

    return names


def _find_speed_in_feature_matrix(
    npz: Mapping[str, np.ndarray],
    *,
    explicit_name: Optional[str],
) -> Optional[Tuple[np.ndarray, str]]:
    """Resolve speed from X using either embedded names or frozen manifest.

    DS-006 *_core_raw.npz intentionally stores only:
        X
        bout_id

    The frozen feature order is stored in:
        data/processed/DS-006/baseline/feature_manifest.json
    """
    feature_matrix_keys = (
        "X",
        "features",
        "X_raw",
        "baseline_features",
    )

    feature_name_keys = (
        "feature_names",
        "features_names",
        "columns",
    )

    matrix_key = next(
        (
            key
            for key
            in feature_matrix_keys
            if key in npz
        ),
        None,
    )

    if matrix_key is None:
        return None

    matrix = np.asarray(
        npz[
            matrix_key
        ]
    )

    if matrix.ndim != 2:
        return None

    # Prefer embedded names if present; otherwise use the authoritative
    # frozen DS-006 feature manifest written by prepare_ds006.py.
    names_key = next(
        (
            key
            for key
            in feature_name_keys
            if key in npz
        ),
        None,
    )

    if names_key is not None:
        names = np.asarray(
            npz[
                names_key
            ]
        ).astype(str).tolist()

        name_source = (
            f"embedded:{names_key}"
        )
    else:
        names = load_frozen_feature_names()

        name_source = str(
            FEATURE_MANIFEST.relative_to(
                REPO_ROOT
            )
        )

    if matrix.shape[
        1
    ] != len(
        names
    ):
        raise RuntimeError(
            f"Feature matrix has {matrix.shape[1]} columns but "
            f"feature manifest has {len(names)} names."
        )

    target_names = (
        (
            explicit_name,
        )
        if explicit_name is not None
        else SPEED_FEATURE_CANDIDATES
    )

    for candidate in target_names:
        if candidate in names:
            index = names.index(
                candidate
            )

            return (
                np.asarray(
                    matrix[
                        :,
                        index
                    ],
                    dtype=np.float64,
                ),
                (
                    f"{matrix_key}[{candidate}] "
                    f"(column {index}; names from {name_source})"
                ),
            )

    return None


def load_npz_speed(
    path: Path,
    *,
    explicit_speed_feature: Optional[str],
) -> Dict[str, Any]:
    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        keys = list(
            npz.files
        )

        bout_key = _pick_bout_id_key(
            keys
        )

        if bout_key is None:
            raise RuntimeError(
                f"{path}: could not find bout_id/bout_ids array."
            )

        direct_speed_key = _pick_speed_key(
            keys,
            explicit_speed_feature,
        )

        if direct_speed_key is not None:
            speed = np.asarray(
                npz[
                    direct_speed_key
                ],
                dtype=np.float64,
            )

            speed_source = direct_speed_key

        else:
            matrix_result = _find_speed_in_feature_matrix(
                {
                    key: npz[
                        key
                    ]
                    for key
                    in keys
                },
                explicit_name=(
                    explicit_speed_feature
                ),
            )

            if matrix_result is None:
                raise RuntimeError(
                    f"{path}: no recognized speed feature found.\n"
                    f"Available arrays: {keys}\n"
                    f"Expected one of: {SPEED_FEATURE_CANDIDATES}\n"
                    "Or provide --speed-feature explicitly."
                )

            speed, speed_source = (
                matrix_result
            )

        bout_id = np.asarray(
            npz[
                bout_key
            ]
        ).astype(str)

    return {
        "speed": speed,
        "bout_id": bout_id,
        "speed_source": speed_source,
        "bout_id_source": bout_key,
    }


def load_csv_speed(
    path: Path,
    *,
    explicit_speed_feature: Optional[str],
) -> Dict[str, Any]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        bout_key = _pick_bout_id_key(
            fieldnames
        )

        if bout_key is None:
            raise RuntimeError(
                f"{path}: could not find bout_id column."
            )

        speed_key = _pick_speed_key(
            fieldnames,
            explicit_speed_feature,
        )

        if speed_key is None:
            raise RuntimeError(
                f"{path}: no recognized speed feature column.\n"
                f"Columns: {fieldnames}\n"
                f"Expected one of: {SPEED_FEATURE_CANDIDATES}"
            )

        bout_ids: List[
            str
        ] = []

        speeds: List[
            float
        ] = []

        for row in reader:
            bout_ids.append(
                str(
                    row[
                        bout_key
                    ]
                )
            )

            speeds.append(
                float(
                    row[
                        speed_key
                    ]
                )
            )

    return {
        "speed": np.asarray(
            speeds,
            dtype=np.float64,
        ),
        "bout_id": np.asarray(
            bout_ids,
            dtype=str,
        ),
        "speed_source": speed_key,
        "bout_id_source": bout_key,
    }


def load_speed_source(
    path: Path,
    *,
    partition: str,
    explicit_speed_feature: Optional[str],
) -> Dict[str, Any]:
    if "test" in path.name.lower():
        raise RuntimeError(
            f"Protected TEST source reached: {path}"
        )

    if path.suffix.lower() == ".npz":
        result = load_npz_speed(
            path,
            explicit_speed_feature=(
                explicit_speed_feature
            ),
        )

    elif path.suffix.lower() == ".csv":
        result = load_csv_speed(
            path,
            explicit_speed_feature=(
                explicit_speed_feature
            ),
        )

    else:
        raise RuntimeError(
            f"Unsupported feature source format: {path}"
        )

    speed = np.asarray(
        result[
            "speed"
        ],
        dtype=np.float64,
    )

    bout_id = np.asarray(
        result[
            "bout_id"
        ]
    ).astype(str)

    expected_rows = EXPECTED_ROWS[
        partition
    ]

    if speed.ndim != 1:
        raise RuntimeError(
            f"{path}: speed must be one-dimensional."
        )

    if bout_id.ndim != 1:
        raise RuntimeError(
            f"{path}: bout_id must be one-dimensional."
        )

    if len(
        speed
    ) != expected_rows:
        raise RuntimeError(
            f"{path}: expected {expected_rows:,} {partition} rows, "
            f"got {len(speed):,}."
        )

    if len(
        bout_id
    ) != expected_rows:
        raise RuntimeError(
            f"{path}: bout ID row count mismatch."
        )

    if not np.isfinite(
        speed
    ).all():
        raise RuntimeError(
            f"{path}: non-finite speed values."
        )

    if len(
        np.unique(
            bout_id
        )
    ) != len(
        bout_id
    ):
        raise RuntimeError(
            f"{path}: duplicate bout IDs."
        )

    result[
        "path"
    ] = path

    result[
        "sha256"
    ] = sha256_file(
        path
    )

    return result


def load_cluster_labels(
    cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    path = (
        cluster_root
        / f"seed{seed}"
        / f"{partition}_labels.npy"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST label path reached."
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

    expected_rows = EXPECTED_ROWS[
        partition
    ]

    if labels.shape != (
        expected_rows,
    ):
        raise RuntimeError(
            f"{path}: expected {expected_rows:,} labels, "
            f"got {labels.shape}."
        )

    unique = np.unique(
        labels
    )

    if not np.array_equal(
        unique,
        np.arange(
            FROZEN_K,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"{path}: expected all labels 0..{FROZEN_K - 1}; got {unique}."
        )

    return labels


def load_transfer_bout_ids(
    cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    """Get frozen row ordering from the source transfer-embedding artifact."""
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

    source_key = (
        "source_train_embedding"
        if partition == "train"
        else "source_validation_embedding"
    )

    source_rel = manifest.get(
        source_key
    )

    if not source_rel:
        raise RuntimeError(
            f"{manifest_path}: missing {source_key}."
        )

    source_path = (
        REPO_ROOT
        / source_rel
    ).resolve()

    if "test" in source_path.name.lower():
        raise RuntimeError(
            "Protected TEST embedding source reached."
        )

    with np.load(
        source_path,
        allow_pickle=False,
    ) as npz:
        bout_id = np.asarray(
            npz[
                "bout_id"
            ]
        ).astype(str)

    return bout_id


def align_speed_to_cluster_rows(
    *,
    feature_bout_ids: np.ndarray,
    feature_speed: np.ndarray,
    cluster_bout_ids: np.ndarray,
) -> np.ndarray:
    if np.array_equal(
        feature_bout_ids,
        cluster_bout_ids,
    ):
        return feature_speed.astype(
            np.float64,
            copy=False,
        )

    lookup = {
        bout_id: index
        for index, bout_id
        in enumerate(
            feature_bout_ids
        )
    }

    missing = [
        bout_id
        for bout_id
        in cluster_bout_ids
        if bout_id not in lookup
    ]

    if missing:
        raise RuntimeError(
            f"Could not align {len(missing):,} cluster rows to speed rows. "
            f"Example missing bout ID: {missing[0]}"
        )

    indices = np.asarray(
        [
            lookup[
                bout_id
            ]
            for bout_id
            in cluster_bout_ids
        ],
        dtype=np.int64,
    )

    aligned = feature_speed[
        indices
    ]

    if len(
        np.unique(
            indices
        )
    ) != len(
        indices
    ):
        raise RuntimeError(
            "Bout-ID alignment reused feature rows unexpectedly."
        )

    return aligned.astype(
        np.float64,
        copy=False,
    )


def eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    """One-way ANOVA effect size eta^2 = SS_between / SS_total."""
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    grand_mean = float(
        np.mean(
            values
        )
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

    for cluster in range(
        FROZEN_K
    ):
        mask = (
            labels
            == cluster
        )

        if not np.any(
            mask
        ):
            continue

        cluster_values = values[
            mask
        ]

        cluster_mean = float(
            np.mean(
                cluster_values
            )
        )

        ss_between += (
            int(
                mask.sum()
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


def cluster_speed_summary(
    values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {}

    for cluster in range(
        FROZEN_K
    ):
        x = values[
            labels
            == cluster
        ]

        output[
            str(
                cluster
            )
        ] = {
            "n": int(
                len(
                    x
                )
            ),
            "mean": float(
                np.mean(
                    x
                )
            ),
            "std": float(
                np.std(
                    x
                )
            ),
            "median": float(
                np.median(
                    x
                )
            ),
            "p05": float(
                np.percentile(
                    x,
                    5,
                )
            ),
            "p95": float(
                np.percentile(
                    x,
                    95,
                )
            ),
        }

    return output


def fit_speed_only_classifier(
    train_speed: np.ndarray,
    train_labels: np.ndarray,
    validation_speed: np.ndarray,
    validation_labels: np.ndarray,
    *,
    max_iter: int,
) -> Dict[str, Any]:
    # Fit scaling on TRAIN only.
    train_mean = float(
        np.mean(
            train_speed
        )
    )

    train_std = float(
        np.std(
            train_speed
        )
    )

    if train_std <= 0:
        raise RuntimeError(
            "TRAIN mean speed has zero variance."
        )

    X_train = (
        (
            train_speed
            - train_mean
        )
        / train_std
    ).reshape(
        -1,
        1,
    )

    X_validation = (
        (
            validation_speed
            - train_mean
        )
        / train_std
    ).reshape(
        -1,
        1,
    )

    # Multinomial logistic regression as the frozen simple speed-only probe.
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=CLASSIFICATION_RANDOM_SEED,
        multi_class="auto",
    )

    classifier.fit(
        X_train,
        train_labels,
    )

    train_pred = classifier.predict(
        X_train
    )

    validation_pred = classifier.predict(
        X_validation
    )

    chance_balanced_accuracy = (
        1.0
        / FROZEN_K
    )

    return {
        "train_speed_mean": (
            train_mean
        ),
        "train_speed_std": (
            train_std
        ),
        "classifier": (
            "multinomial_logistic_regression"
        ),
        "features": [
            "mean_speed"
        ],
        "fit_partition": (
            "train"
        ),
        "train": {
            "accuracy": float(
                accuracy_score(
                    train_labels,
                    train_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    train_labels,
                    train_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    train_labels,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "validation": {
            "accuracy": float(
                accuracy_score(
                    validation_labels,
                    validation_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    validation_labels,
                    validation_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    validation_labels,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "chance_balanced_accuracy": float(
            chance_balanced_accuracy
        ),
        "validation_balanced_accuracy_over_chance": float(
            balanced_accuracy_score(
                validation_labels,
                validation_pred,
            )
            / chance_balanced_accuracy
        ),
    }


def write_checksums(
    output_root: Path,
    files: Sequence[
        Path
    ],
) -> Path:
    path = (
        output_root
        / "DS006_TRANSFER_SPEED_DEPENDENCE_SHA256SUMS"
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
        cluster_root,
        output_root,
    )

    if args.max_iter < 1:
        raise ValueError(
            "--max-iter must be >= 1."
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

    train_source = resolve_source(
        "train",
        args.train_source,
    )

    validation_source = resolve_source(
        "validation",
        args.validation_source,
    )

    train_features = load_speed_source(
        train_source,
        partition="train",
        explicit_speed_feature=(
            args.speed_feature
        ),
    )

    validation_features = load_speed_source(
        validation_source,
        partition="validation",
        explicit_speed_feature=(
            args.speed_feature
        ),
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 TRANSFER SPEED DEPENDENCE"
    )
    print(
        "=" * 80
    )
    print(
        f"TRAIN source:      {train_source}"
    )
    print(
        f"VALIDATION source: {validation_source}"
    )
    print(
        f"Speed feature:     {train_features['speed_source']}"
    )
    print(
        f"Feature manifest:  {FEATURE_MANIFEST}"
    )
    print(
        f"Seeds:             {list(SEEDS)}"
    )
    print(
        f"k:                 {FROZEN_K}"
    )
    print(
        "Speed probe fit:   TRAIN only"
    )
    print(
        "TEST partition:    PROTECTED / NOT LOADED"
    )
    print()

    results: Dict[
        int,
        Dict[str, Any],
    ] = {}

    written: List[
        Path
    ] = []

    reference_train_ids: Optional[
        np.ndarray
    ] = None

    reference_validation_ids: Optional[
        np.ndarray
    ] = None

    for seed in SEEDS:
        print(
            "=" * 80
        )
        print(
            f"SEED {seed}"
        )
        print(
            "=" * 80
        )

        train_labels = load_cluster_labels(
            cluster_root,
            seed=seed,
            partition="train",
        )

        validation_labels = load_cluster_labels(
            cluster_root,
            seed=seed,
            partition="validation",
        )

        cluster_train_ids = load_transfer_bout_ids(
            cluster_root,
            seed=seed,
            partition="train",
        )

        cluster_validation_ids = load_transfer_bout_ids(
            cluster_root,
            seed=seed,
            partition="validation",
        )

        if reference_train_ids is None:
            reference_train_ids = cluster_train_ids
            reference_validation_ids = cluster_validation_ids
        else:
            if not np.array_equal(
                reference_train_ids,
                cluster_train_ids,
            ):
                raise RuntimeError(
                    f"TRAIN bout ordering differs at seed {seed}."
                )

            if not np.array_equal(
                reference_validation_ids,
                cluster_validation_ids,
            ):
                raise RuntimeError(
                    f"VALIDATION bout ordering differs at seed {seed}."
                )

        train_speed = align_speed_to_cluster_rows(
            feature_bout_ids=(
                train_features[
                    "bout_id"
                ]
            ),
            feature_speed=(
                train_features[
                    "speed"
                ]
            ),
            cluster_bout_ids=(
                cluster_train_ids
            ),
        )

        validation_speed = align_speed_to_cluster_rows(
            feature_bout_ids=(
                validation_features[
                    "bout_id"
                ]
            ),
            feature_speed=(
                validation_features[
                    "speed"
                ]
            ),
            cluster_bout_ids=(
                cluster_validation_ids
            ),
        )

        train_eta2 = eta_squared(
            train_speed,
            train_labels,
        )

        validation_eta2 = eta_squared(
            validation_speed,
            validation_labels,
        )

        speed_probe = fit_speed_only_classifier(
            train_speed,
            train_labels,
            validation_speed,
            validation_labels,
            max_iter=args.max_iter,
        )

        result = {
            "dataset_id": "DS-006",
            "analysis": (
                "transfer_cluster_speed_dependence"
            ),
            "source_encoder_dataset": "DS-005",
            "ssl_encoder_seed": int(
                seed
            ),
            "k": int(
                FROZEN_K
            ),
            "speed_feature": (
                train_features[
                    "speed_source"
                ]
            ),
            "train_eta_squared": float(
                train_eta2
            ),
            "validation_eta_squared": float(
                validation_eta2
            ),
            "train_cluster_speed_summary": cluster_speed_summary(
                train_speed,
                train_labels,
            ),
            "validation_cluster_speed_summary": cluster_speed_summary(
                validation_speed,
                validation_labels,
            ),
            "speed_only_probe": (
                speed_probe
            ),
            "train_feature_source": str(
                train_source.relative_to(
                    REPO_ROOT
                )
            )
            if is_relative_to(
                train_source,
                REPO_ROOT
            )
            else str(
                train_source
            ),
            "validation_feature_source": str(
                validation_source.relative_to(
                    REPO_ROOT
                )
            )
            if is_relative_to(
                validation_source,
                REPO_ROOT
            )
            else str(
                validation_source
            ),
            "train_feature_source_sha256": (
                train_features[
                    "sha256"
                ]
            ),
            "validation_feature_source_sha256": (
                validation_features[
                    "sha256"
                ]
            ),
            "feature_manifest": str(
                FEATURE_MANIFEST.relative_to(
                    REPO_ROOT
                )
            ),
            "feature_manifest_sha256": sha256_file(
                FEATURE_MANIFEST
            ),
            "bout_id_alignment_verified": True,
            "speed_probe_fit_on_train_only": True,
            "test_partition_used": False,
        }

        results[
            seed
        ] = result

        out_path = (
            output_root
            / f"seed{seed}.json"
        )

        atomic_write_json(
            out_path,
            result,
        )

        written.append(
            out_path
        )

        print(
            f"TRAIN eta^2:                    {train_eta2:.6f}"
        )
        print(
            f"VALIDATION eta^2:               {validation_eta2:.6f}"
        )
        print(
            "Speed-only VAL balanced acc:    "
            f"{speed_probe['validation']['balanced_accuracy']:.6f}"
        )
        print(
            "Speed-only VAL macro F1:         "
            f"{speed_probe['validation']['macro_f1']:.6f}"
        )
        print(
            "Speed-only VAL accuracy:         "
            f"{speed_probe['validation']['accuracy']:.6f}"
        )
        print(
            "Chance balanced accuracy:        "
            f"{speed_probe['chance_balanced_accuracy']:.6f}"
        )
        print(
            "TEST partition used:             NO"
        )
        print()

    train_eta2_values = np.asarray(
        [
            results[
                seed
            ][
                "train_eta_squared"
            ]
            for seed
            in SEEDS
        ],
        dtype=np.float64,
    )

    validation_eta2_values = np.asarray(
        [
            results[
                seed
            ][
                "validation_eta_squared"
            ]
            for seed
            in SEEDS
        ],
        dtype=np.float64,
    )

    val_bal_acc = np.asarray(
        [
            results[
                seed
            ][
                "speed_only_probe"
            ][
                "validation"
            ][
                "balanced_accuracy"
            ]
            for seed
            in SEEDS
        ],
        dtype=np.float64,
    )

    val_macro_f1 = np.asarray(
        [
            results[
                seed
            ][
                "speed_only_probe"
            ][
                "validation"
            ][
                "macro_f1"
            ]
            for seed
            in SEEDS
        ],
        dtype=np.float64,
    )

    val_accuracy = np.asarray(
        [
            results[
                seed
            ][
                "speed_only_probe"
            ][
                "validation"
            ][
                "accuracy"
            ]
            for seed
            in SEEDS
        ],
        dtype=np.float64,
    )

    chance = (
        1.0
        / FROZEN_K
    )

    summary = {
        "dataset_id": "DS-006",
        "analysis": (
            "transfer_cluster_speed_dependence"
        ),
        "source_encoder_dataset": "DS-005",
        "seeds": list(
            SEEDS
        ),
        "k": int(
            FROZEN_K
        ),
        "chance_balanced_accuracy": float(
            chance
        ),
        "aggregate": {
            "train_eta_squared": {
                "mean": float(
                    np.mean(
                        train_eta2_values
                    )
                ),
                "std": float(
                    np.std(
                        train_eta2_values
                    )
                ),
                "min": float(
                    np.min(
                        train_eta2_values
                    )
                ),
                "max": float(
                    np.max(
                        train_eta2_values
                    )
                ),
            },
            "validation_eta_squared": {
                "mean": float(
                    np.mean(
                        validation_eta2_values
                    )
                ),
                "std": float(
                    np.std(
                        validation_eta2_values
                    )
                ),
                "min": float(
                    np.min(
                        validation_eta2_values
                    )
                ),
                "max": float(
                    np.max(
                        validation_eta2_values
                    )
                ),
            },
            "speed_only_validation_balanced_accuracy": {
                "mean": float(
                    np.mean(
                        val_bal_acc
                    )
                ),
                "std": float(
                    np.std(
                        val_bal_acc
                    )
                ),
                "min": float(
                    np.min(
                        val_bal_acc
                    )
                ),
                "max": float(
                    np.max(
                        val_bal_acc
                    )
                ),
                "mean_over_chance": float(
                    np.mean(
                        val_bal_acc
                    )
                    / chance
                ),
            },
            "speed_only_validation_macro_f1": {
                "mean": float(
                    np.mean(
                        val_macro_f1
                    )
                ),
                "std": float(
                    np.std(
                        val_macro_f1
                    )
                ),
                "min": float(
                    np.min(
                        val_macro_f1
                    )
                ),
                "max": float(
                    np.max(
                        val_macro_f1
                    )
                ),
            },
            "speed_only_validation_accuracy": {
                "mean": float(
                    np.mean(
                        val_accuracy
                    )
                ),
                "std": float(
                    np.std(
                        val_accuracy
                    )
                ),
                "min": float(
                    np.min(
                        val_accuracy
                    )
                ),
                "max": float(
                    np.max(
                        val_accuracy
                    )
                ),
            },
        },
        "per_seed": {
            str(
                seed
            ): results[
                seed
            ]
            for seed
            in SEEDS
        },
        "interpretation_guardrails": {
            "eta_squared_measures_association_not_causality": True,
            "speed_only_probe_uses_mean_speed_only": True,
            "speed_only_probe_fit_on_train_only": True,
            "no_cluster_refitting": True,
            "no_k_selection": True,
            "test_partition_used": False,
        },
        "test_partition_used": False,
    }

    atomic_write_json(
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
        "DS-006 TRANSFER SPEED DEPENDENCE SUMMARY"
    )
    print(
        "=" * 80
    )
    print(
        "Mean TRAIN eta^2:                   "
        f"{np.mean(train_eta2_values):.6f}"
    )
    print(
        "Mean VALIDATION eta^2:              "
        f"{np.mean(validation_eta2_values):.6f}"
    )
    print(
        "Mean speed-only VAL balanced acc:   "
        f"{np.mean(val_bal_acc):.6f}"
    )
    print(
        "Mean speed-only VAL macro F1:        "
        f"{np.mean(val_macro_f1):.6f}"
    )
    print(
        "Mean speed-only VAL accuracy:        "
        f"{np.mean(val_accuracy):.6f}"
    )
    print(
        "Chance balanced accuracy:           "
        f"{chance:.6f}"
    )
    print(
        "TEST partition used:                NO"
    )
    print(
        f"Summary:   {summary_path}"
    )
    print(
        f"Checksums: {checksum_path}"
    )


if __name__ == "__main__":
    main()
