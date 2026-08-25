#!/usr/bin/env python3
"""Compare frozen DS-006 handcrafted baseline structure against transferred SSL clusters.

Purpose
-------
Directly test whether the recomputed DS-006 handcrafted representation explains
the frozen transferred SSL k=8 partition.

This mirrors the frozen DS-005 baseline-vs-SSL comparison logic:

1. compare frozen baseline cluster labels against SSL cluster labels using:
   - ARI
   - NMI
   - AMI
   - H(SSL | baseline) / H(SSL)
   - H(baseline | SSL) / H(baseline)

2. fit a linear 18-feature -> SSL-cluster probe on DS-006 TRAIN and evaluate
   on held-out VALIDATION;

3. fit a nonlinear 18-feature -> SSL-cluster probe on DS-006 TRAIN and evaluate
   on held-out VALIDATION, using the same frozen sensitivity-model family used
   in DS-005.

Scientific interpretation
-------------------------
If baseline clustering differs strongly from SSL but a nonlinear probe recovers
SSL labels well, then the SSL organization is better interpreted as a nonlinear
reorganization of information already present in the handcrafted feature set,
rather than evidence that SSL contains wholly novel information absent from
those features.

This script does NOT:
- access DS-006 TEST;
- refit SSL clustering;
- change k;
- tune probe hyperparameters;
- select among probe families;
- modify DS-005 artifacts.

Expected inputs
---------------
Frozen DS-006 handcrafted features:
    data/processed/DS-006/baseline/train_core_raw.npz
    data/processed/DS-006/baseline/validation_core_raw.npz
    data/processed/DS-006/baseline/feature_manifest.json

Frozen DS-006 baseline clustering:
    data/processed/DS-006/baseline_clustering/
or equivalent frozen baseline label artifact.

Frozen DS-006 transferred SSL clustering:
    data/processed/DS-006/transfer_clustering/seedXX/
        train_labels.npy
        validation_labels.npy
        train_labels_aligned.npy
        validation_labels_aligned.npy

Outputs
-------
data/processed/DS-006/baseline_vs_transfer_ssl/
    seed11.json
    seed23.json
    seed37.json
    seed51.json
    seed79.json
    summary.json
    DS006_BASELINE_VS_TRANSFER_SSL_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ds006_baseline_vs_transfer_ssl.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ds006_baseline_vs_transfer_ssl.py \
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "baseline"
)

TRANSFER_CLUSTER_ROOT = (
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
    / "baseline_vs_transfer_ssl"
)

FEATURE_MANIFEST = (
    BASELINE_ROOT
    / "feature_manifest.json"
)

SEEDS = (11, 23, 37, 51, 79)
PARTITIONS = ("train", "validation")

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

SSL_K = 8

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

# Frozen probe settings.
LINEAR_RANDOM_SEED = 20260822

NONLINEAR_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.0,
    "random_state": 20260822,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare frozen DS-006 handcrafted baseline against frozen "
            "transferred SSL clusters. TRAIN/VALIDATION only."
        )
    )

    p.add_argument(
        "--baseline-root",
        type=Path,
        default=BASELINE_ROOT,
    )

    p.add_argument(
        "--transfer-cluster-root",
        type=Path,
        default=TRANSFER_CLUSTER_ROOT,
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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    os.replace(
        tmp,
        path,
    )


def is_relative_to(
    path: Path,
    parent: Path,
) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_paths(
    baseline_root: Path,
    transfer_cluster_root: Path,
    output_root: Path,
) -> None:
    baseline_root = baseline_root.resolve()
    transfer_cluster_root = transfer_cluster_root.resolve()
    output_root = output_root.resolve()

    if baseline_root != BASELINE_ROOT.resolve():
        raise RuntimeError(
            "--baseline-root must resolve exactly to the frozen DS-006 "
            f"baseline directory: {BASELINE_ROOT.resolve()}"
        )

    if transfer_cluster_root != TRANSFER_CLUSTER_ROOT.resolve():
        raise RuntimeError(
            "--transfer-cluster-root must resolve exactly to the frozen "
            f"DS-006 transfer clustering directory: "
            f"{TRANSFER_CLUSTER_ROOT.resolve()}"
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

    if "test" in str(
        output_root
    ).lower():
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


def baseline_npz_path(
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
            "Protected TEST baseline path reached."
        )

    return path


def load_baseline_features(
    baseline_root: Path,
    partition: str,
) -> Dict[str, Any]:
    path = baseline_npz_path(
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
        if "X" not in npz.files or "bout_id" not in npz.files:
            raise RuntimeError(
                f"{path}: expected arrays X and bout_id."
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

    if len(
        np.unique(
            bout_id
        )
    ) != expected:
        raise RuntimeError(
            f"{path}: duplicate bout IDs."
        )

    return {
        "path": path,
        "sha256": sha256_file(
            path
        ),
        "X": X,
        "bout_id": bout_id,
    }


def candidate_baseline_label_paths(
    partition: str,
) -> List[Path]:
    """Search known frozen DS-006 baseline clustering artifact layouts."""
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    names = [
        f"{partition}_labels.npy",
        f"{partition}_cluster_labels.npy",
        f"{partition}_pred.npy",
        f"{partition}_predictions.npy",
    ]

    roots = [
        REPO_ROOT
        / "data"
        / "processed"
        / "DS-006"
        / "baseline_clustering",

        BASELINE_ROOT
        / "clustering",

        BASELINE_ROOT,
    ]

    paths: List[
        Path
    ] = []

    for root in roots:
        for name in names:
            paths.append(
                root
                / name
            )

    return paths


def discover_baseline_label_path(
    partition: str,
) -> Path:
    existing = [
        path.resolve()
        for path
        in candidate_baseline_label_paths(
            partition
        )
        if path.exists()
    ]

    if not existing:
        tried = "\n".join(
            "  "
            + str(
                p
            )
            for p
            in candidate_baseline_label_paths(
                partition
            )
        )

        raise FileNotFoundError(
            "Could not locate frozen DS-006 baseline-cluster labels for "
            f"{partition.upper()}.\n"
            "Tried:\n"
            f"{tried}\n\n"
            "The handcrafted baseline has been recomputed, but this comparison "
            "requires the frozen DS-006 baseline cluster labels. If your "
            "baseline labels use a different filename, add that path to "
            "candidate_baseline_label_paths()."
        )

    # Deterministic preference order from candidate list.
    preferred = candidate_baseline_label_paths(
        partition
    )

    for candidate in preferred:
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        "Baseline label discovery failed unexpectedly."
    )


def load_baseline_labels(
    partition: str,
) -> Dict[str, Any]:
    path = discover_baseline_label_path(
        partition
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST baseline-label path reached."
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
            f"{path}: expected {expected:,} labels, got {labels.shape}."
        )

    unique = np.unique(
        labels
    )

    if unique.size < 2:
        raise RuntimeError(
            f"{path}: baseline clustering has fewer than two clusters."
        )

    return {
        "path": path,
        "sha256": sha256_file(
            path
        ),
        "labels": labels,
        "unique_labels": unique.tolist(),
        "k": int(
            unique.size
        ),
    }


def transfer_manifest_path(
    transfer_cluster_root: Path,
    seed: int,
) -> Path:
    return (
        transfer_cluster_root
        / f"seed{seed}"
        / "manifest.json"
    )


def load_transfer_bout_ids(
    transfer_cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    manifest_path = transfer_manifest_path(
        transfer_cluster_root,
        seed,
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

    source_rel = manifest.get(
        key
    )

    if not source_rel:
        raise RuntimeError(
            f"{manifest_path}: missing {key}."
        )

    source_path = (
        REPO_ROOT
        / source_rel
    ).resolve()

    if "test" in source_path.name.lower():
        raise RuntimeError(
            "Protected TEST transfer-embedding source reached."
        )

    with np.load(
        source_path,
        allow_pickle=False,
    ) as npz:
        bout_id = np.asarray(
            npz["bout_id"]
        ).astype(str)

    return bout_id


def transfer_label_path(
    transfer_cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> Path:
    path = (
        transfer_cluster_root
        / f"seed{seed}"
        / f"{partition}_labels.npy"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST SSL-label path reached."
        )

    return path


def load_transfer_labels(
    transfer_cluster_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    path = transfer_label_path(
        transfer_cluster_root,
        seed=seed,
        partition=partition,
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
            f"{path}: expected {expected:,} labels, got {labels.shape}."
        )

    unique = np.unique(
        labels
    )

    if not np.array_equal(
        unique,
        np.arange(
            SSL_K
        ),
    ):
        raise RuntimeError(
            f"{path}: expected SSL labels 0..{SSL_K - 1}, got {unique}."
        )

    return labels


def align_rows(
    *,
    feature_ids: np.ndarray,
    X: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    if np.array_equal(
        feature_ids,
        cluster_ids,
    ):
        return X

    lookup = {
        bout_id: i
        for i, bout_id
        in enumerate(
            feature_ids
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
            f"Bout alignment failed; missing {missing[0]!r}."
        )

    idx = np.asarray(
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
            idx
        )
    ) != len(
        idx
    ):
        raise RuntimeError(
            "Bout alignment reused feature rows."
        )

    return X[
        idx
    ]


def entropy_from_labels(
    labels: np.ndarray,
) -> float:
    values, counts = np.unique(
        labels,
        return_counts=True,
    )

    p = counts.astype(
        np.float64
    )

    p /= p.sum()

    return float(
        -np.sum(
            p
            * np.log(
                p
            )
        )
    )


def conditional_entropy(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    """Compute H(Y|X) in natural-log units."""
    total = len(
        y
    )

    result = 0.0

    for x_value in np.unique(
        x
    ):
        mask = (
            x
            == x_value
        )

        weight = float(
            mask.sum()
            / total
        )

        result += (
            weight
            * entropy_from_labels(
                y[
                    mask
                ]
            )
        )

    return float(
        result
    )


def clustering_comparison(
    baseline_labels: np.ndarray,
    ssl_labels: np.ndarray,
) -> Dict[str, Any]:
    h_ssl = entropy_from_labels(
        ssl_labels
    )

    h_baseline = entropy_from_labels(
        baseline_labels
    )

    h_ssl_given_baseline = conditional_entropy(
        ssl_labels,
        baseline_labels,
    )

    h_baseline_given_ssl = conditional_entropy(
        baseline_labels,
        ssl_labels,
    )

    return {
        "adjusted_rand_index": float(
            adjusted_rand_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "entropy_ssl": float(
            h_ssl
        ),
        "entropy_baseline": float(
            h_baseline
        ),
        "conditional_entropy_ssl_given_baseline": float(
            h_ssl_given_baseline
        ),
        "conditional_entropy_baseline_given_ssl": float(
            h_baseline_given_ssl
        ),
        "normalized_conditional_entropy_ssl_given_baseline": float(
            h_ssl_given_baseline
            / h_ssl
            if h_ssl > 0
            else 0.0
        ),
        "normalized_conditional_entropy_baseline_given_ssl": float(
            h_baseline_given_ssl
            / h_baseline
            if h_baseline > 0
            else 0.0
        ),
    }


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Dict[str, Any]:
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=LINEAR_RANDOM_SEED,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    train_pred = pipeline.predict(
        X_train
    )

    validation_pred = pipeline.predict(
        X_validation
    )

    return {
        "model": (
            "median_imputer + StandardScaler + "
            "multinomial_logistic_regression"
        ),
        "fit_partition": "train",
        "feature_count": int(
            X_train.shape[
                1
            ]
        ),
        "train": {
            "accuracy": float(
                accuracy_score(
                    y_train,
                    train_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_train,
                    train_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_train,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "validation": {
            "accuracy": float(
                accuracy_score(
                    y_validation,
                    validation_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_validation,
                    validation_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_validation,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "chance_balanced_accuracy": float(
            1.0
            / SSL_K
        ),
    }


def fit_nonlinear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Dict[str, Any]:
    # Median imputation is fit on TRAIN only.
    imputer = SimpleImputer(
        strategy="median",
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_validation_imp = imputer.transform(
        X_validation
    )

    classifier = HistGradientBoostingClassifier(
        **NONLINEAR_PARAMS
    )

    classifier.fit(
        X_train_imp,
        y_train,
    )

    train_pred = classifier.predict(
        X_train_imp
    )

    validation_pred = classifier.predict(
        X_validation_imp
    )

    return {
        "model": (
            "median_imputer + HistGradientBoostingClassifier"
        ),
        "parameters": dict(
            NONLINEAR_PARAMS
        ),
        "fit_partition": "train",
        "feature_count": int(
            X_train.shape[
                1
            ]
        ),
        "train": {
            "accuracy": float(
                accuracy_score(
                    y_train,
                    train_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_train,
                    train_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_train,
                    train_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "validation": {
            "accuracy": float(
                accuracy_score(
                    y_validation,
                    validation_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_validation,
                    validation_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_validation,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "chance_balanced_accuracy": float(
            1.0
            / SSL_K
        ),
    }


def summarize(
    values: Sequence[
        float
    ],
) -> Dict[str, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
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
        "min": float(
            np.min(
                x
            )
        ),
        "max": float(
            np.max(
                x
            )
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
        / "DS006_BASELINE_VS_TRANSFER_SSL_SHA256SUMS"
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

    transfer_cluster_root = (
        args.transfer_cluster_root
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
        transfer_cluster_root,
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

    feature_manifest = load_feature_manifest()

    baseline_features = {
        partition: load_baseline_features(
            baseline_root,
            partition,
        )
        for partition
        in PARTITIONS
    }

    baseline_labels = {
        partition: load_baseline_labels(
            partition
        )
        for partition
        in PARTITIONS
    }

    if baseline_labels["train"]["k"] != baseline_labels["validation"]["k"]:
        raise RuntimeError(
            "Frozen baseline TRAIN and VALIDATION label sets imply "
            "different cluster counts."
        )

    baseline_k = baseline_labels[
        "train"
    ][
        "k"
    ]

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 BASELINE vs TRANSFER SSL"
    )
    print(
        "=" * 80
    )
    print(
        f"Baseline cluster k: {baseline_k}"
    )
    print(
        f"SSL cluster k:      {SSL_K}"
    )
    print(
        "Features:           18 frozen handcrafted features"
    )
    print(
        "Linear probe fit:   TRAIN only"
    )
    print(
        "Nonlinear probe:    frozen HistGradientBoosting sensitivity model"
    )
    print(
        "Model selection:    NONE"
    )
    print(
        "TEST partition:     PROTECTED / NOT LOADED"
    )
    print()
    print(
        f"Baseline TRAIN labels:      {baseline_labels['train']['path']}"
    )
    print(
        f"Baseline VALIDATION labels: {baseline_labels['validation']['path']}"
    )
    print()

    results: Dict[
        int,
        Dict[
            str,
            Any,
        ],
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
            f"SSL SEED {seed}"
        )
        print(
            "=" * 80
        )

        ssl_train = load_transfer_labels(
            transfer_cluster_root,
            seed=seed,
            partition="train",
        )

        ssl_validation = load_transfer_labels(
            transfer_cluster_root,
            seed=seed,
            partition="validation",
        )

        train_ids = load_transfer_bout_ids(
            transfer_cluster_root,
            seed=seed,
            partition="train",
        )

        validation_ids = load_transfer_bout_ids(
            transfer_cluster_root,
            seed=seed,
            partition="validation",
        )

        if reference_train_ids is None:
            reference_train_ids = train_ids
            reference_validation_ids = validation_ids
        else:
            if not np.array_equal(
                reference_train_ids,
                train_ids,
            ):
                raise RuntimeError(
                    f"TRAIN bout ordering differs at seed {seed}."
                )

            if not np.array_equal(
                reference_validation_ids,
                validation_ids,
            ):
                raise RuntimeError(
                    f"VALIDATION bout ordering differs at seed {seed}."
                )

        X_train = align_rows(
            feature_ids=baseline_features[
                "train"
            ][
                "bout_id"
            ],
            X=baseline_features[
                "train"
            ][
                "X"
            ],
            cluster_ids=train_ids,
        )

        X_validation = align_rows(
            feature_ids=baseline_features[
                "validation"
            ][
                "bout_id"
            ],
            X=baseline_features[
                "validation"
            ][
                "X"
            ],
            cluster_ids=validation_ids,
        )

        # Baseline labels are assumed row-aligned with the frozen baseline NPZ.
        baseline_train = baseline_labels[
            "train"
        ][
            "labels"
        ]

        baseline_validation = baseline_labels[
            "validation"
        ][
            "labels"
        ]

        if not np.array_equal(
            baseline_features[
                "train"
            ][
                "bout_id"
            ],
            train_ids,
        ):
            # If the baseline feature rows were reordered to match SSL, labels
            # need the same reordering.
            lookup = {
                bout_id: i
                for i, bout_id
                in enumerate(
                    baseline_features[
                        "train"
                    ][
                        "bout_id"
                    ]
                )
            }

            idx = np.asarray(
                [
                    lookup[
                        bout_id
                    ]
                    for bout_id
                    in train_ids
                ],
                dtype=np.int64,
            )

            baseline_train = baseline_train[
                idx
            ]

        if not np.array_equal(
            baseline_features[
                "validation"
            ][
                "bout_id"
            ],
            validation_ids,
        ):
            lookup = {
                bout_id: i
                for i, bout_id
                in enumerate(
                    baseline_features[
                        "validation"
                    ][
                        "bout_id"
                    ]
                )
            }

            idx = np.asarray(
                [
                    lookup[
                        bout_id
                    ]
                    for bout_id
                    in validation_ids
                ],
                dtype=np.int64,
            )

            baseline_validation = baseline_validation[
                idx
            ]

        train_cluster_metrics = clustering_comparison(
            baseline_train,
            ssl_train,
        )

        validation_cluster_metrics = clustering_comparison(
            baseline_validation,
            ssl_validation,
        )

        linear_probe = fit_linear_probe(
            X_train,
            ssl_train,
            X_validation,
            ssl_validation,
        )

        nonlinear_probe = fit_nonlinear_probe(
            X_train,
            ssl_train,
            X_validation,
            ssl_validation,
        )

        result = {
            "dataset_id": "DS-006",
            "analysis": (
                "baseline_vs_transfer_ssl"
            ),
            "ssl_encoder_seed": int(
                seed
            ),
            "baseline_cluster_k": int(
                baseline_k
            ),
            "ssl_cluster_k": int(
                SSL_K
            ),
            "feature_count": int(
                len(
                    EXPECTED_FEATURE_NAMES
                )
            ),
            "feature_manifest": str(
                FEATURE_MANIFEST.relative_to(
                    REPO_ROOT
                )
            ),
            "feature_manifest_sha256": sha256_file(
                FEATURE_MANIFEST
            ),
            "baseline_train_labels_source": str(
                baseline_labels[
                    "train"
                ][
                    "path"
                ].relative_to(
                    REPO_ROOT
                )
            )
            if is_relative_to(
                baseline_labels[
                    "train"
                ][
                    "path"
                ],
                REPO_ROOT,
            )
            else str(
                baseline_labels[
                    "train"
                ][
                    "path"
                ]
            ),
            "baseline_validation_labels_source": str(
                baseline_labels[
                    "validation"
                ][
                    "path"
                ].relative_to(
                    REPO_ROOT
                )
            )
            if is_relative_to(
                baseline_labels[
                    "validation"
                ][
                    "path"
                ],
                REPO_ROOT,
            )
            else str(
                baseline_labels[
                    "validation"
                ][
                    "path"
                ]
            ),
            "train_cluster_comparison": train_cluster_metrics,
            "validation_cluster_comparison": validation_cluster_metrics,
            "linear_18_feature_probe": linear_probe,
            "nonlinear_18_feature_probe": nonlinear_probe,
            "bout_id_alignment_verified": True,
            "probe_hyperparameter_selection_performed": False,
            "cluster_refitting_performed": False,
            "test_partition_used": False,
        }

        results[
            seed
        ] = result

        out_path = (
            output_root
            / f"seed{seed}.json"
        )

        atomic_json(
            out_path,
            result,
        )

        written.append(
            out_path
        )

        v = validation_cluster_metrics

        print(
            f"VAL ARI:                              "
            f"{v['adjusted_rand_index']:.6f}"
        )
        print(
            f"VAL NMI:                              "
            f"{v['normalized_mutual_information']:.6f}"
        )
        print(
            f"VAL AMI:                              "
            f"{v['adjusted_mutual_information']:.6f}"
        )
        print(
            "VAL H(SSL|baseline)/H(SSL):          "
            f"{v['normalized_conditional_entropy_ssl_given_baseline']:.6f}"
        )
        print(
            "VAL H(baseline|SSL)/H(baseline):     "
            f"{v['normalized_conditional_entropy_baseline_given_ssl']:.6f}"
        )
        print(
            "Linear probe VAL balanced accuracy:  "
            f"{linear_probe['validation']['balanced_accuracy']:.6f}"
        )
        print(
            "Linear probe VAL macro F1:           "
            f"{linear_probe['validation']['macro_f1']:.6f}"
        )
        print(
            "Nonlinear probe VAL balanced acc:    "
            f"{nonlinear_probe['validation']['balanced_accuracy']:.6f}"
        )
        print(
            "Nonlinear probe VAL macro F1:        "
            f"{nonlinear_probe['validation']['macro_f1']:.6f}"
        )
        print(
            "TEST partition used:                 NO"
        )
        print()

    metric_extractors = {
        "validation_ari": lambda r: r[
            "validation_cluster_comparison"
        ][
            "adjusted_rand_index"
        ],

        "validation_nmi": lambda r: r[
            "validation_cluster_comparison"
        ][
            "normalized_mutual_information"
        ],

        "validation_ami": lambda r: r[
            "validation_cluster_comparison"
        ][
            "adjusted_mutual_information"
        ],

        "validation_h_ssl_given_baseline_norm": lambda r: r[
            "validation_cluster_comparison"
        ][
            "normalized_conditional_entropy_ssl_given_baseline"
        ],

        "validation_h_baseline_given_ssl_norm": lambda r: r[
            "validation_cluster_comparison"
        ][
            "normalized_conditional_entropy_baseline_given_ssl"
        ],

        "linear_validation_balanced_accuracy": lambda r: r[
            "linear_18_feature_probe"
        ][
            "validation"
        ][
            "balanced_accuracy"
        ],

        "linear_validation_macro_f1": lambda r: r[
            "linear_18_feature_probe"
        ][
            "validation"
        ][
            "macro_f1"
        ],

        "linear_validation_accuracy": lambda r: r[
            "linear_18_feature_probe"
        ][
            "validation"
        ][
            "accuracy"
        ],

        "nonlinear_validation_balanced_accuracy": lambda r: r[
            "nonlinear_18_feature_probe"
        ][
            "validation"
        ][
            "balanced_accuracy"
        ],

        "nonlinear_validation_macro_f1": lambda r: r[
            "nonlinear_18_feature_probe"
        ][
            "validation"
        ][
            "macro_f1"
        ],

        "nonlinear_validation_accuracy": lambda r: r[
            "nonlinear_18_feature_probe"
        ][
            "validation"
        ][
            "accuracy"
        ],
    }

    aggregate: Dict[
        str,
        Any,
    ] = {}

    for metric_name, extractor in metric_extractors.items():
        values = [
            float(
                extractor(
                    results[
                        seed
                    ]
                )
            )
            for seed
            in SEEDS
        ]

        aggregate[
            metric_name
        ] = summarize(
            values
        )

    summary = {
        "dataset_id": "DS-006",
        "analysis": (
            "baseline_vs_transfer_ssl"
        ),
        "seeds": list(
            SEEDS
        ),
        "baseline_cluster_k": int(
            baseline_k
        ),
        "ssl_cluster_k": int(
            SSL_K
        ),
        "feature_count": int(
            len(
                EXPECTED_FEATURE_NAMES
            )
        ),
        "chance_balanced_accuracy": float(
            1.0
            / SSL_K
        ),
        "aggregate": aggregate,
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
            "low_cluster_agreement_does_not_imply_features_lack_information": True,
            "nonlinear_probe_tests_recoverability_from_all_18_features": True,
            "high_nonlinear_probe_accuracy_supports_reorganization_interpretation": True,
            "probe_hyperparameter_selection_performed": False,
            "test_partition_used": False,
        },
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
        "DS-006 BASELINE vs TRANSFER SSL SUMMARY"
    )
    print(
        "=" * 80
    )

    a = aggregate

    print(
        "Mean VAL ARI:                              "
        f"{a['validation_ari']['mean']:.6f}"
    )
    print(
        "Mean VAL NMI:                              "
        f"{a['validation_nmi']['mean']:.6f}"
    )
    print(
        "Mean VAL AMI:                              "
        f"{a['validation_ami']['mean']:.6f}"
    )
    print(
        "Mean VAL H(SSL|baseline)/H(SSL):          "
        f"{a['validation_h_ssl_given_baseline_norm']['mean']:.6f}"
    )
    print(
        "Mean VAL H(baseline|SSL)/H(baseline):     "
        f"{a['validation_h_baseline_given_ssl_norm']['mean']:.6f}"
    )
    print(
        "Mean linear VAL balanced accuracy:        "
        f"{a['linear_validation_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean linear VAL macro F1:                 "
        f"{a['linear_validation_macro_f1']['mean']:.6f}"
    )
    print(
        "Mean nonlinear VAL balanced accuracy:     "
        f"{a['nonlinear_validation_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean nonlinear VAL macro F1:              "
        f"{a['nonlinear_validation_macro_f1']['mean']:.6f}"
    )
    print(
        "Chance balanced accuracy:                 "
        f"{1.0 / SSL_K:.6f}"
    )
    print(
        "TEST partition used:                      NO"
    )
    print(
        f"Summary:   {summary_path}"
    )
    print(
        f"Checksums: {checksum_path}"
    )


if __name__ == "__main__":
    main()
