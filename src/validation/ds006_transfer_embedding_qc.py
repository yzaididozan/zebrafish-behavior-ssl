#!/usr/bin/env python3
"""QC for DS-006 transfer embeddings produced by frozen DS-005 encoders.

Purpose
-------
Determine whether the transferred DS-005 SSL representation forms a healthy,
non-collapsed embedding space on DS-006 before any clustering or biological
interpretation is attempted.

This script evaluates TRAIN and VALIDATION transfer embeddings for the frozen
encoder seeds:

    11, 23, 37, 51, 79

It never loads DS-006 TEST.

Checks
------
For each seed and partition:
- artifact existence
- row count and 64-D shape
- bout-ID and row-index alignment
- finite values
- per-dimension mean/std/min/max
- near-zero-variance dimensions
- total embedding variance
- matrix rank
- covariance eigenvalue diagnostics
- covariance condition number
- embedding norm distribution
- extreme norm outlier rate
- PCA components required for 95% variance

TRAIN vs VALIDATION:
- mean-vector shift
- standardized mean shift
- per-dimension variance ratio
- median variance ratio
- embedding norm distribution shift
- PCA-space mean shift using TRAIN-fitted PCA

Across seeds:
- pairwise similarity of broad representation geometry using
  pairwise-distance Spearman correlation on the same deterministic
  subset of DS-006 bouts
- summary of PCA dimensionality and collapse indicators

Interpretation
--------------
This is a representation-health check, not evidence of biological validity.
Passing QC means the frozen DS-005 encoder produces numerically structured,
non-collapsed DS-006 embeddings suitable for the next replication stage.

Expected input layout
---------------------
data/processed/DS-006/transfer_embeddings/
    seed11/
        train_embeddings.npz
        validation_embeddings.npz
        ...
    seed23/
    seed37/
    seed51/
    seed79/

Expected NPZ arrays:
    embeddings : (N, 64)
    row_index  : (N,)
    bout_id    : (N,)

Expected full counts:
    TRAIN       118,100
    VALIDATION   18,835

Outputs
-------
data/processed/DS-006/transfer_embedding_qc/
    seed11/
        train.json
        validation.json
        train_vs_validation.json
    seed23/
        ...
    cross_seed_summary.json
    summary.json
    DS006_TRANSFER_EMBEDDING_QC_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ds006_transfer_embedding_qc.py

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ds006_transfer_embedding_qc.py \
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "transfer_embeddings"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-006"
    / "transfer_embedding_qc"
)

SEEDS = (11, 23, 37, 51, 79)
PARTITIONS = ("train", "validation")

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

EXPECTED_DIM = 64

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10
COV_EIGENVALUE_FLOOR = 1e-12
NORM_OUTLIER_Z = 6.0
PCA_VARIANCE_TARGET = 0.95
GEOMETRY_SAMPLE_SIZE = 2_000
GEOMETRY_PAIR_COUNT = 20_000
GEOMETRY_RANDOM_SEED = 20260824


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QC DS-006 transfer embeddings before clustering. "
            "TRAIN + VALIDATION only; TEST prohibited."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--near-zero-variance-threshold",
        type=float,
        default=NEAR_ZERO_VARIANCE_THRESHOLD,
    )

    parser.add_argument(
        "--geometry-sample-size",
        type=int,
        default=GEOMETRY_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--geometry-pair-count",
        type=int,
        default=GEOMETRY_PAIR_COUNT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
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


def ensure_safe_paths(input_root: Path, output_root: Path) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    expected_input = DEFAULT_INPUT_ROOT.resolve()

    if input_root != expected_input:
        raise RuntimeError(
            "For replication safety, input root must resolve exactly to "
            f"{expected_input}; got {input_root}"
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

    try:
        output_root.relative_to(ds006_processed)
    except ValueError as exc:
        raise RuntimeError(
            "QC outputs must remain under data/processed/DS-006."
        ) from exc

    try:
        output_root.relative_to(ds005_processed)
        raise RuntimeError(
            "Refusing to write QC outputs inside DS-005."
        )
    except ValueError:
        pass


def artifact_path(
    input_root: Path,
    seed: int,
    partition: str,
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            f"Protected/invalid partition requested: {partition}"
        )

    path = (
        input_root
        / f"seed{seed}"
        / f"{partition}_embeddings.npz"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            f"Protected TEST path reached unexpectedly: {path}"
        )

    return path


def load_embeddings(
    input_root: Path,
    *,
    seed: int,
    partition: str,
) -> Dict[str, Any]:
    path = artifact_path(
        input_root,
        seed,
        partition,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        required = {
            "embeddings",
            "row_index",
            "bout_id",
        }

        missing = required - set(npz.files)

        if missing:
            raise RuntimeError(
                f"{path} missing arrays: {sorted(missing)}"
            )

        embeddings = np.asarray(
            npz["embeddings"],
            dtype=np.float64,
        )

        row_index = np.asarray(
            npz["row_index"],
            dtype=np.int64,
        )

        bout_id = np.asarray(
            npz["bout_id"],
        ).astype(str)

    expected_rows = EXPECTED_ROWS[
        partition
    ]

    if embeddings.shape != (
        expected_rows,
        EXPECTED_DIM,
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"({expected_rows}, {EXPECTED_DIM}), "
            f"got {embeddings.shape}."
        )

    if row_index.shape != (
        expected_rows,
    ):
        raise RuntimeError(
            f"{path}: row_index shape mismatch."
        )

    if bout_id.shape != (
        expected_rows,
    ):
        raise RuntimeError(
            f"{path}: bout_id shape mismatch."
        )

    if not np.array_equal(
        row_index,
        np.arange(
            expected_rows,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"{path}: row_index is not exact 0..N-1."
        )

    if len(np.unique(bout_id)) != len(
        bout_id
    ):
        raise RuntimeError(
            f"{path}: duplicate bout_id values."
        )

    if np.any(
        np.char.str_len(
            bout_id
        )
        == 0
    ):
        raise RuntimeError(
            f"{path}: empty bout_id detected."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            f"{path}: NaN/Inf embeddings detected."
        )

    return {
        "path": path,
        "sha256": sha256_file(path),
        "embeddings": embeddings,
        "row_index": row_index,
        "bout_id": bout_id,
    }


def robust_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": float(
            np.min(values)
        ),
        "p01": float(
            np.percentile(
                values,
                1,
            )
        ),
        "p05": float(
            np.percentile(
                values,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "median": float(
            np.median(values)
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),
        "max": float(
            np.max(values)
        ),
    }


def pca_diagnostics(
    embeddings: np.ndarray,
) -> Dict[str, Any]:
    pca = PCA(
        n_components=min(
            embeddings.shape
        ),
        svd_solver="full",
    )

    pca.fit(
        embeddings
    )

    explained = np.asarray(
        pca.explained_variance_ratio_,
        dtype=np.float64,
    )

    cumulative = np.cumsum(
        explained
    )

    components_95 = int(
        np.searchsorted(
            cumulative,
            PCA_VARIANCE_TARGET,
        )
        + 1
    )

    components_90 = int(
        np.searchsorted(
            cumulative,
            0.90,
        )
        + 1
    )

    components_99 = int(
        np.searchsorted(
            cumulative,
            0.99,
        )
        + 1
    )

    return {
        "components_for_90pct_variance": (
            components_90
        ),
        "components_for_95pct_variance": (
            components_95
        ),
        "components_for_99pct_variance": (
            components_99
        ),
        "explained_variance_ratio_first10": [
            float(x)
            for x in explained[:10]
        ],
        "cumulative_variance_first10": [
            float(x)
            for x in cumulative[:10]
        ],
        "first_component_variance_fraction": float(
            explained[0]
        ),
    }


def covariance_diagnostics(
    embeddings: np.ndarray,
) -> Dict[str, Any]:
    cov = np.cov(
        embeddings,
        rowvar=False,
        ddof=0,
    )

    eigvals = np.linalg.eigvalsh(
        cov
    )

    eigvals = np.asarray(
        eigvals,
        dtype=np.float64,
    )

    eigvals_sorted = np.sort(
        eigvals
    )[::-1]

    positive = eigvals_sorted[
        eigvals_sorted
        > COV_EIGENVALUE_FLOOR
    ]

    if positive.size:
        condition_number = float(
            positive[0]
            / positive[-1]
        )
    else:
        condition_number = float(
            "inf"
        )

    return {
        "rank_numeric": int(
            np.linalg.matrix_rank(
                embeddings
            )
        ),
        "covariance_rank": int(
            np.linalg.matrix_rank(
                cov
            )
        ),
        "positive_cov_eigenvalues": int(
            positive.size
        ),
        "largest_cov_eigenvalue": float(
            eigvals_sorted[0]
        ),
        "smallest_cov_eigenvalue": float(
            eigvals_sorted[-1]
        ),
        "smallest_positive_cov_eigenvalue": (
            float(
                positive[-1]
            )
            if positive.size
            else None
        ),
        "covariance_condition_number": (
            condition_number
            if math.isfinite(
                condition_number
            )
            else None
        ),
    }


def norm_diagnostics(
    embeddings: np.ndarray,
) -> Dict[str, Any]:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    mean = float(
        np.mean(norms)
    )

    std = float(
        np.std(norms)
    )

    if std > 0:
        z = np.abs(
            (
                norms
                - mean
            )
            / std
        )

        outlier_fraction = float(
            np.mean(
                z > NORM_OUTLIER_Z
            )
        )

        outlier_count = int(
            np.sum(
                z > NORM_OUTLIER_Z
            )
        )
    else:
        outlier_fraction = 0.0
        outlier_count = 0

    return {
        "distribution": robust_summary(
            norms
        ),
        "zero_norm_count": int(
            np.sum(
                norms == 0
            )
        ),
        "extreme_norm_outlier_z_threshold": float(
            NORM_OUTLIER_Z
        ),
        "extreme_norm_outlier_count": (
            outlier_count
        ),
        "extreme_norm_outlier_fraction": (
            outlier_fraction
        ),
    }


def partition_qc(
    embeddings: np.ndarray,
    *,
    near_zero_threshold: float,
) -> Dict[str, Any]:
    means = np.mean(
        embeddings,
        axis=0,
    )

    stds = np.std(
        embeddings,
        axis=0,
    )

    variances = stds ** 2

    mins = np.min(
        embeddings,
        axis=0,
    )

    maxs = np.max(
        embeddings,
        axis=0,
    )

    near_zero = np.flatnonzero(
        variances
        <= near_zero_threshold
    )

    total_variance = float(
        np.sum(
            variances
        )
    )

    return {
        "rows": int(
            embeddings.shape[0]
        ),
        "embedding_dim": int(
            embeddings.shape[1]
        ),
        "all_finite": bool(
            np.isfinite(
                embeddings
            ).all()
        ),
        "total_variance": (
            total_variance
        ),
        "mean_dimension_variance": float(
            np.mean(
                variances
            )
        ),
        "median_dimension_variance": float(
            np.median(
                variances
            )
        ),
        "min_dimension_variance": float(
            np.min(
                variances
            )
        ),
        "max_dimension_variance": float(
            np.max(
                variances
            )
        ),
        "near_zero_variance_threshold": float(
            near_zero_threshold
        ),
        "near_zero_variance_dimension_count": int(
            near_zero.size
        ),
        "near_zero_variance_dimensions": [
            int(x)
            for x in near_zero
        ],
        "per_dimension": {
            "mean": [
                float(x)
                for x in means
            ],
            "std": [
                float(x)
                for x in stds
            ],
            "variance": [
                float(x)
                for x in variances
            ],
            "min": [
                float(x)
                for x in mins
            ],
            "max": [
                float(x)
                for x in maxs
            ],
        },
        "norms": norm_diagnostics(
            embeddings
        ),
        "covariance": covariance_diagnostics(
            embeddings
        ),
        "pca": pca_diagnostics(
            embeddings
        ),
    }


def train_validation_qc(
    train: np.ndarray,
    validation: np.ndarray,
) -> Dict[str, Any]:
    train_mean = np.mean(
        train,
        axis=0,
    )

    val_mean = np.mean(
        validation,
        axis=0,
    )

    train_std = np.std(
        train,
        axis=0,
    )

    train_var = train_std ** 2

    val_var = np.var(
        validation,
        axis=0,
    )

    eps = 1e-12

    standardized_shift = (
        val_mean
        - train_mean
    ) / np.maximum(
        train_std,
        eps,
    )

    variance_ratio = (
        val_var
        / np.maximum(
            train_var,
            eps,
        )
    )

    train_norm = np.linalg.norm(
        train,
        axis=1,
    )

    val_norm = np.linalg.norm(
        validation,
        axis=1,
    )

    norm_mean_shift_z = float(
        (
            np.mean(
                val_norm
            )
            - np.mean(
                train_norm
            )
        )
        / max(
            float(
                np.std(
                    train_norm
                )
            ),
            eps,
        )
    )

    # TRAIN-fitted scaling + PCA, then compare partition means in the same
    # latent coordinate system.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train
    )

    validation_scaled = scaler.transform(
        validation
    )

    pca = PCA(
        n_components=PCA_VARIANCE_TARGET,
        svd_solver="full",
    )

    train_pca = pca.fit_transform(
        train_scaled
    )

    validation_pca = pca.transform(
        validation_scaled
    )

    pca_mean_shift = np.linalg.norm(
        np.mean(
            validation_pca,
            axis=0,
        )
        - np.mean(
            train_pca,
            axis=0,
        )
    )

    return {
        "mean_vector_l2_shift": float(
            np.linalg.norm(
                val_mean
                - train_mean
            )
        ),
        "standardized_mean_shift": {
            "l2": float(
                np.linalg.norm(
                    standardized_shift
                )
            ),
            "mean_abs": float(
                np.mean(
                    np.abs(
                        standardized_shift
                    )
                )
            ),
            "max_abs": float(
                np.max(
                    np.abs(
                        standardized_shift
                    )
                )
            ),
            "per_dimension": [
                float(x)
                for x in standardized_shift
            ],
        },
        "variance_ratio_validation_over_train": {
            "mean": float(
                np.mean(
                    variance_ratio
                )
            ),
            "median": float(
                np.median(
                    variance_ratio
                )
            ),
            "min": float(
                np.min(
                    variance_ratio
                )
            ),
            "max": float(
                np.max(
                    variance_ratio
                )
            ),
            "per_dimension": [
                float(x)
                for x in variance_ratio
            ],
        },
        "embedding_norm_mean_shift_z": (
            norm_mean_shift_z
        ),
        "train_fitted_pca": {
            "components_retaining_95pct_variance": int(
                pca.n_components_
            ),
            "validation_mean_shift_l2_in_pca_space": float(
                pca_mean_shift
            ),
        },
    }


def deterministic_geometry_indices(
    n_rows: int,
    sample_size: int,
) -> np.ndarray:
    if sample_size < 2:
        raise ValueError(
            "--geometry-sample-size must be >= 2."
        )

    sample_size = min(
        n_rows,
        sample_size,
    )

    rng = np.random.default_rng(
        GEOMETRY_RANDOM_SEED
    )

    return np.sort(
        rng.choice(
            n_rows,
            size=sample_size,
            replace=False,
        )
    )


def deterministic_pair_indices(
    sample_size: int,
    pair_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if pair_count < 1:
        raise ValueError(
            "--geometry-pair-count must be >= 1."
        )

    rng = np.random.default_rng(
        GEOMETRY_RANDOM_SEED
        + 1
    )

    first = rng.integers(
        0,
        sample_size,
        size=pair_count,
    )

    second = rng.integers(
        0,
        sample_size,
        size=pair_count,
    )

    same = first == second

    while np.any(
        same
    ):
        second[
            same
        ] = rng.integers(
            0,
            sample_size,
            size=int(
                np.sum(
                    same
                )
            ),
        )

        same = (
            first
            == second
        )

    return first, second


def sampled_pair_distances(
    embeddings: np.ndarray,
    sample_indices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    sample = embeddings[
        sample_indices
    ]

    diff = (
        sample[
            pair_i
        ]
        - sample[
            pair_j
        ]
    )

    return np.linalg.norm(
        diff,
        axis=1,
    )


def pairwise_geometry_similarity(
    embeddings_by_seed: Mapping[
        int,
        np.ndarray,
    ],
    *,
    sample_size: int,
    pair_count: int,
) -> Dict[str, Any]:
    seeds = sorted(
        embeddings_by_seed
    )

    row_counts = {
        int(
            x.shape[0]
        )
        for x
        in embeddings_by_seed.values()
    }

    if len(
        row_counts
    ) != 1:
        raise RuntimeError(
            "Cross-seed row count mismatch."
        )

    n_rows = next(
        iter(
            row_counts
        )
    )

    sample_indices = (
        deterministic_geometry_indices(
            n_rows,
            sample_size,
        )
    )

    pair_i, pair_j = (
        deterministic_pair_indices(
            len(
                sample_indices
            ),
            pair_count,
        )
    )

    distances = {
        seed: sampled_pair_distances(
            embeddings_by_seed[
                seed
            ],
            sample_indices,
            pair_i,
            pair_j,
        )
        for seed
        in seeds
    }

    pairs: List[
        Dict[str, Any]
    ] = []

    for i, seed_a in enumerate(
        seeds
    ):
        for seed_b in seeds[
            i + 1:
        ]:
            result = spearmanr(
                distances[
                    seed_a
                ],
                distances[
                    seed_b
                ],
            )

            rho = float(
                result.statistic
            )

            pvalue = float(
                result.pvalue
            )

            pairs.append(
                {
                    "seed_a": int(
                        seed_a
                    ),
                    "seed_b": int(
                        seed_b
                    ),
                    "spearman_rho": (
                        rho
                        if math.isfinite(
                            rho
                        )
                        else None
                    ),
                    "pvalue": (
                        pvalue
                        if math.isfinite(
                            pvalue
                        )
                        else None
                    ),
                }
            )

    finite_rho = np.asarray(
        [
            x[
                "spearman_rho"
            ]
            for x
            in pairs
            if x[
                "spearman_rho"
            ]
            is not None
        ],
        dtype=np.float64,
    )

    return {
        "sample_rows": int(
            len(
                sample_indices
            )
        ),
        "sampled_pairs": int(
            pair_count
        ),
        "sampling_seed": int(
            GEOMETRY_RANDOM_SEED
        ),
        "metric": (
            "Spearman correlation of sampled pairwise Euclidean "
            "distance profiles"
        ),
        "mean_pairwise_rho": (
            float(
                np.mean(
                    finite_rho
                )
            )
            if finite_rho.size
            else None
        ),
        "median_pairwise_rho": (
            float(
                np.median(
                    finite_rho
                )
            )
            if finite_rho.size
            else None
        ),
        "min_pairwise_rho": (
            float(
                np.min(
                    finite_rho
                )
            )
            if finite_rho.size
            else None
        ),
        "max_pairwise_rho": (
            float(
                np.max(
                    finite_rho
                )
            )
            if finite_rho.size
            else None
        ),
        "pairs": pairs,
    }


def write_checksums(
    output_root: Path,
    files: Sequence[Path],
) -> Path:
    checksum_path = (
        output_root
        / "DS006_TRANSFER_EMBEDDING_QC_SHA256SUMS"
    )

    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.relative_to(output_root)}\n"
            for path
            in sorted(
                files,
                key=lambda p: str(p),
            )
        ),
        encoding="utf-8",
    )

    return checksum_path


def main() -> None:
    args = parse_args()

    input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    ensure_safe_paths(
        input_root,
        output_root,
    )

    if (
        args.near_zero_variance_threshold
        < 0
    ):
        raise ValueError(
            "--near-zero-variance-threshold must be >= 0."
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

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 TRANSFER EMBEDDING QC"
    )
    print(
        "=" * 80
    )
    print(
        f"Input root:       {input_root}"
    )
    print(
        f"Seeds:            {list(SEEDS)}"
    )
    print(
        "Partitions:       TRAIN + VALIDATION"
    )
    print(
        "TEST partition:   PROTECTED / NOT LOADED"
    )
    print(
        f"Expected dim:     {EXPECTED_DIM}"
    )
    print()

    loaded: Dict[
        int,
        Dict[
            str,
            Dict[str, Any],
        ],
    ] = {}

    per_seed_results: Dict[
        int,
        Dict[str, Any],
    ] = {}

    written_files: List[
        Path
    ] = []

    for seed in SEEDS:
        loaded[
            seed
        ] = {}

        per_seed_results[
            seed
        ] = {}

        seed_dir = (
            output_root
            / f"seed{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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
            artifact = load_embeddings(
                input_root,
                seed=seed,
                partition=partition,
            )

            loaded[
                seed
            ][
                partition
            ] = artifact

            result = partition_qc(
                artifact[
                    "embeddings"
                ],
                near_zero_threshold=(
                    args.near_zero_variance_threshold
                ),
            )

            result[
                "source_path"
            ] = str(
                artifact[
                    "path"
                ].relative_to(
                    REPO_ROOT
                )
            )

            result[
                "source_sha256"
            ] = artifact[
                "sha256"
            ]

            result[
                "row_alignment_verified"
            ] = True

            result[
                "bout_id_uniqueness_verified"
            ] = True

            result[
                "test_partition_loaded"
            ] = False

            per_seed_results[
                seed
            ][
                partition
            ] = result

            output_path = (
                seed_dir
                / f"{partition}.json"
            )

            atomic_write_json(
                output_path,
                result,
            )

            written_files.append(
                output_path
            )

            print(
                f"{partition.upper():<11} "
                f"rows={result['rows']:,}  "
                f"total_var={result['total_variance']:.6f}  "
                f"near_zero_dims="
                f"{result['near_zero_variance_dimension_count']}  "
                f"rank={result['covariance']['rank_numeric']}  "
                f"PCA95={result['pca']['components_for_95pct_variance']}"
            )

        train = loaded[
            seed
        ][
            "train"
        ][
            "embeddings"
        ]

        validation = loaded[
            seed
        ][
            "validation"
        ][
            "embeddings"
        ]

        tv = train_validation_qc(
            train,
            validation,
        )

        tv[
            "test_partition_loaded"
        ] = False

        tv_path = (
            seed_dir
            / "train_vs_validation.json"
        )

        atomic_write_json(
            tv_path,
            tv,
        )

        written_files.append(
            tv_path
        )

        per_seed_results[
            seed
        ][
            "train_vs_validation"
        ] = tv

        print(
            "TRAIN->VALIDATION  "
            f"std-mean-shift-L2="
            f"{tv['standardized_mean_shift']['l2']:.4f}  "
            f"median-var-ratio="
            f"{tv['variance_ratio_validation_over_train']['median']:.4f}  "
            f"norm-shift-z="
            f"{tv['embedding_norm_mean_shift_z']:.4f}"
        )
        print(
            "TEST partition used: NO"
        )
        print()

    # Cross-seed geometry on TRAIN and VALIDATION independently.
    cross_seed = {
        partition: pairwise_geometry_similarity(
            {
                seed: loaded[
                    seed
                ][
                    partition
                ][
                    "embeddings"
                ]
                for seed
                in SEEDS
            },
            sample_size=(
                args.geometry_sample_size
            ),
            pair_count=(
                args.geometry_pair_count
            ),
        )
        for partition
        in PARTITIONS
    }

    # Aggregate simple QC metrics across seeds.
    aggregate: Dict[
        str,
        Any,
    ] = {}

    for partition in PARTITIONS:
        aggregate[
            partition
        ] = {
            "near_zero_variance_dimension_count": {
                "values_by_seed": {
                    str(seed): int(
                        per_seed_results[
                            seed
                        ][
                            partition
                        ][
                            "near_zero_variance_dimension_count"
                        ]
                    )
                    for seed
                    in SEEDS
                },
                "max": int(
                    max(
                        per_seed_results[
                            seed
                        ][
                            partition
                        ][
                            "near_zero_variance_dimension_count"
                        ]
                        for seed
                        in SEEDS
                    )
                ),
            },
            "pca_components_for_95pct_variance": {
                "values_by_seed": {
                    str(seed): int(
                        per_seed_results[
                            seed
                        ][
                            partition
                        ][
                            "pca"
                        ][
                            "components_for_95pct_variance"
                        ]
                    )
                    for seed
                    in SEEDS
                },
                "mean": float(
                    np.mean(
                        [
                            per_seed_results[
                                seed
                            ][
                                partition
                            ][
                                "pca"
                            ][
                                "components_for_95pct_variance"
                            ]
                            for seed
                            in SEEDS
                        ]
                    )
                ),
            },
            "total_variance": {
                "values_by_seed": {
                    str(seed): float(
                        per_seed_results[
                            seed
                        ][
                            partition
                        ][
                            "total_variance"
                        ]
                    )
                    for seed
                    in SEEDS
                },
                "mean": float(
                    np.mean(
                        [
                            per_seed_results[
                                seed
                            ][
                                partition
                            ][
                                "total_variance"
                            ]
                            for seed
                            in SEEDS
                        ]
                    )
                ),
            },
        }

    cross_seed_path = (
        output_root
        / "cross_seed_summary.json"
    )

    atomic_write_json(
        cross_seed_path,
        {
            "dataset_id": "DS-006",
            "analysis": (
                "transfer_embedding_geometry_similarity"
            ),
            "source_encoder_dataset": "DS-005",
            "seeds": list(
                SEEDS
            ),
            "train": cross_seed[
                "train"
            ],
            "validation": cross_seed[
                "validation"
            ],
            "test_partition_loaded": False,
        },
    )

    written_files.append(
        cross_seed_path
    )

    max_near_zero_train = aggregate[
        "train"
    ][
        "near_zero_variance_dimension_count"
    ][
        "max"
    ]

    max_near_zero_val = aggregate[
        "validation"
    ][
        "near_zero_variance_dimension_count"
    ][
        "max"
    ]

    min_rank_train = min(
        per_seed_results[
            seed
        ][
            "train"
        ][
            "covariance"
        ][
            "rank_numeric"
        ]
        for seed
        in SEEDS
    )

    min_rank_val = min(
        per_seed_results[
            seed
        ][
            "validation"
        ][
            "covariance"
        ][
            "rank_numeric"
        ]
        for seed
        in SEEDS
    )

    healthy_noncollapsed = bool(
        max_near_zero_train == 0
        and max_near_zero_val == 0
        and min_rank_train == EXPECTED_DIM
        and min_rank_val == EXPECTED_DIM
    )

    summary = {
        "dataset_id": "DS-006",
        "analysis": "transfer_embedding_qc",
        "source_encoder_dataset": "DS-005",
        "seeds": list(
            SEEDS
        ),
        "partitions": list(
            PARTITIONS
        ),
        "expected_embedding_dim": int(
            EXPECTED_DIM
        ),
        "expected_rows": {
            key: int(
                value
            )
            for key, value
            in EXPECTED_ROWS.items()
        },
        "near_zero_variance_threshold": float(
            args.near_zero_variance_threshold
        ),
        "aggregate": aggregate,
        "cross_seed_geometry": cross_seed,
        "healthy_noncollapsed_by_strict_basic_gate": (
            healthy_noncollapsed
        ),
        "strict_basic_gate_definition": {
            "zero_near_zero_variance_dimensions_all_seeds": True,
            "full_numeric_rank_all_seeds": True,
            "applies_to_train_and_validation": True,
        },
        "interpretation_guardrails": {
            "passing_qc_not_biological_validation": True,
            "no_clustering_performed": True,
            "no_model_selection_performed": True,
            "no_test_access": True,
        },
        "test_partition_loaded": False,
    }

    atomic_write_json(
        summary_path,
        summary,
    )

    written_files.append(
        summary_path
    )

    checksum_path = write_checksums(
        output_root,
        written_files,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 TRANSFER EMBEDDING QC SUMMARY"
    )
    print(
        "=" * 80
    )
    print(
        "Strict non-collapse gate: "
        f"{'PASS' if healthy_noncollapsed else 'REVIEW'}"
    )
    print(
        f"Minimum TRAIN rank:       {min_rank_train}/{EXPECTED_DIM}"
    )
    print(
        f"Minimum VALIDATION rank:  {min_rank_val}/{EXPECTED_DIM}"
    )
    print(
        f"Max TRAIN near-zero dims: {max_near_zero_train}"
    )
    print(
        f"Max VAL near-zero dims:   {max_near_zero_val}"
    )
    print(
        "Mean cross-seed TRAIN geometry rho:      "
        f"{cross_seed['train']['mean_pairwise_rho']}"
    )
    print(
        "Mean cross-seed VALIDATION geometry rho: "
        f"{cross_seed['validation']['mean_pairwise_rho']}"
    )
    print(
        "TEST partition used: NO"
    )
    print(
        f"Summary:     {summary_path}"
    )
    print(
        f"Cross-seed:  {cross_seed_path}"
    )
    print(
        f"Checksums:   {checksum_path}"
    )


if __name__ == "__main__":
    main()
