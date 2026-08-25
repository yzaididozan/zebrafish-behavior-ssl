#!/usr/bin/env python3
"""Frozen DS-005 SSL clustering recipe applied to DS-006 transfer embeddings.

Scientific role
---------------
This is a method-replication clustering stage for DS-006 transfer embeddings.

It DOES NOT re-select:
- clustering method;
- number of clusters;
- PCA target;
- clustering seed.

The frozen DS-005-selected recipe is:

    representation      transferred DS-005 encoder embeddings
    scaling             StandardScaler fit on DS-006 TRAIN only
    PCA                 95% variance, fit on DS-006 TRAIN only
    clustering          KMeans
    k                   8
    random_state        20260822
    n_init              10

For every frozen SSL encoder seed (11, 23, 37, 51, 79), the script:

1. loads DS-006 TRAIN/VALIDATION transfer embeddings only;
2. verifies row/bout-ID alignment across seeds;
3. fits scaler on TRAIN only;
4. fits PCA on scaled TRAIN only;
5. fits frozen KMeans(k=8) on TRAIN only;
6. predicts VALIDATION labels;
7. reports occupancy, silhouette and KMeans distance-margin confidence;
8. estimates repeated-fit TRAIN stability on a deterministic TRAIN subsample;
9. aligns cluster identities to seed 11 using TRAIN-only Hungarian matching;
10. applies the TRAIN-derived mapping unchanged to VALIDATION;
11. reports cross-seed ARI/NMI/aligned agreement;
12. saves fitted scaler/PCA/KMeans objects for later frozen TEST evaluation;
13. writes manifests and SHA-256 checksums.

TEST is never loaded.

Important interpretation
------------------------
This does not test whether DS-005 KMeans centroids themselves transfer.
It tests whether the already frozen DS-005 clustering *recipe* yields
reproducible k=8 organization when applied to DS-006 representations.

Expected input
--------------
data/processed/DS-006/transfer_embeddings/
    seed11/
        train_embeddings.npz
        validation_embeddings.npz
        train_manifest.json
        validation_manifest.json
    ...
    seed79/

Expected rows:
    TRAIN       118,100
    VALIDATION   18,835

Expected embedding dimension:
    64

Outputs
-------
data/processed/DS-006/transfer_clustering/
    seed11/
        train_labels.npy
        validation_labels.npy
        train_labels_aligned.npy
        validation_labels_aligned.npy
        scaler.joblib
        pca.joblib
        kmeans.joblib
        cluster_centers.npy
        pca_explained_variance_ratio.npy
        metrics.json
        manifest.json
        SHA256SUMS
    ...
    seed79/
    cross_seed_stability.json
    summary.json
    DS006_TRANSFER_CLUSTERING_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/discovery/ds006_transfer_clustering.py

Intentional rerun:

    PYTHONPATH=. python3 src/discovery/ds006_transfer_clustering.py \
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

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
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
    / "transfer_clustering"
)

SSL_SEEDS = (11, 23, 37, 51, 79)
REFERENCE_SSL_SEED = 11
PARTITIONS = ("train", "validation")

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

EXPECTED_DIM = 64

# Frozen DS-005-selected discovery configuration.
FROZEN_METHOD = "kmeans"
FROZEN_K = 8
FROZEN_CLUSTERING_SEED = 20260822
FROZEN_PCA_VARIANCE = 0.95
FROZEN_KMEANS_N_INIT = 10

# Diagnostic settings. These are not used for method/k selection.
DEFAULT_SILHOUETTE_SAMPLE = 20_000
DEFAULT_STABILITY_SAMPLE = 50_000
DEFAULT_STABILITY_REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen KMeans(k=8) DS-005 clustering recipe to "
            "DS-006 transfer embeddings. TEST is prohibited."
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
        "--silhouette-sample",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE,
    )

    parser.add_argument(
        "--stability-sample",
        type=int,
        default=DEFAULT_STABILITY_SAMPLE,
    )

    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=DEFAULT_STABILITY_REPEATS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_json(
    path: Path,
    obj: Any,
) -> None:
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
    ) as handle:
        handle.write(payload)
        tmp = handle.name

    os.replace(
        tmp,
        path,
    )


def is_relative_to(
    path: Path,
    parent: Path,
) -> bool:
    try:
        path.relative_to(
            parent
        )
        return True
    except ValueError:
        return False


def assert_safe_paths(
    input_root: Path,
    output_root: Path,
) -> None:
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    expected_input = (
        DEFAULT_INPUT_ROOT.resolve()
    )

    if input_root != expected_input:
        raise RuntimeError(
            "For replication safety, --input-root must resolve exactly to "
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


def embedding_path(
    input_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    path = (
        input_root
        / f"seed{ssl_seed}"
        / f"{partition}_embeddings.npz"
    )

    if "test" in path.name.lower():
        raise RuntimeError(
            "Protected TEST path reached unexpectedly."
        )

    return path


def manifest_path(
    input_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Path:
    if partition not in PARTITIONS:
        raise RuntimeError(
            "Only TRAIN and VALIDATION are permitted."
        )

    return (
        input_root
        / f"seed{ssl_seed}"
        / f"{partition}_manifest.json"
    )


def load_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        obj,
        dict,
    ):
        raise RuntimeError(
            f"{path} is not a JSON object."
        )

    return obj


def verify_transfer_manifest(
    input_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Dict[str, Any]:
    path = manifest_path(
        input_root,
        ssl_seed=ssl_seed,
        partition=partition,
    )

    manifest = load_json(
        path
    )

    expected = {
        "dataset_id": "DS-006",
        "source_encoder_dataset": "DS-005",
        "partition": partition,
        "training_seed": ssl_seed,
        "embedding_dim": EXPECTED_DIM,
        "representation": "encoder_embedding",
        "projection_head_executed": False,
        "encoder_fine_tuned_on_ds006": False,
        "encoder_parameters_updated": False,
        "test_partition_loaded": False,
        "capped_debug_export": False,
        "rows_exported": EXPECTED_ROWS[
            partition
        ],
    }

    for key, value in expected.items():
        if manifest.get(
            key
        ) != value:
            raise RuntimeError(
                "Transfer manifest mismatch "
                f"seed={ssl_seed} partition={partition}: "
                f"{key}={manifest.get(key)!r}, expected={value!r}"
            )

    return manifest


def load_embeddings(
    input_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Dict[str, Any]:
    verify_transfer_manifest(
        input_root,
        ssl_seed=ssl_seed,
        partition=partition,
    )

    path = embedding_path(
        input_root,
        ssl_seed=ssl_seed,
        partition=partition,
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
            "embeddings",
            "row_index",
            "bout_id",
        }

        missing = (
            required
            - set(
                npz.files
            )
        )

        if missing:
            raise RuntimeError(
                f"{path} missing: {sorted(missing)}"
            )

        embeddings = np.asarray(
            npz[
                "embeddings"
            ],
            dtype=np.float32,
        )

        row_index = np.asarray(
            npz[
                "row_index"
            ],
            dtype=np.int64,
        )

        bout_id = np.asarray(
            npz[
                "bout_id"
            ]
        ).astype(
            str
        )

    expected_rows = (
        EXPECTED_ROWS[
            partition
        ]
    )

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
            f"{path}: row_index is not 0..N-1."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            f"{path}: NaN/Inf embeddings."
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

    return {
        "path": path,
        "sha256": sha256_file(
            path
        ),
        "embeddings": (
            embeddings
        ),
        "row_index": (
            row_index
        ),
        "bout_id": (
            bout_id
        ),
    }


def verify_cross_seed_alignment(
    loaded: Mapping[
        int,
        Mapping[
            str,
            Dict[str, Any],
        ],
    ],
) -> None:
    for partition in PARTITIONS:
        reference = loaded[
            REFERENCE_SSL_SEED
        ][
            partition
        ]

        for seed in SSL_SEEDS:
            candidate = loaded[
                seed
            ][
                partition
            ]

            if not np.array_equal(
                reference[
                    "row_index"
                ],
                candidate[
                    "row_index"
                ],
            ):
                raise RuntimeError(
                    "Cross-seed row_index alignment failed: "
                    f"partition={partition}, seed={seed}."
                )

            if not np.array_equal(
                reference[
                    "bout_id"
                ],
                candidate[
                    "bout_id"
                ],
            ):
                raise RuntimeError(
                    "Cross-seed bout_id ordering failed: "
                    f"partition={partition}, seed={seed}."
                )


def deterministic_indices(
    n_rows: int,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if (
        max_rows <= 0
        or n_rows <= max_rows
    ):
        return np.arange(
            n_rows,
            dtype=np.int64,
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        n_rows,
        size=max_rows,
        replace=False,
    )

    indices.sort()

    return indices.astype(
        np.int64,
        copy=False,
    )


def safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> float:
    if np.unique(
        labels
    ).size < 2:
        return -1.0

    idx = deterministic_indices(
        len(
            labels
        ),
        max_rows=max_rows,
        seed=seed,
    )

    sampled_labels = labels[
        idx
    ]

    if np.unique(
        sampled_labels
    ).size < 2:
        return -1.0

    return float(
        silhouette_score(
            X[
                idx
            ],
            sampled_labels,
            metric="euclidean",
        )
    )


def cluster_occupancy(
    labels: np.ndarray,
    *,
    k: int,
) -> Dict[str, Any]:
    counts = np.bincount(
        labels.astype(
            np.int64
        ),
        minlength=k,
    )

    total = int(
        counts.sum()
    )

    fractions = (
        counts
        / total
        if total
        else np.zeros(
            k,
            dtype=float,
        )
    )

    return {
        "counts": [
            int(
                x
            )
            for x
            in counts
        ],
        "fractions": [
            float(
                x
            )
            for x
            in fractions
        ],
        "min_fraction": float(
            np.min(
                fractions
            )
        ),
        "max_fraction": float(
            np.max(
                fractions
            )
        ),
        "empty_clusters": int(
            np.sum(
                counts == 0
            )
        ),
    }


def kmeans_confidence(
    model: KMeans,
    X: np.ndarray,
) -> float:
    """Mean normalized nearest-vs-second-nearest distance margin."""
    distances = model.transform(
        X
    )

    if distances.shape[
        1
    ] < 2:
        return 0.0

    nearest_two = np.partition(
        distances,
        kth=1,
        axis=1,
    )[
        :,
        :2,
    ]

    nearest = np.min(
        nearest_two,
        axis=1,
    )

    second = np.max(
        nearest_two,
        axis=1,
    )

    margin = (
        second
        - nearest
    ) / np.maximum(
        second,
        1e-12,
    )

    return float(
        np.mean(
            margin
        )
    )


def repeated_fit_stability(
    train_pca: np.ndarray,
    *,
    repeats: int,
    max_rows: int,
) -> Dict[str, Any]:
    if repeats < 2:
        raise ValueError(
            "--stability-repeats must be >= 2."
        )

    idx = deterministic_indices(
        train_pca.shape[
            0
        ],
        max_rows=max_rows,
        seed=FROZEN_CLUSTERING_SEED,
    )

    X = train_pca[
        idx
    ]

    labels_by_repeat: List[
        np.ndarray
    ] = []

    seeds: List[
        int
    ] = []

    for repeat in range(
        repeats
    ):
        seed = (
            FROZEN_CLUSTERING_SEED
            + repeat
            * 1009
        )

        model = KMeans(
            n_clusters=FROZEN_K,
            random_state=seed,
            n_init=FROZEN_KMEANS_N_INIT,
        )

        labels = model.fit_predict(
            X
        ).astype(
            np.int16,
            copy=False,
        )

        labels_by_repeat.append(
            labels
        )

        seeds.append(
            int(
                seed
            )
        )

    pairwise: List[
        float
    ] = []

    for i in range(
        len(
            labels_by_repeat
        )
    ):
        for j in range(
            i + 1,
            len(
                labels_by_repeat
            ),
        ):
            pairwise.append(
                float(
                    adjusted_rand_score(
                        labels_by_repeat[
                            i
                        ],
                        labels_by_repeat[
                            j
                        ],
                    )
                )
            )

    return {
        "sample_rows": int(
            len(
                idx
            )
        ),
        "repeats": int(
            repeats
        ),
        "repeat_seeds": (
            seeds
        ),
        "mean_pairwise_ari": float(
            np.mean(
                pairwise
            )
        ),
        "std_pairwise_ari": float(
            np.std(
                pairwise
            )
        ),
        "min_pairwise_ari": float(
            np.min(
                pairwise
            )
        ),
        "max_pairwise_ari": float(
            np.max(
                pairwise
            )
        ),
    }


def fit_frozen_recipe(
    train: np.ndarray,
    validation: np.ndarray,
) -> Dict[str, Any]:
    scaler = StandardScaler(
        copy=True
    )

    train_scaled = scaler.fit_transform(
        train
    )

    validation_scaled = scaler.transform(
        validation
    )

    pca = PCA(
        n_components=FROZEN_PCA_VARIANCE,
        svd_solver="full",
        random_state=FROZEN_CLUSTERING_SEED,
    )

    train_pca = pca.fit_transform(
        train_scaled
    )

    validation_pca = pca.transform(
        validation_scaled
    )

    model = KMeans(
        n_clusters=FROZEN_K,
        random_state=FROZEN_CLUSTERING_SEED,
        n_init=FROZEN_KMEANS_N_INIT,
    )

    train_labels = model.fit_predict(
        train_pca
    ).astype(
        np.int16,
        copy=False,
    )

    validation_labels = model.predict(
        validation_pca
    ).astype(
        np.int16,
        copy=False,
    )

    return {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "train_pca": train_pca,
        "validation_pca": validation_pca,
        "train_labels": train_labels,
        "validation_labels": validation_labels,
    }


def hungarian_mapping(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[int, int]:
    matrix = confusion_matrix(
        reference_labels,
        candidate_labels,
        labels=np.arange(
            FROZEN_K
        ),
    )

    row_ind, col_ind = (
        linear_sum_assignment(
            -matrix
        )
    )

    return {
        int(
            candidate
        ): int(
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
    mapping: Mapping[
        int,
        int,
    ],
) -> np.ndarray:
    return np.asarray(
        [
            mapping[
                int(
                    label
                )
            ]
            for label
            in labels
        ],
        dtype=np.int16,
    )


def aligned_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    return float(
        np.mean(
            reference
            == candidate
        )
    )


def pairwise_metrics(
    labels_by_seed: Mapping[
        int,
        np.ndarray,
    ],
) -> List[Dict[str, Any]]:
    seeds = sorted(
        labels_by_seed
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for i, seed_a in enumerate(
        seeds
    ):
        for seed_b in seeds[
            i + 1:
        ]:
            a = labels_by_seed[
                seed_a
            ]

            b = labels_by_seed[
                seed_b
            ]

            mapping = hungarian_mapping(
                a,
                b,
            )

            b_aligned = apply_mapping(
                b,
                mapping,
            )

            rows.append(
                {
                    "seed_a": int(
                        seed_a
                    ),
                    "seed_b": int(
                        seed_b
                    ),
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
                    "aligned_agreement": (
                        aligned_agreement(
                            a,
                            b_aligned,
                        )
                    ),
                }
            )

    return rows


def summarize_pairwise(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {
        "pair_count": int(
            len(
                rows
            )
        ),
        "pairs": list(
            rows
        ),
    }

    for metric in (
        "ari",
        "nmi",
        "aligned_agreement",
    ):
        values = np.asarray(
            [
                float(
                    row[
                        metric
                    ]
                )
                for row
                in rows
            ],
            dtype=np.float64,
        )

        output[
            metric
        ] = {
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

    return output


def write_seed_checksums(
    seed_dir: Path,
    files: Sequence[
        Path
    ],
) -> Path:
    path = (
        seed_dir
        / "SHA256SUMS"
    )

    path.write_text(
        "".join(
            f"{sha256_file(file)}  "
            f"{file.name}\n"
            for file
            in files
        ),
        encoding="utf-8",
    )

    return path


def write_global_checksums(
    output_root: Path,
    files: Sequence[
        Path
    ],
) -> Path:
    path = (
        output_root
        / "DS006_TRANSFER_CLUSTERING_SHA256SUMS"
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

    assert_safe_paths(
        input_root,
        output_root,
    )

    if (
        args.silhouette_sample
        < 1
    ):
        raise ValueError(
            "--silhouette-sample must be >= 1."
        )

    if (
        args.stability_sample
        < 1
    ):
        raise ValueError(
            "--stability-sample must be >= 1."
        )

    if (
        args.stability_repeats
        < 2
    ):
        raise ValueError(
            "--stability-repeats must be >= 2."
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
            "Use --overwrite only for an intentional rerun."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 FROZEN TRANSFER CLUSTERING"
    )
    print(
        "=" * 80
    )
    print(
        "Scientific mode:    frozen-method replication"
    )
    print(
        f"Method:             {FROZEN_METHOD}"
    )
    print(
        f"k:                  {FROZEN_K}"
    )
    print(
        f"Clustering seed:    {FROZEN_CLUSTERING_SEED}"
    )
    print(
        f"KMeans n_init:      {FROZEN_KMEANS_N_INIT}"
    )
    print(
        f"PCA target:         {FROZEN_PCA_VARIANCE:.2f}"
    )
    print(
        "Scaler fit:         DS-006 TRAIN only"
    )
    print(
        "PCA fit:            DS-006 TRAIN only"
    )
    print(
        "KMeans fit:         DS-006 TRAIN only"
    )
    print(
        "Method/k selection: NONE"
    )
    print(
        "TEST partition:     PROTECTED / NOT LOADED"
    )
    print()

    loaded: Dict[
        int,
        Dict[
            str,
            Dict[str, Any],
        ],
    ] = {}

    for seed in SSL_SEEDS:
        loaded[
            seed
        ] = {}

        for partition in PARTITIONS:
            loaded[
                seed
            ][
                partition
            ] = load_embeddings(
                input_root,
                ssl_seed=seed,
                partition=partition,
            )

    verify_cross_seed_alignment(
        loaded
    )

    print(
        "Cross-seed bout ordering: VERIFIED"
    )
    print()

    fitted: Dict[
        int,
        Dict[str, Any],
    ] = {}

    seed_metrics: Dict[
        int,
        Dict[str, Any],
    ] = {}

    written: List[
        Path
    ] = []

    for ssl_seed in SSL_SEEDS:
        print(
            "=" * 80
        )
        print(
            f"TRANSFER ENCODER SEED {ssl_seed}"
        )
        print(
            "=" * 80
        )

        train = loaded[
            ssl_seed
        ][
            "train"
        ][
            "embeddings"
        ]

        validation = loaded[
            ssl_seed
        ][
            "validation"
        ][
            "embeddings"
        ]

        result = fit_frozen_recipe(
            train,
            validation,
        )

        fitted[
            ssl_seed
        ] = result

        pca = result[
            "pca"
        ]

        model = result[
            "model"
        ]

        train_pca = result[
            "train_pca"
        ]

        validation_pca = result[
            "validation_pca"
        ]

        train_labels = result[
            "train_labels"
        ]

        validation_labels = result[
            "validation_labels"
        ]

        train_occ = cluster_occupancy(
            train_labels,
            k=FROZEN_K,
        )

        val_occ = cluster_occupancy(
            validation_labels,
            k=FROZEN_K,
        )

        train_sil = safe_silhouette(
            train_pca,
            train_labels,
            max_rows=(
                args.silhouette_sample
            ),
            seed=(
                FROZEN_CLUSTERING_SEED
            ),
        )

        val_sil = safe_silhouette(
            validation_pca,
            validation_labels,
            max_rows=(
                args.silhouette_sample
            ),
            seed=(
                FROZEN_CLUSTERING_SEED
            ),
        )

        val_confidence = (
            kmeans_confidence(
                model,
                validation_pca,
            )
        )

        stability = (
            repeated_fit_stability(
                train_pca,
                repeats=(
                    args.stability_repeats
                ),
                max_rows=(
                    args.stability_sample
                ),
            )
        )

        metrics = {
            "dataset_id": "DS-006",
            "source_encoder_dataset": "DS-005",
            "ssl_encoder_seed": int(
                ssl_seed
            ),
            "method": FROZEN_METHOD,
            "k": int(
                FROZEN_K
            ),
            "clustering_seed": int(
                FROZEN_CLUSTERING_SEED
            ),
            "kmeans_n_init": int(
                FROZEN_KMEANS_N_INIT
            ),
            "pca_variance_target": float(
                FROZEN_PCA_VARIANCE
            ),
            "pca_components": int(
                pca.n_components_
            ),
            "pca_variance_retained": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
            "train_rows": int(
                train.shape[
                    0
                ]
            ),
            "validation_rows": int(
                validation.shape[
                    0
                ]
            ),
            "train_silhouette": float(
                train_sil
            ),
            "validation_silhouette": float(
                val_sil
            ),
            "validation_confidence": float(
                val_confidence
            ),
            "train_occupancy": (
                train_occ
            ),
            "validation_occupancy": (
                val_occ
            ),
            "train_repeated_fit_stability": (
                stability
            ),
            "scaler_fit_on_train_only": True,
            "pca_fit_on_train_only": True,
            "kmeans_fit_on_train_only": True,
            "method_selection_performed": False,
            "k_selection_performed": False,
            "test_partition_used": False,
        }

        seed_metrics[
            ssl_seed
        ] = metrics

        seed_dir = (
            output_root
            / f"seed{ssl_seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {
            "train_labels": (
                seed_dir
                / "train_labels.npy"
            ),
            "validation_labels": (
                seed_dir
                / "validation_labels.npy"
            ),
            "centers": (
                seed_dir
                / "cluster_centers.npy"
            ),
            "pca_ratio": (
                seed_dir
                / "pca_explained_variance_ratio.npy"
            ),
            "scaler": (
                seed_dir
                / "scaler.joblib"
            ),
            "pca": (
                seed_dir
                / "pca.joblib"
            ),
            "kmeans": (
                seed_dir
                / "kmeans.joblib"
            ),
            "metrics": (
                seed_dir
                / "metrics.json"
            ),
            "manifest": (
                seed_dir
                / "manifest.json"
            ),
        }

        if not args.overwrite:
            existing = [
                path
                for path
                in paths.values()
                if path.exists()
            ]

            if existing:
                raise FileExistsError(
                    "Clustering artifacts already exist. "
                    "Use --overwrite for intentional rerun:\n"
                    + "\n".join(
                        str(
                            path
                        )
                        for path
                        in existing
                    )
                )

        np.save(
            paths[
                "train_labels"
            ],
            train_labels,
        )

        np.save(
            paths[
                "validation_labels"
            ],
            validation_labels,
        )

        np.save(
            paths[
                "centers"
            ],
            model.cluster_centers_.astype(
                np.float32,
                copy=False,
            ),
        )

        np.save(
            paths[
                "pca_ratio"
            ],
            np.asarray(
                pca.explained_variance_ratio_,
                dtype=np.float32,
            ),
        )

        joblib.dump(
            result[
                "scaler"
            ],
            paths[
                "scaler"
            ],
        )

        joblib.dump(
            pca,
            paths[
                "pca"
            ],
        )

        joblib.dump(
            model,
            paths[
                "kmeans"
            ],
        )

        atomic_write_json(
            paths[
                "metrics"
            ],
            metrics,
        )

        manifest = {
            "dataset_id": "DS-006",
            "analysis": (
                "frozen_ds005_clustering_recipe_on_transfer_embeddings"
            ),
            "source_encoder_dataset": "DS-005",
            "ssl_encoder_seed": int(
                ssl_seed
            ),
            "source_train_embedding": str(
                loaded[
                    ssl_seed
                ][
                    "train"
                ][
                    "path"
                ].relative_to(
                    REPO_ROOT
                )
            ),
            "source_validation_embedding": str(
                loaded[
                    ssl_seed
                ][
                    "validation"
                ][
                    "path"
                ].relative_to(
                    REPO_ROOT
                )
            ),
            "source_train_sha256": loaded[
                ssl_seed
            ][
                "train"
            ][
                "sha256"
            ],
            "source_validation_sha256": loaded[
                ssl_seed
            ][
                "validation"
            ][
                "sha256"
            ],
            "method": FROZEN_METHOD,
            "k": int(
                FROZEN_K
            ),
            "clustering_seed": int(
                FROZEN_CLUSTERING_SEED
            ),
            "kmeans_n_init": int(
                FROZEN_KMEANS_N_INIT
            ),
            "pca_variance_target": float(
                FROZEN_PCA_VARIANCE
            ),
            "pca_components": int(
                pca.n_components_
            ),
            "pca_variance_retained": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
            "train_rows": int(
                train.shape[
                    0
                ]
            ),
            "validation_rows": int(
                validation.shape[
                    0
                ]
            ),
            "scaler_fit_partition": "train",
            "pca_fit_partition": "train",
            "kmeans_fit_partition": "train",
            "method_selection_performed": False,
            "k_selection_performed": False,
            "test_partition_loaded": False,
            "fitted_objects_saved_for_later_frozen_test_application": True,
        }

        atomic_write_json(
            paths[
                "manifest"
            ],
            manifest,
        )

        seed_files = list(
            paths.values()
        )

        checksum = write_seed_checksums(
            seed_dir,
            seed_files,
        )

        written.extend(
            seed_files
        )

        written.append(
            checksum
        )

        print(
            f"TRAIN rows:               {train.shape[0]:,}"
        )
        print(
            f"VALIDATION rows:          {validation.shape[0]:,}"
        )
        print(
            f"PCA retained:             {pca.n_components_} components "
            f"({np.sum(pca.explained_variance_ratio_):.4f} variance)"
        )
        print(
            f"TRAIN silhouette:         {train_sil:.6f}"
        )
        print(
            f"VALIDATION silhouette:    {val_sil:.6f}"
        )
        print(
            f"VALIDATION confidence:    {val_confidence:.6f}"
        )
        print(
            f"TRAIN min occupancy:      {train_occ['min_fraction']:.6f}"
        )
        print(
            f"VALIDATION min occupancy: {val_occ['min_fraction']:.6f}"
        )
        print(
            f"TRAIN repeated-fit ARI:   "
            f"{stability['mean_pairwise_ari']:.6f}"
        )
        print(
            "TEST partition used:      NO"
        )
        print()

    # TRAIN-derived mapping to reference seed 11, reused unchanged on VAL.
    reference_train = fitted[
        REFERENCE_SSL_SEED
    ][
        "train_labels"
    ]

    mappings: Dict[
        int,
        Dict[int, int],
    ] = {
        REFERENCE_SSL_SEED: {
            cluster: cluster
            for cluster
            in range(
                FROZEN_K
            )
        }
    }

    aligned_train: Dict[
        int,
        np.ndarray,
    ] = {}

    aligned_validation: Dict[
        int,
        np.ndarray,
    ] = {}

    for seed in SSL_SEEDS:
        if seed != REFERENCE_SSL_SEED:
            mappings[
                seed
            ] = hungarian_mapping(
                reference_train,
                fitted[
                    seed
                ][
                    "train_labels"
                ],
            )

        aligned_train[
            seed
        ] = apply_mapping(
            fitted[
                seed
            ][
                "train_labels"
            ],
            mappings[
                seed
            ],
        )

        aligned_validation[
            seed
        ] = apply_mapping(
            fitted[
                seed
            ][
                "validation_labels"
            ],
            mappings[
                seed
            ],
        )

        seed_dir = (
            output_root
            / f"seed{seed}"
        )

        train_aligned_path = (
            seed_dir
            / "train_labels_aligned.npy"
        )

        validation_aligned_path = (
            seed_dir
            / "validation_labels_aligned.npy"
        )

        if (
            not args.overwrite
            and (
                train_aligned_path.exists()
                or validation_aligned_path.exists()
            )
        ):
            raise FileExistsError(
                "Aligned label artifacts already exist. "
                "Use --overwrite for intentional rerun."
            )

        np.save(
            train_aligned_path,
            aligned_train[
                seed
            ],
        )

        np.save(
            validation_aligned_path,
            aligned_validation[
                seed
            ],
        )

        written.extend(
            [
                train_aligned_path,
                validation_aligned_path,
            ]
        )

    # Pairwise metrics remain permutation-invariant for ARI/NMI.
    # Aligned agreement uses pair-specific Hungarian mapping.
    train_pairwise = summarize_pairwise(
        pairwise_metrics(
            {
                seed: fitted[
                    seed
                ][
                    "train_labels"
                ]
                for seed
                in SSL_SEEDS
            }
        )
    )

    validation_pairwise = summarize_pairwise(
        pairwise_metrics(
            {
                seed: fitted[
                    seed
                ][
                    "validation_labels"
                ]
                for seed
                in SSL_SEEDS
            }
        )
    )

    # Also report agreement after the global TRAIN->seed11 mappings are
    # reused unchanged on validation, which is stricter and leakage-safe.
    val_ref = aligned_validation[
        REFERENCE_SSL_SEED
    ]

    train_ref = aligned_train[
        REFERENCE_SSL_SEED
    ]

    train_ref_agreement = {
        str(
            seed
        ): float(
            np.mean(
                aligned_train[
                    seed
                ]
                == train_ref
            )
        )
        for seed
        in SSL_SEEDS
    }

    validation_ref_agreement = {
        str(
            seed
        ): float(
            np.mean(
                aligned_validation[
                    seed
                ]
                == val_ref
            )
        )
        for seed
        in SSL_SEEDS
    }

    cross_seed = {
        "dataset_id": "DS-006",
        "analysis": (
            "cross_seed_frozen_transfer_clustering_stability"
        ),
        "reference_ssl_seed": int(
            REFERENCE_SSL_SEED
        ),
        "train_derived_mappings_to_reference": {
            str(
                seed
            ): {
                str(
                    src
                ): int(
                    dst
                )
                for src, dst
                in mapping.items()
            }
            for seed, mapping
            in mappings.items()
        },
        "train_pairwise": (
            train_pairwise
        ),
        "validation_pairwise": (
            validation_pairwise
        ),
        "train_agreement_to_seed11_after_train_alignment": (
            train_ref_agreement
        ),
        "validation_agreement_to_seed11_using_train_mapping": (
            validation_ref_agreement
        ),
        "mean_validation_agreement_to_seed11_using_train_mapping": float(
            np.mean(
                [
                    value
                    for key, value
                    in validation_ref_agreement.items()
                    if int(
                        key
                    )
                    != REFERENCE_SSL_SEED
                ]
            )
        ),
        "test_partition_used": False,
    }

    cross_seed_path = (
        output_root
        / "cross_seed_stability.json"
    )

    atomic_write_json(
        cross_seed_path,
        cross_seed,
    )

    written.append(
        cross_seed_path
    )

    pca_components = np.asarray(
        [
            seed_metrics[
                seed
            ][
                "pca_components"
            ]
            for seed
            in SSL_SEEDS
        ],
        dtype=np.float64,
    )

    val_silhouettes = np.asarray(
        [
            seed_metrics[
                seed
            ][
                "validation_silhouette"
            ]
            for seed
            in SSL_SEEDS
        ],
        dtype=np.float64,
    )

    train_stability = np.asarray(
        [
            seed_metrics[
                seed
            ][
                "train_repeated_fit_stability"
            ][
                "mean_pairwise_ari"
            ]
            for seed
            in SSL_SEEDS
        ],
        dtype=np.float64,
    )

    val_min_occupancy = np.asarray(
        [
            seed_metrics[
                seed
            ][
                "validation_occupancy"
            ][
                "min_fraction"
            ]
            for seed
            in SSL_SEEDS
        ],
        dtype=np.float64,
    )

    summary = {
        "dataset_id": "DS-006",
        "analysis": "frozen_transfer_clustering",
        "source_encoder_dataset": "DS-005",
        "scientific_mode": "frozen_method_replication",
        "frozen_configuration": {
            "method": FROZEN_METHOD,
            "k": int(
                FROZEN_K
            ),
            "clustering_seed": int(
                FROZEN_CLUSTERING_SEED
            ),
            "kmeans_n_init": int(
                FROZEN_KMEANS_N_INIT
            ),
            "pca_variance_target": float(
                FROZEN_PCA_VARIANCE
            ),
            "standard_scaler_fit_on_ds006_train_only": True,
            "pca_fit_on_ds006_train_only": True,
            "kmeans_fit_on_ds006_train_only": True,
        },
        "ssl_encoder_seeds": list(
            SSL_SEEDS
        ),
        "aggregate": {
            "mean_pca_components": float(
                np.mean(
                    pca_components
                )
            ),
            "pca_components_by_seed": {
                str(
                    seed
                ): int(
                    seed_metrics[
                        seed
                    ][
                        "pca_components"
                    ]
                )
                for seed
                in SSL_SEEDS
            },
            "mean_validation_silhouette": float(
                np.mean(
                    val_silhouettes
                )
            ),
            "validation_silhouette_range": [
                float(
                    np.min(
                        val_silhouettes
                    )
                ),
                float(
                    np.max(
                        val_silhouettes
                    )
                ),
            ],
            "mean_train_repeated_fit_stability_ari": float(
                np.mean(
                    train_stability
                )
            ),
            "train_repeated_fit_stability_ari_range": [
                float(
                    np.min(
                        train_stability
                    )
                ),
                float(
                    np.max(
                        train_stability
                    )
                ),
            ],
            "mean_validation_min_cluster_fraction": float(
                np.mean(
                    val_min_occupancy
                )
            ),
            "validation_pairwise_ari": (
                validation_pairwise[
                    "ari"
                ]
            ),
            "validation_pairwise_nmi": (
                validation_pairwise[
                    "nmi"
                ]
            ),
            "validation_pairwise_aligned_agreement": (
                validation_pairwise[
                    "aligned_agreement"
                ]
            ),
            "mean_validation_agreement_to_seed11_using_train_mapping": (
                cross_seed[
                    "mean_validation_agreement_to_seed11_using_train_mapping"
                ]
            ),
        },
        "per_seed": {
            str(
                seed
            ): seed_metrics[
                seed
            ]
            for seed
            in SSL_SEEDS
        },
        "interpretation_guardrails": {
            "no_method_selection_on_ds006": True,
            "no_k_selection_on_ds006": True,
            "not_direct_transfer_of_ds005_kmeans_centroids": True,
            "this_is_transfer_representation_plus_frozen_clustering_recipe": True,
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

    global_checksum = write_global_checksums(
        output_root,
        written,
    )

    print(
        "=" * 80
    )
    print(
        "DS-006 FROZEN TRANSFER CLUSTERING SUMMARY"
    )
    print(
        "=" * 80
    )
    print(
        f"Mean PCA components:                "
        f"{np.mean(pca_components):.2f}"
    )
    print(
        f"Mean VALIDATION silhouette:         "
        f"{np.mean(val_silhouettes):.6f}"
    )
    print(
        f"Mean TRAIN repeated-fit ARI:        "
        f"{np.mean(train_stability):.6f}"
    )
    print(
        f"Mean cross-seed VALIDATION ARI:     "
        f"{validation_pairwise['ari']['mean']:.6f}"
    )
    print(
        f"Mean cross-seed VALIDATION NMI:     "
        f"{validation_pairwise['nmi']['mean']:.6f}"
    )
    print(
        f"Mean pairwise aligned agreement:    "
        f"{validation_pairwise['aligned_agreement']['mean']:.6f}"
    )
    print(
        f"Mean VAL agreement to seed11 "
        f"(TRAIN mapping): "
        f"{cross_seed['mean_validation_agreement_to_seed11_using_train_mapping']:.6f}"
    )
    print(
        "Method/k selection performed:          NO"
    )
    print(
        "TEST partition used:                    NO"
    )
    print(
        f"Summary:     {summary_path}"
    )
    print(
        f"Cross-seed:  {cross_seed_path}"
    )
    print(
        f"Checksums:   {global_checksum}"
    )


if __name__ == "__main__":
    main()
