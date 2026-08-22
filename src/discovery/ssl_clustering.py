#!/usr/bin/env python3
"""TRAIN/VALIDATION-only clustering selection for DS-005 SSL embeddings.

This module performs discovery model selection on exported encoder embeddings.
It deliberately refuses to load TEST embeddings.

Primary workflow
----------------
1. Load TRAIN and VALIDATION encoder embeddings for one SSL seed.
2. Fit PCA on TRAIN only.
3. Transform TRAIN and VALIDATION with the frozen TRAIN PCA.
4. Evaluate preregistered candidate clusterers:
   - KMeans
   - GaussianMixture
5. Select k/method using VALIDATION silhouette plus TRAIN subsample stability.
6. Save selection artifacts and checksums.

Important
---------
- TEST is not loaded here.
- Encoder embeddings are used; projection-head outputs are not used.
- This is discovery-model selection, not final held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_clustering"
)

DEFAULT_SEED = 20260822
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 12
DEFAULT_PCA_VARIANCE = 0.95

# Candidate composite rule. Keep explicit and auditable.
# If your preregistration specifies different weights, change these BEFORE
# examining final SSL clustering outcomes and record the change in decisions.md.
SILHOUETTE_WEIGHT = 0.60
STABILITY_WEIGHT = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select clustering configuration for SSL embeddings."
    )
    parser.add_argument(
        "command",
        choices=("select",),
        nargs="?",
        default="select",
    )
    parser.add_argument(
        "--ssl-seed",
        type=int,
        default=11,
        help="SSL training seed whose embeddings should be analyzed.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBED_DIR,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=DEFAULT_K_MIN,
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=DEFAULT_K_MAX,
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=DEFAULT_PCA_VARIANCE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Clustering/model-selection seed.",
    )
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=20000,
        help="Maximum validation rows used to estimate silhouette.",
    )
    parser.add_argument(
        "--stability-sample",
        type=int,
        default=20000,
        help="Maximum TRAIN rows used for stability calculation.",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=3,
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_partition(
    embedding_dir: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> np.ndarray:
    if partition == "test":
        raise RuntimeError(
            "TEST embeddings are protected and cannot be loaded during selection."
        )

    path = (
        embedding_dir
        / f"ssl_seed{ssl_seed}_{partition}_embeddings.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Export TRAIN/VALIDATION embeddings for this SSL seed first."
        )

    x = np.load(path, mmap_mode="r")
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {x.shape}.")

    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"Invalid embedding shape: {x.shape}.")

    if not np.isfinite(x).all():
        raise ValueError(f"{partition} embeddings contain NaN or Inf.")

    return x


def deterministic_subsample(
    x: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        x.shape[0],
        size=max_rows,
        replace=False,
    )
    indices.sort()
    return x[indices]


def fit_preprocessing(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    pca_variance: float,
    seed: int,
) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray]:
    """Fit scaler and PCA on TRAIN only."""
    if not 0.0 < pca_variance <= 1.0:
        raise ValueError("--pca-variance must be in (0, 1].")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)

    pca = PCA(
        n_components=pca_variance,
        svd_solver="full",
        random_state=seed,
    )

    train_pca = pca.fit_transform(train_scaled)
    validation_pca = pca.transform(validation_scaled)

    return scaler, pca, train_pca, validation_pca


def fit_labels(
    method: str,
    *,
    k: int,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "kmeans":
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
        )
        train_labels = model.fit_predict(train_x)
        validation_labels = model.predict(validation_x)
        return train_labels, validation_labels

    if method == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=1,
            reg_covar=1e-6,
        )
        model.fit(train_x)
        train_labels = model.predict(train_x)
        validation_labels = model.predict(validation_x)
        return train_labels, validation_labels

    raise ValueError(f"Unknown method: {method}")


def validation_silhouette(
    validation_x: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return -1.0

    if validation_x.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            validation_x.shape[0],
            size=max_rows,
            replace=False,
        )
        x = validation_x[indices]
        y = labels[indices]
    else:
        x = validation_x
        y = labels

    if np.unique(y).size < 2:
        return -1.0

    return float(
        silhouette_score(
            x,
            y,
            metric="euclidean",
        )
    )


def clustering_stability(
    method: str,
    *,
    k: int,
    train_x: np.ndarray,
    base_seed: int,
    repeats: int,
    max_rows: int,
) -> float:
    """Estimate seed stability using ARI on a fixed TRAIN subsample."""
    x = deterministic_subsample(
        train_x,
        max_rows=max_rows,
        seed=base_seed,
    )

    label_sets: List[np.ndarray] = []

    for repeat in range(repeats):
        seed = base_seed + repeat * 1009

        if method == "kmeans":
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
            )
            labels = model.fit_predict(x)
        elif method == "gmm":
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                n_init=1,
                reg_covar=1e-6,
            )
            labels = model.fit_predict(x)
        else:
            raise ValueError(method)

        label_sets.append(labels)

    if len(label_sets) < 2:
        return 1.0

    aris: List[float] = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            aris.append(
                float(
                    adjusted_rand_score(
                        label_sets[i],
                        label_sets[j],
                    )
                )
            )

    return float(np.mean(aris))


def composite_score(
    *,
    val_silhouette: float,
    stability: float,
) -> float:
    return (
        SILHOUETTE_WEIGHT * float(val_silhouette)
        + STABILITY_WEIGHT * float(stability)
    )


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select(args: argparse.Namespace) -> None:
    if args.k_min < 2:
        raise ValueError("--k-min must be >= 2.")

    if args.k_max < args.k_min:
        raise ValueError("--k-max must be >= --k-min.")

    train = load_partition(
        args.embedding_dir,
        ssl_seed=args.ssl_seed,
        partition="train",
    )
    validation = load_partition(
        args.embedding_dir,
        ssl_seed=args.ssl_seed,
        partition="validation",
    )

    if train.shape[1] != validation.shape[1]:
        raise ValueError(
            "TRAIN and VALIDATION embedding dimensions do not match."
        )

    print()
    print("SSL CLUSTERING MODEL SELECTION")
    print("=" * 35)
    print("TEST partition status: NOT LOADED")
    print(f"SSL seed:         {args.ssl_seed}")
    print(f"Train rows:       {train.shape[0]:,}")
    print(f"Validation rows:  {validation.shape[0]:,}")
    print(f"Embedding dim:    {train.shape[1]}")

    scaler, pca, train_pca, validation_pca = fit_preprocessing(
        train,
        validation,
        pca_variance=args.pca_variance,
        seed=args.seed,
    )

    retained = float(np.sum(pca.explained_variance_ratio_))

    print(
        f"PCA fit on TRAIN only: {pca.n_components_} components "
        f"retain {retained:.4f} variance"
    )

    results: List[Dict[str, Any]] = []

    for method in ("kmeans", "gmm"):
        for k in range(args.k_min, args.k_max + 1):
            _train_labels, val_labels = fit_labels(
                method,
                k=k,
                train_x=train_pca,
                validation_x=validation_pca,
                seed=args.seed,
            )

            val_sil = validation_silhouette(
                validation_pca,
                val_labels,
                max_rows=args.silhouette_sample,
                seed=args.seed + k,
            )

            stability = clustering_stability(
                method,
                k=k,
                train_x=train_pca,
                base_seed=args.seed,
                repeats=args.stability_repeats,
                max_rows=args.stability_sample,
            )

            score = composite_score(
                val_silhouette=val_sil,
                stability=stability,
            )

            results.append(
                {
                    "method": method,
                    "k": int(k),
                    "seed": int(args.seed),
                    "score": float(score),
                    "validation_silhouette": float(val_sil),
                    "stability_ari": float(stability),
                }
            )

    results.sort(
        key=lambda row: (
            row["score"],
            row["validation_silhouette"],
            row["stability_ari"],
        ),
        reverse=True,
    )

    print()
    print("Top candidates:")
    for row in results[:10]:
        print(
            f"  {row['method']:6s} "
            f"k={row['k']:2d} "
            f"score={row['score']:7.4f} "
            f"val_sil={row['validation_silhouette']:7.4f} "
            f"stability={row['stability_ari']:7.4f}"
        )

    best = results[0]

    print()
    print("Selected configuration:")
    print(f"  method: {best['method']}")
    print(f"  k:      {best['k']}")
    print(f"  seed:   {best['seed']}")
    print(f"  score:  {best['score']:.6f}")
    print()
    print("TEST partition remains untouched.")

    output_dir = args.output_root / f"seed{args.ssl_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "ssl_training_seed": int(args.ssl_seed),
        "test_partition_loaded": False,
        "train_rows": int(train.shape[0]),
        "validation_rows": int(validation.shape[0]),
        "input_embedding_dim": int(train.shape[1]),
        "pca_fit_partition": "train",
        "pca_components": int(pca.n_components_),
        "pca_variance_retained": retained,
        "k_range": [int(args.k_min), int(args.k_max)],
        "methods": ["kmeans", "gmm"],
        "selection_score": {
            "formula": (
                f"{SILHOUETTE_WEIGHT:.2f} * validation_silhouette + "
                f"{STABILITY_WEIGHT:.2f} * stability_ari"
            ),
            "silhouette_weight": SILHOUETTE_WEIGHT,
            "stability_weight": STABILITY_WEIGHT,
        },
        "results": results,
    }

    selected_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "ssl_training_seed": int(args.ssl_seed),
        "test_partition_loaded": False,
        "method": best["method"],
        "k": best["k"],
        "seed": best["seed"],
        "score": best["score"],
        "validation_silhouette": best["validation_silhouette"],
        "stability_ari": best["stability_ari"],
        "pca_components": int(pca.n_components_),
        "pca_variance_retained": retained,
    }

    pca_payload = {
        "fit_partition": "train",
        "input_dim": int(train.shape[1]),
        "components": int(pca.n_components_),
        "variance_retained": retained,
        "explained_variance_ratio": [
            float(x) for x in pca.explained_variance_ratio_
        ],
    }

    selection_path = output_dir / "selection_results.json"
    selected_path = output_dir / "selected_configuration.json"
    pca_path = output_dir / "pca_diagnostics.json"

    save_json(selection_path, selection_payload)
    save_json(selected_path, selected_payload)
    save_json(pca_path, pca_payload)

    checksums = output_dir / "SELECTION_SHA256SUMS"
    checksum_lines = []
    for path in (selection_path, selected_path, pca_path):
        checksum_lines.append(
            f"{sha256_file(path)}  {path.name}"
        )

    checksums.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Selection artifacts:")
    print(f"  {selection_path}")
    print(f"  {pca_path}")
    print(f"  {selected_path}")
    print(f"  {checksums}")


def main() -> None:
    args = parse_args()

    if args.command == "select":
        select(args)
        return

    raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
