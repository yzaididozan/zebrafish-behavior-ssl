#!/usr/bin/env python3
"""TRAIN/VALIDATION-only clustering selection for DS-005 SSL embeddings.

This module performs clustering model selection on the frozen encoder
embeddings exported by ``scripts/extract_ssl_embeddings.py``.

Protocol
--------
For each frozen SSL training seed:
1. Load TRAIN and VALIDATION encoder embeddings only.
2. Fit StandardScaler on TRAIN only.
3. Fit PCA on TRAIN only and retain the configured variance target.
4. Transform TRAIN and VALIDATION with those TRAIN-fitted transforms.
5. Evaluate KMeans and GaussianMixture for candidate k values.
6. Measure validation silhouette and TRAIN subsample clustering stability.
7. Record per-seed candidate results.
8. Aggregate each method/k candidate across all SSL training seeds.
9. Select one global SSL clustering configuration using TRAIN/VALIDATION only.

TEST protection
---------------
- This file has no final-test command.
- It never constructs a TEST embedding path.
- ``load_partition`` refuses any partition except TRAIN/VALIDATION.
- It scans the SSL embedding root and refuses to run if TEST export artifacts
  are found.
- Final TEST evaluation belongs in a separate, explicitly authorized stage.

Default inputs
--------------
data/processed/DS-005/ssl/
    seed11/
        train_embeddings.npz
        validation_embeddings.npz
        train_manifest.json
        validation_manifest.json
    seed23/
    seed37/
    seed51/
    seed79/

Default outputs
---------------
data/processed/DS-005/ssl_clustering/
    selection_results.json
    selected_configuration.json
    pca_diagnostics.json
    SELECTION_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/discovery/ssl_clustering.py select

Intentional rerun:

    PYTHONPATH=. python3 src/discovery/ssl_clustering.py select --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "ssl_clustering.py requires numpy, PyYAML, and scikit-learn. "
        "Install project dependencies before running."
    ) from exc


Method = Literal["kmeans", "gmm"]
Partition = Literal["train", "validation"]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDING_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_clustering"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_CLUSTERING_CONFIG = (
    REPO_ROOT / "configs" / "ssl" / "ssl_clustering.yaml"
)

DEFAULT_SSL_SEEDS: Tuple[int, ...] = (11, 23, 37, 51, 79)
DEFAULT_CLUSTERING_SEED = 20260822
DEFAULT_K_VALUES: Tuple[int, ...] = tuple(range(2, 13))
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_SILHOUETTE_SAMPLE = 20_000
DEFAULT_STABILITY_SAMPLE = 20_000
DEFAULT_STABILITY_REPEATS = 3
EXPECTED_EMBEDDING_DIM = 64
EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}

# Mirrors the baseline discovery selection rule:
# 0.60 validation silhouette
# 0.30 stability ARI
# 0.10 validation occupancy reward
# -1.00 if an empty validation cluster exists
SILHOUETTE_WEIGHT = 0.60
STABILITY_WEIGHT = 0.30
OCCUPANCY_WEIGHT = 0.10
EMPTY_CLUSTER_PENALTY = 1.00


@dataclass(frozen=True)
class CandidateResult:
    ssl_seed: int
    method: Method
    k: int
    clustering_seed: int
    validation_silhouette: float
    stability_ari: float
    train_min_cluster_fraction: float
    validation_min_cluster_fraction: float
    train_empty_clusters: int
    validation_empty_clusters: int
    validation_confidence: float
    train_bic: Optional[float]
    train_aic: Optional[float]
    validation_log_likelihood: Optional[float]
    selection_score: float


@dataclass(frozen=True)
class AggregatedCandidate:
    method: Method
    k: int
    ssl_seed_count: int
    mean_selection_score: float
    std_selection_score: float
    mean_validation_silhouette: float
    std_validation_silhouette: float
    mean_stability_ari: float
    std_stability_ari: float
    mean_validation_min_cluster_fraction: float
    max_validation_empty_clusters: int
    per_ssl_seed_scores: Mapping[str, float]


@dataclass(frozen=True)
class SelectedConfiguration:
    method: Method
    k: int
    clustering_seed: int
    ssl_training_seeds: Tuple[int, ...]
    pca_variance_target: float
    standardize_fit_partition: str
    pca_fit_partition: str
    mean_selection_score: float
    mean_validation_silhouette: float
    mean_stability_ari: float
    test_partition_used: bool


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
        temp_name = handle.name

    os.replace(temp_name, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(path: Path, artifacts: Sequence[Path]) -> None:
    text = "".join(
        f"{sha256_file(artifact)}  {artifact.name}\n"
        for artifact in artifacts
    )
    path.write_text(text, encoding="utf-8")


def _parse_int_list(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(
            int(item.strip())
            for item in text.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated integer list."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError("Integer list cannot be empty.")

    return values


def load_yaml_if_present(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    return obj if isinstance(obj, dict) else {}


def configured_ssl_seeds(training_config: Path) -> Tuple[int, ...]:
    obj = load_yaml_if_present(training_config)
    training = obj.get("training", {}) if isinstance(obj, dict) else {}
    seeds = training.get("seeds", {}).get("values")

    if isinstance(seeds, list) and seeds:
        return tuple(int(seed) for seed in seeds)

    return DEFAULT_SSL_SEEDS


def assert_no_test_artifacts(embedding_dir: Path) -> None:
    """Refuse selection if TEST embedding exports exist under this root."""
    hits: List[Path] = []

    if not embedding_dir.exists():
        raise FileNotFoundError(embedding_dir)

    for path in embedding_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            name.startswith("test_")
            or "_test_" in name
            or name == "test.npz"
            or name == "test.npy"
        ):
            hits.append(path)

    if hits:
        display = "\n".join(str(path) for path in hits[:20])
        raise RuntimeError(
            "Protected TEST artifacts were found beneath the SSL embedding "
            "directory. Model selection is refusing to continue:\n"
            f"{display}"
        )


def load_partition(
    embedding_dir: Path,
    *,
    ssl_seed: int,
    partition: Partition,
) -> np.ndarray:
    if partition not in ("train", "validation"):
        raise RuntimeError(
            f"Protected/unknown partition {partition!r} requested. "
            "SSL clustering selection permits TRAIN and VALIDATION only."
        )

    path = (
        embedding_dir
        / f"seed{ssl_seed}"
        / f"{partition}_embeddings.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Run scripts/extract_ssl_embeddings.py first."
        )

    with np.load(path, allow_pickle=False) as npz:
        if "embeddings" not in npz.files:
            raise ValueError(f"{path} does not contain 'embeddings'.")
        X = np.asarray(npz["embeddings"], dtype=np.float32)

    expected_rows = EXPECTED_ROWS[partition]

    if X.shape != (expected_rows, EXPECTED_EMBEDDING_DIM):
        raise ValueError(
            f"{partition} seed {ssl_seed}: expected "
            f"({expected_rows}, {EXPECTED_EMBEDDING_DIM}), got {X.shape}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{partition} seed {ssl_seed} contains NaN/Inf."
        )

    return X


def verify_manifest(
    embedding_dir: Path,
    *,
    ssl_seed: int,
    partition: Partition,
) -> None:
    path = (
        embedding_dir
        / f"seed{ssl_seed}"
        / f"{partition}_manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    manifest = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "representation": "encoder_embedding",
        "projection_head_output_saved": False,
        "test_partition_loaded": False,
        "capped_debug_export": False,
        "partition": partition,
        "training_seed": ssl_seed,
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "rows": EXPECTED_ROWS[partition],
    }

    for key, expected in required.items():
        observed = manifest.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Manifest verification failed for seed {ssl_seed} "
                f"{partition}: {key}={observed!r}, expected {expected!r}."
            )


def fit_preprocessing(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    pca_variance: float,
    random_state: int,
) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray]:
    """Fit scaler and PCA using TRAIN only."""
    if not 0.0 < pca_variance <= 1.0:
        raise ValueError("pca_variance must be in (0, 1].")

    scaler = StandardScaler(copy=True)
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)

    pca = PCA(
        n_components=pca_variance,
        svd_solver="full",
        random_state=random_state,
    )
    train_pca = pca.fit_transform(train_scaled)
    validation_pca = pca.transform(validation_scaled)

    return scaler, pca, train_pca, validation_pca


def deterministic_indices(
    n_rows: int,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> float:
    labels = np.asarray(labels)

    if np.unique(labels).size < 2:
        return -1.0

    idx = deterministic_indices(
        len(labels),
        max_rows=max_rows,
        seed=seed,
    )
    sampled_labels = labels[idx]

    if np.unique(sampled_labels).size < 2:
        return -1.0

    return float(silhouette_score(X[idx], sampled_labels))


def cluster_occupancy(
    labels: np.ndarray,
    k: int,
) -> Tuple[float, int]:
    counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=k,
    )
    empty = int(np.sum(counts == 0))
    total = int(counts.sum())
    minimum_fraction = (
        float(np.min(counts) / total)
        if total > 0
        else 0.0
    )
    return minimum_fraction, empty


def kmeans_confidence(model: KMeans, X: np.ndarray) -> float:
    distances = model.transform(X)
    if distances.shape[1] < 2:
        return 0.0

    nearest_two = np.partition(distances, kth=1, axis=1)[:, :2]
    nearest = np.min(nearest_two, axis=1)
    second = np.max(nearest_two, axis=1)

    margin = (second - nearest) / np.maximum(second, 1e-12)
    return float(np.mean(margin))


def gmm_confidence(model: GaussianMixture, X: np.ndarray) -> float:
    posterior = model.predict_proba(X)
    return float(np.mean(np.max(posterior, axis=1)))


def clustering_stability(
    method: Method,
    *,
    k: int,
    train_x: np.ndarray,
    base_seed: int,
    repeats: int,
    max_rows: int,
) -> float:
    """Mean pairwise ARI across repeated fits on a fixed TRAIN subsample."""
    idx = deterministic_indices(
        train_x.shape[0],
        max_rows=max_rows,
        seed=base_seed,
    )
    X = train_x[idx]

    label_sets: List[np.ndarray] = []

    for repeat in range(repeats):
        seed = base_seed + repeat * 1009

        if method == "kmeans":
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
            )
            labels = model.fit_predict(X)

        elif method == "gmm":
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                n_init=1,
                reg_covar=1e-6,
            )
            labels = model.fit_predict(X)

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


def selection_score(
    *,
    validation_silhouette: float,
    stability_ari: float,
    validation_min_cluster_fraction: float,
    validation_empty_clusters: int,
) -> float:
    """Same discovery score used by the handcrafted baseline pipeline."""
    occupancy_reward = min(
        validation_min_cluster_fraction / 0.05,
        1.0,
    )

    return float(
        SILHOUETTE_WEIGHT * validation_silhouette
        + STABILITY_WEIGHT * stability_ari
        + OCCUPANCY_WEIGHT * occupancy_reward
        - EMPTY_CLUSTER_PENALTY
        * float(validation_empty_clusters > 0)
    )


def evaluate_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    ssl_seed: int,
    method: Method,
    k: int,
    clustering_seed: int,
    silhouette_sample: int,
    stability_sample: int,
    stability_repeats: int,
) -> CandidateResult:
    if method == "kmeans":
        model = KMeans(
            n_clusters=k,
            random_state=clustering_seed,
            n_init=10,
        )
        train_labels = model.fit_predict(train)
        validation_labels = model.predict(validation)

        confidence = kmeans_confidence(model, validation)
        train_bic = None
        train_aic = None
        validation_log_likelihood = None

    elif method == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=clustering_seed,
            n_init=1,
            reg_covar=1e-6,
        )
        model.fit(train)

        train_labels = model.predict(train)
        validation_labels = model.predict(validation)

        confidence = gmm_confidence(model, validation)
        train_bic = float(model.bic(train))
        train_aic = float(model.aic(train))
        validation_log_likelihood = float(model.score(validation))

    else:
        raise ValueError(method)

    train_min, train_empty = cluster_occupancy(train_labels, k)
    val_min, val_empty = cluster_occupancy(validation_labels, k)

    val_silhouette = safe_silhouette(
        validation,
        validation_labels,
        max_rows=silhouette_sample,
        seed=clustering_seed,
    )

    stability = clustering_stability(
        method,
        k=k,
        train_x=train,
        base_seed=clustering_seed,
        repeats=stability_repeats,
        max_rows=stability_sample,
    )

    score = selection_score(
        validation_silhouette=val_silhouette,
        stability_ari=stability,
        validation_min_cluster_fraction=val_min,
        validation_empty_clusters=val_empty,
    )

    return CandidateResult(
        ssl_seed=ssl_seed,
        method=method,
        k=k,
        clustering_seed=clustering_seed,
        validation_silhouette=val_silhouette,
        stability_ari=stability,
        train_min_cluster_fraction=train_min,
        validation_min_cluster_fraction=val_min,
        train_empty_clusters=train_empty,
        validation_empty_clusters=val_empty,
        validation_confidence=confidence,
        train_bic=train_bic,
        train_aic=train_aic,
        validation_log_likelihood=validation_log_likelihood,
        selection_score=score,
    )


def aggregate_candidates(
    results: Sequence[CandidateResult],
) -> List[AggregatedCandidate]:
    groups: Dict[Tuple[Method, int], List[CandidateResult]] = {}

    for result in results:
        groups.setdefault((result.method, result.k), []).append(result)

    aggregated: List[AggregatedCandidate] = []

    for (method, k), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        scores = np.asarray(
            [r.selection_score for r in group],
            dtype=np.float64,
        )
        silhouettes = np.asarray(
            [r.validation_silhouette for r in group],
            dtype=np.float64,
        )
        stability = np.asarray(
            [r.stability_ari for r in group],
            dtype=np.float64,
        )
        occupancy = np.asarray(
            [r.validation_min_cluster_fraction for r in group],
            dtype=np.float64,
        )

        aggregated.append(
            AggregatedCandidate(
                method=method,
                k=k,
                ssl_seed_count=len(group),
                mean_selection_score=float(np.mean(scores)),
                std_selection_score=float(np.std(scores)),
                mean_validation_silhouette=float(np.mean(silhouettes)),
                std_validation_silhouette=float(np.std(silhouettes)),
                mean_stability_ari=float(np.mean(stability)),
                std_stability_ari=float(np.std(stability)),
                mean_validation_min_cluster_fraction=float(
                    np.mean(occupancy)
                ),
                max_validation_empty_clusters=max(
                    r.validation_empty_clusters for r in group
                ),
                per_ssl_seed_scores={
                    str(r.ssl_seed): float(r.selection_score)
                    for r in group
                },
            )
        )

    return aggregated


def select_best(
    candidates: Sequence[AggregatedCandidate],
) -> AggregatedCandidate:
    if not candidates:
        raise ValueError("No candidate results to select from.")

    # Explicit tie-break order:
    # 1. higher mean composite score
    # 2. higher mean validation silhouette
    # 3. higher mean stability
    # 4. smaller k (parsimony)
    # 5. kmeans before gmm only as a final deterministic tie-break
    return max(
        candidates,
        key=lambda c: (
            c.mean_selection_score,
            c.mean_validation_silhouette,
            c.mean_stability_ari,
            -c.k,
            1 if c.method == "kmeans" else 0,
        ),
    )


def run_selection(
    *,
    embedding_dir: Path,
    output_dir: Path,
    ssl_seeds: Sequence[int],
    k_values: Sequence[int],
    methods: Sequence[Method],
    clustering_seed: int,
    pca_variance: float,
    silhouette_sample: int,
    stability_sample: int,
    stability_repeats: int,
    overwrite: bool,
) -> SelectedConfiguration:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "selection_results.json"
    selected_path = output_dir / "selected_configuration.json"
    pca_path = output_dir / "pca_diagnostics.json"
    checksum_path = output_dir / "SELECTION_SHA256SUMS"

    outputs = (results_path, selected_path, pca_path, checksum_path)

    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "SSL clustering selection outputs already exist. "
                "Use --overwrite only for an intentional rerun:\n"
                + "\n".join(str(path) for path in existing)
            )

    assert_no_test_artifacts(embedding_dir)

    print("SSL CLUSTERING MODEL SELECTION")
    print("=" * 80)
    print("TEST partition status: NOT LOADED")
    print(f"SSL training seeds: {list(ssl_seeds)}")
    print(f"Candidate methods: {list(methods)}")
    print(f"Candidate k: {list(k_values)}")
    print(f"Clustering seed: {clustering_seed}")
    print(
        "Selection score: "
        "0.60*validation_silhouette + "
        "0.30*TRAIN_stability + "
        "0.10*occupancy_reward"
    )
    print()

    all_results: List[CandidateResult] = []
    pca_diagnostics: Dict[str, Any] = {}

    for ssl_seed in ssl_seeds:
        print("=" * 80)
        print(f"SSL TRAINING SEED {ssl_seed}")
        print("=" * 80)

        verify_manifest(
            embedding_dir,
            ssl_seed=ssl_seed,
            partition="train",
        )
        verify_manifest(
            embedding_dir,
            ssl_seed=ssl_seed,
            partition="validation",
        )

        train = load_partition(
            embedding_dir,
            ssl_seed=ssl_seed,
            partition="train",
        )
        validation = load_partition(
            embedding_dir,
            ssl_seed=ssl_seed,
            partition="validation",
        )

        print(f"TRAIN rows:      {train.shape[0]:,}")
        print(f"VALIDATION rows: {validation.shape[0]:,}")
        print(f"Embedding dim:   {train.shape[1]}")
        print("Scaler fit:      TRAIN only")
        print("PCA fit:         TRAIN only")

        scaler, pca, train_z, validation_z = fit_preprocessing(
            train,
            validation,
            pca_variance=pca_variance,
            random_state=clustering_seed,
        )

        explained = np.asarray(
            pca.explained_variance_ratio_,
            dtype=np.float64,
        )

        pca_diagnostics[str(ssl_seed)] = {
            "ssl_seed": ssl_seed,
            "input_dim": int(train.shape[1]),
            "components_retained": int(pca.n_components_),
            "variance_target": float(pca_variance),
            "explained_variance_ratio": explained.tolist(),
            "cumulative_explained_variance": np.cumsum(
                explained
            ).tolist(),
            "total_variance_retained": float(np.sum(explained)),
            "standardizer_fit_partition": "train",
            "pca_fit_partition": "train",
            "test_partition_used": False,
        }

        print(
            f"PCA retained:    {pca.n_components_} components "
            f"({np.sum(explained):.4f} variance)"
        )
        print()

        for method in methods:
            for k in k_values:
                print(
                    f"Evaluating seed={ssl_seed} "
                    f"method={method} k={k} ...",
                    flush=True,
                )

                result = evaluate_candidate(
                    train_z,
                    validation_z,
                    ssl_seed=ssl_seed,
                    method=method,
                    k=k,
                    clustering_seed=clustering_seed,
                    silhouette_sample=silhouette_sample,
                    stability_sample=stability_sample,
                    stability_repeats=stability_repeats,
                )
                all_results.append(result)

                print(
                    f"  validation silhouette="
                    f"{result.validation_silhouette:.4f} | "
                    f"stability={result.stability_ari:.4f} | "
                    f"min occupancy="
                    f"{result.validation_min_cluster_fraction:.4f} | "
                    f"score={result.selection_score:.4f}",
                    flush=True,
                )

        # Release large arrays before the next encoder seed.
        del train, validation, train_z, validation_z, scaler, pca

    aggregated = aggregate_candidates(all_results)
    selected = select_best(aggregated)

    selected_config = SelectedConfiguration(
        method=selected.method,
        k=selected.k,
        clustering_seed=clustering_seed,
        ssl_training_seeds=tuple(int(s) for s in ssl_seeds),
        pca_variance_target=pca_variance,
        standardize_fit_partition="train",
        pca_fit_partition="train",
        mean_selection_score=selected.mean_selection_score,
        mean_validation_silhouette=(
            selected.mean_validation_silhouette
        ),
        mean_stability_ari=selected.mean_stability_ari,
        test_partition_used=False,
    )

    results_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "projection_head_used": False,
        "test_partition_used": False,
        "ssl_training_seeds": list(ssl_seeds),
        "candidate_methods": list(methods),
        "candidate_k_values": list(k_values),
        "clustering_seed": clustering_seed,
        "pca_variance_target": pca_variance,
        "silhouette_sample": silhouette_sample,
        "stability_sample": stability_sample,
        "stability_repeats": stability_repeats,
        "selection_rule": {
            "validation_silhouette_weight": SILHOUETTE_WEIGHT,
            "train_stability_weight": STABILITY_WEIGHT,
            "validation_occupancy_weight": OCCUPANCY_WEIGHT,
            "empty_cluster_penalty": EMPTY_CLUSTER_PENALTY,
            "occupancy_reward_cap_fraction": 0.05,
            "aggregation_across_ssl_seeds": "arithmetic_mean",
            "tie_break_order": [
                "higher_mean_selection_score",
                "higher_mean_validation_silhouette",
                "higher_mean_stability_ari",
                "smaller_k",
                "kmeans_final_deterministic_tiebreak",
            ],
        },
        "per_seed_candidates": [
            asdict(result) for result in all_results
        ],
        "aggregated_candidates": [
            asdict(candidate) for candidate in aggregated
        ],
    }

    atomic_write_json(results_path, results_payload)
    atomic_write_json(pca_path, pca_diagnostics)
    atomic_write_json(selected_path, asdict(selected_config))

    write_checksums(
        checksum_path,
        (results_path, pca_path, selected_path),
    )

    print()
    print("=" * 80)
    print("SELECTED SSL CLUSTERING CONFIGURATION")
    print("=" * 80)
    print(f"Method:                     {selected.method}")
    print(f"k:                          {selected.k}")
    print(
        f"Mean selection score:       "
        f"{selected.mean_selection_score:.6f}"
    )
    print(
        f"Mean validation silhouette: "
        f"{selected.mean_validation_silhouette:.6f}"
    )
    print(
        f"Mean TRAIN stability ARI:    "
        f"{selected.mean_stability_ari:.6f}"
    )
    print(f"SSL seeds represented:       {selected.ssl_seed_count}")
    print("TEST partition used:         NO")
    print()
    print(f"Results:   {results_path}")
    print(f"PCA:       {pca_path}")
    print(f"Selected:  {selected_path}")
    print(f"Checksums: {checksum_path}")

    return selected_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DS-005 SSL TRAIN/VALIDATION clustering selection "
            "without touching TEST."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser(
        "select",
        help="Run SSL clustering model selection.",
    )

    select.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
    )
    select.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    select.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    select.add_argument(
        "--ssl-seeds",
        type=_parse_int_list,
        default=None,
        help=(
            "Comma-separated SSL training seeds. "
            "Default: values from configs/ssl/training.yaml."
        ),
    )
    select.add_argument(
        "--k-values",
        type=_parse_int_list,
        default=DEFAULT_K_VALUES,
        help="Comma-separated candidate k values.",
    )
    select.add_argument(
        "--methods",
        nargs="+",
        choices=("kmeans", "gmm"),
        default=("kmeans", "gmm"),
    )
    select.add_argument(
        "--clustering-seed",
        type=int,
        default=DEFAULT_CLUSTERING_SEED,
    )
    select.add_argument(
        "--pca-variance",
        type=float,
        default=DEFAULT_PCA_VARIANCE,
    )
    select.add_argument(
        "--silhouette-sample",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE,
    )
    select.add_argument(
        "--stability-sample",
        type=int,
        default=DEFAULT_STABILITY_SAMPLE,
    )
    select.add_argument(
        "--stability-repeats",
        type=int,
        default=DEFAULT_STABILITY_REPEATS,
    )
    select.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command != "select":
        raise RuntimeError("Only TRAIN/VALIDATION selection is supported.")

    ssl_seeds = (
        args.ssl_seeds
        if args.ssl_seeds is not None
        else configured_ssl_seeds(args.training_config)
    )

    if not ssl_seeds:
        raise ValueError("No SSL seeds configured.")

    if any(k < 2 for k in args.k_values):
        raise ValueError("All candidate k values must be >= 2.")

    if args.stability_repeats < 2:
        raise ValueError("--stability-repeats must be >= 2.")

    run_selection(
        embedding_dir=args.embedding_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        ssl_seeds=ssl_seeds,
        k_values=args.k_values,
        methods=args.methods,
        clustering_seed=args.clustering_seed,
        pca_variance=args.pca_variance,
        silhouette_sample=args.silhouette_sample,
        stability_sample=args.stability_sample,
        stability_repeats=args.stability_repeats,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
