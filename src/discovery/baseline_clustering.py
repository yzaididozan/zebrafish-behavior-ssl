"""Baseline clustering for frozen DS-005 Input A features.

This module performs diagnostic PCA and preregistered clustering for the
hand-engineered DS-005 baseline representation.

Core protocol
-------------
1. Use the frozen scaled Input A feature matrices.
2. Fit PCA on TRAIN only.
3. Transform validation/test using the TRAIN-fitted PCA.
4. Select clustering method and number of clusters using TRAIN + VALIDATION
   only.
5. Do not use TEST for hyperparameter selection.
6. Permit TEST evaluation only through an explicit final-evaluation function.
7. Record all candidate metrics and random seeds.

Supported clustering methods
----------------------------
- KMeans
- GaussianMixture

Cluster-number selection
------------------------
Candidate k values are evaluated on training and validation data using:

KMeans:
- train silhouette
- validation silhouette
- validation assignment confidence proxy via distance margin
- train/validation cluster occupancy
- multi-seed stability using adjusted Rand index (ARI)

GaussianMixture:
- train silhouette
- validation silhouette
- validation average log likelihood
- validation posterior confidence
- BIC / AIC on training data
- train/validation cluster occupancy
- multi-seed stability using adjusted Rand index (ARI)

The default model-selection score prioritizes:
1. validation silhouette,
2. multi-seed stability,
3. validation non-degenerate occupancy,
while avoiding use of the held-out test partition.

This module intentionally does not declare a biological interpretation for
clusters. Clusters are candidate behavioral states until independently
validated.

Expected inputs
---------------
data/processed/DS-005/baseline/
├── train_core_scaled.npz
├── validation_core_scaled.npz
└── test_core_scaled.npz
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score, silhouette_score
except ImportError as exc:
    raise ImportError(
        "baseline_clustering.py requires scikit-learn. "
        "Install it with: python3 -m pip install scikit-learn"
    ) from exc


Method = Literal["kmeans", "gmm"]
Partition = Literal["train", "validation", "test"]

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline_clustering"
)

DEFAULT_K_VALUES: Tuple[int, ...] = tuple(range(2, 13))
DEFAULT_SEEDS: Tuple[int, ...] = (20260822, 20260823, 20260824, 20260825, 20260826)
DEFAULT_PCA_VARIANCE = 0.95

EXPECTED_FEATURE_COUNT = 18


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionMatrix:
    X: np.ndarray
    fish_id: np.ndarray
    partition: np.ndarray
    feature_names: Tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class PCAResult:
    model: PCA
    train: np.ndarray
    validation: np.ndarray
    test: Optional[np.ndarray]
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray


@dataclass(frozen=True)
class CandidateResult:
    method: Method
    k: int
    seed: int

    train_silhouette: float
    validation_silhouette: float

    train_min_cluster_fraction: float
    validation_min_cluster_fraction: float
    train_empty_clusters: int
    validation_empty_clusters: int

    validation_confidence: float

    train_bic: Optional[float] = None
    train_aic: Optional[float] = None
    validation_log_likelihood: Optional[float] = None


@dataclass(frozen=True)
class AggregatedCandidate:
    method: Method
    k: int
    n_seeds: int

    validation_silhouette_mean: float
    validation_silhouette_std: float

    train_silhouette_mean: float
    train_silhouette_std: float

    validation_confidence_mean: float
    validation_confidence_std: float

    stability_ari_mean: float
    stability_ari_std: float

    train_min_cluster_fraction_mean: float
    validation_min_cluster_fraction_mean: float

    train_empty_clusters_max: int
    validation_empty_clusters_max: int

    train_bic_mean: Optional[float]
    train_aic_mean: Optional[float]
    validation_log_likelihood_mean: Optional[float]

    selection_score: float


@dataclass(frozen=True)
class SelectedConfiguration:
    method: Method
    k: int
    seed: int
    selection_score: float
    pca_components: int
    pca_variance_target: float


@dataclass(frozen=True)
class FinalTestResult:
    method: Method
    k: int
    seed: int
    test_silhouette: float
    test_min_cluster_fraction: float
    test_empty_clusters: int
    test_confidence: float
    test_log_likelihood: Optional[float]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_partition_npz(path: Path, expected_partition: Partition) -> PartitionMatrix:
    """Load one frozen baseline NPZ safely."""
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {
            "X",
            "feature_names",
            "fish_id",
            "partition",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(
                f"{path.name} missing arrays: {sorted(missing)}"
            )

        X = np.asarray(data["X"], dtype=np.float64)
        feature_names = tuple(str(x) for x in data["feature_names"].tolist())
        fish_id = np.asarray(data["fish_id"]).astype(str)
        partition = np.asarray(data["partition"]).astype(str)

    if X.ndim != 2:
        raise RuntimeError(f"{path.name}: X must be two-dimensional.")

    if X.shape[0] != fish_id.shape[0] or X.shape[0] != partition.shape[0]:
        raise RuntimeError(f"{path.name}: metadata row-count mismatch.")

    observed = set(partition.tolist())
    if observed != {expected_partition}:
        raise RuntimeError(
            f"{path.name}: expected partition {expected_partition!r}, "
            f"observed {observed}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"{path.name}: non-finite feature values detected.")

    return PartitionMatrix(
        X=X,
        fish_id=fish_id,
        partition=partition,
        feature_names=feature_names,
    )


def load_frozen_baseline(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    *,
    load_test: bool = False,
) -> Dict[str, PartitionMatrix]:
    """Load frozen baseline partitions.

    By default test is deliberately NOT loaded.

    This is the recommended mode during model selection.
    """
    train = load_partition_npz(
        baseline_dir / "train_core_scaled.npz",
        "train",
    )
    validation = load_partition_npz(
        baseline_dir / "validation_core_scaled.npz",
        "validation",
    )

    if train.feature_names != validation.feature_names:
        raise RuntimeError("Train/validation feature schema mismatch.")

    if train.n_features != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} baseline features, "
            f"found {train.n_features}."
        )

    result: Dict[str, PartitionMatrix] = {
        "train": train,
        "validation": validation,
    }

    if load_test:
        test = load_partition_npz(
            baseline_dir / "test_core_scaled.npz",
            "test",
        )

        if train.feature_names != test.feature_names:
            raise RuntimeError("Train/test feature schema mismatch.")

        result["test"] = test

    assert_no_fish_overlap(result)
    return result


def assert_no_fish_overlap(
    partitions: Mapping[str, PartitionMatrix],
) -> None:
    fish = {
        name: set(matrix.fish_id.tolist())
        for name, matrix in partitions.items()
    }

    names = list(fish)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = fish[a] & fish[b]
            if overlap:
                raise RuntimeError(
                    f"Fish overlap between {a} and {b}: "
                    f"{sorted(overlap)[:10]}"
                )


def atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        temp_name = tmp.name

    os.replace(temp_name, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_diagnostic_pca(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    test: Optional[np.ndarray] = None,
    variance_target: float = DEFAULT_PCA_VARIANCE,
    random_state: int = 20260822,
) -> PCAResult:
    """Fit PCA on TRAIN only, then transform other partitions."""
    if not (0.0 < variance_target <= 1.0):
        raise ValueError("variance_target must be in (0, 1].")

    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("train and validation must be 2D.")

    if train.shape[1] != validation.shape[1]:
        raise ValueError("train/validation feature dimensions differ.")

    if test is not None and test.shape[1] != train.shape[1]:
        raise ValueError("train/test feature dimensions differ.")

    pca = PCA(
        n_components=variance_target,
        svd_solver="full",
        random_state=random_state,
    )

    train_z = pca.fit_transform(train)
    validation_z = pca.transform(validation)
    test_z = pca.transform(test) if test is not None else None

    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    return PCAResult(
        model=pca,
        train=train_z,
        validation=validation_z,
        test=test_z,
        explained_variance_ratio=explained,
        cumulative_explained_variance=np.cumsum(explained),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int = 50_000,
    random_state: int = 20260822,
) -> float:
    """Silhouette with deterministic subsampling for large matrices."""
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return float("-inf")

    if len(labels) <= max_samples:
        return float(silhouette_score(X, labels))

    rng = np.random.default_rng(random_state)
    idx = rng.choice(
        len(labels),
        size=max_samples,
        replace=False,
    )

    sampled_labels = labels[idx]

    # Extremely unlikely, but guard against losing a cluster in sampling.
    if len(np.unique(sampled_labels)) < 2:
        return float("-inf")

    return float(
        silhouette_score(
            X[idx],
            sampled_labels,
        )
    )


def cluster_occupancy(
    labels: np.ndarray,
    k: int,
) -> Tuple[float, int]:
    counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=k,
    )

    empty = int(np.sum(counts == 0))

    if counts.sum() == 0:
        return 0.0, empty

    return float(np.min(counts) / counts.sum()), empty


def kmeans_confidence(
    model: KMeans,
    X: np.ndarray,
) -> float:
    """Mean normalized nearest-vs-second-nearest distance margin."""
    distances = model.transform(X)

    if distances.shape[1] < 2:
        return 0.0

    sorted_dist = np.sort(distances, axis=1)
    nearest = sorted_dist[:, 0]
    second = sorted_dist[:, 1]

    denom = np.maximum(second, 1e-12)
    margin = (second - nearest) / denom

    return float(np.mean(margin))


def gmm_confidence(
    model: GaussianMixture,
    X: np.ndarray,
) -> float:
    """Mean maximum posterior component probability."""
    posterior = model.predict_proba(X)
    return float(np.mean(np.max(posterior, axis=1)))


def pairwise_seed_stability(
    labelings: Sequence[np.ndarray],
) -> Tuple[float, float]:
    """Mean/std pairwise ARI across seed-specific labelings."""
    if len(labelings) < 2:
        return 1.0, 0.0

    scores: List[float] = []

    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            scores.append(
                float(
                    adjusted_rand_score(
                        labelings[i],
                        labelings[j],
                    )
                )
            )

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Candidate fitting
# ---------------------------------------------------------------------------

def evaluate_kmeans_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    k: int,
    seed: int,
    n_init: int = 20,
) -> Tuple[CandidateResult, np.ndarray]:
    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
    )
    train_labels = model.fit_predict(train)
    validation_labels = model.predict(validation)

    train_min, train_empty = cluster_occupancy(train_labels, k)
    val_min, val_empty = cluster_occupancy(validation_labels, k)

    result = CandidateResult(
        method="kmeans",
        k=k,
        seed=seed,
        train_silhouette=_safe_silhouette(
            train,
            train_labels,
            random_state=seed,
        ),
        validation_silhouette=_safe_silhouette(
            validation,
            validation_labels,
            random_state=seed,
        ),
        train_min_cluster_fraction=train_min,
        validation_min_cluster_fraction=val_min,
        train_empty_clusters=train_empty,
        validation_empty_clusters=val_empty,
        validation_confidence=kmeans_confidence(
            model,
            validation,
        ),
    )

    return result, validation_labels


def evaluate_gmm_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    k: int,
    seed: int,
    covariance_type: str = "full",
    n_init: int = 5,
    reg_covar: float = 1e-6,
) -> Tuple[CandidateResult, np.ndarray]:
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=seed,
        n_init=n_init,
        reg_covar=reg_covar,
    )

    model.fit(train)

    train_labels = model.predict(train)
    validation_labels = model.predict(validation)

    train_min, train_empty = cluster_occupancy(train_labels, k)
    val_min, val_empty = cluster_occupancy(validation_labels, k)

    result = CandidateResult(
        method="gmm",
        k=k,
        seed=seed,
        train_silhouette=_safe_silhouette(
            train,
            train_labels,
            random_state=seed,
        ),
        validation_silhouette=_safe_silhouette(
            validation,
            validation_labels,
            random_state=seed,
        ),
        train_min_cluster_fraction=train_min,
        validation_min_cluster_fraction=val_min,
        train_empty_clusters=train_empty,
        validation_empty_clusters=val_empty,
        validation_confidence=gmm_confidence(
            model,
            validation,
        ),
        train_bic=float(model.bic(train)),
        train_aic=float(model.aic(train)),
        validation_log_likelihood=float(
            model.score(validation)
        ),
    )

    return result, validation_labels


# ---------------------------------------------------------------------------
# Aggregation and selection
# ---------------------------------------------------------------------------

def selection_score(
    *,
    validation_silhouette: float,
    stability_ari: float,
    validation_min_cluster_fraction: float,
    empty_clusters: int,
) -> float:
    """Protocol-level candidate score using validation information only.

    Weights are explicit and should be frozen before confirmatory selection.

    score =
        0.60 * validation silhouette
      + 0.30 * stability ARI
      + 0.10 * occupancy reward
      - 1.00 * empty-cluster penalty

    Occupancy reward is capped at 0.05 so a few balanced clusters do not
    dominate silhouette/stability.
    """
    occupancy_reward = min(
        validation_min_cluster_fraction / 0.05,
        1.0,
    )

    return float(
        0.60 * validation_silhouette
        + 0.30 * stability_ari
        + 0.10 * occupancy_reward
        - 1.00 * float(empty_clusters > 0)
    )


def aggregate_candidate_group(
    results: Sequence[CandidateResult],
    validation_labelings: Sequence[np.ndarray],
) -> AggregatedCandidate:
    if not results:
        raise ValueError("results cannot be empty.")

    methods = {r.method for r in results}
    ks = {r.k for r in results}

    if len(methods) != 1 or len(ks) != 1:
        raise ValueError("Candidate group must share method and k.")

    stability_mean, stability_std = pairwise_seed_stability(
        validation_labelings
    )

    val_sil = np.asarray(
        [r.validation_silhouette for r in results],
        dtype=float,
    )
    train_sil = np.asarray(
        [r.train_silhouette for r in results],
        dtype=float,
    )
    confidence = np.asarray(
        [r.validation_confidence for r in results],
        dtype=float,
    )

    train_min = np.asarray(
        [r.train_min_cluster_fraction for r in results],
        dtype=float,
    )
    val_min = np.asarray(
        [r.validation_min_cluster_fraction for r in results],
        dtype=float,
    )

    bic_values = [
        r.train_bic
        for r in results
        if r.train_bic is not None
    ]
    aic_values = [
        r.train_aic
        for r in results
        if r.train_aic is not None
    ]
    ll_values = [
        r.validation_log_likelihood
        for r in results
        if r.validation_log_likelihood is not None
    ]

    empty_max = max(r.validation_empty_clusters for r in results)

    score = selection_score(
        validation_silhouette=float(np.mean(val_sil)),
        stability_ari=stability_mean,
        validation_min_cluster_fraction=float(np.mean(val_min)),
        empty_clusters=empty_max,
    )

    return AggregatedCandidate(
        method=results[0].method,
        k=results[0].k,
        n_seeds=len(results),
        validation_silhouette_mean=float(np.mean(val_sil)),
        validation_silhouette_std=float(np.std(val_sil)),
        train_silhouette_mean=float(np.mean(train_sil)),
        train_silhouette_std=float(np.std(train_sil)),
        validation_confidence_mean=float(np.mean(confidence)),
        validation_confidence_std=float(np.std(confidence)),
        stability_ari_mean=stability_mean,
        stability_ari_std=stability_std,
        train_min_cluster_fraction_mean=float(np.mean(train_min)),
        validation_min_cluster_fraction_mean=float(np.mean(val_min)),
        train_empty_clusters_max=max(
            r.train_empty_clusters for r in results
        ),
        validation_empty_clusters_max=empty_max,
        train_bic_mean=(
            float(np.mean(bic_values))
            if bic_values
            else None
        ),
        train_aic_mean=(
            float(np.mean(aic_values))
            if aic_values
            else None
        ),
        validation_log_likelihood_mean=(
            float(np.mean(ll_values))
            if ll_values
            else None
        ),
        selection_score=score,
    )


def run_model_selection(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    methods: Sequence[Method] = ("kmeans", "gmm"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Tuple[List[CandidateResult], List[AggregatedCandidate], SelectedConfiguration]:
    """Select method/k using TRAIN + VALIDATION only."""
    if not k_values:
        raise ValueError("k_values cannot be empty.")

    if any(k < 2 for k in k_values):
        raise ValueError("All k values must be >= 2.")

    if not seeds:
        raise ValueError("seeds cannot be empty.")

    individual: List[CandidateResult] = []
    aggregated: List[AggregatedCandidate] = []

    grouped: Dict[Tuple[str, int], Tuple[List[CandidateResult], List[np.ndarray]]] = {}

    for method in methods:
        if method not in {"kmeans", "gmm"}:
            raise ValueError(f"Unsupported method: {method}")

        for k in k_values:
            results: List[CandidateResult] = []
            labelings: List[np.ndarray] = []

            for seed in seeds:
                if method == "kmeans":
                    result, labels = evaluate_kmeans_candidate(
                        train,
                        validation,
                        k=k,
                        seed=seed,
                    )
                else:
                    result, labels = evaluate_gmm_candidate(
                        train,
                        validation,
                        k=k,
                        seed=seed,
                    )

                individual.append(result)
                results.append(result)
                labelings.append(labels)

            grouped[(method, k)] = (results, labelings)

            aggregated.append(
                aggregate_candidate_group(
                    results,
                    labelings,
                )
            )

    # Highest validation-derived score wins.
    ranked = sorted(
        aggregated,
        key=lambda x: (
            x.selection_score,
            x.validation_silhouette_mean,
            x.stability_ari_mean,
            -x.k,  # conservative tie-break toward smaller k
        ),
        reverse=True,
    )

    winner = ranked[0]

    # Refit with the first preregistered seed for deterministic final model.
    selected = SelectedConfiguration(
        method=winner.method,
        k=winner.k,
        seed=int(seeds[0]),
        selection_score=winner.selection_score,
        pca_components=train.shape[1],
        pca_variance_target=DEFAULT_PCA_VARIANCE,
    )

    return individual, aggregated, selected


# ---------------------------------------------------------------------------
# Final model fitting
# ---------------------------------------------------------------------------

def fit_selected_model(
    train: np.ndarray,
    validation: np.ndarray,
    config: SelectedConfiguration,
):
    """Refit selected model using TRAIN + VALIDATION after selection.

    This is allowed only after method/k selection has been completed.
    The held-out TEST partition remains unused.
    """
    combined = np.concatenate(
        [train, validation],
        axis=0,
    )

    if config.method == "kmeans":
        model = KMeans(
            n_clusters=config.k,
            random_state=config.seed,
            n_init=20,
        )
        model.fit(combined)
        return model

    model = GaussianMixture(
        n_components=config.k,
        covariance_type="full",
        random_state=config.seed,
        n_init=5,
        reg_covar=1e-6,
    )
    model.fit(combined)
    return model


def evaluate_final_test(
    model,
    test: np.ndarray,
    config: SelectedConfiguration,
    *,
    final_evaluation_confirmed: bool = False,
) -> FinalTestResult:
    """Evaluate the final selected model on held-out TEST.

    The explicit confirmation argument is intentionally required to make
    accidental test use during model selection harder.
    """
    if not final_evaluation_confirmed:
        raise PermissionError(
            "Held-out TEST evaluation is locked. "
            "Set final_evaluation_confirmed=True only after model-selection "
            "configuration has been frozen."
        )

    labels = model.predict(test)

    min_fraction, empty = cluster_occupancy(
        labels,
        config.k,
    )

    silhouette = _safe_silhouette(
        test,
        labels,
        random_state=config.seed,
    )

    if config.method == "kmeans":
        confidence = kmeans_confidence(
            model,
            test,
        )
        log_likelihood = None
    else:
        confidence = gmm_confidence(
            model,
            test,
        )
        log_likelihood = float(
            model.score(test)
        )

    return FinalTestResult(
        method=config.method,
        k=config.k,
        seed=config.seed,
        test_silhouette=silhouette,
        test_min_cluster_fraction=min_fraction,
        test_empty_clusters=empty,
        test_confidence=confidence,
        test_log_likelihood=log_likelihood,
    )


# ---------------------------------------------------------------------------
# Selection workflow
# ---------------------------------------------------------------------------

def run_selection_workflow(
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    methods: Sequence[Method] = ("kmeans", "gmm"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pca_variance_target: float = DEFAULT_PCA_VARIANCE,
    overwrite: bool = False,
) -> SelectedConfiguration:
    """Run PCA + clustering selection without opening the test matrix."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = output_dir / "selection_results.json"
    pca_path = output_dir / "pca_diagnostics.json"
    selected_path = output_dir / "selected_configuration.json"

    outputs = [selection_path, pca_path, selected_path]

    if not overwrite:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(
                "Selection outputs already exist. "
                "Use --overwrite only for an intentional rerun:\n"
                + "\n".join(str(p) for p in existing)
            )

    # Critical: test is not loaded during selection.
    partitions = load_frozen_baseline(
        baseline_dir,
        load_test=False,
    )

    train = partitions["train"]
    validation = partitions["validation"]

    print("BASELINE CLUSTERING MODEL SELECTION")
    print("===================================")
    print("TEST partition status: NOT LOADED")
    print(f"Train rows:      {train.n_rows:,}")
    print(f"Validation rows: {validation.n_rows:,}")
    print(f"Input features:  {train.n_features}")
    print()

    pca = fit_diagnostic_pca(
        train.X,
        validation.X,
        variance_target=pca_variance_target,
        random_state=int(seeds[0]),
    )

    print(
        f"PCA fit on TRAIN only: "
        f"{pca.train.shape[1]} components retain "
        f"{pca.cumulative_explained_variance[-1]:.4f} variance"
    )

    individual, aggregated, selected = run_model_selection(
        pca.train,
        pca.validation,
        k_values=k_values,
        methods=methods,
        seeds=seeds,
    )

    # Correct PCA metadata in selected config.
    selected = SelectedConfiguration(
        method=selected.method,
        k=selected.k,
        seed=selected.seed,
        selection_score=selected.selection_score,
        pca_components=int(pca.train.shape[1]),
        pca_variance_target=pca_variance_target,
    )

    ranked = sorted(
        aggregated,
        key=lambda x: x.selection_score,
        reverse=True,
    )

    print()
    print("Top candidates:")
    for candidate in ranked[:10]:
        print(
            f"  {candidate.method:6s} "
            f"k={candidate.k:2d} "
            f"score={candidate.selection_score: .4f} "
            f"val_sil={candidate.validation_silhouette_mean: .4f} "
            f"stability={candidate.stability_ari_mean: .4f}"
        )

    print()
    print("Selected configuration:")
    print(f"  method: {selected.method}")
    print(f"  k:      {selected.k}")
    print(f"  seed:   {selected.seed}")
    print(f"  score:  {selected.selection_score:.6f}")
    print()
    print("TEST partition remains untouched.")

    pca_payload = {
        "fit_partition": "train",
        "test_loaded": False,
        "variance_target": pca_variance_target,
        "n_input_features": train.n_features,
        "n_components": int(pca.train.shape[1]),
        "explained_variance_ratio": (
            pca.explained_variance_ratio.tolist()
        ),
        "cumulative_explained_variance": (
            pca.cumulative_explained_variance.tolist()
        ),
    }

    selection_payload = {
        "protocol": {
            "selection_partitions": [
                "train",
                "validation",
            ],
            "test_loaded": False,
            "methods": list(methods),
            "k_values": list(k_values),
            "seeds": list(seeds),
            "selection_score": {
                "validation_silhouette_weight": 0.60,
                "stability_ari_weight": 0.30,
                "occupancy_weight": 0.10,
                "empty_cluster_penalty": 1.00,
                "occupancy_reward_cap_fraction": 0.05,
            },
        },
        "individual_candidates": [
            asdict(x)
            for x in individual
        ],
        "aggregated_candidates": [
            asdict(x)
            for x in ranked
        ],
    }

    atomic_json_dump(
        pca_payload,
        pca_path,
    )
    atomic_json_dump(
        selection_payload,
        selection_path,
    )
    atomic_json_dump(
        asdict(selected),
        selected_path,
    )

    hashes = {
        path.name: sha256_file(path)
        for path in outputs
    }

    hash_path = output_dir / "SELECTION_SHA256SUMS"
    hash_path.write_text(
        "\n".join(
            f"{digest}  {name}"
            for name, digest in sorted(hashes.items())
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Selection artifacts:")
    for path in outputs:
        print(f"  {path}")
    print(f"  {hash_path}")

    return selected


# ---------------------------------------------------------------------------
# Final held-out test workflow
# ---------------------------------------------------------------------------

def run_final_test_workflow(
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    selection_dir: Path = DEFAULT_OUTPUT_DIR,
    output_path: Optional[Path] = None,
    final_evaluation_confirmed: bool = False,
) -> FinalTestResult:
    """Run held-out test only after selection configuration is frozen."""
    if not final_evaluation_confirmed:
        raise PermissionError(
            "Final test workflow locked. Use --final-evaluation-confirmed "
            "only after recording/fixing the selected configuration."
        )

    config_path = (
        selection_dir / "selected_configuration.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            "No frozen selected_configuration.json found."
        )

    payload = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    config = SelectedConfiguration(**payload)

    # Only now is TEST loaded.
    partitions = load_frozen_baseline(
        baseline_dir,
        load_test=True,
    )

    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]

    # Refit PCA on TRAIN only, using frozen component count.
    pca = PCA(
        n_components=config.pca_components,
        svd_solver="full",
        random_state=config.seed,
    )
    train_z = pca.fit_transform(train.X)
    validation_z = pca.transform(validation.X)
    test_z = pca.transform(test.X)

    model = fit_selected_model(
        train_z,
        validation_z,
        config,
    )

    result = evaluate_final_test(
        model,
        test_z,
        config,
        final_evaluation_confirmed=True,
    )

    if output_path is None:
        output_path = (
            selection_dir / "final_test_result.json"
        )

    atomic_json_dump(
        asdict(result),
        output_path,
    )

    print("FINAL HELD-OUT TEST EVALUATION")
    print("==============================")
    print(f"method:      {result.method}")
    print(f"k:           {result.k}")
    print(f"silhouette:  {result.test_silhouette:.6f}")
    print(f"confidence:  {result.test_confidence:.6f}")
    print(
        f"min cluster: {result.test_min_cluster_fraction:.6f}"
    )
    print(
        f"empty:       {result.test_empty_clusters}"
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(text: str) -> Tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in text.split(",")
        if item.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DS-005 baseline PCA and clustering selection "
            "without touching the held-out test partition."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    select = sub.add_parser(
        "select",
        help=(
            "Run TRAIN/VALIDATION-only PCA and clustering "
            "model selection."
        ),
    )
    select.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )
    select.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    select.add_argument(
        "--k-values",
        type=_parse_int_list,
        default=DEFAULT_K_VALUES,
        help="Comma-separated candidate k values, e.g. 2,3,4,5,6.",
    )
    select.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated random seeds.",
    )
    select.add_argument(
        "--pca-variance",
        type=float,
        default=DEFAULT_PCA_VARIANCE,
    )
    select.add_argument(
        "--methods",
        nargs="+",
        choices=["kmeans", "gmm"],
        default=["kmeans", "gmm"],
    )
    select.add_argument(
        "--overwrite",
        action="store_true",
    )

    final = sub.add_parser(
        "final-test",
        help=(
            "Evaluate the already-selected configuration on TEST. "
            "Use only after selection is frozen."
        ),
    )
    final.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )
    final.add_argument(
        "--selection-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    final.add_argument(
        "--final-evaluation-confirmed",
        action="store_true",
        help=(
            "Required acknowledgement that model selection is frozen "
            "and held-out test evaluation is now authorized."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "select":
        run_selection_workflow(
            baseline_dir=args.baseline_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            k_values=args.k_values,
            methods=args.methods,
            seeds=args.seeds,
            pca_variance_target=args.pca_variance,
            overwrite=args.overwrite,
        )
        return

    run_final_test_workflow(
        baseline_dir=args.baseline_dir.resolve(),
        selection_dir=args.selection_dir.resolve(),
        final_evaluation_confirmed=(
            args.final_evaluation_confirmed
        ),
    )


if __name__ == "__main__":
    main()
