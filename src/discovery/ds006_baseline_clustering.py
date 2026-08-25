#!/usr/bin/env python3
"""Apply the frozen DS-005 handcrafted baseline clustering recipe to DS-006.

Frozen replication recipe
-------------------------
- 18 handcrafted features
- TRAIN-only median imputation
- TRAIN-only StandardScaler
- PCA with exactly 6 components
- GaussianMixture with exactly k=2
- random_state=20260822
- fit preprocessing and GMM on TRAIN only
- predict TRAIN and VALIDATION
- no method selection
- no k selection
- no PCA-dimension selection
- TEST never loaded

Outputs
-------
data/processed/DS-006/baseline_clustering/
    train_labels.npy
    validation_labels.npy
    train_bout_id.npy
    validation_bout_id.npy
    imputer.joblib
    scaler.joblib
    pca.joblib
    gmm.joblib
    pca_explained_variance_ratio.npy
    summary.json
    manifest.json
    DS006_BASELINE_CLUSTERING_SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-006" / "baseline"
)

OUTPUT_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-006" / "baseline_clustering"
)

FEATURE_MANIFEST = BASELINE_ROOT / "feature_manifest.json"

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

EXPECTED_ROWS = {
    "train": 118_100,
    "validation": 18_835,
}

FROZEN_PCA_COMPONENTS = 6
FROZEN_GMM_K = 2
FROZEN_SEED = 20260822

# Diagnostic only; never used for selection.
SILHOUETTE_SAMPLE_SIZE = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_paths(baseline_root: Path, output_root: Path) -> None:
    if baseline_root.resolve() != BASELINE_ROOT.resolve():
        raise RuntimeError(
            "--baseline-root must resolve exactly to "
            f"{BASELINE_ROOT.resolve()}"
        )

    ds006_root = (
        REPO_ROOT / "data" / "processed" / "DS-006"
    ).resolve()

    try:
        output_root.resolve().relative_to(ds006_root)
    except ValueError as exc:
        raise RuntimeError(
            "Outputs must remain under data/processed/DS-006."
        ) from exc

    if "test" in str(output_root).lower():
        raise RuntimeError("Output path unexpectedly contains TEST.")


def load_manifest() -> Dict[str, Any]:
    if not FEATURE_MANIFEST.exists():
        raise FileNotFoundError(FEATURE_MANIFEST)

    obj = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))

    if obj.get("feature_names") != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Frozen DS-006 feature order differs from expected 18-feature order."
        )

    return obj


def load_partition(
    baseline_root: Path,
    partition: str,
) -> Dict[str, Any]:
    if partition not in ("train", "validation"):
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = baseline_root / f"{partition}_core_raw.npz"

    if "test" in path.name.lower():
        raise RuntimeError("Protected TEST path reached.")

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        if "X" not in npz.files or "bout_id" not in npz.files:
            raise RuntimeError(f"{path}: expected X and bout_id arrays.")

        X = np.asarray(npz["X"], dtype=np.float64)
        bout_id = np.asarray(npz["bout_id"]).astype(str)

    expected = EXPECTED_ROWS[partition]

    if X.shape != (expected, 18):
        raise RuntimeError(
            f"{path}: expected shape {(expected, 18)}, got {X.shape}."
        )

    if bout_id.shape != (expected,):
        raise RuntimeError(
            f"{path}: unexpected bout_id shape {bout_id.shape}."
        )

    if len(np.unique(bout_id)) != expected:
        raise RuntimeError(f"{path}: duplicate bout IDs.")

    return {
        "path": path,
        "sha256": sha256_file(path),
        "X": X,
        "bout_id": bout_id,
    }


def occupancy(labels: np.ndarray) -> Dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    total = labels.size

    return {
        str(int(label)): {
            "count": int(count),
            "fraction": float(count / total),
        }
        for label, count in zip(values, counts)
    }


def sampled_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    if np.unique(labels).size < 2:
        return 0.0

    if X.shape[0] <= SILHOUETTE_SAMPLE_SIZE:
        return float(silhouette_score(X, labels))

    rng = np.random.default_rng(FROZEN_SEED)
    idx = rng.choice(
        X.shape[0],
        size=SILHOUETTE_SAMPLE_SIZE,
        replace=False,
    )

    return float(silhouette_score(X[idx], labels[idx]))


def write_checksums(
    output_root: Path,
    files: Sequence[Path],
) -> Path:
    checksum_path = (
        output_root / "DS006_BASELINE_CLUSTERING_SHA256SUMS"
    )

    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_root)}\n"
            for path in sorted(files, key=lambda x: str(x))
        ),
        encoding="utf-8",
    )

    return checksum_path


def main() -> None:
    args = parse_args()

    baseline_root = args.baseline_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    verify_paths(baseline_root, output_root)

    summary_path = output_root / "summary.json"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    manifest = load_manifest()
    train = load_partition(baseline_root, "train")
    validation = load_partition(baseline_root, "validation")

    print("=" * 80)
    print("DS-006 FROZEN HANDCRAFTED BASELINE CLUSTERING")
    print("=" * 80)
    print("Scientific mode:    frozen-method replication")
    print("Features:           18 handcrafted")
    print("Imputation:         median, fit TRAIN only")
    print("Scaling:            StandardScaler, fit TRAIN only")
    print(f"PCA components:     {FROZEN_PCA_COMPONENTS} (frozen)")
    print("PCA selection:      NONE")
    print("Clustering:         GaussianMixture")
    print(f"k:                  {FROZEN_GMM_K} (frozen)")
    print(f"Random seed:        {FROZEN_SEED}")
    print("Method/k selection: NONE")
    print("TEST partition:     PROTECTED / NOT LOADED")
    print()
    print(f"TRAIN rows:         {train['X'].shape[0]:,}")
    print(f"VALIDATION rows:    {validation['X'].shape[0]:,}")
    print()

    # TRAIN-only preprocessing.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(train["X"])
    X_val_imp = imputer.transform(validation["X"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    pca = PCA(
        n_components=FROZEN_PCA_COMPONENTS,
        random_state=FROZEN_SEED,
    )
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    retained_variance = float(pca.explained_variance_ratio_.sum())

    # Frozen GMM(k=2).
    gmm = GaussianMixture(
        n_components=FROZEN_GMM_K,
        covariance_type="full",
        random_state=FROZEN_SEED,
    )
    gmm.fit(X_train_pca)

    train_labels = np.asarray(
        gmm.predict(X_train_pca),
        dtype=np.int64,
    )
    validation_labels = np.asarray(
        gmm.predict(X_val_pca),
        dtype=np.int64,
    )

    train_counts = np.bincount(
        train_labels,
        minlength=FROZEN_GMM_K,
    )
    validation_counts = np.bincount(
        validation_labels,
        minlength=FROZEN_GMM_K,
    )

    train_silhouette = sampled_silhouette(
        X_train_pca,
        train_labels,
    )
    validation_silhouette = sampled_silhouette(
        X_val_pca,
        validation_labels,
    )

    print(f"PCA retained variance: {retained_variance:.6f}")
    print(f"TRAIN counts:          {train_counts.tolist()}")
    print(f"VALIDATION counts:     {validation_counts.tolist()}")
    print(f"TRAIN silhouette:      {train_silhouette:.6f}")
    print(f"VALIDATION silhouette: {validation_silhouette:.6f}")
    print(f"GMM converged:         {bool(gmm.converged_)}")
    print(f"GMM iterations:        {int(gmm.n_iter_)}")
    print("TEST partition used:   NO")
    print()

    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "train_labels": output_root / "train_labels.npy",
        "validation_labels": output_root / "validation_labels.npy",
        "train_bout_id": output_root / "train_bout_id.npy",
        "validation_bout_id": output_root / "validation_bout_id.npy",
        "imputer": output_root / "imputer.joblib",
        "scaler": output_root / "scaler.joblib",
        "pca": output_root / "pca.joblib",
        "gmm": output_root / "gmm.joblib",
        "pca_evr": output_root / "pca_explained_variance_ratio.npy",
        "summary": summary_path,
        "manifest": output_root / "manifest.json",
    }

    np.save(paths["train_labels"], train_labels, allow_pickle=False)
    np.save(
        paths["validation_labels"],
        validation_labels,
        allow_pickle=False,
    )
    np.save(
        paths["train_bout_id"],
        train["bout_id"],
        allow_pickle=False,
    )
    np.save(
        paths["validation_bout_id"],
        validation["bout_id"],
        allow_pickle=False,
    )
    np.save(
        paths["pca_evr"],
        pca.explained_variance_ratio_,
        allow_pickle=False,
    )

    joblib.dump(imputer, paths["imputer"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(pca, paths["pca"])
    joblib.dump(gmm, paths["gmm"])

    save_json(
        paths["manifest"],
        {
            "dataset_id": "DS-006",
            "analysis": "frozen_handcrafted_baseline_clustering",
            "scientific_mode": "frozen_method_replication",
            "feature_count": 18,
            "feature_manifest": str(
                FEATURE_MANIFEST.relative_to(REPO_ROOT)
            ),
            "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST),
            "imputation": "median fit on TRAIN only",
            "scaling": "StandardScaler fit on TRAIN only",
            "pca_components": FROZEN_PCA_COMPONENTS,
            "pca_fit_partition": "train",
            "method": "GaussianMixture",
            "k": FROZEN_GMM_K,
            "gmm_fit_partition": "train",
            "random_seed": FROZEN_SEED,
            "model_selection_performed": False,
            "k_selection_performed": False,
            "pca_dimension_selection_performed": False,
            "train_source": str(train["path"].relative_to(REPO_ROOT)),
            "validation_source": str(
                validation["path"].relative_to(REPO_ROOT)
            ),
            "train_source_sha256": train["sha256"],
            "validation_source_sha256": validation["sha256"],
            "test_partition_used": False,
        },
    )

    save_json(
        paths["summary"],
        {
            "dataset_id": "DS-006",
            "analysis": "frozen_handcrafted_baseline_clustering",
            "scientific_mode": "frozen_method_replication",
            "train_rows": int(train["X"].shape[0]),
            "validation_rows": int(validation["X"].shape[0]),
            "feature_count": 18,
            "pca_components": FROZEN_PCA_COMPONENTS,
            "pca_retained_variance": retained_variance,
            "gmm_k": FROZEN_GMM_K,
            "gmm_converged": bool(gmm.converged_),
            "gmm_iterations": int(gmm.n_iter_),
            "train_counts": train_counts.astype(int).tolist(),
            "validation_counts": validation_counts.astype(int).tolist(),
            "train_occupancy": occupancy(train_labels),
            "validation_occupancy": occupancy(validation_labels),
            "train_silhouette": train_silhouette,
            "validation_silhouette": validation_silhouette,
            "model_selection_performed": False,
            "k_selection_performed": False,
            "pca_dimension_selection_performed": False,
            "test_partition_used": False,
        },
    )

    checksum_path = write_checksums(
        output_root,
        list(paths.values()),
    )

    print("=" * 80)
    print("DS-006 BASELINE CLUSTERING SUMMARY")
    print("=" * 80)
    print(f"PCA components:        {FROZEN_PCA_COMPONENTS}")
    print(f"Retained variance:     {retained_variance:.6f}")
    print(f"GMM k:                 {FROZEN_GMM_K}")
    print(f"TRAIN counts:          {train_counts.tolist()}")
    print(f"VALIDATION counts:     {validation_counts.tolist()}")
    print(f"TRAIN silhouette:      {train_silhouette:.6f}")
    print(f"VALIDATION silhouette: {validation_silhouette:.6f}")
    print("Method/k selection:    NO")
    print("TEST partition used:   NO")
    print(f"Summary:   {summary_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()
