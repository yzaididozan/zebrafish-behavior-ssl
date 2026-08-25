#!/usr/bin/env python3
"""Cross-seed stability analysis for selected DS-005 SSL clusters.

Evaluates whether independently trained SSL encoders produce comparable
behavioral partitions under the already-selected KMeans(k=8) configuration.

TRAIN and VALIDATION only. TEST is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
DEFAULT_SELECTION_FILE = (
    REPO_ROOT / "data" / "processed" / "DS-005"
    / "ssl_clustering" / "selected_configuration.json"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_cluster_stability"
)

EXPECTED_ROWS = {"train": 842_841, "validation": 168_464}
EXPECTED_DIM = 64
PARTITIONS = ("train", "validation")
DEFAULT_REFERENCE_SEED = 11


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temp_name = handle.name
    os.replace(temp_name, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_checksums(path: Path, artifacts: Sequence[Path]) -> None:
    path.write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in artifacts),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return obj


def configured_ssl_seeds(training_config: Path) -> Tuple[int, ...]:
    if not training_config.exists():
        raise FileNotFoundError(training_config)
    with training_config.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    training = obj.get("training", {})
    seeds = training.get("seeds", {}).get("values")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No frozen SSL seeds found in training.yaml.")
    return tuple(int(seed) for seed in seeds)


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
            or name in {"test.npz", "test.npy"}
        ):
            hits.append(path)
    if hits:
        raise RuntimeError(
            "TEST artifacts found under SSL embedding root; refusing to continue:\n"
            + "\n".join(str(p) for p in hits[:20])
        )


def verify_manifest(
    embedding_dir: Path, *, ssl_seed: int, partition: str
) -> None:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")
    path = embedding_dir / f"seed{ssl_seed}" / f"{partition}_manifest.json"
    manifest = load_json(path)

    required = {
        "representation": "encoder_embedding",
        "projection_head_output_saved": False,
        "test_partition_loaded": False,
        "capped_debug_export": False,
        "partition": partition,
        "training_seed": ssl_seed,
        "embedding_dim": EXPECTED_DIM,
        "rows": EXPECTED_ROWS[partition],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Manifest mismatch seed={ssl_seed} partition={partition}: "
                f"{key}={manifest.get(key)!r}, expected={expected!r}"
            )


def load_embeddings(
    embedding_dir: Path, *, ssl_seed: int, partition: str
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError(
            f"Partition {partition!r} prohibited. TRAIN/VALIDATION only."
        )
    path = embedding_dir / f"seed{ssl_seed}" / f"{partition}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as npz:
        if "embeddings" not in npz.files:
            raise ValueError(f"{path} has no embeddings array.")
        X = np.asarray(npz["embeddings"], dtype=np.float32)

    expected = (EXPECTED_ROWS[partition], EXPECTED_DIM)
    if X.shape != expected:
        raise ValueError(
            f"seed {ssl_seed} {partition}: expected {expected}, got {X.shape}"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"seed {ssl_seed} {partition}: NaN/Inf detected.")
    return X


def fit_selected_clustering(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    k: int,
    clustering_seed: int,
    pca_variance_target: float,
):
    scaler = StandardScaler(copy=True)
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)

    pca = PCA(
        n_components=pca_variance_target,
        svd_solver="full",
        random_state=clustering_seed,
    )
    train_pca = pca.fit_transform(train_scaled)
    validation_pca = pca.transform(validation_scaled)

    model = KMeans(
        n_clusters=k,
        random_state=clustering_seed,
        n_init=10,
    )
    train_labels = model.fit_predict(train_pca).astype(np.int16, copy=False)
    validation_labels = model.predict(validation_pca).astype(np.int16, copy=False)

    diagnostics = {
        "pca_components": int(pca.n_components_),
        "pca_variance_retained": float(np.sum(pca.explained_variance_ratio_)),
        "input_dim": int(train.shape[1]),
        "train_rows": int(train.shape[0]),
        "validation_rows": int(validation.shape[0]),
        "k": int(k),
        "clustering_seed": int(clustering_seed),
        "test_partition_used": False,
    }

    return (
        train_labels,
        validation_labels,
        model.cluster_centers_.astype(np.float32, copy=False),
        np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        diagnostics,
    )


def cluster_sizes(labels: np.ndarray, k: int) -> Dict[str, Any]:
    counts = np.bincount(labels.astype(np.int64), minlength=k)
    fractions = counts / counts.sum()
    return {
        "counts": counts.astype(int).tolist(),
        "fractions": fractions.astype(float).tolist(),
        "min_fraction": float(np.min(fractions)),
        "max_fraction": float(np.max(fractions)),
        "empty_clusters": int(np.sum(counts == 0)),
    }


def hungarian_alignment(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    k: int,
):
    matrix = confusion_matrix(
        reference_labels, candidate_labels, labels=np.arange(k)
    )
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {
        int(candidate): int(reference)
        for reference, candidate in zip(row_ind, col_ind)
    }
    aligned = np.asarray(
        [mapping[int(label)] for label in candidate_labels], dtype=np.int16
    )
    aligned_matrix = confusion_matrix(
        reference_labels, aligned, labels=np.arange(k)
    )
    return mapping, aligned, aligned_matrix


def pairwise_metrics(
    labels_by_seed: Mapping[int, np.ndarray],
    *,
    k: int,
    partition: str,
) -> List[Dict[str, Any]]:
    seeds = sorted(labels_by_seed)
    results: List[Dict[str, Any]] = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            a = labels_by_seed[seed_a]
            b = labels_by_seed[seed_b]

            mapping, b_aligned, matrix = hungarian_alignment(a, b, k=k)

            results.append(
                {
                    "partition": partition,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "ari": float(adjusted_rand_score(a, b)),
                    "nmi": float(normalized_mutual_info_score(a, b)),
                    "aligned_agreement": float(np.mean(a == b_aligned)),
                    "alignment_map_b_to_a": {
                        str(src): int(dst) for src, dst in mapping.items()
                    },
                    "aligned_confusion_matrix": matrix.astype(int).tolist(),
                }
            )
    return results


def summarize_pairwise(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ari = np.asarray([r["ari"] for r in results], dtype=np.float64)
    nmi = np.asarray([r["nmi"] for r in results], dtype=np.float64)
    agreement = np.asarray(
        [r["aligned_agreement"] for r in results], dtype=np.float64
    )

    def stats(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return {
        "pair_count": int(len(results)),
        "ari": stats(ari),
        "nmi": stats(nmi),
        "aligned_agreement": stats(agreement),
    }


def reference_alignment_maps(
    labels_by_seed: Mapping[int, np.ndarray],
    *,
    reference_seed: int,
    k: int,
) -> Dict[str, Any]:
    reference = labels_by_seed[reference_seed]
    out: Dict[str, Any] = {}

    for seed, labels in sorted(labels_by_seed.items()):
        if seed == reference_seed:
            out[str(seed)] = {
                "mapping_to_reference": {str(i): i for i in range(k)},
                "aligned_agreement": 1.0,
            }
            continue

        mapping, aligned, matrix = hungarian_alignment(reference, labels, k=k)
        out[str(seed)] = {
            "mapping_to_reference": {
                str(src): int(dst) for src, dst in mapping.items()
            },
            "aligned_agreement": float(np.mean(reference == aligned)),
            "aligned_confusion_matrix": matrix.astype(int).tolist(),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-seed stability of selected DS-005 SSL clusters."
    )
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-seed", type=int, default=DEFAULT_REFERENCE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_dir = args.embedding_dir.resolve()
    selection_file = args.selection_file.resolve()
    training_config = args.training_config.resolve()
    output_dir = args.output_dir.resolve()

    assert_no_test_artifacts(embedding_dir)

    selected = load_json(selection_file)

    method = selected.get("method")
    k = int(selected.get("k", -1))
    clustering_seed = int(selected.get("clustering_seed", 20260822))
    pca_variance_target = float(selected.get("pca_variance_target", 0.95))

    if method != "kmeans":
        raise RuntimeError(f"Expected selected method kmeans, got {method!r}.")
    if k != 8:
        raise RuntimeError(f"Expected frozen selected k=8, got k={k}.")
    if selected.get("test_partition_used") is not False:
        raise RuntimeError("Selection file does not confirm TEST remained unused.")

    ssl_seeds = configured_ssl_seeds(training_config)

    if args.reference_seed not in ssl_seeds:
        raise ValueError(
            f"Reference seed {args.reference_seed} not in frozen seeds {ssl_seeds}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    final_paths = [
        output_dir / "pairwise_metrics.json",
        output_dir / "alignment_maps.json",
        output_dir / "stability_summary.json",
        output_dir / "STABILITY_SHA256SUMS",
    ]

    if not args.overwrite:
        existing = [p for p in final_paths if p.exists()]
        if existing:
            raise FileExistsError(
                "Stability outputs already exist. Use --overwrite for rerun:\n"
                + "\n".join(str(p) for p in existing)
            )

    print("=" * 80)
    print("DS-005 SSL CROSS-SEED CLUSTER STABILITY")
    print("=" * 80)
    print(f"Selected method:      {method}")
    print(f"Selected k:           {k}")
    print(f"Clustering seed:      {clustering_seed}")
    print(f"PCA variance target:  {pca_variance_target}")
    print(f"SSL training seeds:   {list(ssl_seeds)}")
    print(f"Reference seed:       {args.reference_seed}")
    print("Scaler fit:           TRAIN only")
    print("PCA fit:              TRAIN only")
    print("TEST partition:       PROTECTED / NOT LOADED")
    print()

    train_labels_by_seed: Dict[int, np.ndarray] = {}
    validation_labels_by_seed: Dict[int, np.ndarray] = {}
    diagnostics_by_seed: Dict[str, Any] = {}

    for ssl_seed in ssl_seeds:
        print("=" * 80)
        print(f"SSL SEED {ssl_seed}")
        print("=" * 80)

        verify_manifest(embedding_dir, ssl_seed=ssl_seed, partition="train")
        verify_manifest(embedding_dir, ssl_seed=ssl_seed, partition="validation")

        train = load_embeddings(
            embedding_dir, ssl_seed=ssl_seed, partition="train"
        )
        validation = load_embeddings(
            embedding_dir, ssl_seed=ssl_seed, partition="validation"
        )

        (
            train_labels,
            validation_labels,
            centers,
            explained_variance_ratio,
            diagnostics,
        ) = fit_selected_clustering(
            train,
            validation,
            k=k,
            clustering_seed=clustering_seed,
            pca_variance_target=pca_variance_target,
        )

        train_labels_by_seed[ssl_seed] = train_labels
        validation_labels_by_seed[ssl_seed] = validation_labels

        train_sizes = cluster_sizes(train_labels, k)
        validation_sizes = cluster_sizes(validation_labels, k)

        diagnostics_by_seed[str(ssl_seed)] = {
            **diagnostics,
            "train_cluster_sizes": train_sizes,
            "validation_cluster_sizes": validation_sizes,
            "pca_explained_variance_ratio": (
                explained_variance_ratio.astype(float).tolist()
            ),
        }

        seed_dir = output_dir / f"seed{ssl_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        np.save(seed_dir / "train_labels.npy", train_labels)
        np.save(seed_dir / "validation_labels.npy", validation_labels)
        np.save(seed_dir / "cluster_centers_pca.npy", centers)

        atomic_write_json(
            seed_dir / "cluster_sizes.json",
            {
                "ssl_seed": int(ssl_seed),
                "k": int(k),
                "train": train_sizes,
                "validation": validation_sizes,
                "test_partition_used": False,
            },
        )

        print(
            f"PCA retained: {diagnostics['pca_components']} components "
            f"({diagnostics['pca_variance_retained']:.4f} variance)"
        )
        print(
            f"TRAIN occupancy: min={train_sizes['min_fraction']:.4f}, "
            f"max={train_sizes['max_fraction']:.4f}"
        )
        print(
            f"VALIDATION occupancy: min={validation_sizes['min_fraction']:.4f}, "
            f"max={validation_sizes['max_fraction']:.4f}"
        )
        print("TEST partition used: NO")
        print()

        del train, validation

    train_pairwise = pairwise_metrics(
        train_labels_by_seed, k=k, partition="train"
    )
    validation_pairwise = pairwise_metrics(
        validation_labels_by_seed, k=k, partition="validation"
    )

    train_summary = summarize_pairwise(train_pairwise)
    validation_summary = summarize_pairwise(validation_pairwise)

    pairwise_path = output_dir / "pairwise_metrics.json"
    alignment_path = output_dir / "alignment_maps.json"
    summary_path = output_dir / "stability_summary.json"
    checksum_path = output_dir / "STABILITY_SHA256SUMS"

    atomic_write_json(
        pairwise_path,
        {
            "dataset_id": "DS-005",
            "representation": "ssl_encoder_embedding",
            "method": method,
            "k": k,
            "clustering_seed": clustering_seed,
            "ssl_training_seeds": list(ssl_seeds),
            "train": train_pairwise,
            "validation": validation_pairwise,
            "test_partition_used": False,
        },
    )

    atomic_write_json(
        alignment_path,
        {
            "reference_seed": args.reference_seed,
            "train": reference_alignment_maps(
                train_labels_by_seed,
                reference_seed=args.reference_seed,
                k=k,
            ),
            "validation": reference_alignment_maps(
                validation_labels_by_seed,
                reference_seed=args.reference_seed,
                k=k,
            ),
            "test_partition_used": False,
        },
    )

    atomic_write_json(
        summary_path,
        {
            "dataset_id": "DS-005",
            "representation": "ssl_encoder_embedding",
            "method": method,
            "k": k,
            "clustering_seed": clustering_seed,
            "pca_variance_target": pca_variance_target,
            "ssl_training_seeds": list(ssl_seeds),
            "reference_seed": args.reference_seed,
            "train_pairwise_summary": train_summary,
            "validation_pairwise_summary": validation_summary,
            "seed_diagnostics": diagnostics_by_seed,
            "test_partition_used": False,
        },
    )

    checksum_targets = [pairwise_path, alignment_path, summary_path]
    for ssl_seed in ssl_seeds:
        seed_dir = output_dir / f"seed{ssl_seed}"
        checksum_targets.extend(
            [
                seed_dir / "train_labels.npy",
                seed_dir / "validation_labels.npy",
                seed_dir / "cluster_centers_pca.npy",
                seed_dir / "cluster_sizes.json",
            ]
        )

    write_checksums(checksum_path, checksum_targets)

    print("=" * 80)
    print("CROSS-SEED STABILITY SUMMARY")
    print("=" * 80)
    print(f"TRAIN mean ARI:                    {train_summary['ari']['mean']:.6f}")
    print(f"TRAIN mean NMI:                    {train_summary['nmi']['mean']:.6f}")
    print(
        f"TRAIN mean aligned agreement:      "
        f"{train_summary['aligned_agreement']['mean']:.6f}"
    )
    print()
    print(
        f"VALIDATION mean ARI:               "
        f"{validation_summary['ari']['mean']:.6f}"
    )
    print(
        f"VALIDATION mean NMI:               "
        f"{validation_summary['nmi']['mean']:.6f}"
    )
    print(
        f"VALIDATION mean aligned agreement: "
        f"{validation_summary['aligned_agreement']['mean']:.6f}"
    )
    print()
    print("TEST partition used: NO")
    print(f"Summary:    {summary_path}")
    print(f"Pairwise:   {pairwise_path}")
    print(f"Alignments: {alignment_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
