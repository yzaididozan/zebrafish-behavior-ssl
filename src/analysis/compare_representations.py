#!/usr/bin/env python3
"""Compare handcrafted baseline and SSL representations for DS-005.

This script is intended for TRAIN/VALIDATION-only comparison before final
held-out TEST evaluation.

It compares:
- cluster assignment agreement (ARI / NMI),
- cluster balance,
- cross-representation recoverability with simple logistic regression,
- representation dimensionality / row alignment checks.

Important
---------
- TEST is refused by default.
- Metadata is used only for row alignment/auditing, never as model input.
- This script assumes baseline and SSL artifacts refer to the same bout rows
  within each partition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SSL_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"
)
DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "representation_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and SSL representations."
    )
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--ssl-seed", type=int, default=11)
    parser.add_argument(
        "--ssl-embeddings",
        type=Path,
        default=None,
        help="Override SSL embedding .npy path.",
    )
    parser.add_argument(
        "--baseline-features",
        type=Path,
        default=None,
        help="Override baseline feature .npz path.",
    )
    parser.add_argument(
        "--baseline-labels",
        type=Path,
        required=False,
        help="Optional baseline cluster labels .npy path.",
    )
    parser.add_argument(
        "--ssl-labels",
        type=Path,
        required=False,
        help="Optional SSL cluster labels .npy path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Optional cap for recoverability models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260822,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def load_ssl_embeddings(
    path: Path,
) -> np.ndarray:
    x = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"SSL embeddings must be 2D, got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("SSL embeddings contain NaN/Inf.")
    return x


def load_baseline_features(path: Path) -> np.ndarray:
    data = np.load(path)

    preferred = (
        "X",
        "features",
        "x",
        "data",
    )
    for key in preferred:
        if key in data:
            x = np.asarray(data[key], dtype=np.float32)
            break
    else:
        candidates = [
            key for key in data.files
            if np.asarray(data[key]).ndim == 2
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely identify 2D feature matrix in {path}. "
                f"Keys: {data.files}"
            )
        x = np.asarray(data[candidates[0]], dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Baseline features must be 2D, got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Baseline features contain NaN/Inf.")
    return x


def load_labels(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    y = np.asarray(np.load(path))
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {y.shape}.")
    return y


def cluster_balance(labels: np.ndarray) -> Dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return {
        "n_clusters": int(values.size),
        "counts": {
            str(v): int(c)
            for v, c in zip(values, counts)
        },
        "proportions": {
            str(v): float(p)
            for v, p in zip(values, proportions)
        },
        "smallest_cluster_fraction": float(proportions.min()),
        "largest_cluster_fraction": float(proportions.max()),
    }


def deterministic_indices(
    n: int,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    idx.sort()
    return idx


def recoverability(
    source_x: np.ndarray,
    target_labels: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> Dict[str, float]:
    idx = deterministic_indices(
        source_x.shape[0],
        max_rows=max_rows,
        seed=seed,
    )

    x = source_x[idx]
    y = target_labels[idx]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))

    split = max(2, int(0.8 * len(idx)))
    train_idx = perm[:split]
    test_idx = perm[split:]

    if test_idx.size < 2:
        raise ValueError("Not enough rows for recoverability holdout.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        ),
    )
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])

    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[test_idx], pred)
        ),
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
    }


def main() -> None:
    args = parse_args()

    if args.partition == "test" and not args.allow_test:
        raise SystemExit(
            "Refusing TEST comparison without --allow-test."
        )

    ssl_path = args.ssl_embeddings or (
        DEFAULT_SSL_DIR
        / f"ssl_seed{args.ssl_seed}_{args.partition}_embeddings.npy"
    )

    baseline_path = args.baseline_features or (
        DEFAULT_BASELINE_DIR / f"{args.partition}_core_raw.npz"
    )

    ssl_x = load_ssl_embeddings(ssl_path)
    baseline_x = load_baseline_features(baseline_path)

    if ssl_x.shape[0] != baseline_x.shape[0]:
        raise RuntimeError(
            "Baseline and SSL row counts do not match. "
            f"baseline={baseline_x.shape[0]}, ssl={ssl_x.shape[0]}. "
            "Do not compare until bout-row alignment is verified."
        )

    baseline_labels = load_labels(args.baseline_labels)
    ssl_labels = load_labels(args.ssl_labels)

    if baseline_labels is not None and len(baseline_labels) != len(ssl_x):
        raise ValueError("Baseline label count does not match representation rows.")

    if ssl_labels is not None and len(ssl_labels) != len(ssl_x):
        raise ValueError("SSL label count does not match representation rows.")

    result: Dict[str, Any] = {
        "dataset_id": "DS-005",
        "partition": args.partition,
        "test_partition_used": args.partition == "test",
        "ssl_seed": args.ssl_seed,
        "rows": int(ssl_x.shape[0]),
        "baseline_dim": int(baseline_x.shape[1]),
        "ssl_dim": int(ssl_x.shape[1]),
        "baseline_features": str(baseline_path),
        "ssl_embeddings": str(ssl_path),
    }

    if baseline_labels is not None:
        result["baseline_cluster_balance"] = cluster_balance(
            baseline_labels
        )

    if ssl_labels is not None:
        result["ssl_cluster_balance"] = cluster_balance(
            ssl_labels
        )

    if baseline_labels is not None and ssl_labels is not None:
        result["cluster_agreement"] = {
            "adjusted_rand_index": float(
                adjusted_rand_score(
                    baseline_labels,
                    ssl_labels,
                )
            ),
            "normalized_mutual_info": float(
                normalized_mutual_info_score(
                    baseline_labels,
                    ssl_labels,
                )
            ),
        }

        result["recoverability"] = {
            "baseline_features_predict_ssl_clusters": recoverability(
                baseline_x,
                ssl_labels,
                max_rows=args.max_rows,
                seed=args.seed,
            ),
            "ssl_embeddings_predict_baseline_clusters": recoverability(
                ssl_x,
                baseline_labels,
                max_rows=args.max_rows,
                seed=args.seed,
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = (
        args.output_dir
        / f"seed{args.ssl_seed}_{args.partition}_comparison.json"
    )
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("REPRESENTATION COMPARISON")
    print("=" * 25)
    print(f"Partition: {args.partition}")
    print(f"Rows: {ssl_x.shape[0]:,}")
    print(f"Baseline dim: {baseline_x.shape[1]}")
    print(f"SSL dim: {ssl_x.shape[1]}")
    print(f"TEST used: {'YES' if args.partition == 'test' else 'NO'}")

    if "cluster_agreement" in result:
        ca = result["cluster_agreement"]
        print(
            f"ARI: {ca['adjusted_rand_index']:.4f} | "
            f"NMI: {ca['normalized_mutual_info']:.4f}"
        )

    print(f"Artifact: {out}")


if __name__ == "__main__":
    main()
