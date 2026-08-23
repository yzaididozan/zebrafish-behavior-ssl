#!/usr/bin/env python3
"""Fish-identity leakage test for DS-005 SSL embeddings.

Question
--------
Can fish identity be predicted from the learned embedding substantially above
chance? Strong identity predictability may indicate nuisance leakage.

Default behavior uses TRAIN embeddings only and creates a deterministic
within-TRAIN holdout. TEST is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "validation" / "identity_leakage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-seed", type=int, default=11)
    parser.add_argument(
        "--partition",
        choices=("train", "validation"),
        default="train",
    )
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()

    emb_path = (
        DEFAULT_EMBED_DIR
        / f"ssl_seed{args.ssl_seed}_{args.partition}_embeddings.npy"
    )
    meta_path = (
        DEFAULT_EMBED_DIR
        / f"ssl_seed{args.ssl_seed}_{args.partition}_metadata.csv"
    )

    x = np.asarray(np.load(emb_path, mmap_mode="r"), dtype=np.float32)
    metadata = load_metadata(meta_path)

    if len(metadata) != x.shape[0]:
        raise RuntimeError("Metadata row count does not match embeddings.")

    fish = np.asarray([row["fish_id"] for row in metadata], dtype=object)

    if np.any(fish == ""):
        raise RuntimeError("Some metadata rows have missing fish_id.")

    if args.max_rows > 0 and x.shape[0] > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            x.shape[0],
            size=args.max_rows,
            replace=False,
        )
        idx.sort()
        x = x[idx]
        fish = fish[idx]

    # Keep fish with enough examples to support a within-fish train/test split.
    values, counts = np.unique(fish, return_counts=True)
    eligible = values[counts >= 4]
    mask = np.isin(fish, eligible)

    x = x[mask]
    fish = fish[mask]

    if np.unique(fish).size < 2:
        raise RuntimeError("Need at least two fish classes.")

    rng = np.random.default_rng(args.seed)
    train_idx = []
    test_idx = []

    for f in np.unique(fish):
        idx = np.flatnonzero(fish == f)
        idx = rng.permutation(idx)
        split = max(1, int(0.8 * len(idx)))
        split = min(split, len(idx) - 1)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=args.seed,
            class_weight="balanced",
        ),
    )
    model.fit(x[train_idx], fish[train_idx])
    pred = model.predict(x[test_idx])

    n_classes = int(np.unique(fish).size)
    majority = max(
        np.mean(fish[test_idx] == cls)
        for cls in np.unique(fish[test_idx])
    )

    result = {
        "dataset_id": "DS-005",
        "analysis": "identity_leakage",
        "ssl_seed": args.ssl_seed,
        "partition": args.partition,
        "test_partition_used": False,
        "rows_used": int(x.shape[0]),
        "fish_classes": n_classes,
        "accuracy": float(
            accuracy_score(fish[test_idx], pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(fish[test_idx], pred)
        ),
        "chance_uniform": float(1.0 / n_classes),
        "majority_class_baseline": float(majority),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (
        f"seed{args.ssl_seed}_{args.partition}_identity_leakage.json"
    )
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("IDENTITY LEAKAGE TEST")
    print("=" * 21)
    print(f"Rows used: {x.shape[0]:,}")
    print(f"Fish classes: {n_classes}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{result['balanced_accuracy']:.4f}"
    )
    print(f"Uniform chance: {result['chance_uniform']:.4f}")
    print(f"TEST used: NO")
    print(f"Artifact: {out}")


if __name__ == "__main__":
    main()
