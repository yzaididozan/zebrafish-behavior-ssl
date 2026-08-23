#!/usr/bin/env python3
"""Context/session leakage test for DS-005 SSL embeddings.

This evaluates whether experimental context can be predicted from embeddings.
High predictability is not automatically invalid because behavior can truly
differ across contexts, so interpret alongside biological expectations and
other nuisance controls.

TEST is never loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "validation" / "context_leakage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-seed", type=int, default=11)
    parser.add_argument(
        "--partition",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument(
        "--target",
        choices=("context_name", "session_id"),
        default="context_name",
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

    y = np.asarray(
        [row.get(args.target, "") for row in metadata],
        dtype=object,
    )

    valid = y != ""
    x = x[valid]
    y = y[valid]

    if args.max_rows > 0 and x.shape[0] > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            x.shape[0],
            size=args.max_rows,
            replace=False,
        )
        idx.sort()
        x = x[idx]
        y = y[idx]

    classes, counts = np.unique(y, return_counts=True)
    eligible = classes[counts >= 4]
    keep = np.isin(y, eligible)
    x = x[keep]
    y = y[keep]

    if np.unique(y).size < 2:
        raise RuntimeError("Need at least two target classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=args.seed,
            class_weight="balanced",
        ),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    n_classes = int(np.unique(y).size)
    majority = max(
        np.mean(y_test == cls)
        for cls in np.unique(y_test)
    )

    result = {
        "dataset_id": "DS-005",
        "analysis": "context_leakage",
        "target": args.target,
        "ssl_seed": args.ssl_seed,
        "partition": args.partition,
        "test_partition_used": False,
        "rows_used": int(x.shape[0]),
        "classes": n_classes,
        "accuracy": float(
            accuracy_score(y_test, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "chance_uniform": float(1.0 / n_classes),
        "majority_class_baseline": float(majority),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (
        f"seed{args.ssl_seed}_{args.partition}_{args.target}_leakage.json"
    )
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("CONTEXT / SESSION LEAKAGE TEST")
    print("=" * 31)
    print(f"Target: {args.target}")
    print(f"Rows used: {x.shape[0]:,}")
    print(f"Classes: {n_classes}")
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
