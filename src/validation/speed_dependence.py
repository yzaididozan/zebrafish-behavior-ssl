#!/usr/bin/env python3
"""Speed-dependence analysis for DS-005 SSL embeddings.

This quantifies how strongly learned embeddings are explained by bout-level
speed summaries.

Primary diagnostics:
- linear R^2 for predicting mean speed from SSL embeddings,
- correlation between embedding PC1 and mean speed,
- optional cluster-level speed separation if cluster labels are supplied.

TEST is never loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.ds005 import DS005
from src.ssl.input import bout_to_ssl_input


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_embeddings"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "validation" / "speed_dependence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-seed", type=int, default=11)
    parser.add_argument(
        "--partition",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--cluster-labels",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def derive_mean_speed(
    *,
    partition: str,
    expected_rows: int,
) -> np.ndarray:
    values: List[float] = []

    with DS005(
        repo_root=REPO_ROOT,
        validate=True,
        verify_split_hash=True,
    ) as dataset:
        for bout in dataset.iter_bouts(
            partition=partition,
            primary_qc_only=True,
            include_optional=False,
        ):
            x = bout_to_ssl_input(bout)
            speed = np.asarray(x[:, 2], dtype=np.float32)
            values.append(float(np.mean(speed)))

            if len(values) >= expected_rows:
                break

    if len(values) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} speed rows, got {len(values)}."
        )

    return np.asarray(values, dtype=np.float32)


def cluster_speed_summary(
    labels: np.ndarray,
    mean_speed: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label in np.unique(labels):
        mask = labels == label
        out[str(label)] = {
            "rows": int(mask.sum()),
            "mean_speed": float(mean_speed[mask].mean()),
            "std_speed": float(mean_speed[mask].std()),
        }
    return out


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

    mean_speed = derive_mean_speed(
        partition=args.partition,
        expected_rows=x.shape[0],
    )

    if args.max_rows > 0 and x.shape[0] > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            x.shape[0],
            size=args.max_rows,
            replace=False,
        )
        idx.sort()
        x_eval = x[idx]
        speed_eval = mean_speed[idx]
    else:
        x_eval = x
        speed_eval = mean_speed

    x_train, x_test, y_train, y_test = train_test_split(
        x_eval,
        speed_eval,
        test_size=0.2,
        random_state=args.seed,
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    r2 = float(r2_score(y_test, pred))

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(
        StandardScaler().fit_transform(x_eval)
    )[:, 0]

    corr, p_value = pearsonr(pc1, speed_eval)

    result: Dict[str, Any] = {
        "dataset_id": "DS-005",
        "analysis": "speed_dependence",
        "ssl_seed": args.ssl_seed,
        "partition": args.partition,
        "test_partition_used": False,
        "rows_used": int(x_eval.shape[0]),
        "ridge_speed_prediction_r2": r2,
        "pc1_speed_pearson_r": float(corr),
        "pc1_speed_p_value": float(p_value),
        "pc1_variance_fraction": float(
            pca.explained_variance_ratio_[0]
        ),
    }

    if args.cluster_labels is not None:
        labels = np.asarray(np.load(args.cluster_labels))
        if labels.ndim != 1 or len(labels) != len(mean_speed):
            raise ValueError(
                "Cluster labels must be 1D and match full partition row count."
            )

        result["cluster_speed_summary"] = cluster_speed_summary(
            labels,
            mean_speed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (
        f"seed{args.ssl_seed}_{args.partition}_speed_dependence.json"
    )
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("SPEED DEPENDENCE TEST")
    print("=" * 21)
    print(f"Rows used: {x_eval.shape[0]:,}")
    print(f"Ridge speed R^2: {r2:.4f}")
    print(f"PC1-speed Pearson r: {corr:.4f}")
    print(f"TEST used: NO")
    print(f"Artifact: {out}")


if __name__ == "__main__":
    main()
