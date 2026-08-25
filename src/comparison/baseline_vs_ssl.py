#!/usr/bin/env python3
"""Compare frozen DS-005 handcrafted baseline (Input A) with SSL (Input B).

Purpose
-------
Directly test the primary comparison question:

    Does the SSL representation contain behavioral structure not fully captured
    by the frozen 18-feature handcrafted baseline?

Frozen Input A
--------------
- 18 hand-engineered features
- TRAIN-only scaling
- PCA(6)
- GaussianMixture(k=2, seed=20260822)

Frozen Input B
--------------
- SSL encoder embeddings
- selected clustering: KMeans(k=8)
- five SSL training seeds: 11, 23, 37, 51, 79

TRAIN and VALIDATION only. TEST is never loaded.

Analyses
--------
1. Recompute the frozen Input A baseline clustering using TRAIN only.
2. Verify exact row alignment between Input A and Input B using:
   fish_id, session_id, bout_index, partition, context_id.
3. Compare baseline-cluster vs SSL-cluster assignments for every SSL seed:
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)
   - Adjusted Mutual Information (AMI)
   - contingency matrix
   - normalized conditional entropy H(SSL | baseline)
   - normalized conditional entropy H(baseline | SSL)
4. Quantify whether the two baseline states are subdivided by SSL:
   - SSL cluster distribution within each baseline state
   - effective number of SSL clusters within each baseline state
5. Fit a TRAIN-only multinomial logistic-regression probe using all 18
   handcrafted baseline features to predict each seed's SSL k=8 labels.
   Evaluate on VALIDATION:
   - accuracy
   - balanced accuracy
   - macro F1
   - majority baseline
   - uniform chance (1/8)

Interpretation guardrail
------------------------
The linear probe tests LINEAR reconstructability from Input A. Failure of a
linear probe does not by itself prove that no nonlinear mapping from Input A
could reconstruct SSL clusters. This script therefore reports overlap and
linear-decoding evidence without making the final biological claim automatically.

TEST safety
-----------
- Only train_core_* and validation_core_* files are referenced.
- No TEST path is constructed.
- Any path containing "test" is rejected.
- SSL metadata/label roots are scanned for TEST artifacts and execution stops
  if any are found.

Outputs
-------
data/processed/DS-005/baseline_vs_ssl/
    baseline_labels/
        train_labels.npy
        validation_labels.npy
        baseline_summary.json
    seed11/
        comparison.json
        linear_probe.json
    seed23/
        ...
    aggregate_summary.json
    BASELINE_VS_SSL_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/comparison/baseline_vs_ssl.py

Intentional rerun:

    PYTHONPATH=. python3 src/comparison/baseline_vs_ssl.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
DEFAULT_BASELINE_SELECTION = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "baseline_clustering"
    / "selected_configuration.json"
)
DEFAULT_SSL_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_SSL_LABEL_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline_vs_ssl"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_BASELINE_FEATURES = 18
EXPECTED_BASELINE_K = 2
EXPECTED_SSL_K = 8
BASELINE_PCA_COMPONENTS = 6
BASELINE_SEED = 20260822
PARTITIONS = ("train", "validation")


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
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_checksums(path: Path, artifacts: Sequence[Path]) -> None:
    path.write_text(
        "".join(
            f"{sha256_file(artifact)}  {artifact.relative_to(path.parent)}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )


def prohibit_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(
            f"TEST access prohibited during baseline-vs-SSL comparison: {path}"
        )


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
            or name in {"test.npy", "test.npz", "test.csv", "test.json"}
        ):
            hits.append(path)

    if hits:
        raise RuntimeError(
            "Protected TEST artifacts were found under an Input B comparison "
            "root. Refusing to continue:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def load_json(path: Path) -> Dict[str, Any]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    obj = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object.")

    return obj


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)

    training = obj.get("training", {})
    seeds = training.get("seeds", {}).get("values")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No frozen SSL seeds found in training.yaml.")

    return tuple(int(seed) for seed in seeds)


def load_baseline_npz(
    baseline_dir: Path,
    *,
    partition: str,
    scaled: bool,
) -> Dict[str, np.ndarray]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    suffix = "scaled" if scaled else "raw"
    path = baseline_dir / f"{partition}_core_{suffix}.npz"
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        data = {key: np.asarray(npz[key]) for key in npz.files}

    required = {
        "X",
        "feature_names",
        "fish_id",
        "session_id",
        "bout_index",
        "partition",
        "context_id",
    }

    missing = required - set(data)

    if missing:
        raise RuntimeError(
            f"{path.name} missing required arrays: {sorted(missing)}"
        )

    X = np.asarray(data["X"])

    if X.shape != (
        EXPECTED_ROWS[partition],
        EXPECTED_BASELINE_FEATURES,
    ):
        raise RuntimeError(
            f"{path.name}: expected "
            f"({EXPECTED_ROWS[partition]}, {EXPECTED_BASELINE_FEATURES}), "
            f"got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(f"{path.name}: non-finite baseline features.")

    observed_partition = set(data["partition"].astype(str).tolist())

    if observed_partition != {partition}:
        raise RuntimeError(
            f"{path.name}: expected partition={partition!r}, "
            f"observed {sorted(observed_partition)}"
        )

    return data


def load_ssl_metadata(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> Dict[str, np.ndarray]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    columns = {
        "fish_id": [],
        "session_id": [],
        "bout_index": [],
        "partition": [],
        "context_id": [],
        "bout_id": [],
    }

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header.")

        required = {
            "row_index",
            "fish_id",
            "session_id",
            "bout_index",
            "partition",
            "context_id",
            "bout_id",
            "training_seed",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"{path} missing metadata columns: {sorted(missing)}"
            )

        for expected_row, row in enumerate(reader):
            if int(row["row_index"]) != expected_row:
                raise RuntimeError(
                    f"{path}: row_index mismatch at row {expected_row}."
                )

            if row["partition"] != partition:
                raise RuntimeError(
                    f"{path}: partition mismatch at row {expected_row}."
                )

            if int(row["training_seed"]) != ssl_seed:
                raise RuntimeError(
                    f"{path}: SSL seed mismatch at row {expected_row}."
                )

            columns["fish_id"].append(row["fish_id"])
            columns["session_id"].append(row["session_id"])
            columns["bout_index"].append(int(row["bout_index"]))
            columns["partition"].append(row["partition"])
            columns["context_id"].append(row["context_id"])
            columns["bout_id"].append(row["bout_id"])

    if len(columns["fish_id"]) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"observed {len(columns['fish_id']):,}"
        )

    return {
        "fish_id": np.asarray(columns["fish_id"], dtype=str),
        "session_id": np.asarray(columns["session_id"], dtype=str),
        "bout_index": np.asarray(columns["bout_index"], dtype=np.int64),
        "partition": np.asarray(columns["partition"], dtype=str),
        "context_id": np.asarray(columns["context_id"], dtype=str),
        "bout_id": np.asarray(columns["bout_id"], dtype=str),
    }


def verify_row_alignment(
    baseline: Mapping[str, np.ndarray],
    ssl_metadata: Mapping[str, np.ndarray],
    *,
    partition: str,
    ssl_seed: int,
) -> None:
    fields = (
        "fish_id",
        "session_id",
        "bout_index",
        "partition",
        "context_id",
    )

    for field in fields:
        baseline_values = np.asarray(baseline[field]).astype(str)
        ssl_values = np.asarray(ssl_metadata[field]).astype(str)

        if baseline_values.shape != ssl_values.shape:
            raise RuntimeError(
                f"Row-alignment shape mismatch seed={ssl_seed} "
                f"partition={partition} field={field}: "
                f"{baseline_values.shape} != {ssl_values.shape}"
            )

        unequal = np.flatnonzero(baseline_values != ssl_values)

        if unequal.size:
            idx = int(unequal[0])
            raise RuntimeError(
                f"Input A/Input B row alignment FAILED at row {idx}, "
                f"seed={ssl_seed}, partition={partition}, field={field}: "
                f"baseline={baseline_values[idx]!r}, "
                f"ssl={ssl_values[idx]!r}"
            )


def load_ssl_labels(
    label_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        label_root
        / f"seed{ssl_seed}"
        / f"{partition}_labels.npy"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.int64,
    )

    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected ({EXPECTED_ROWS[partition]},), "
            f"got {labels.shape}"
        )

    if not np.array_equal(
        np.unique(labels),
        np.arange(EXPECTED_SSL_K),
    ):
        raise RuntimeError(
            f"{path}: expected all SSL labels 0..{EXPECTED_SSL_K - 1}."
        )

    return labels


def verify_frozen_baseline_selection(path: Path) -> Dict[str, Any]:
    selected = load_json(path)

    observed_method = str(selected.get("method", "")).lower()
    observed_k = int(selected.get("k", -1))
    observed_seed = int(selected.get("seed", selected.get("clustering_seed", -1)))

    if observed_method != "gmm":
        raise RuntimeError(
            f"Frozen baseline method mismatch: {observed_method!r} != 'gmm'."
        )

    if observed_k != EXPECTED_BASELINE_K:
        raise RuntimeError(
            f"Frozen baseline k mismatch: {observed_k} != {EXPECTED_BASELINE_K}."
        )

    if observed_seed != BASELINE_SEED:
        raise RuntimeError(
            f"Frozen baseline seed mismatch: {observed_seed} != {BASELINE_SEED}."
        )

    pca_components = int(
        selected.get("pca_components", BASELINE_PCA_COMPONENTS)
    )

    if pca_components != BASELINE_PCA_COMPONENTS:
        raise RuntimeError(
            "Frozen baseline PCA component count mismatch: "
            f"{pca_components} != {BASELINE_PCA_COMPONENTS}."
        )

    return selected


def fit_frozen_baseline(
    train_scaled: np.ndarray,
    validation_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(
        n_components=BASELINE_PCA_COMPONENTS,
        svd_solver="auto",
    )

    train_pca = pca.fit_transform(train_scaled)
    validation_pca = pca.transform(validation_scaled)

    explained = float(
        np.sum(pca.explained_variance_ratio_)
    )

    model = GaussianMixture(
        n_components=EXPECTED_BASELINE_K,
        covariance_type="full",
        random_state=BASELINE_SEED,
        n_init=5,
        reg_covar=1e-6,
    )

    model.fit(train_pca)

    train_labels = model.predict(train_pca).astype(np.int16)
    validation_labels = model.predict(validation_pca).astype(np.int16)

    return train_labels, validation_labels, explained


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))

    if total <= 0:
        return 0.0

    p = counts[counts > 0] / total

    return float(
        -np.sum(p * np.log(p))
    )


def normalized_conditional_entropy(
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    *,
    x_k: int,
    y_k: int,
) -> float:
    """Return H(Y|X) / H(Y), in [0,1] when H(Y)>0."""
    table = np.zeros(
        (x_k, y_k),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (x_labels.astype(int), y_labels.astype(int)),
        1,
    )

    y_counts = np.sum(table, axis=0)
    h_y = entropy_from_counts(y_counts)

    if h_y <= 0:
        return 0.0

    total = float(np.sum(table))
    h_y_given_x = 0.0

    for x in range(x_k):
        row = table[x]
        row_total = float(np.sum(row))

        if row_total <= 0:
            continue

        h_y_given_x += (
            row_total / total
        ) * entropy_from_counts(row)

    return float(h_y_given_x / h_y)


def effective_category_count(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]

    if p.size == 0:
        return 0.0

    return float(
        math.exp(
            -np.sum(p * np.log(p))
        )
    )


def comparison_metrics(
    baseline_labels: np.ndarray,
    ssl_labels: np.ndarray,
) -> Dict[str, Any]:
    table = np.zeros(
        (EXPECTED_BASELINE_K, EXPECTED_SSL_K),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (
            baseline_labels.astype(int),
            ssl_labels.astype(int),
        ),
        1,
    )

    ssl_within_baseline: Dict[str, Any] = {}

    for baseline_cluster in range(EXPECTED_BASELINE_K):
        counts = table[baseline_cluster]
        total = int(np.sum(counts))
        fractions = counts / max(total, 1)

        ssl_within_baseline[str(baseline_cluster)] = {
            "count": total,
            "ssl_cluster_counts": counts.astype(int).tolist(),
            "ssl_cluster_fractions": fractions.astype(float).tolist(),
            "effective_ssl_cluster_count": effective_category_count(
                fractions
            ),
            "dominant_ssl_cluster_fraction": float(
                np.max(fractions)
            ),
        }

    return {
        "adjusted_rand_index": float(
            adjusted_rand_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(
                baseline_labels,
                ssl_labels,
            )
        ),
        "normalized_H_ssl_given_baseline": (
            normalized_conditional_entropy(
                baseline_labels,
                ssl_labels,
                x_k=EXPECTED_BASELINE_K,
                y_k=EXPECTED_SSL_K,
            )
        ),
        "normalized_H_baseline_given_ssl": (
            normalized_conditional_entropy(
                ssl_labels,
                baseline_labels,
                x_k=EXPECTED_SSL_K,
                y_k=EXPECTED_BASELINE_K,
            )
        ),
        "contingency_baseline_x_ssl": (
            table.astype(int).tolist()
        ),
        "ssl_within_each_baseline_cluster": (
            ssl_within_baseline
        ),
    }


def majority_accuracy(labels: np.ndarray) -> float:
    counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_SSL_K,
    )
    return float(
        np.max(counts) / np.sum(counts)
    )


def fit_linear_probe(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    validation_x: np.ndarray,
    validation_labels: np.ndarray,
) -> Dict[str, Any]:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=BASELINE_SEED,
    )

    model.fit(
        train_x,
        train_labels,
    )

    train_pred = model.predict(train_x)
    validation_pred = model.predict(validation_x)

    def metrics(
        truth: np.ndarray,
        pred: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "accuracy": float(
                accuracy_score(truth, pred)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    truth,
                    pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    truth,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "majority_accuracy": (
                majority_accuracy(truth)
            ),
            "uniform_chance": float(
                1.0 / EXPECTED_SSL_K
            ),
        }

    return {
        "probe": "multinomial_logistic_regression",
        "input": "18 frozen TRAIN-scaled handcrafted features",
        "target": "SSL k=8 cluster labels",
        "fit_partition": "train",
        "train": metrics(
            train_labels,
            train_pred,
        ),
        "validation": metrics(
            validation_labels,
            validation_pred,
        ),
        "interpretation": (
            "Tests linear reconstructability of SSL cluster labels from Input A. "
            "Does not rule out nonlinear reconstructability."
        ),
        "test_partition_used": False,
    }


def aggregate_seed_results(
    per_seed: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    names = (
        "train_ari",
        "validation_ari",
        "train_nmi",
        "validation_nmi",
        "train_ami",
        "validation_ami",
        "train_h_ssl_given_baseline",
        "validation_h_ssl_given_baseline",
        "train_h_baseline_given_ssl",
        "validation_h_baseline_given_ssl",
        "train_probe_balanced_accuracy",
        "validation_probe_balanced_accuracy",
        "train_probe_macro_f1",
        "validation_probe_macro_f1",
    )

    values: Dict[str, List[float]] = {
        name: []
        for name in names
    }

    for result in per_seed.values():
        train_cmp = result["train_comparison"]
        val_cmp = result["validation_comparison"]
        probe = result["linear_probe"]

        values["train_ari"].append(
            train_cmp["adjusted_rand_index"]
        )
        values["validation_ari"].append(
            val_cmp["adjusted_rand_index"]
        )
        values["train_nmi"].append(
            train_cmp["normalized_mutual_information"]
        )
        values["validation_nmi"].append(
            val_cmp["normalized_mutual_information"]
        )
        values["train_ami"].append(
            train_cmp["adjusted_mutual_information"]
        )
        values["validation_ami"].append(
            val_cmp["adjusted_mutual_information"]
        )
        values["train_h_ssl_given_baseline"].append(
            train_cmp["normalized_H_ssl_given_baseline"]
        )
        values["validation_h_ssl_given_baseline"].append(
            val_cmp["normalized_H_ssl_given_baseline"]
        )
        values["train_h_baseline_given_ssl"].append(
            train_cmp["normalized_H_baseline_given_ssl"]
        )
        values["validation_h_baseline_given_ssl"].append(
            val_cmp["normalized_H_baseline_given_ssl"]
        )
        values["train_probe_balanced_accuracy"].append(
            probe["train"]["balanced_accuracy"]
        )
        values["validation_probe_balanced_accuracy"].append(
            probe["validation"]["balanced_accuracy"]
        )
        values["train_probe_macro_f1"].append(
            probe["train"]["macro_f1"]
        )
        values["validation_probe_macro_f1"].append(
            probe["validation"]["macro_f1"]
        )

    output: Dict[str, Any] = {}

    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        output[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen DS-005 handcrafted baseline clusters with "
            "selected SSL clusters using TRAIN/VALIDATION only."
        )
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )
    parser.add_argument(
        "--baseline-selection",
        type=Path,
        default=DEFAULT_BASELINE_SELECTION,
    )
    parser.add_argument(
        "--ssl-metadata-root",
        type=Path,
        default=DEFAULT_SSL_METADATA_ROOT,
    )
    parser.add_argument(
        "--ssl-label-root",
        type=Path,
        default=DEFAULT_SSL_LABEL_ROOT,
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_dir = args.baseline_dir.resolve()
    baseline_selection = args.baseline_selection.resolve()
    ssl_metadata_root = args.ssl_metadata_root.resolve()
    ssl_label_root = args.ssl_label_root.resolve()
    training_config = args.training_config.resolve()
    output_dir = args.output_dir.resolve()

    for path in (
        baseline_dir,
        baseline_selection,
        ssl_metadata_root,
        ssl_label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    # Input B safety: there should still be no exported TEST artifacts.
    assert_no_test_artifacts(
        ssl_metadata_root
    )
    assert_no_test_artifacts(
        ssl_label_root
    )

    selected_baseline = verify_frozen_baseline_selection(
        baseline_selection
    )
    ssl_seeds = configured_ssl_seeds(
        training_config
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        output_dir / "aggregate_summary.json"
    )
    checksum_path = (
        output_dir / "BASELINE_VS_SSL_SHA256SUMS"
    )

    if not args.overwrite and aggregate_path.exists():
        raise FileExistsError(
            f"{aggregate_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 BASELINE VS SSL COMPARISON")
    print("=" * 80)
    print("Input A: PCA(6) -> GMM(k=2, seed=20260822)")
    print("Input B: SSL -> KMeans(k=8)")
    print(f"SSL seeds: {list(ssl_seeds)}")
    print("Evaluation: TRAIN + VALIDATION")
    print("TEST partition: PROTECTED / NOT LOADED")
    print()

    train_scaled = load_baseline_npz(
        baseline_dir,
        partition="train",
        scaled=True,
    )
    validation_scaled = load_baseline_npz(
        baseline_dir,
        partition="validation",
        scaled=True,
    )

    train_raw = load_baseline_npz(
        baseline_dir,
        partition="train",
        scaled=False,
    )
    validation_raw = load_baseline_npz(
        baseline_dir,
        partition="validation",
        scaled=False,
    )

    np.testing.assert_array_equal(
        train_scaled["feature_names"],
        train_raw["feature_names"],
    )
    np.testing.assert_array_equal(
        validation_scaled["feature_names"],
        validation_raw["feature_names"],
    )
    np.testing.assert_array_equal(
        train_scaled["feature_names"],
        validation_scaled["feature_names"],
    )

    feature_names = (
        train_scaled["feature_names"]
        .astype(str)
        .tolist()
    )

    print(f"TRAIN rows:      {train_scaled['X'].shape[0]:,}")
    print(f"VALIDATION rows: {validation_scaled['X'].shape[0]:,}")
    print(f"Input A features: {len(feature_names)}")
    print()

    baseline_train_labels, baseline_validation_labels, explained = (
        fit_frozen_baseline(
            np.asarray(
                train_scaled["X"],
                dtype=np.float32,
            ),
            np.asarray(
                validation_scaled["X"],
                dtype=np.float32,
            ),
        )
    )

    print("Frozen Input A recomputed:")
    print(f"  PCA variance retained: {explained:.4f}")
    print(
        "  TRAIN cluster counts: "
        f"{np.bincount(baseline_train_labels, minlength=2).tolist()}"
    )
    print(
        "  VALIDATION cluster counts: "
        f"{np.bincount(baseline_validation_labels, minlength=2).tolist()}"
    )
    print()

    baseline_label_dir = (
        output_dir / "baseline_labels"
    )
    baseline_label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_baseline_labels_path = (
        baseline_label_dir / "train_labels.npy"
    )
    validation_baseline_labels_path = (
        baseline_label_dir
        / "validation_labels.npy"
    )
    baseline_summary_path = (
        baseline_label_dir
        / "baseline_summary.json"
    )

    np.save(
        train_baseline_labels_path,
        baseline_train_labels,
    )
    np.save(
        validation_baseline_labels_path,
        baseline_validation_labels,
    )

    atomic_write_json(
        baseline_summary_path,
        {
            "dataset_id": "DS-005",
            "representation": "handcrafted_baseline",
            "selected_configuration": {
                "method": "gmm",
                "k": EXPECTED_BASELINE_K,
                "seed": BASELINE_SEED,
                "pca_components": BASELINE_PCA_COMPONENTS,
            },
            "pca_variance_retained": explained,
            "feature_names": feature_names,
            "train_cluster_counts": (
                np.bincount(
                    baseline_train_labels,
                    minlength=EXPECTED_BASELINE_K,
                ).astype(int).tolist()
            ),
            "validation_cluster_counts": (
                np.bincount(
                    baseline_validation_labels,
                    minlength=EXPECTED_BASELINE_K,
                ).astype(int).tolist()
            ),
            "frozen_selection_file": str(
                baseline_selection
            ),
            "frozen_selection_score": (
                selected_baseline.get(
                    "selection_score"
                )
            ),
            "test_partition_used": False,
        },
    )

    written_artifacts: List[Path] = [
        train_baseline_labels_path,
        validation_baseline_labels_path,
        baseline_summary_path,
    ]

    per_seed_results: Dict[int, Dict[str, Any]] = {}

    for ssl_seed in ssl_seeds:
        print("=" * 80)
        print(f"SSL SEED {ssl_seed}")
        print("=" * 80)

        train_meta = load_ssl_metadata(
            ssl_metadata_root,
            ssl_seed=ssl_seed,
            partition="train",
        )
        validation_meta = load_ssl_metadata(
            ssl_metadata_root,
            ssl_seed=ssl_seed,
            partition="validation",
        )

        verify_row_alignment(
            train_scaled,
            train_meta,
            partition="train",
            ssl_seed=ssl_seed,
        )
        verify_row_alignment(
            validation_scaled,
            validation_meta,
            partition="validation",
            ssl_seed=ssl_seed,
        )

        print("Input A/Input B bout alignment: PASS")

        ssl_train_labels = load_ssl_labels(
            ssl_label_root,
            ssl_seed=ssl_seed,
            partition="train",
        )
        ssl_validation_labels = load_ssl_labels(
            ssl_label_root,
            ssl_seed=ssl_seed,
            partition="validation",
        )

        train_comparison = comparison_metrics(
            baseline_train_labels,
            ssl_train_labels,
        )
        validation_comparison = comparison_metrics(
            baseline_validation_labels,
            ssl_validation_labels,
        )

        linear_probe = fit_linear_probe(
            np.asarray(
                train_scaled["X"],
                dtype=np.float32,
            ),
            ssl_train_labels,
            np.asarray(
                validation_scaled["X"],
                dtype=np.float32,
            ),
            ssl_validation_labels,
        )

        seed_result = {
            "ssl_seed": int(ssl_seed),
            "train_comparison": train_comparison,
            "validation_comparison": validation_comparison,
            "linear_probe": linear_probe,
            "row_alignment_verified": True,
            "test_partition_used": False,
        }

        per_seed_results[
            ssl_seed
        ] = seed_result

        seed_dir = (
            output_dir / f"seed{ssl_seed}"
        )
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        comparison_path = (
            seed_dir / "comparison.json"
        )
        probe_path = (
            seed_dir / "linear_probe.json"
        )

        atomic_write_json(
            comparison_path,
            {
                "ssl_seed": int(ssl_seed),
                "train": train_comparison,
                "validation": validation_comparison,
                "row_alignment_verified": True,
                "test_partition_used": False,
            },
        )

        atomic_write_json(
            probe_path,
            {
                "ssl_seed": int(ssl_seed),
                **linear_probe,
            },
        )

        written_artifacts.extend(
            [
                comparison_path,
                probe_path,
            ]
        )

        print(
            "VALIDATION baseline-vs-SSL ARI: "
            f"{validation_comparison['adjusted_rand_index']:.6f}"
        )
        print(
            "VALIDATION baseline-vs-SSL NMI: "
            f"{validation_comparison['normalized_mutual_information']:.6f}"
        )
        print(
            "VALIDATION H(SSL|baseline)/H(SSL): "
            f"{validation_comparison['normalized_H_ssl_given_baseline']:.6f}"
        )
        print(
            "VALIDATION H(baseline|SSL)/H(baseline): "
            f"{validation_comparison['normalized_H_baseline_given_ssl']:.6f}"
        )
        print(
            "VALIDATION Input-A linear probe balanced accuracy: "
            f"{linear_probe['validation']['balanced_accuracy']:.6f}"
        )
        print(
            "VALIDATION Input-A linear probe macro F1: "
            f"{linear_probe['validation']['macro_f1']:.6f}"
        )
        print(
            "VALIDATION majority baseline: "
            f"{linear_probe['validation']['majority_accuracy']:.6f}"
        )
        print(
            "Uniform chance (k=8): "
            f"{linear_probe['validation']['uniform_chance']:.6f}"
        )
        print("TEST partition used: NO")
        print()

    aggregate = aggregate_seed_results(
        per_seed_results
    )

    aggregate_payload = {
        "dataset_id": "DS-005",
        "input_a": {
            "representation": "18 handcrafted core features",
            "selected_clustering": {
                "pca_components": BASELINE_PCA_COMPONENTS,
                "method": "gmm",
                "k": EXPECTED_BASELINE_K,
                "seed": BASELINE_SEED,
            },
        },
        "input_b": {
            "representation": "ssl_encoder_embedding",
            "selected_clustering": {
                "method": "kmeans",
                "k": EXPECTED_SSL_K,
            },
            "ssl_training_seeds": list(
                ssl_seeds
            ),
        },
        "row_alignment": {
            "verified_fields": [
                "fish_id",
                "session_id",
                "bout_index",
                "partition",
                "context_id",
            ],
            "verified_for_all_ssl_seeds": True,
        },
        "aggregate_metrics": aggregate,
        "per_seed": {
            str(seed): result
            for seed, result
            in per_seed_results.items()
        },
        "interpretation_guardrails": {
            "low_overlap": (
                "Low ARI/NMI indicates the partitions differ, but difference "
                "alone does not establish that SSL structure is biologically "
                "meaningful."
            ),
            "conditional_entropy": (
                "High H(SSL|baseline)/H(SSL) means knowing the baseline state "
                "leaves substantial uncertainty about the SSL state."
            ),
            "linear_probe": (
                "The 18-feature logistic-regression probe tests only linear "
                "reconstructability. A later nonlinear sensitivity probe is "
                "needed before claiming SSL structure cannot be reconstructed "
                "from Input A by any reasonable mapping."
            ),
            "test_partition_used": False,
        },
        "test_partition_used": False,
    }

    atomic_write_json(
        aggregate_path,
        aggregate_payload,
    )
    written_artifacts.append(
        aggregate_path
    )

    write_checksums(
        checksum_path,
        written_artifacts,
    )

    print("=" * 80)
    print("BASELINE VS SSL SUMMARY")
    print("=" * 80)
    print(
        "Mean VALIDATION ARI:                    "
        f"{aggregate['validation_ari']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION NMI:                    "
        f"{aggregate['validation_nmi']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION AMI:                    "
        f"{aggregate['validation_ami']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION H(SSL|baseline)/H(SSL): "
        f"{aggregate['validation_h_ssl_given_baseline']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION H(baseline|SSL)/H(baseline): "
        f"{aggregate['validation_h_baseline_given_ssl']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION linear-probe balanced accuracy: "
        f"{aggregate['validation_probe_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION linear-probe macro F1:         "
        f"{aggregate['validation_probe_macro_f1']['mean']:.6f}"
    )
    print()
    print(f"Uniform chance for SSL k=8:          {1.0 / EXPECTED_SSL_K:.6f}")
    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
