#!/usr/bin/env python3
"""Speed-dependence analysis for selected DS-005 SSL clusters.

Tests whether SSL cluster assignments are largely explainable by mean bout speed.

TRAIN and VALIDATION only. TEST is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_ROOT = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
DEFAULT_LABEL_ROOT = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_cluster_stability"
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_speed_dependence"

EXPECTED_ROWS = {"train": 842_841, "validation": 168_464}
EXPECTED_K = 8
PARTITIONS = ("train", "validation")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        tmp = handle.name
    os.replace(tmp, path)


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


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
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
            or name in {"test.npz", "test.npy", "test.csv", "test.json"}
        ):
            hits.append(path)
    if hits:
        raise RuntimeError(
            "TEST artifacts detected under SSL root; refusing speed analysis:\n"
            + "\n".join(str(p) for p in hits[:20])
        )


def load_metadata_speed(
    metadata_root: Path, *, ssl_seed: int, partition: str
) -> Tuple[np.ndarray, List[str]]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = metadata_root / f"seed{ssl_seed}" / f"{partition}_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    speeds: List[float] = []
    bout_ids: List[str] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")

        required = {"row_index", "bout_id", "speed_mean", "partition", "training_seed"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for expected_row, row in enumerate(reader):
            row_index = int(row["row_index"])
            if row_index != expected_row:
                raise RuntimeError(
                    f"{path}: row_index mismatch at row {expected_row}: observed {row_index}"
                )
            if row["partition"] != partition:
                raise RuntimeError(f"{path}: partition mismatch {row['partition']!r}")
            if int(row["training_seed"]) != ssl_seed:
                raise RuntimeError(
                    f"{path}: training_seed mismatch {row['training_seed']!r}"
                )

            speed = float(row["speed_mean"])
            if not np.isfinite(speed):
                raise RuntimeError(
                    f"{path}: non-finite speed_mean at row {expected_row}"
                )

            speeds.append(speed)
            bout_ids.append(row["bout_id"])

    speed_arr = np.asarray(speeds, dtype=np.float64)
    if speed_arr.shape[0] != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"observed {speed_arr.shape[0]:,}"
        )

    return speed_arr, bout_ids


def load_labels(
    label_root: Path, *, ssl_seed: int, partition: str
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = label_root / f"seed{ssl_seed}" / f"{partition}_labels.npy"
    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected shape ({EXPECTED_ROWS[partition]},), got {labels.shape}"
        )

    unique = np.unique(labels)
    expected = np.arange(EXPECTED_K)
    if not np.array_equal(unique, expected):
        raise RuntimeError(
            f"{path}: expected cluster IDs {expected.tolist()}, observed {unique.tolist()}"
        )

    return labels


def summarize_speed_by_cluster(
    speed: np.ndarray, labels: np.ndarray, *, k: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cluster in range(k):
        values = speed[labels == cluster]
        if values.size == 0:
            raise RuntimeError(f"Cluster {cluster} is empty.")
        out[str(cluster)] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return out


def eta_squared(speed: np.ndarray, labels: np.ndarray) -> float:
    grand_mean = float(np.mean(speed))
    ss_total = float(np.sum((speed - grand_mean) ** 2))
    if ss_total <= 0:
        return 0.0

    ss_between = 0.0
    for cluster in np.unique(labels):
        values = speed[labels == cluster]
        cluster_mean = float(np.mean(values))
        ss_between += float(values.size) * (cluster_mean - grand_mean) ** 2

    return float(ss_between / ss_total)


def majority_accuracy(labels: np.ndarray) -> float:
    counts = np.bincount(labels, minlength=EXPECTED_K)
    return float(np.max(counts) / np.sum(counts))


def fit_speed_only_classifier(
    train_speed: np.ndarray,
    train_labels: np.ndarray,
    validation_speed: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
) -> Dict[str, Any]:
    train_x = train_speed.reshape(-1, 1).astype(np.float64, copy=False)
    validation_x = validation_speed.reshape(-1, 1).astype(np.float64, copy=False)

    mean = float(np.mean(train_x[:, 0]))
    std = float(np.std(train_x[:, 0]))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid TRAIN speed normalization.")

    train_z = (train_x - mean) / std
    validation_z = (validation_x - mean) / std

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        random_state=seed,
    )
    model.fit(train_z, train_labels)

    train_pred = model.predict(train_z)
    validation_pred = model.predict(validation_z)

    return {
        "model": "multinomial_logistic_regression",
        "input_features": ["speed_mean"],
        "normalization": {
            "fit_partition": "train",
            "mean": mean,
            "std": std,
        },
        "train": {
            "accuracy": float(accuracy_score(train_labels, train_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(train_labels, train_pred)
            ),
            "macro_f1": float(
                f1_score(train_labels, train_pred, average="macro", zero_division=0)
            ),
            "majority_accuracy": majority_accuracy(train_labels),
            "chance_accuracy": float(1.0 / EXPECTED_K),
        },
        "validation": {
            "accuracy": float(accuracy_score(validation_labels, validation_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(validation_labels, validation_pred)
            ),
            "macro_f1": float(
                f1_score(
                    validation_labels,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "majority_accuracy": majority_accuracy(validation_labels),
            "chance_accuracy": float(1.0 / EXPECTED_K),
        },
        "test_partition_used": False,
    }


def aggregate_seed_results(results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "train_eta_squared": [],
        "validation_eta_squared": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_balanced_accuracy": [],
        "validation_balanced_accuracy": [],
        "train_macro_f1": [],
        "validation_macro_f1": [],
    }

    for result in results.values():
        metrics["train_eta_squared"].append(result["eta_squared"]["train"])
        metrics["validation_eta_squared"].append(result["eta_squared"]["validation"])

        classifier = result["speed_only_classifier"]
        metrics["train_accuracy"].append(classifier["train"]["accuracy"])
        metrics["validation_accuracy"].append(classifier["validation"]["accuracy"])
        metrics["train_balanced_accuracy"].append(
            classifier["train"]["balanced_accuracy"]
        )
        metrics["validation_balanced_accuracy"].append(
            classifier["validation"]["balanced_accuracy"]
        )
        metrics["train_macro_f1"].append(classifier["train"]["macro_f1"])
        metrics["validation_macro_f1"].append(classifier["validation"]["macro_f1"])

    summary: Dict[str, Any] = {}
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether selected SSL clusters are explainable by speed alone."
    )
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_root = args.metadata_root.resolve()
    label_root = args.label_root.resolve()
    training_config = args.training_config.resolve()
    output_dir = args.output_dir.resolve()

    assert_no_test_artifacts(metadata_root)
    seeds = configured_ssl_seeds(training_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "aggregate_summary.json"
    checksum_path = output_dir / "SPEED_DEPENDENCE_SHA256SUMS"

    if not args.overwrite and aggregate_path.exists():
        raise FileExistsError(
            f"{aggregate_path} already exists. Use --overwrite for intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL SPEED-DEPENDENCE ANALYSIS")
    print("=" * 80)
    print(f"SSL seeds:            {list(seeds)}")
    print(f"Selected clusters:    k={EXPECTED_K}")
    print("Speed feature:        speed_mean")
    print("Classifier fit:       TRAIN only")
    print("Evaluation:           TRAIN + VALIDATION")
    print("TEST partition:       PROTECTED / NOT LOADED")
    print()

    seed_results: Dict[int, Dict[str, Any]] = {}
    written_artifacts: List[Path] = []

    for seed in seeds:
        print("=" * 80)
        print(f"SSL SEED {seed}")
        print("=" * 80)

        train_speed, train_bouts = load_metadata_speed(
            metadata_root, ssl_seed=seed, partition="train"
        )
        validation_speed, validation_bouts = load_metadata_speed(
            metadata_root, ssl_seed=seed, partition="validation"
        )

        train_labels = load_labels(
            label_root, ssl_seed=seed, partition="train"
        )
        validation_labels = load_labels(
            label_root, ssl_seed=seed, partition="validation"
        )

        if len(train_bouts) != train_labels.shape[0]:
            raise RuntimeError("TRAIN metadata/label length mismatch.")
        if len(validation_bouts) != validation_labels.shape[0]:
            raise RuntimeError("VALIDATION metadata/label length mismatch.")

        train_summary = summarize_speed_by_cluster(
            train_speed, train_labels, k=EXPECTED_K
        )
        validation_summary = summarize_speed_by_cluster(
            validation_speed, validation_labels, k=EXPECTED_K
        )

        train_eta = eta_squared(train_speed, train_labels)
        validation_eta = eta_squared(validation_speed, validation_labels)

        classifier = fit_speed_only_classifier(
            train_speed,
            train_labels,
            validation_speed,
            validation_labels,
            seed=20260822,
        )

        seed_result = {
            "ssl_seed": int(seed),
            "k": EXPECTED_K,
            "cluster_speed_summary": {
                "train": train_summary,
                "validation": validation_summary,
            },
            "eta_squared": {
                "train": train_eta,
                "validation": validation_eta,
            },
            "speed_only_classifier": classifier,
            "test_partition_used": False,
        }
        seed_results[seed] = seed_result

        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cluster_summary_path = seed_dir / "cluster_speed_summary.json"
        classifier_path = seed_dir / "speed_only_classifier.json"

        atomic_write_json(
            cluster_summary_path,
            {
                "ssl_seed": int(seed),
                "k": EXPECTED_K,
                "train": train_summary,
                "validation": validation_summary,
                "eta_squared": {
                    "train": train_eta,
                    "validation": validation_eta,
                },
                "test_partition_used": False,
            },
        )
        atomic_write_json(
            classifier_path,
            {
                "ssl_seed": int(seed),
                "k": EXPECTED_K,
                **classifier,
            },
        )
        written_artifacts.extend([cluster_summary_path, classifier_path])

        print(f"TRAIN eta^2(speed ~ cluster):       {train_eta:.6f}")
        print(f"VALIDATION eta^2(speed ~ cluster):  {validation_eta:.6f}")
        print(
            f"TRAIN speed-only balanced accuracy: "
            f"{classifier['train']['balanced_accuracy']:.6f}"
        )
        print(
            f"VALIDATION balanced accuracy:       "
            f"{classifier['validation']['balanced_accuracy']:.6f}"
        )
        print(
            f"VALIDATION speed-only macro F1:     "
            f"{classifier['validation']['macro_f1']:.6f}"
        )
        print(
            f"VALIDATION ordinary accuracy:       "
            f"{classifier['validation']['accuracy']:.6f}"
        )
        print(
            f"VALIDATION majority baseline:       "
            f"{classifier['validation']['majority_accuracy']:.6f}"
        )
        print(f"Chance level:                       {1.0 / EXPECTED_K:.6f}")
        print("TEST partition used: NO")
        print()

    aggregate_stats = aggregate_seed_results(seed_results)

    aggregate_payload = {
        "dataset_id": "DS-005",
        "representation": "ssl_encoder_embedding",
        "selected_clustering": {
            "method": "kmeans",
            "k": EXPECTED_K,
        },
        "speed_feature": "speed_mean",
        "ssl_training_seeds": list(seeds),
        "aggregate_metrics": aggregate_stats,
        "per_seed": {
            str(seed): result for seed, result in seed_results.items()
        },
        "interpretation_guardrails": {
            "eta_squared": (
                "Fraction of speed variance explained by cluster labels; "
                "higher values indicate stronger speed separation."
            ),
            "balanced_accuracy": (
                "How well speed alone predicts cluster membership while "
                "weighting clusters equally."
            ),
            "macro_f1": "Class-balanced F1 for speed-only prediction.",
            "no_fixed_pass_fail_threshold": True,
            "note": (
                "Interpret relative to chance, majority baseline, and "
                "TRAIN-to-VALIDATION generalization rather than using an "
                "invented universal cutoff."
            ),
        },
        "test_partition_used": False,
    }

    atomic_write_json(aggregate_path, aggregate_payload)
    written_artifacts.append(aggregate_path)
    write_checksums(checksum_path, written_artifacts)

    print("=" * 80)
    print("SPEED-DEPENDENCE SUMMARY")
    print("=" * 80)
    print(
        "Mean TRAIN eta^2:                  "
        f"{aggregate_stats['train_eta_squared']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION eta^2:             "
        f"{aggregate_stats['validation_eta_squared']['mean']:.6f}"
    )
    print(
        "Mean TRAIN balanced accuracy:      "
        f"{aggregate_stats['train_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION balanced accuracy: "
        f"{aggregate_stats['validation_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION macro F1:          "
        f"{aggregate_stats['validation_macro_f1']['mean']:.6f}"
    )
    print()
    print(f"Chance accuracy for k=8:            {1.0 / EXPECTED_K:.6f}")
    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
