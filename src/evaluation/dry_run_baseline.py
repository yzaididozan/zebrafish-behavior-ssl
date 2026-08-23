"""TRAIN/VALIDATION-only dry run of the frozen DS-005 baseline evaluation.

Purpose
-------
Exercise the already-frozen evaluation procedures on real project-shaped
baseline data without opening or using the DS-005 TEST partition.

This script:
1. Loads DS-005 TRAIN and VALIDATION baseline arrays only.
2. Fits PCA(6) on TRAIN scaled features only.
3. Fits GMM(k=2, seed=20260822) on TRAIN PCA scores only.
4. Predicts TRAIN and VALIDATION baseline cluster labels.
5. Runs held-out cluster occupancy on VALIDATION.
6. Runs a speed-only GMM(k=2) control and compares it with baseline clusters.
7. Runs a context-prediction nuisance probe on TRAIN -> VALIDATION.
8. Reports existing tracking-QC proxy rates.
9. Writes a dry-run JSON report outside the frozen baseline artifacts.

It NEVER loads any path containing "test".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.evaluation.nuisance_prediction import fit_context_probe
from src.evaluation.reproducibility import heldout_cluster_occupancy
from src.evaluation.speed_controls import (
    cluster_speed_summaries,
    compare_speed_only_to_ssl,
    speed_only_cluster_labels,
)


SEED = 20260822
PCA_COMPONENTS = 6
GMM_K = 2

ROOT = Path("data/processed/DS-005")
BASELINE_DIR = ROOT / "baseline"

TRAIN_SCALED = BASELINE_DIR / "train_core_scaled.npz"
VALIDATION_SCALED = BASELINE_DIR / "validation_core_scaled.npz"
TRAIN_RAW = BASELINE_DIR / "train_core_raw.npz"
VALIDATION_RAW = BASELINE_DIR / "validation_core_raw.npz"

DEFAULT_OUTPUT = Path(
    "results/dry_runs/ds005_baseline_train_validation_dry_run.json"
)


def prohibit_test_path(path: Path) -> None:
    """Hard-stop if a path could refer to TEST data."""
    if "test" in str(path).lower():
        raise RuntimeError(
            f"TEST access prohibited during TRAIN/VALIDATION dry run: {path}"
        )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def assert_partition(data: dict[str, np.ndarray], expected: str) -> None:
    values = np.unique(data["partition"])
    if values.size != 1 or str(values[0]).lower() != expected.lower():
        raise RuntimeError(
            f"Expected only partition={expected!r}; found {values.tolist()!r}"
        )


def jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass-like values into JSON-safe objects."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: jsonable(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def deterministic_subsample(
    n_rows: int,
    max_rows: int | None,
    *,
    seed: int,
) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DS-005 baseline evaluation dry run using TRAIN and "
            "VALIDATION only. TEST access is prohibited."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--probe-max-train",
        type=int,
        default=100_000,
        help=(
            "Maximum TRAIN rows used only for the context nuisance probe. "
            "PCA/GMM still use full TRAIN. Use 0 for all rows."
        ),
    )
    parser.add_argument(
        "--probe-max-validation",
        type=int,
        default=50_000,
        help=(
            "Maximum VALIDATION rows used only for the context nuisance probe. "
            "PCA/GMM still use full VALIDATION. Use 0 for all rows."
        ),
    )
    args = parser.parse_args()

    for path in (
        TRAIN_SCALED,
        VALIDATION_SCALED,
        TRAIN_RAW,
        VALIDATION_RAW,
        args.output,
    ):
        prohibit_test_path(path)

    print("DS-005 BASELINE TRAIN/VALIDATION DRY RUN")
    print("=" * 44)
    print("TEST partition status: PROHIBITED / NOT LOADED")
    print()

    train_scaled = load_npz(TRAIN_SCALED)
    validation_scaled = load_npz(VALIDATION_SCALED)
    train_raw = load_npz(TRAIN_RAW)
    validation_raw = load_npz(VALIDATION_RAW)

    assert_partition(train_scaled, "train")
    assert_partition(train_raw, "train")
    assert_partition(validation_scaled, "validation")
    assert_partition(validation_raw, "validation")

    feature_names = train_raw["feature_names"].astype(str)
    validation_feature_names = validation_raw["feature_names"].astype(str)
    np.testing.assert_array_equal(feature_names, validation_feature_names)

    if "speed_mean" not in set(feature_names):
        raise RuntimeError("Required frozen feature 'speed_mean' was not found.")

    speed_idx = int(np.flatnonzero(feature_names == "speed_mean")[0])

    x_train = np.asarray(train_scaled["X"], dtype=np.float32)
    x_validation = np.asarray(validation_scaled["X"], dtype=np.float32)

    print(f"TRAIN rows:      {x_train.shape[0]:,}")
    print(f"VALIDATION rows: {x_validation.shape[0]:,}")
    print(f"Input features:  {x_train.shape[1]}")
    print()

    # ------------------------------------------------------------------
    # Frozen baseline: PCA(6) -> GMM(k=2)
    # ------------------------------------------------------------------
    print("1. Frozen baseline clustering")
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="auto")
    train_pca = pca.fit_transform(x_train)
    validation_pca = pca.transform(x_validation)

    explained = float(np.sum(pca.explained_variance_ratio_))

    gmm = GaussianMixture(
        n_components=GMM_K,
        random_state=SEED,
    )
    gmm.fit(train_pca)

    train_labels = gmm.predict(train_pca)
    validation_labels = gmm.predict(validation_pca)

    print(
        f"   PCA({PCA_COMPONENTS}) TRAIN explained variance: "
        f"{explained:.4f}"
    )
    print(
        "   TRAIN cluster counts: "
        + str(np.bincount(train_labels, minlength=GMM_K).tolist())
    )
    print(
        "   VALIDATION cluster counts: "
        + str(np.bincount(validation_labels, minlength=GMM_K).tolist())
    )

    # ------------------------------------------------------------------
    # Held-out fish occupancy
    # ------------------------------------------------------------------
    print("\n2. VALIDATION held-out cluster occupancy")
    occupancy = heldout_cluster_occupancy(
        validation_labels,
        validation_scaled["fish_id"],
    )
    for row in occupancy:
        print(
            f"   cluster={row.cluster}: "
            f"fish_with_cluster={row.n_fish_with_cluster}, "
            f"median_per_fish_occupancy={row.median_per_fish_occupancy:.4f}"
        )

    # ------------------------------------------------------------------
    # Speed-only control
    # ------------------------------------------------------------------
    print("\n3. Speed-only control")
    train_speed = np.asarray(train_raw["X"][:, speed_idx], dtype=float)
    validation_speed = np.asarray(
        validation_raw["X"][:, speed_idx], dtype=float
    )

    validation_speed_labels = speed_only_cluster_labels(
        train_speed,
        validation_speed,
        method="gmm",
        k=GMM_K,
        seed=SEED,
    )

    speed_comparison = compare_speed_only_to_ssl(
        validation_speed_labels,
        validation_labels,
    )

    speed_summaries, speed_h, speed_p = cluster_speed_summaries(
        validation_labels,
        validation_speed,
    )

    print(f"   speed-only vs baseline ARI: {speed_comparison.ari:.4f}")
    print(f"   speed-only vs baseline NMI: {speed_comparison.nmi:.4f}")
    print(f"   cluster-speed Kruskal H:    {speed_h:.4f}")
    print(f"   cluster-speed p:            {speed_p:.6g}")

    # ------------------------------------------------------------------
    # Context nuisance-probe plumbing
    # ------------------------------------------------------------------
    print("\n4. Context nuisance probe (TRAIN -> VALIDATION)")
    probe_train_idx = deterministic_subsample(
        x_train.shape[0],
        None if args.probe_max_train == 0 else args.probe_max_train,
        seed=SEED,
    )
    probe_validation_idx = deterministic_subsample(
        x_validation.shape[0],
        None if args.probe_max_validation == 0
        else args.probe_max_validation,
        seed=SEED + 1,
    )

    train_context = train_scaled["context_id"].astype(str)
    validation_context = validation_scaled["context_id"].astype(str)

    # A classifier cannot predict a target class it never encountered in
    # TRAIN. Fail explicitly rather than silently dropping classes.
    unseen_contexts = set(np.unique(validation_context)) - set(
        np.unique(train_context)
    )
    if unseen_contexts:
        raise RuntimeError(
            "VALIDATION contains context classes absent from TRAIN: "
            f"{sorted(unseen_contexts)}"
        )

    _, context_result = fit_context_probe(
        x_train[probe_train_idx],
        train_context[probe_train_idx],
        x_validation[probe_validation_idx],
        validation_context[probe_validation_idx],
    )

    print(f"   probe TRAIN rows:      {probe_train_idx.size:,}")
    print(f"   probe VALIDATION rows: {probe_validation_idx.size:,}")
    print(
        f"   balanced accuracy:     "
        f"{context_result.balanced_accuracy:.4f}"
    )
    print(f"   macro F1:              {context_result.macro_f1:.4f}")
    print(f"   uniform chance:        {context_result.uniform_chance:.4f}")
    print(f"   chance ratio:          {context_result.chance_ratio:.4f}")

    # ------------------------------------------------------------------
    # Existing QC proxies; no new exclusions.
    # ------------------------------------------------------------------
    print("\n5. Existing tracking/QC proxy summaries")
    qc = {}
    for partition_name, data in (
        ("train", train_raw),
        ("validation", validation_raw),
    ):
        qc[partition_name] = {
            "all_zero_speed_rate": float(
                np.mean(data["all_zero_speed"].astype(bool))
            ),
            "extreme_speed_gt_100_rate": float(
                np.mean(data["extreme_speed_gt_100"].astype(bool))
            ),
        }
        print(
            f"   {partition_name}: "
            f"all_zero_speed={qc[partition_name]['all_zero_speed_rate']:.6f}, "
            f"extreme_speed_gt_100="
            f"{qc[partition_name]['extreme_speed_gt_100_rate']:.6f}"
        )

    report = {
        "status": "DRY_RUN_ONLY",
        "dataset_id": "DS-005",
        "partitions_loaded": ["train", "validation"],
        "test_loaded": False,
        "governance": {
            "test_access_prohibited": True,
            "new_post_clustering_exclusions_applied": False,
            "method_selection_performed": False,
            "frozen_configuration_used": {
                "pca_components": PCA_COMPONENTS,
                "method": "gmm",
                "k": GMM_K,
                "seed": SEED,
            },
        },
        "data": {
            "train_rows": int(x_train.shape[0]),
            "validation_rows": int(x_validation.shape[0]),
            "n_features": int(x_train.shape[1]),
            "feature_names": feature_names.tolist(),
            "train_unique_fish": int(
                np.unique(train_scaled["fish_id"]).size
            ),
            "validation_unique_fish": int(
                np.unique(validation_scaled["fish_id"]).size
            ),
        },
        "baseline": {
            "pca_explained_variance": explained,
            "train_cluster_counts": np.bincount(
                train_labels, minlength=GMM_K
            ).tolist(),
            "validation_cluster_counts": np.bincount(
                validation_labels, minlength=GMM_K
            ).tolist(),
        },
        "heldout_cluster_occupancy": jsonable(occupancy),
        "speed_control": {
            "speed_feature": "speed_mean",
            "speed_only_method": "gmm",
            "speed_only_k": GMM_K,
            "speed_only_vs_baseline": jsonable(speed_comparison),
            "baseline_cluster_speed_summaries": jsonable(speed_summaries),
            "kruskal_h": speed_h,
            "kruskal_p": speed_p,
        },
        "context_probe": {
            "train_rows_used": int(probe_train_idx.size),
            "validation_rows_used": int(probe_validation_idx.size),
            "result": jsonable(context_result),
        },
        "qc_proxies": qc,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nDRY RUN COMPLETE")
    print("TEST partition status: NOT LOADED")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
