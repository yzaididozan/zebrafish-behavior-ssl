#!/usr/bin/env python3
"""One-time, inference-only DS-005 held-out TEST evaluation.

This is the only project program authorized to read the DS-005 TEST baseline
matrices or TEST bouts in the canonical HDF5.  It exits before opening TEST
unless both ``--confirm-open-test`` and the exact clean freeze commit are
provided.  All estimators are loaded from the TRAIN/VALIDATION-only object
freeze; this file performs no fitting, tuning, alignment, or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
    balanced_accuracy_score, f1_score, normalized_mutual_info_score,
    silhouette_score,
)

from src.data.ds005 import DS005
from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.input import bout_to_ssl_input
from src.ssl.train import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 23, 37, 51, 79)
N_TEST = 192_104
K = 8
OUTPUT = ROOT / "data/processed/DS-005/final_test_evaluation"
OBJECT_ROOT = ROOT / "data/processed/DS-005/frozen_final_test_objects"
RAW_HDF5 = ROOT / "data/raw/DS-005/DS-005-v1/Datasets/JM_data/filtered_jmpool_kin.h5"
BASELINE_RAW = ROOT / "data/processed/DS-005/baseline/test_core_raw.npz"
BASELINE_SCALED = ROOT / "data/processed/DS-005/baseline/test_core_scaled.npz"
TRAINING_CONFIG = ROOT / "configs/ssl/training.yaml"
NORMALIZATION = ROOT / "configs/ssl/normalization.json"
ALIGNMENTS = ROOT / "data/processed/DS-005/ssl_cluster_stability/alignment_maps.json"
SILHOUETTE_MAX = 20_000
RANDOM_SEED = 20260822

FEATURES = (
    "bout_duration_s", "inter_bout_interval_s", "speed_mean", "speed_std",
    "speed_median", "speed_max", "speed_p95", "speed_rms",
    "accel_abs_mean", "accel_abs_std", "accel_abs_max", "accel_rms",
    "turn_abs_total_rad", "turn_net_rad", "turn_abs_mean_rad",
    "turn_abs_std_rad", "turn_abs_max_rad", "turn_rms_rad",
)
LONG_CS_FEATURES = ("bout_duration_s", "accel_rms", "accel_abs_std")
LLC_FEATURE = "turn_net_rad"
# Frozen project label-normalization logic maps numeric 1 -> Long_CS and 9 -> LLC.
LONG_CS_CODE = 1.0
LLC_CODE = 9.0

FROZEN_HASHES = {
    "data/raw/DS-005/DS-005-v1/Datasets/JM_data/filtered_jmpool_kin.h5": "7aa22dad1005d4a7d7929d590899e04ea7337a0d3db134587704c30be17ab4a3",
    "data/splits/DS-005-fish-split-v1.csv": "19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935",
    "data/processed/DS-005/baseline/test_core_raw.npz": "bbb5c53348720f873cc7be492a6a27e719c21586bd763b64b3b10d88c5f4f911",
    "data/processed/DS-005/baseline/test_core_scaled.npz": "bd9f9e4086fae94835409ca37c85163e22db607f3a76020c481dac12ab3474d6",
    "data/processed/DS-005/baseline/feature_schema_core.json": "a8b2fe73f3251f7788e99e6fb1fde2688256afbe34a39e40daa8431018a5e91a",
    "configs/ssl/training.yaml": "d70da9c8ac1064025f3eec0f8784770d2405cf543275ec01289dba32401271dc",
    "configs/ssl/normalization.json": "47110c19c4cb64cdfe0dd85f3adcf7926062d4f84c9e8ab284c93919d4b7857a",
    "data/processed/DS-005/ssl_cluster_stability/alignment_maps.json": "d06ef611e879fef9f33a2645846c935d376697d71a6596222c77bf943d559ba2",
    "results/ssl/checkpoints/ssl_seed11_best.pt": "f837bce2a8ba3fca50ff7689c51cf2f7c56904445e83941e7d0bfc83370ceeda",
    "results/ssl/checkpoints/ssl_seed23_best.pt": "2773586a0ddfbdb1427f5e344764a7fd37998fdc2c089cbf39001667e5b9b076",
    "results/ssl/checkpoints/ssl_seed37_best.pt": "21a048aee177492c21829b16099b792d722371e1ebaf1b9607130ef3bb2f8086",
    "results/ssl/checkpoints/ssl_seed51_best.pt": "66d6d2837f67ec9b4682a20ef001dd9af275d860e60c83147ba3ecea2f2a268f",
    "results/ssl/checkpoints/ssl_seed79_best.pt": "0c646674c8e1c0b35847a33f823552fb4f7b61ddb01921bcda34d120318525e8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-open-test", action="store_true", help="Required explicit authorization")
    p.add_argument("--freeze-commit", help="Required full pre-TEST freeze commit SHA")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Frozen SHA-256 mismatch for {path}: {observed} != {expected}")
    return observed


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    os.replace(tmp, path)


def require_authorization(args: argparse.Namespace) -> str:
    # This function is deliberately called before any TEST or HDF5 path opens.
    if not args.confirm_open_test:
        raise SystemExit("Refusing to open DS-005 TEST without --confirm-open-test")
    value = args.freeze_commit
    if value is None or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise SystemExit("--freeze-commit requires the full 40-character SHA")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    if head != value:
        raise SystemExit(f"Refusing TEST: HEAD {head} != freeze commit {value}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True, capture_output=True).stdout
    if status:
        raise SystemExit("Refusing TEST: working tree is not clean")
    if OUTPUT.exists():
        raise SystemExit(f"Refusing one-time rerun/overwrite: {OUTPUT}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    return head


def summarize(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    x, y = values[np.isfinite(values)], labels[np.isfinite(values)]
    total = float(np.sum((x - x.mean()) ** 2))
    if total <= 0:
        return 0.0
    return float(sum(np.sum(y == c) * (x[y == c].mean() - x.mean()) ** 2 for c in np.unique(y)) / total)


def profile(values: np.ndarray, labels: np.ndarray, statistic: str = "mean") -> np.ndarray:
    fn = np.nanmean if statistic == "mean" else np.nanmedian
    return np.asarray([fn(values[labels == c]) if np.any(labels == c) else np.nan for c in range(K)])


def contingency(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(values.astype(str), return_inverse=True)
    table = np.zeros((int(encoded.max()) + 1, K), dtype=np.int64)
    np.add.at(table, (encoded, labels), 1)
    return table


def association(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    _, encoded = np.unique(values.astype(str), return_inverse=True)
    table = contingency(values, labels)
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    mask = expected > 0
    d = min(table.shape[0] - 1, table.shape[1] - 1)
    v = math.sqrt((np.sum((table[mask] - expected[mask]) ** 2 / expected[mask]) / n) / d) if d > 0 else 0.0
    column = table / np.maximum(table.sum(0, keepdims=True), 1)
    p = column[column > 0]
    return {
        "nmi": float(normalized_mutual_info_score(encoded, labels)),
        "ami": float(adjusted_mutual_info_score(encoded, labels)),
        "cramers_v": float(v),
        "mean_normalized_entropy": float(-np.sum(p * np.log(p)) / K / math.log(max(table.shape[0], 2))),
        "maximum_concentration": float(column.max()),
    }


def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    value = 0.0
    for group in np.unique(x):
        subset = y[x == group]
        _, count = np.unique(subset, return_counts=True)
        p = count / count.sum()
        value += len(subset) / len(y) * float(-np.sum(p * np.log(p)))
    return value


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    def entropy(x: np.ndarray) -> float:
        _, count = np.unique(x, return_counts=True)
        p = count / count.sum()
        return float(-np.sum(p * np.log(p)))
    return {
        "ari": float(adjusted_rand_score(a, b)), "nmi": float(normalized_mutual_info_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
        "normalized_conditional_entropy_ssl_given_baseline": conditional_entropy(b, a) / entropy(b),
        "normalized_conditional_entropy_baseline_given_ssl": conditional_entropy(a, b) / entropy(a),
    }


def classification(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    return torch.device("mps" if name == "mps" or (name == "auto" and torch.backends.mps.is_available()) else "cpu")


def training_config() -> Mapping[str, Any]:
    cfg = yaml.safe_load(TRAINING_CONFIG.read_text())["training"]
    if tuple(cfg["seeds"]["values"]) != SEEDS or int(cfg["encoder"]["embedding_dim"]) != 64:
        raise RuntimeError("Frozen encoder configuration changed")
    return cfg


def model_for_seed(cfg: Mapping[str, Any], seed: int, device: torch.device) -> ContrastiveModel:
    enc, proj = cfg["encoder"], cfg["projection_head"]
    model = ContrastiveModel(EncoderConfig(
        input_channels=int(enc["architecture"]["input_channels"]), embedding_dim=64,
        projection_dim=int(proj["projection_dim"]), dropout=float(enc["architecture"]["dropout"]),
    )).to(device)
    payload = load_checkpoint(
        path=ROOT / f"results/ssl/checkpoints/ssl_seed{seed}_best.pt",
        model=model,
        optimizer=None,
        map_location=device,
    )
    if int(payload.get("training_seed", seed)) != seed:
        raise RuntimeError("Checkpoint seed mismatch")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_baseline() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(BASELINE_RAW, allow_pickle=False) as raw, np.load(BASELINE_SCALED, allow_pickle=False) as scaled:
        x_raw = np.asarray(raw["X"], dtype=np.float32)
        x_scaled = np.asarray(scaled["X"], dtype=np.float32)
        metadata = {name: np.asarray(raw[name]) for name in ("fish_id", "session_id", "fish_index", "bout_index", "partition", "context_id", "context_name", "bout_type")}
        names = tuple(np.asarray(raw["feature_names"]).astype(str))
    if x_raw.shape != (N_TEST, 18) or x_scaled.shape != (N_TEST, 18) or names != FEATURES:
        raise RuntimeError("Unexpected TEST baseline schema")
    if not np.isfinite(x_raw).all() or not np.isfinite(x_scaled).all() or np.any(metadata["partition"].astype(str) != "test"):
        raise RuntimeError("Invalid TEST baseline values/partition")
    keys = np.asarray([f"{f}::{int(b):06d}" for f, b in zip(metadata["fish_id"].astype(str), metadata["bout_index"])])
    if len(np.unique(keys)) != N_TEST:
        raise RuntimeError("TEST bout identities are not unique")
    metadata["bout_key"] = keys
    return x_raw, x_scaled, metadata


@torch.inference_mode()
def encode_test(model: ContrastiveModel, batch_size: int, device: torch.device, expected: dict[str, np.ndarray], speed_mean: float, speed_std: float) -> np.ndarray:
    embeddings = np.empty((N_TEST, 64), dtype=np.float32)
    batch: list[np.ndarray] = []
    row = 0
    with DS005() as ds:
        for bout in ds.iter_bouts(partition="test", primary_qc_only=True, include_optional=True):
            key = bout.key
            if str(key.fish_id) != str(expected["fish_id"][row]) or int(key.bout_index) != int(expected["bout_index"][row]):
                raise RuntimeError(f"SSL/baseline TEST row mismatch at {row}")
            x = bout_to_ssl_input(bout)
            x[:, 2] = (x[:, 2] - speed_mean) / speed_std
            batch.append(x)
            if len(batch) == batch_size:
                tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(device)
                embeddings[row - len(batch) + 1:row + 1] = model.encoder(tensor).cpu().numpy()
                batch.clear()
            row += 1
        if batch:
            tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(device)
            embeddings[row - len(batch):row] = model.encoder(tensor).cpu().numpy()
    if row != N_TEST or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Expected {N_TEST} TEST embeddings, got {row}")
    return embeddings


def class_analysis(mask: np.ndarray, x: np.ndarray, labels: np.ndarray, features: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": int(mask.sum()), "features": {}}
    for name in features:
        values = x[mask, FEATURES.index(name)]
        y = labels[mask]
        result["features"][name] = {
            "eta_squared": eta_squared(values, y),
            "mean_profile": profile(values, y, "mean").tolist(),
            "median_profile": profile(values, y, "median").tolist(),
        }
    return result


def final_test(args: argparse.Namespace) -> None:
    freeze = require_authorization(args)

    # Verify all hashes and serialized-object manifest before either TEST NPZ or HDF5 opens.
    input_hashes = {rel: verify(ROOT / rel, digest) for rel, digest in FROZEN_HASHES.items()}
    manifest_path = OBJECT_ROOT / "object_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing TRAIN/VALIDATION-only frozen objects; run scripts/prepare_ds005_final_test_objects.py before the freeze commit")
    object_manifest = json.loads(manifest_path.read_text())
    if object_manifest.get("test_opened") is not False or object_manifest.get("test_path_accessed") is not False:
        raise RuntimeError("Invalid frozen-object provenance")
    for rel, digest in object_manifest["artifact_hashes"].items():
        input_hashes[rel] = verify(ROOT / rel, digest)

    # All guards passed. These calls constitute the one-time TEST opening.
    x_raw, x_scaled, meta = load_baseline()
    OUTPUT.mkdir(parents=True)
    pca_b = joblib.load(OBJECT_ROOT / "baseline_pca.joblib")
    gmm = joblib.load(OBJECT_ROOT / "baseline_gmm.joblib")
    baseline_labels = np.asarray(gmm.predict(pca_b.transform(x_scaled)), dtype=np.int64)
    np.save(OUTPUT / "baseline_test_labels.npy", baseline_labels)

    normalization = json.loads(NORMALIZATION.read_text())
    speed_mean = float(normalization["normalization"]["speed_head"]["mean"])
    speed_std = float(normalization["normalization"]["speed_head"]["std"])
    cfg, device = training_config(), choose_device(args.device)
    alignment = json.loads(ALIGNMENTS.read_text())["train"]
    all_labels: dict[int, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    nuisance: dict[str, Any] = {}
    long_cs: dict[str, Any] = {"class": "Long_CS", "features": list(LONG_CS_FEATURES), "by_seed": {}}
    llc: dict[str, Any] = {"class": "LLC", "feature": LLC_FEATURE, "direction_threshold_rad": 0.10, "by_seed": {}}

    for seed in SEEDS:
        model = model_for_seed(cfg, seed, device)
        embeddings = encode_test(model, args.batch_size, device, meta, speed_mean, speed_std)
        seed_dir = OUTPUT / f"seed{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "test_embeddings.npz", embeddings=embeddings, fish_id=meta["fish_id"], bout_index=meta["bout_index"])
        scaler = joblib.load(OBJECT_ROOT / f"seed{seed}/scaler.joblib")
        pca = joblib.load(OBJECT_ROOT / f"seed{seed}/pca.joblib")
        kmeans = joblib.load(OBJECT_ROOT / f"seed{seed}/kmeans.joblib")
        reduced = pca.transform(scaler.transform(embeddings))
        raw_labels = np.asarray(kmeans.predict(reduced), dtype=np.int64)
        mapping = {int(k): int(v) for k, v in alignment[str(seed)]["mapping_to_reference"].items()}
        labels = np.asarray([mapping[int(x)] for x in raw_labels], dtype=np.int64)
        all_labels[seed] = labels
        np.save(seed_dir / "test_labels.npy", labels)

        distances = kmeans.transform(reduced)
        nearest = np.partition(distances, 1, axis=1)[:, :2]
        margin = (nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 1], 1e-12)
        sample = np.random.default_rng(RANDOM_SEED).choice(N_TEST, SILHOUETTE_MAX, replace=False)
        linear = joblib.load(OBJECT_ROOT / f"seed{seed}/linear_probe.joblib")
        nonlinear = joblib.load(OBJECT_ROOT / f"seed{seed}/nonlinear_probe.joblib")
        speed_scaler = joblib.load(OBJECT_ROOT / f"seed{seed}/speed_scaler.joblib")
        speed_probe = joblib.load(OBJECT_ROOT / f"seed{seed}/speed_probe.joblib")
        speed = x_raw[:, FEATURES.index("speed_mean")]
        metrics[str(seed)] = {
            "occupancy": {str(c): int(np.sum(labels == c)) for c in range(K)},
            "contributing_fish": {str(c): int(len(np.unique(meta["fish_id"][labels == c]))) for c in range(K)},
            "contributing_contexts": {str(c): int(len(np.unique(meta["context_id"][labels == c]))) for c in range(K)},
            "silhouette": float(silhouette_score(reduced[sample], raw_labels[sample])),
            "silhouette_sample_size": SILHOUETTE_MAX,
            "distance_margin_confidence": summarize(margin),
            "baseline_vs_ssl": compare(baseline_labels, labels),
            "linear_18_feature_probe": classification(raw_labels, linear.predict(x_scaled)),
            "nonlinear_18_feature_probe": classification(raw_labels, nonlinear.predict(x_scaled)),
            "test_used_for_fitting": False,
        }
        nuisance[str(seed)] = {
            "mean_speed_eta_squared": eta_squared(speed, labels),
            "speed_only": classification(raw_labels, speed_probe.predict(speed_scaler.transform(speed.reshape(-1, 1)))),
            "fish_identity": association(meta["fish_id"], labels),
            "session_identity": association(meta["session_id"], labels),
            "context_id": association(meta["context_id"], labels),
            "context_name": association(meta["context_name"], labels),
        }
        long_cs["by_seed"][str(seed)] = class_analysis(meta["bout_type"] == LONG_CS_CODE, x_raw, labels, LONG_CS_FEATURES)
        llc_result = class_analysis(meta["bout_type"] == LLC_CODE, x_raw, labels, (LLC_FEATURE,))
        means = llc_result["features"][LLC_FEATURE]["mean_profile"]
        llc_result["cluster_directions"] = {str(c): "positive" if means[c] > 0.10 else "negative" if means[c] < -0.10 else "neutral" for c in range(K)}
        llc_result["cluster_0_positive"] = means[0] > 0.10
        llc_result["cluster_6_negative"] = means[6] < -0.10
        llc["by_seed"][str(seed)] = llc_result
        atomic_json(seed_dir / "metrics.json", metrics[str(seed)])

    pairs = []
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            pairs.append({"seed_a": a, "seed_b": b, "ari": float(adjusted_rand_score(all_labels[a], all_labels[b])), "nmi": float(normalized_mutual_info_score(all_labels[a], all_labels[b])), "aligned_agreement": float(np.mean(all_labels[a] == all_labels[b]))})
    cross = {"pairs": pairs, **{name: summarize([p[name] for p in pairs]) for name in ("ari", "nmi", "aligned_agreement")}}

    # Frozen TRAIN profiles are used only as references; TEST never changes alignment.
    for seed in SEEDS:
        train_long = json.loads((ROOT / f"data/processed/DS-005/ssl_long_cs_kinematic_reproducibility/seed{seed}/train.json").read_text())
        for name in LONG_CS_FEATURES:
            test_profile = np.asarray(long_cs["by_seed"][str(seed)]["features"][name]["mean_profile"])
            train_profile = np.asarray(train_long["features"][name]["mean_profile"])
            long_cs["by_seed"][str(seed)]["features"][name]["train_to_test_spearman"] = float(spearmanr(train_profile, test_profile).statistic)
        train_llc = json.loads((ROOT / f"data/processed/DS-005/ssl_llc_turn_reproducibility/seed{seed}/train.json").read_text())
        test_profile = np.asarray(llc["by_seed"][str(seed)]["features"][LLC_FEATURE]["mean_profile"])
        llc["by_seed"][str(seed)]["train_to_test_spearman"] = float(spearmanr(np.asarray(train_llc["mean_profile"]), test_profile).statistic)

    for report, names in ((long_cs, LONG_CS_FEATURES), (llc, (LLC_FEATURE,))):
        report["cross_seed_test_profile_reproducibility"] = {}
        for name in names:
            profiles = [np.asarray(report["by_seed"][str(s)]["features"][name]["mean_profile"]) for s in SEEDS]
            rhos = [float(spearmanr(profiles[i], profiles[j]).statistic) for i in range(5) for j in range(i + 1, 5)]
            report["cross_seed_test_profile_reproducibility"][name] = summarize(rhos)

    mean_long_eta = {name: float(np.mean([long_cs["by_seed"][str(s)]["features"][name]["eta_squared"] for s in SEEDS])) for name in LONG_CS_FEATURES}
    mean_long_rho = {name: float(np.mean([long_cs["by_seed"][str(s)]["features"][name]["train_to_test_spearman"] for s in SEEDS])) for name in LONG_CS_FEATURES}
    mean_llc_eta = float(np.mean([llc["by_seed"][str(s)]["features"][LLC_FEATURE]["eta_squared"] for s in SEEDS]))
    mean_llc_rho = float(np.mean([llc["by_seed"][str(s)]["train_to_test_spearman"] for s in SEEDS]))
    long_supported = all(mean_long_eta[x] >= 0.25 and mean_long_rho[x] >= 0.50 for x in LONG_CS_FEATURES)
    long_weakened = all(mean_long_eta[x] >= 0.10 and mean_long_rho[x] >= 0.25 for x in LONG_CS_FEATURES)
    directional = sum(bool(llc["by_seed"][str(s)]["cluster_0_positive"] and llc["by_seed"][str(s)]["cluster_6_negative"]) for s in SEEDS)
    claims = {
        "allowed_statuses": ["SUPPORTED", "WEAKENED", "CONTRADICTED", "NOT_TESTABLE"],
        "rules_frozen_before_test": True,
        "assessments": {
            "general_ssl_k8_structure": "SUPPORTED" if cross["ari"]["mean"] >= 0.30 else "WEAKENED" if cross["ari"]["mean"] >= 0.15 else "CONTRADICTED",
            "baseline_and_ssl_organizations_differ": "SUPPORTED" if np.mean([metrics[str(s)]["baseline_vs_ssl"]["ari"] for s in SEEDS]) < 0.30 else "WEAKENED",
            "ssl_is_nonlinearly_recoverable_from_handcrafted_features": "SUPPORTED" if np.mean([metrics[str(s)]["nonlinear_18_feature_probe"]["balanced_accuracy"] for s in SEEDS]) >= 0.80 else "WEAKENED" if np.mean([metrics[str(s)]["nonlinear_18_feature_probe"]["balanced_accuracy"] for s in SEEDS]) >= 0.65 else "CONTRADICTED",
            "low_fish_identity_leakage": "SUPPORTED" if np.mean([nuisance[str(s)]["fish_identity"]["cramers_v"] for s in SEEDS]) <= 0.20 else "WEAKENED" if np.mean([nuisance[str(s)]["fish_identity"]["cramers_v"] for s in SEEDS]) <= 0.35 else "CONTRADICTED",
            "low_context_leakage": "SUPPORTED" if np.mean([nuisance[str(s)]["context_id"]["cramers_v"] for s in SEEDS]) <= 0.20 else "WEAKENED" if np.mean([nuisance[str(s)]["context_id"]["cramers_v"] for s in SEEDS]) <= 0.35 else "CONTRADICTED",
            "speed_related_but_not_speed_only": "SUPPORTED" if np.mean([nuisance[str(s)]["mean_speed_eta_squared"] for s in SEEDS]) >= 0.10 and np.mean([nuisance[str(s)]["speed_only"]["balanced_accuracy"] for s in SEEDS]) < 0.50 else "WEAKENED",
            "long_cs_primary_interpretation": "SUPPORTED" if long_supported else "WEAKENED" if long_weakened else "CONTRADICTED",
            "llc_secondary_interpretation": "SUPPORTED" if mean_llc_eta >= 0.10 and mean_llc_rho >= 0.50 and directional >= 4 else "WEAKENED" if mean_llc_eta >= 0.03 and mean_llc_rho >= 0.25 else "CONTRADICTED",
            "eight_clusters_are_distinct_novel_behaviors": "NOT_TESTABLE",
        },
        "thresholds": {"long_cs_supported_eta2": 0.25, "long_cs_supported_rho": 0.50, "llc_supported_eta2": 0.10, "llc_supported_rho": 0.50, "llc_directional_seeds": 4},
    }

    atomic_json(OUTPUT / "baseline_vs_ssl_summary.json", {str(s): {"comparison": metrics[str(s)]["baseline_vs_ssl"], "linear_probe": metrics[str(s)]["linear_18_feature_probe"], "nonlinear_probe": metrics[str(s)]["nonlinear_18_feature_probe"]} for s in SEEDS})
    atomic_json(OUTPUT / "cross_seed_summary.json", cross)
    atomic_json(OUTPUT / "nuisance_summary.json", nuisance)
    atomic_json(OUTPUT / "long_cs_primary_summary.json", long_cs)
    atomic_json(OUTPUT / "llc_secondary_summary.json", llc)
    atomic_json(OUTPUT / "claim_assessment.json", claims)

    outputs = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name not in {"run_manifest.json", "FINAL_TEST_SHA256SUMS"})
    output_hashes = {str(p.relative_to(OUTPUT)): sha256(p) for p in outputs}
    atomic_json(OUTPUT / "run_manifest.json", {
        "freeze_commit": freeze, "command": " ".join(sys.argv), "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {x: importlib.metadata.version(x) for x in ("numpy", "scipy", "scikit-learn", "torch", "joblib", "PyYAML")},
        "input_hashes": input_hashes, "output_hashes": output_hashes,
        "test_bouts": N_TEST, "input_shape": [N_TEST, 175, 3], "embedding_shape": [N_TEST, 64],
        "ssl_seeds": list(SEEDS), "long_cs_primary": list(LONG_CS_FEATURES), "llc_secondary": LLC_FEATURE,
        "test_used_for_fitting": False, "configuration_changed": False,
        "new_alignment_performed": False, "method_selection_performed": False,
        "prohibited_operations_performed": [], "exclusive_output_directory": str(OUTPUT.relative_to(ROOT)),
    })
    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "FINAL_TEST_SHA256SUMS")
    (OUTPUT / "FINAL_TEST_SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(OUTPUT)}\n" for p in files), encoding="utf-8")


def main() -> None:
    args = parse_args()
    final_test(args)


if __name__ == "__main__":
    main()
