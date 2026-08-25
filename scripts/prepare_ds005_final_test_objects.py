#!/usr/bin/env python3
"""Freeze DS-005 TRAIN-fitted objects needed by the final TEST runner.

This program cannot address a path containing ``test`` and loads only the
frozen TRAIN and VALIDATION artifacts.  It reconstructs the already-selected
models, verifies their predictions against the frozen labels, and serializes
the objects so the one-time TEST run performs transforms/predictions only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.evaluation.frozen_models import FrozenKMeansPredictor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/DS-005/frozen_final_test_objects"
SEEDS = (11, 23, 37, 51, 79)
N_TRAIN = 842_841
N_VALIDATION = 168_464
K = 8
RANDOM_SEED = 20260822

EMBEDDING_HASHES = {
    11: ("7a0c301929ec94dcc1d4c0ff38d6dd1d0956bcce55a8fc927e3d328b1cec9ddf", "0a03a93be5adc237826d63479a0d8356ce11fb3b21a82fa90a410dc03a9d450b"),
    23: ("f0285715b4436a45510c42d29a7d6b42ee3fe43b13ff4a352526b3a1e029896d", "f5b3407f630d7b8c5767ae1db1121747d6a5b20467f98453ac8b8067b19b2973"),
    37: ("92ac58eafe3edc1c901ac13695efbc77396b3e0a3d66033e111f4ea2f51e9f3e", "2b8fefdd7ab51bd84931b22040f1d85f714a5ac04651353de2d3bd75cc2ddc10"),
    51: ("5adf8d5336569a62d97e808e42a2cf6069a88732887f764ad0d21871cf7d8ae7", "ce8082ceee792ca72b4aa071a6163776b10edd934c440ef0ca70a8832a1ad066"),
    79: ("af56cfa0a2d8ca5e8431e6cedbbee9225af3bf18649f52301b7457b8b58daad1", "56f0506b83da04d2cc6d1087953da9ddd635aeeebf50a16a453a8d5d53fe8d46"),
}
BASELINE_HASHES = {
    "train": "b0f21568c7ef933f4d1341d9999afd09984c19523a7ea7f5da79ddbf742b2806",
    "validation": "05696e864da5460e18b52bc3500263222ead4bd887580812bfb192c5bacff229",
}
CENTER_HASHES = {
    11: "8f56cb2ba4bdc017ea9fa56bfd8850d02c1bf665c4b29148253da0fa2644ae3c",
    23: "6eb77b4759bb7c39935d87914544a05249e1e839ab1c5e751eddd1f464f7810a",
    37: "eb0c95082a41040db736705140eba239aa4a46df01e92ca656d6fbc3311c8460",
    51: "c69201722e3e3e055f335edc56c643267bb75a2ffd3c4e0573eae31fb0a2fba6",
    79: "2a882aa40e04a59c78cbb8c26e2d2af706a0fd0ee18996ea4c5cdbf4dd260542",
}
LABEL_HASHES = {
    11: ("5fac3d27fcb4c6be45dca0da4d5a76c5e2307ad89ad0f134f652c7e214e16bce", "f4a41a3f70402a2278301992e2dff5763177722c1e3e5dfdb60575657222d15b"),
    23: ("0b115bab0c929c8234be83cd8d04541c769e32163f8a43ced72c703760bb73de", "01c37c08f920f0f8f44b21a00718c7d32a9978caa96fac9d2be093e9ccbc074d"),
    37: ("87cd8f7a0c299e82c81480c3e47a94590068d06da7da62a6c346b35f64b9abfb", "8198fae6d892dcbc28e3b1ba66bf4bef5a67c19eff48bc1866a47d3ce3ebe7b6"),
    51: ("743349acd9512bcd4b7ab08083df45811330b18376754d01ba0db6c8304c5968", "0e3c0944670168ad099ae4d1e79e51fa0632dd21fa141a8154f5af9c047a6d35"),
    79: ("f9599de1fca15d952da9995b85a5bcf4cbe251dde3d5d9fdcbf1ef01053fca2b", "32aa6503d43fae0fccba297c1cc64468befe23c1a93ca88697bbc59a0abe8172"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str) -> None:
    if "test" in path.name.lower():
        raise RuntimeError(f"TEST access is prohibited during preparation: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def atomic_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    os.replace(tmp, path)


def load_x(path: Path, expected: tuple[int, int]) -> np.ndarray:
    if "test" in path.name.lower():
        raise RuntimeError("TEST path prohibited")
    with np.load(path, allow_pickle=False) as z:
        key = "embeddings" if "embeddings" in z.files else "X"
        x = np.asarray(z[key], dtype=np.float32)
    if x.shape != expected or not np.isfinite(x).all():
        raise RuntimeError(f"Invalid frozen matrix {path}: {x.shape}")
    return x


def dump(obj: object, path: Path, artifacts: dict[str, str]) -> None:
    joblib.dump(obj, path)
    artifacts[str(path.relative_to(ROOT))] = sha256(path)


def main() -> None:
    if (OUT / "object_manifest.json").exists():
        raise SystemExit(f"Refusing to overwrite completed frozen objects: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    sources: dict[str, str] = {}

    baseline = {}
    for partition, rows in (("train", N_TRAIN), ("validation", N_VALIDATION)):
        path = ROOT / f"data/processed/DS-005/baseline/{partition}_core_scaled.npz"
        verify(path, BASELINE_HASHES[partition])
        sources[str(path.relative_to(ROOT))] = sha256(path)
        baseline[partition] = load_x(path, (rows, 18))

    pca_b = PCA(n_components=6, svd_solver="auto")
    train_b = pca_b.fit_transform(baseline["train"])
    validation_b = pca_b.transform(baseline["validation"])
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=RANDOM_SEED,
        n_init=5, reg_covar=1e-6,
    ).fit(train_b)
    baseline_label_paths = (
        ROOT / "data/processed/DS-005/baseline_vs_ssl/baseline_labels/train_labels.npy",
        ROOT / "data/processed/DS-005/baseline_vs_ssl/baseline_labels/validation_labels.npy",
    )
    for path, digest in zip(baseline_label_paths, ("e8dd9c8d6c3614b36034191b3cb4020205827b65fd10b074e45a9afa7b205f54", "61b13f275cfff148b1f858ac266876b7eb350bac0802189afd40b825ef6ef3bf")):
        verify(path, digest)
        sources[str(path.relative_to(ROOT))] = digest
    expected_train = np.load(baseline_label_paths[0], allow_pickle=False)
    expected_validation = np.load(baseline_label_paths[1], allow_pickle=False)
    if not np.array_equal(gmm.predict(train_b), expected_train) or not np.array_equal(gmm.predict(validation_b), expected_validation):
        raise RuntimeError("Reconstructed baseline PCA/GMM does not reproduce frozen labels")
    for name, obj in (("baseline_pca.joblib", pca_b), ("baseline_gmm.joblib", gmm)):
        path = OUT / name
        if not path.exists():
            joblib.dump(obj, path)
        artifacts[str(path.relative_to(ROOT))] = sha256(path)

    alignment_path = ROOT / "data/processed/DS-005/ssl_cluster_stability/alignment_maps.json"
    verify(alignment_path, "d06ef611e879fef9f33a2645846c935d376697d71a6596222c77bf943d559ba2")
    sources[str(alignment_path.relative_to(ROOT))] = sha256(alignment_path)
    alignment = json.loads(alignment_path.read_text())
    for seed in SEEDS:
        seed_out = OUT / f"seed{seed}"
        seed_out.mkdir(exist_ok=True)
        arrays = {}
        for i, (partition, rows) in enumerate((("train", N_TRAIN), ("validation", N_VALIDATION))):
            path = ROOT / f"data/processed/DS-005/ssl/seed{seed}/{partition}_embeddings.npz"
            verify(path, EMBEDDING_HASHES[seed][i])
            sources[str(path.relative_to(ROOT))] = sha256(path)
            arrays[partition] = load_x(path, (rows, 64))

        scaler = StandardScaler(copy=True)
        train_z = scaler.fit_transform(arrays["train"])
        validation_z = scaler.transform(arrays["validation"])
        pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_SEED)
        train_p = pca.fit_transform(train_z)
        validation_p = pca.transform(validation_z)
        centers_path = ROOT / f"data/processed/DS-005/ssl_cluster_stability/seed{seed}/cluster_centers_pca.npy"
        verify(centers_path, CENTER_HASHES[seed])
        sources[str(centers_path.relative_to(ROOT))] = sha256(centers_path)
        label_paths = (
            ROOT / f"data/processed/DS-005/ssl_cluster_stability/seed{seed}/train_labels.npy",
            ROOT / f"data/processed/DS-005/ssl_cluster_stability/seed{seed}/validation_labels.npy",
        )
        for path, digest in zip(label_paths, LABEL_HASHES[seed]):
            verify(path, digest)
            sources[str(path.relative_to(ROOT))] = digest
        expected_train = np.load(label_paths[0], allow_pickle=False)
        expected_validation = np.load(label_paths[1], allow_pickle=False)
        # Use the checksum-verified centers from the original frozen fit.
        # FrozenKMeansPredictor promotes them for stable distance arithmetic.
        centers = np.asarray(np.load(centers_path, allow_pickle=False))
        kmeans = FrozenKMeansPredictor(centers)
        raw_train = kmeans.predict(train_p)
        raw_validation = kmeans.predict(validation_p)
        mapping = {int(k): int(v) for k, v in alignment["train"][str(seed)]["mapping_to_reference"].items()}
        aligned_train = np.asarray([mapping[int(x)] for x in raw_train])
        aligned_validation = np.asarray([mapping[int(x)] for x in raw_validation])
        if not np.array_equal(raw_train, expected_train) or not np.array_equal(raw_validation, expected_validation):
            raise RuntimeError(f"Reconstructed seed {seed} clustering does not reproduce frozen labels")

        for name, obj in (("scaler.joblib", scaler), ("pca.joblib", pca), ("kmeans.joblib", kmeans)):
            path = seed_out / name
            joblib.dump(obj, path)
            artifacts[str(path.relative_to(ROOT))] = sha256(path)

        probe_paths = [seed_out / name for name in ("linear_probe.joblib", "nonlinear_probe.joblib", "speed_scaler.joblib", "speed_probe.joblib")]
        if not all(path.exists() for path in probe_paths):
            linear = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED).fit(baseline["train"], raw_train)
            nonlinear = HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
                max_depth=None, min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.10,
                n_iter_no_change=20, random_state=RANDOM_SEED,
            ).fit(baseline["train"], raw_train)
            speed_index = 2
            speed_scaler = StandardScaler().fit(baseline["train"][:, [speed_index]])
            speed_model = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=500,
                random_state=seed,
            ).fit(speed_scaler.transform(baseline["train"][:, [speed_index]]), raw_train)
            for path, obj in zip(probe_paths, (linear, nonlinear, speed_scaler, speed_model)):
                joblib.dump(obj, path)
        for path in probe_paths:
            artifacts[str(path.relative_to(ROOT))] = sha256(path)

    manifest = {
        "mode": "DS005_TRAIN_VALIDATION_ONLY_OBJECT_FREEZE",
        "test_opened": False,
        "test_path_accessed": False,
        "frozen_configuration": {
            "ssl_seeds": list(SEEDS), "ssl_k": K, "reference_seed": 11,
            "baseline": {"pca_components": 6, "method": "GMM", "k": 2},
            "probe_definitions_unchanged": True,
        },
        "source_hashes": sources,
        "artifact_hashes": artifacts,
        "train_label_reproduction_verified": True,
        "validation_label_reproduction_verified": True,
    }
    atomic_json(OUT / "object_manifest.json", manifest)
    print(f"Prepared {len(artifacts)} frozen objects without opening TEST.")


if __name__ == "__main__":
    main()
