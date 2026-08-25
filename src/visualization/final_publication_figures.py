#!/usr/bin/env python3
"""Generate final publication figures and tables from frozen result summaries.

This script is deliberately read-only with respect to scientific artifacts. It
does not load raw bouts, embeddings, or model objects and performs no fitting,
selection, clustering, or statistical testing. All outputs are deterministic
summaries of the committed DS-005 and DS-006 final TEST JSON files.

Outputs
-------
results/publication/
    figures/figure_01_study_design.{pdf,svg,png}
    figures/figure_02_stability_and_baseline.{pdf,svg,png}
    figures/figure_03_nuisance_controls.{pdf,svg,png}
    figures/figure_04_long_cs_primary.{pdf,svg,png}
    figures/figure_05_llc_secondary.{pdf,svg,png}
    figures/figure_06_external_replication.{pdf,svg,png}
    tables/table_01_datasets.{csv,md}
    tables/table_02_frozen_methods.{csv,md}
    tables/table_03_ds005_final_test.{csv,md}
    tables/table_04_biological_results.{csv,md}
    tables/table_05_external_replication.{csv,md}
    publication_manifest.json
    PUBLICATION_SHA256SUMS

Usage
-----
    PYTHONPATH=. python3 src/visualization/final_publication_figures.py

Intentional regeneration:
    PYTHONPATH=. python3 src/visualization/final_publication_figures.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zebrafish-publication-matplotlib")
)
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DS005 = ROOT / "data/processed/DS-005/final_test_evaluation"
DS006 = ROOT / "data/processed/DS-006/final_test_evaluation"
DEFAULT_OUTPUT = ROOT / "results/publication"
SEEDS = (11, 23, 37, 51, 79)
CLUSTERS = np.arange(8)

# Okabe-Ito colorblind-safe palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#222222"
GRAY = "#8A8A8A"
CLUSTER_COLORS = (BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, BLACK)

SOURCE_FILES = (
    DS005 / "run_manifest.json",
    DS005 / "claim_assessment.json",
    DS005 / "cross_seed_summary.json",
    DS005 / "baseline_vs_ssl_summary.json",
    DS005 / "nuisance_summary.json",
    DS005 / "long_cs_primary_summary.json",
    DS005 / "llc_secondary_summary.json",
    DS006 / "run_manifest.json",
    DS006 / "claim_assessment.json",
    DS006 / "cross_seed_summary.json",
    DS006 / "baseline_vs_ssl_summary.json",
    DS006 / "nuisance_summary.json",
    DS006 / "kinematic_axes_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def mean(values: Iterable[float]) -> float:
    return float(np.mean(np.asarray(list(values), dtype=float)))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "zebrafish-behavior-ssl-publication-v1",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: Path) -> List[Path]:
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def grouped_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    ylabel: str,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / len(series)
    for index, (name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.9, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)


def figure_study_design(output: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.3, 3.7, 2.25, 1.45, "DS-005 primary", "463 fish\n1,203,409 bouts\nfish-level split", BLUE),
        (0.3, 0.9, 2.25, 1.45, "DS-006 replication", "374 usable fish-wells\n163,065 bouts\nrecording-level split", GREEN),
        (3.1, 3.7, 2.45, 1.45, "Input A", "18 handcrafted features\nPCA(6) → GMM(k=2)", ORANGE),
        (3.1, 0.9, 2.45, 1.45, "Input B", "175 × 3 temporal bout\n1D CNN → 64-D embedding", PURPLE),
        (6.15, 2.3, 2.35, 1.45, "Frozen discovery", "Five SSL seeds\nPCA → KMeans(k=8)\nTRAIN-only alignment", SKY),
        (9.1, 2.3, 2.45, 1.45, "Final evaluation", "One-time held-out TEST\ncontrols + biological cases\nchecksum-verified", RED),
    ]
    for x, y, w, h, title, body, color in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + 0.14, y + h - 0.28, title, fontweight="bold", color=color, fontsize=10)
        ax.text(x + 0.14, y + h - 0.58, body, va="top", linespacing=1.35)
    for start, end in (
        ((2.55, 4.42), (3.1, 4.42)), ((2.55, 1.62), (3.1, 1.62)),
        ((5.55, 4.42), (6.15, 3.3)), ((5.55, 1.62), (6.15, 2.75)),
        ((8.5, 3.02), (9.1, 3.02)),
    ):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": BLACK})
    ax.text(6.0, 5.72, "Frozen, leakage-resistant analysis workflow", ha="center", fontsize=14, fontweight="bold")
    ax.text(6.0, 0.25, "No TEST fitting, tuning, new alignment, or method selection", ha="center", color=RED, fontweight="bold")
    return save_figure(fig, output / "figure_01_study_design")


def figure_stability_baseline(ds5: Mapping[str, Any], ds6: Mapping[str, Any], output: Path) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    cross5, base5 = ds5["cross"], ds5["baseline"]
    pair_labels = [f"{p['seed_a']}–{p['seed_b']}" for p in cross5["pairs"]]
    grouped_bars(
        axes[0, 0], pair_labels,
        (("ARI", [p["ari"] for p in cross5["pairs"]], BLUE), ("NMI", [p["nmi"] for p in cross5["pairs"]], ORANGE)),
        "Agreement metric",
    )
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_title("DS-005 cross-seed TEST stability")
    axes[0, 0].legend(frameon=False, ncol=2)

    metrics = ("ari", "nmi", "aligned_agreement")
    grouped_bars(
        axes[0, 1], ("ARI", "NMI", "Aligned\nagreement"),
        (("DS-005", [cross5[x]["mean"] for x in metrics], BLUE), ("DS-006", [ds6["cross"][x]["mean"] for x in metrics], GREEN)),
        "Mean across seed pairs",
    )
    axes[0, 1].set_title("Held-out stability across datasets")
    axes[0, 1].legend(frameon=False)

    seeds = [str(x) for x in SEEDS]
    grouped_bars(
        axes[1, 0], seeds,
        (("ARI", [base5[s]["comparison"]["ari"] for s in seeds], BLUE), ("NMI", [base5[s]["comparison"]["nmi"] for s in seeds], ORANGE), ("AMI", [base5[s]["comparison"]["ami"] for s in seeds], GREEN)),
        "Baseline–SSL agreement",
    )
    axes[1, 0].set_xlabel("SSL seed")
    axes[1, 0].set_title("DS-005 coarse baseline differs from SSL")
    axes[1, 0].legend(frameon=False, ncol=3)

    grouped_bars(
        axes[1, 1], seeds,
        (("Linear", [base5[s]["linear_probe"]["balanced_accuracy"] for s in seeds], SKY), ("Nonlinear", [base5[s]["nonlinear_probe"]["balanced_accuracy"] for s in seeds], PURPLE)),
        "Balanced accuracy",
    )
    axes[1, 1].axhline(1 / 8, color=GRAY, linestyle="--", linewidth=1, label="Uniform chance")
    axes[1, 1].set_xlabel("SSL seed")
    axes[1, 1].set_title("Handcrafted features predict DS-005 SSL labels")
    axes[1, 1].legend(frameon=False, ncol=3)
    for label, ax in zip("ABCD", axes.flat):
        panel_label(ax, label)
    fig.suptitle("Stability and representation comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output / "figure_02_stability_and_baseline")


def nuisance_arrays(nuisance: Mapping[str, Any], ds006: bool = False) -> Dict[str, List[float]]:
    values = {"speed_eta": [], "speed_ba": [], "identity_v": [], "context_v": []}
    for seed in map(str, SEEDS):
        item = nuisance[seed]
        values["speed_eta"].append(item["mean_speed_eta_squared"])
        values["speed_ba"].append(item["speed_only"]["balanced_accuracy"])
        identity_key = "fish_well_identity" if ds006 else "fish_identity"
        values["identity_v"].append(item[identity_key]["cramers_v"])
        if ds006:
            values["context_v"].append(mean(x["cramers_v"] for x in item["contexts"].values()))
        else:
            values["context_v"].append(item["context_id"]["cramers_v"])
    return values


def figure_nuisance(ds5: Mapping[str, Any], ds6: Mapping[str, Any], output: Path) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    a, b = nuisance_arrays(ds5["nuisance"]), nuisance_arrays(ds6["nuisance"], ds006=True)
    specifications = (
        ("speed_eta", "Mean-speed eta-squared", None),
        ("speed_ba", "Speed-only balanced accuracy", 1 / 8),
        ("identity_v", "Identity Cramér's V", None),
        ("context_v", "Context Cramér's V", None),
    )
    for label, ax, (key, title, reference) in zip("ABCD", axes.flat, specifications):
        grouped_bars(ax, [str(x) for x in SEEDS], (("DS-005", a[key], BLUE), ("DS-006", b[key], GREEN)), title)
        if reference is not None:
            ax.axhline(reference, color=GRAY, linestyle="--", linewidth=1, label="Uniform chance")
        ax.set_xlabel("SSL seed")
        ax.set_title(title)
        ax.legend(frameon=False)
        panel_label(ax, label)
    fig.suptitle("Held-out nuisance and validity controls", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output / "figure_03_nuisance_controls")


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    center = matrix.mean(axis=1, keepdims=True)
    spread = matrix.std(axis=1, keepdims=True)
    return (matrix - center) / np.where(spread > 0, spread, 1)


def figure_long_cs(ds5: Mapping[str, Any], output: Path) -> List[Path]:
    result = ds5["long_cs"]["by_seed"]
    features = ("bout_duration_s", "accel_rms", "accel_abs_std")
    display = ("Bout duration", "Acceleration RMS", "Acceleration absolute SD")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    grouped_bars(
        axes[0, 0], [str(x) for x in SEEDS],
        tuple((name, [result[str(s)]["features"][feature]["eta_squared"] for s in SEEDS], color) for feature, name, color in zip(features, display, (BLUE, ORANGE, GREEN))),
        "TEST eta-squared",
    )
    axes[0, 0].axhline(0.25, color=GRAY, linestyle="--", linewidth=1, label="Frozen support threshold")
    axes[0, 0].set_xlabel("SSL seed")
    axes[0, 0].set_title("Subcluster-associated variation")
    axes[0, 0].legend(frameon=False, fontsize=7)

    grouped_bars(
        axes[0, 1], [str(x) for x in SEEDS],
        tuple((name, [result[str(s)]["features"][feature]["train_to_test_spearman"] for s in SEEDS], color) for feature, name, color in zip(features, display, (BLUE, ORANGE, GREEN))),
        "TRAIN-to-TEST Spearman rho",
    )
    axes[0, 1].axhline(0.5, color=GRAY, linestyle="--", linewidth=1, label="Frozen support threshold")
    axes[0, 1].set_xlabel("SSL seed")
    axes[0, 1].set_title("Profile reproducibility")
    axes[0, 1].legend(frameon=False, fontsize=7)

    for ax, feature, title, label in zip(axes[1], (features[0], features[1]), (display[0], "Acceleration profiles"), "CD"):
        profiles = np.asarray([result[str(s)]["features"][feature]["mean_profile"] for s in SEEDS])
        image = ax.imshow(zscore_rows(profiles), aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
        ax.set_xticks(CLUSTERS, [str(x) for x in CLUSTERS])
        ax.set_yticks(np.arange(5), [str(x) for x in SEEDS])
        ax.set_xlabel("Aligned SSL cluster")
        ax.set_ylabel("SSL seed")
        ax.set_title(f"{title}: standardized mean profiles")
        fig.colorbar(image, ax=ax, shrink=0.78, label="Within-seed z-score")
        panel_label(ax, label)
    panel_label(axes[0, 0], "A")
    panel_label(axes[0, 1], "B")
    fig.suptitle("Long_CS primary held-out result", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, output / "figure_04_long_cs_primary")


def figure_llc(ds5: Mapping[str, Any], output: Path) -> List[Path]:
    result = ds5["llc"]["by_seed"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for seed, color in zip(SEEDS, (BLUE, ORANGE, GREEN, PURPLE, RED)):
        profile = result[str(seed)]["features"]["turn_net_rad"]["mean_profile"]
        axes[0].plot(CLUSTERS, profile, marker="o", linewidth=1.5, label=f"Seed {seed}", color=color)
    axes[0].axhspan(-0.10, 0.10, color="#EEEEEE", label="Frozen neutral band")
    axes[0].axhline(0, color=BLACK, linewidth=0.8)
    axes[0].set_xticks(CLUSTERS)
    axes[0].set_xlabel("Aligned SSL cluster")
    axes[0].set_ylabel("Mean turn_net_rad")
    axes[0].set_title("Aligned TEST turn profiles")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    eta = [result[str(s)]["features"]["turn_net_rad"]["eta_squared"] for s in SEEDS]
    rho = [result[str(s)]["train_to_test_spearman"] for s in SEEDS]
    axes[1].bar([str(x) for x in SEEDS], eta, color=BLUE)
    axes[1].axhline(0.10, color=GRAY, linestyle="--", label="Frozen support threshold")
    axes[1].set_xlabel("SSL seed")
    axes[1].set_ylabel("TEST eta-squared")
    axes[1].set_title("Turning association")
    axes[1].legend(frameon=False)
    axes[2].bar([str(x) for x in SEEDS], rho, color=GREEN)
    axes[2].axhline(0.50, color=GRAY, linestyle="--", label="Frozen support threshold")
    axes[2].set_xlabel("SSL seed")
    axes[2].set_ylabel("TRAIN-to-TEST Spearman rho")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Profile reproducibility")
    axes[2].legend(frameon=False)
    for label, ax in zip("ABC", axes):
        panel_label(ax, label)
    fig.suptitle("LLC secondary held-out result: cluster 0 positive, cluster 6 negative in 5/5 seeds", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output / "figure_05_llc_secondary")


def replication_metrics(ds5: Mapping[str, Any], ds6: Mapping[str, Any]) -> tuple[List[str], List[float], List[float]]:
    n5, n6 = nuisance_arrays(ds5["nuisance"]), nuisance_arrays(ds6["nuisance"], ds006=True)
    base5, base6 = ds5["baseline"], ds6["baseline"]
    labels = ["Cross-seed\nARI", "Speed\neta²", "Speed-only\nBA", "Identity\nV", "Context\nV", "Baseline–SSL\nARI", "Nonlinear\nprobe BA"]
    values5 = [
        ds5["cross"]["ari"]["mean"], mean(n5["speed_eta"]), mean(n5["speed_ba"]),
        mean(n5["identity_v"]), mean(n5["context_v"]),
        mean(base5[str(s)]["comparison"]["ari"] for s in SEEDS),
        mean(base5[str(s)]["nonlinear_probe"]["balanced_accuracy"] for s in SEEDS),
    ]
    values6 = [
        ds6["cross"]["ari"]["mean"], mean(n6["speed_eta"]), mean(n6["speed_ba"]),
        mean(n6["identity_v"]), mean(n6["context_v"]),
        mean(base6[str(s)]["ari"] for s in SEEDS),
        mean(base6[str(s)]["nonlinear_probe"]["balanced_accuracy"] for s in SEEDS),
    ]
    return labels, values5, values6


def figure_replication(ds5: Mapping[str, Any], ds6: Mapping[str, Any], output: Path) -> List[Path]:
    labels, values5, values6 = replication_metrics(ds5, ds6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grouped_bars(axes[0], labels, (("DS-005", values5, BLUE), ("DS-006", values6, GREEN)), "Held-out metric")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Generalization and nuisance controls")
    axes[0].legend(frameon=False)

    observed = ds6["claims"]["observed_summary"]
    biological_labels = ["Acceleration /\nspeed change", "Duration", "Signed\nnet turn", "Turning\nmagnitude"]
    ds5_values = [
        mean([
            mean(ds5["long_cs"]["by_seed"][str(s)]["features"]["accel_rms"]["eta_squared"] for s in SEEDS),
            mean(ds5["long_cs"]["by_seed"][str(s)]["features"]["accel_abs_std"]["eta_squared"] for s in SEEDS),
        ]),
        mean(ds5["long_cs"]["by_seed"][str(s)]["features"]["bout_duration_s"]["eta_squared"] for s in SEEDS),
        mean(ds5["llc"]["by_seed"][str(s)]["features"]["turn_net_rad"]["eta_squared"] for s in SEEDS),
        np.nan,
    ]
    ds6_values = [
        observed["mean_speed_change_eta_squared"], observed["mean_duration_eta_squared"],
        observed["mean_turn_net_eta_squared"], observed["mean_turn_total_abs_eta_squared"],
    ]
    x = np.arange(4)
    axes[1].bar(x - 0.2, np.nan_to_num(ds5_values), width=0.38, color=BLUE, label="DS-005 class-specific")
    axes[1].bar(x + 0.2, ds6_values, width=0.38, color=GREEN, label="DS-006 analogue")
    axes[1].text(3 - 0.2, 0.01, "N/A", ha="center", va="bottom", color=GRAY, rotation=90)
    axes[1].set_xticks(x, biological_labels)
    axes[1].set_ylabel("Mean TEST eta-squared")
    axes[1].set_title("Biological axes: direct and analogous evidence")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.suptitle("Independent external replication is mixed but informative", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output / "figure_06_external_replication")


def write_table(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[Path]:
    csv_path = path.with_suffix(".csv")
    md_path = path.with_suffix(".md")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    def render(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def generate_tables(ds5: Mapping[str, Any], ds6: Mapping[str, Any], output: Path) -> List[Path]:
    paths: List[Path] = []
    paths += write_table(
        output / "table_01_datasets",
        ("Dataset", "Role", "Units", "Bouts", "Split unit", "TEST bouts", "TEST status"),
        (
            ("DS-005", "Primary", "463 fish", "1,203,409", "Fish", "192,104", "Opened once; complete"),
            ("DS-006", "External replication", "374 usable fish-wells", "163,065", "Recording", "26,130", "Opened once; complete"),
        ),
    )
    paths += write_table(
        output / "table_02_frozen_methods",
        ("Component", "Frozen specification"),
        (
            ("Input A", "18 handcrafted timing, speed, acceleration, and turning features"),
            ("Baseline clustering", "TRAIN-fitted PCA(6) + GaussianMixture(k=2, seed=20260822)"),
            ("Input B", "175×3 temporal orientation/speed sequence; 1D CNN; 64-D embedding"),
            ("SSL seeds", "11, 23, 37, 51, 79"),
            ("SSL clustering", "TRAIN-fitted StandardScaler + PCA(95% variance) + KMeans(k=8)"),
            ("Alignment", "Hungarian mapping derived on TRAIN only; reference seed 11"),
            ("Final evaluation", "One-time inference only; no fitting, tuning, or new selection"),
        ),
    )
    n5 = nuisance_arrays(ds5["nuisance"])
    b5 = ds5["baseline"]
    table3 = (
        ("Cross-seed structure", "Mean ARI", ds5["cross"]["ari"]["mean"], "SUPPORTED"),
        ("Cross-seed structure", "Mean NMI", ds5["cross"]["nmi"]["mean"], "SUPPORTED"),
        ("Baseline differs from SSL", "Mean ARI", mean(b5[str(s)]["comparison"]["ari"] for s in SEEDS), "SUPPORTED"),
        ("Linear feature recovery", "Balanced accuracy", mean(b5[str(s)]["linear_probe"]["balanced_accuracy"] for s in SEEDS), "SUPPORTING"),
        ("Nonlinear feature recovery", "Balanced accuracy", mean(b5[str(s)]["nonlinear_probe"]["balanced_accuracy"] for s in SEEDS), "SUPPORTED"),
        ("Speed dependence", "Mean-speed eta-squared", mean(n5["speed_eta"]), "SUPPORTED"),
        ("Speed-only collapse rejected", "Balanced accuracy", mean(n5["speed_ba"]), "SUPPORTED"),
        ("Low fish identity leakage", "Cramer's V", mean(n5["identity_v"]), "SUPPORTED"),
        ("Low context leakage", "Cramer's V", mean(n5["context_v"]), "SUPPORTED"),
    )
    paths += write_table(output / "table_03_ds005_final_test", ("Finding", "Metric", "TEST value", "Assessment"), table3)

    long_cs, llc = ds5["long_cs"]["by_seed"], ds5["llc"]["by_seed"]
    table4 = []
    for feature, display in (("bout_duration_s", "Long_CS duration"), ("accel_rms", "Long_CS acceleration RMS"), ("accel_abs_std", "Long_CS acceleration absolute SD")):
        table4.append((display, mean(long_cs[str(s)]["features"][feature]["eta_squared"] for s in SEEDS), mean(long_cs[str(s)]["features"][feature]["train_to_test_spearman"] for s in SEEDS), "SUPPORTED"))
    table4.append(("LLC signed net turning", mean(llc[str(s)]["features"]["turn_net_rad"]["eta_squared"] for s in SEEDS), mean(llc[str(s)]["train_to_test_spearman"] for s in SEEDS), "SUPPORTED; directions 5/5"))
    paths += write_table(output / "table_04_biological_results", ("Finding", "Mean TEST eta-squared", "Mean TRAIN-to-TEST rho", "Assessment"), table4)

    labels, values5, values6 = replication_metrics(ds5, ds6)
    status = ("REPLICATED", "REPLICATED", "REPLICATED", "REPLICATED", "REPLICATED", "REPLICATED", "NOT STRONGLY REPLICATED")
    rows = [(label.replace("\n", " "), a, b, s) for label, a, b, s in zip(labels, values5, values6, status)]
    paths += write_table(output / "table_05_external_replication", ("Finding", "DS-005 TEST", "DS-006 TEST", "Interpretation"), rows)
    return paths


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    figures = output / "figures"
    tables = output / "tables"
    expected = output / "publication_manifest.json"
    if expected.exists() and not args.overwrite:
        raise SystemExit(f"Publication outputs already exist; use --overwrite: {output}")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    configure_style()

    loaded = {str(path.relative_to(ROOT)): load_json(path) for path in SOURCE_FILES}
    ds5 = {
        "run": loaded[str((DS005 / "run_manifest.json").relative_to(ROOT))],
        "claims": loaded[str((DS005 / "claim_assessment.json").relative_to(ROOT))],
        "cross": loaded[str((DS005 / "cross_seed_summary.json").relative_to(ROOT))],
        "baseline": loaded[str((DS005 / "baseline_vs_ssl_summary.json").relative_to(ROOT))],
        "nuisance": loaded[str((DS005 / "nuisance_summary.json").relative_to(ROOT))],
        "long_cs": loaded[str((DS005 / "long_cs_primary_summary.json").relative_to(ROOT))],
        "llc": loaded[str((DS005 / "llc_secondary_summary.json").relative_to(ROOT))],
    }
    ds6 = {
        "run": loaded[str((DS006 / "run_manifest.json").relative_to(ROOT))],
        "claims": loaded[str((DS006 / "claim_assessment.json").relative_to(ROOT))],
        "cross": loaded[str((DS006 / "cross_seed_summary.json").relative_to(ROOT))],
        "baseline": loaded[str((DS006 / "baseline_vs_ssl_summary.json").relative_to(ROOT))],
        "nuisance": loaded[str((DS006 / "nuisance_summary.json").relative_to(ROOT))],
        "axes": loaded[str((DS006 / "kinematic_axes_summary.json").relative_to(ROOT))],
    }
    if ds5["run"]["test_used_for_fitting"] or ds6["run"]["test_used_for_fitting"]:
        raise RuntimeError("Final manifest reports TEST fitting")
    if tuple(ds5["run"]["ssl_seeds"]) != SEEDS or tuple(ds6["run"]["ssl_seeds"]) != SEEDS:
        raise RuntimeError("Frozen seed set changed")

    outputs: List[Path] = []
    outputs += figure_study_design(figures)
    outputs += figure_stability_baseline(ds5, ds6, figures)
    outputs += figure_nuisance(ds5, ds6, figures)
    outputs += figure_long_cs(ds5, figures)
    outputs += figure_llc(ds5, figures)
    outputs += figure_replication(ds5, ds6, figures)
    outputs += generate_tables(ds5, ds6, tables)

    manifest = {
        "purpose": "deterministic publication figures and tables from frozen final summaries",
        "scientific_artifacts_read_only": True,
        "raw_bouts_loaded": False,
        "embeddings_loaded": False,
        "fitting_or_selection_performed": False,
        "ds005_freeze_commit": ds5["run"]["freeze_commit"],
        "ds006_freeze_commit": ds6["run"]["freeze_commit"],
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES},
        "output_hashes": {str(path.relative_to(output)): sha256(path) for path in sorted(outputs)},
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")
        },
        "figure_formats": ["pdf", "svg", "png"],
        "png_dpi": 600,
    }
    atomic_json(expected, manifest)
    all_outputs = sorted(outputs + [expected])
    checksum = output / "PUBLICATION_SHA256SUMS"
    checksum.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in all_outputs),
        encoding="utf-8",
    )
    print(f"Generated {len(outputs)} publication artifacts under {output}")
    print(f"Manifest: {expected}")
    print(f"Checksums: {checksum}")


if __name__ == "__main__":
    main()
