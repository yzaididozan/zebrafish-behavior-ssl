#!/usr/bin/env python3
"""Create a frozen DS-005 vs DS-006 directional replication comparison.

This report compares the direction and interpretation of major DS-005 findings
against DS-006 TRAIN/VALIDATION results. It does not use p-values as the
replication criterion and never accesses DS-006 TEST.

Outputs:
  data/processed/DS-006/cross_dataset_comparison/comparison.json
  docs/ds005-ds006-directional-comparison.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "clustering": REPO_ROOT / "data/processed/DS-006/transfer_clustering/summary.json",
    "speed": REPO_ROOT / "data/processed/DS-006/transfer_speed_dependence/summary.json",
    "identity": REPO_ROOT / "data/processed/DS-006/transfer_identity_leakage/aggregate_summary.json",
    "context": REPO_ROOT / "data/processed/DS-006/transfer_context_leakage/aggregate_summary.json",
    "substructure": REPO_ROOT / "data/processed/DS-006/transfer_substructure/summary.json",
    "targeted": REPO_ROOT / "data/processed/DS-006/transfer_substructure/targeted_axes_summary.json",
}

DEFAULT_OUTPUT = REPO_ROOT / "data/processed/DS-006/cross_dataset_comparison"
DEFAULT_DOC = REPO_ROOT / "docs/ds005-ds006-directional-comparison.md"

# Frozen DS-005 validation values from the completed primary analysis.
DS005 = {
    "cluster_ari": 0.358195,
    "cluster_nmi": 0.459763,
    "cluster_aligned": 0.565489,
    "cluster_silhouette": 0.126659,
    "speed_eta2": 0.457877,
    "speed_balacc": 0.294657,
    "speed_macro_f1": 0.235911,
    "identity_nmi": 0.032191,
    "identity_ami": 0.031737,
    "identity_v": 0.184327,
    "identity_entropy": 0.923046,
    "context_nmi": 0.031566,
    "context_ami": 0.031447,
    "context_v": 0.151235,
    "context_entropy": 0.932587,
    "longcs_duration_eta2": 0.525723,
    "longcs_accel_rms_eta2": 0.538565,
    "longcs_accel_abs_std_eta2": 0.532868,
    "longcs_duration_trainval_rho": 0.861905,
    "longcs_accel_rms_trainval_rho": 0.819048,
    "llc_turn_net_eta2": 0.164450,
    "llc_turn_net_trainval_rho": 0.952381,
    "llc_turn_net_crossseed_rho": 0.423810,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("test_partition_used") is True:
        raise RuntimeError(f"{path} unexpectedly records TEST use.")
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_ds006(src: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    c = src["clustering"]["aggregate"]
    s = src["speed"]["aggregate"]
    i = src["identity"]["metrics"]["validation"]
    ctx = src["context"]["metrics"]
    targeted = src["targeted"]["groups"]

    duration = targeted["long_cs_like_duration_speed_change_axes"]["features"]
    turning = targeted["llc_like_turning_axes"]["features"]

    context_nmis = {
        field: float(data["validation"]["nmi"]["mean"])
        for field, data in ctx.items()
    }
    context_amis = {
        field: float(data["validation"]["ami"]["mean"])
        for field, data in ctx.items()
    }
    context_vs = {
        field: float(data["validation"]["cramers_v"]["mean"])
        for field, data in ctx.items()
    }
    context_entropy = {
        field: float(data["validation"]["mean_normalized_cluster_entropy"]["mean"])
        for field, data in ctx.items()
    }

    def f(group, name):
        return group[name]

    return {
        "cluster_ari": float(c["validation_pairwise_ari"]["mean"]),
        "cluster_nmi": float(c["validation_pairwise_nmi"]["mean"]),
        "cluster_aligned": float(c["validation_pairwise_aligned_agreement"]["mean"]),
        "cluster_silhouette": float(c["mean_validation_silhouette"]),
        "speed_eta2": float(s["validation_eta_squared"]["mean"]),
        "speed_balacc": float(s["speed_only_validation_balanced_accuracy"]["mean"]),
        "speed_macro_f1": float(s["speed_only_validation_macro_f1"]["mean"]),
        "identity_nmi": float(i["nmi"]["mean"]),
        "identity_ami": float(i["ami"]["mean"]),
        "identity_v": float(i["cramers_v"]["mean"]),
        "identity_entropy": float(i["mean_normalized_cluster_entropy"]["mean"]),
        "context_max_nmi": max(context_nmis.values()),
        "context_max_ami": max(context_amis.values()),
        "context_max_v": max(context_vs.values()),
        "context_min_entropy": min(context_entropy.values()),
        "context_nmi_by_field": context_nmis,
        "duration_eta2": float(f(duration, "bout_duration")["validation_eta_squared"]["mean"]),
        "duration_trainval_rho": float(
            f(duration, "bout_duration")["train_to_validation_profile_reproducibility"]
            ["mean_profile_spearman_mean_across_seeds"]
        ),
        "speed_change_rms_eta2": float(
            f(duration, "speed_change_rms")["validation_eta_squared"]["mean"]
        ),
        "speed_change_rms_trainval_rho": float(
            f(duration, "speed_change_rms")["train_to_validation_profile_reproducibility"]
            ["mean_profile_spearman_mean_across_seeds"]
        ),
        "speed_change_std_eta2": float(
            f(duration, "speed_change_std")["validation_eta_squared"]["mean"]
        ),
        "turn_net_eta2": float(f(turning, "turn_net")["validation_eta_squared"]["mean"]),
        "turn_net_trainval_rho": float(
            f(turning, "turn_net")["train_to_validation_profile_reproducibility"]
            ["mean_profile_spearman_mean_across_seeds"]
        ),
        "turn_total_abs_eta2": float(
            f(turning, "turn_total_abs")["validation_eta_squared"]["mean"]
        ),
        "turn_total_abs_trainval_rho": float(
            f(turning, "turn_total_abs")["train_to_validation_profile_reproducibility"]
            ["mean_profile_spearman_mean_across_seeds"]
        ),
        "turn_total_abs_crossseed_rho": float(
            f(turning, "turn_total_abs")["cross_seed_validation_mean_profile_spearman"]["mean"]
        ),
        "direct_class_labels_available": bool(
            src["substructure"]["direct_long_cs_or_llc_class_replication_available"]
        ),
    }


def comparison_rows(ds006: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "claim": "Moderately reproducible k=8 organization across encoder seeds",
            "status": "REPLICATED",
            "ds005": f"ARI={DS005['cluster_ari']:.3f}; NMI={DS005['cluster_nmi']:.3f}; aligned={DS005['cluster_aligned']:.3f}; silhouette={DS005['cluster_silhouette']:.3f}",
            "ds006": f"ARI={ds006['cluster_ari']:.3f}; NMI={ds006['cluster_nmi']:.3f}; aligned={ds006['cluster_aligned']:.3f}; silhouette={ds006['cluster_silhouette']:.3f}",
            "reason": "Cross-seed clustering stability is extremely similar in direction and magnitude.",
        },
        {
            "claim": "Speed-related but not reducible to mean speed",
            "status": "REPLICATED",
            "ds005": f"eta²={DS005['speed_eta2']:.3f}; speed-only balacc={DS005['speed_balacc']:.3f}; macro-F1={DS005['speed_macro_f1']:.3f}",
            "ds006": f"eta²={ds006['speed_eta2']:.3f}; speed-only balacc={ds006['speed_balacc']:.3f}; macro-F1={ds006['speed_macro_f1']:.3f}",
            "reason": "Both datasets show strong speed association but poor mean-speed-only recovery of k=8 labels.",
        },
        {
            "claim": "Low subject/fish identity leakage",
            "status": "REPLICATED",
            "ds005": f"NMI={DS005['identity_nmi']:.3f}; AMI={DS005['identity_ami']:.3f}; V={DS005['identity_v']:.3f}",
            "ds006": f"NMI={ds006['identity_nmi']:.3f}; AMI={ds006['identity_ami']:.3f}; V={ds006['identity_v']:.3f}",
            "reason": "DS-006 fish-well identity associations remain low and comparable to DS-005.",
        },
        {
            "claim": "Low recording/context leakage",
            "status": "REPLICATED",
            "ds005": f"NMI={DS005['context_nmi']:.3f}; AMI={DS005['context_ami']:.3f}; V={DS005['context_v']:.3f}",
            "ds006": f"max field NMI={ds006['context_max_nmi']:.3f}; max AMI={ds006['context_max_ami']:.3f}; max V={ds006['context_max_v']:.3f}",
            "reason": "All tested DS-006 context fields show very low held-out associations.",
        },
        {
            "claim": "Acceleration/speed-change heterogeneity",
            "status": "REPLICATED",
            "ds005": f"Long_CS accel_rms eta²={DS005['longcs_accel_rms_eta2']:.3f}; accel_abs_std eta²={DS005['longcs_accel_abs_std_eta2']:.3f}",
            "ds006": f"speed_change_rms eta²={ds006['speed_change_rms_eta2']:.3f}; speed_change_std eta²={ds006['speed_change_std_eta2']:.3f}; train→val rho={ds006['speed_change_rms_trainval_rho']:.3f}",
            "reason": "The exact Long_CS label is unavailable, but the same acceleration/speed-change axis is strong and highly reproducible.",
        },
        {
            "claim": "Long_CS-like duration heterogeneity",
            "status": "NOT_STRONGLY_REPLICATED",
            "ds005": f"duration eta²={DS005['longcs_duration_eta2']:.3f}; train→val rho={DS005['longcs_duration_trainval_rho']:.3f}",
            "ds006": f"duration eta²={ds006['duration_eta2']:.3f}; train→val rho={ds006['duration_trainval_rho']:.3f}",
            "reason": "DS-006 duration association is much weaker, and no direct Long_CS label exists.",
        },
        {
            "claim": "LLC-like signed net-turning structure",
            "status": "NOT_STRONGLY_REPLICATED",
            "ds005": f"turn_net eta²={DS005['llc_turn_net_eta2']:.3f}; train→val rho={DS005['llc_turn_net_trainval_rho']:.3f}",
            "ds006": f"turn_net eta²={ds006['turn_net_eta2']:.3f}; train→val rho={ds006['turn_net_trainval_rho']:.3f}",
            "reason": "Signed net turning is very weak in DS-006 despite stable within-seed profile ordering.",
        },
        {
            "claim": "Turning magnitude/intensity as a reproducible axis",
            "status": "PARTIAL",
            "ds005": "Related turning features present, but signed turn_net was the primary LLC result.",
            "ds006": f"turn_total_abs eta²={ds006['turn_total_abs_eta2']:.3f}; train→val rho={ds006['turn_total_abs_trainval_rho']:.3f}; cross-seed rho={ds006['turn_total_abs_crossseed_rho']:.3f}",
            "reason": "DS-006 shows robust turning magnitude, but this is not the same as reproducing signed turn direction.",
        },
        {
            "claim": "Direct within-Long_CS / within-LLC subdivision replication",
            "status": "NOT_DIRECTLY_TESTABLE",
            "ds005": "Available",
            "ds006": "Unavailable",
            "reason": "DS-006 lacks DS-005-equivalent conventional bout-class labels.",
        },
    ]


def render_markdown(rows: List[Dict[str, Any]], hashes: Dict[str, str]) -> str:
    lines = [
        "# DS-005 vs DS-006 Directional Replication Comparison",
        "",
        "## Scope",
        "",
        "This comparison evaluates whether the **direction and scientific interpretation** of the frozen DS-005 findings reproduce in DS-006. It does **not** use individual p-values as the replication criterion.",
        "",
        "**DS-006 TEST remains sealed.** All DS-006 values below come from TRAIN/VALIDATION analyses.",
        "",
        "DS-006 does not provide the same conventional bout-class labels as DS-005 (`Long_CS`, `LLC`, etc.), so those comparisons are treated as kinematic analogues rather than direct within-class replications.",
        "",
        "## Comparison",
        "",
        "| Claim | DS-005 | DS-006 | Status | Interpretation |",
        "|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['claim']} | {row['ds005']} | {row['ds006']} | "
            f"**{row['status']}** | {row['reason']} |"
        )

    lines += [
        "",
        "## Overall interpretation",
        "",
        "DS-006 supports the broad representation-level conclusions from DS-005: the transferred representation remains structured, the frozen `k=8` organization shows moderate cross-encoder reproducibility, the clusters are substantially speed-related but not reducible to mean speed, and fish-well/context leakage is low.",
        "",
        "At the finer kinematic level, **speed-change/acceleration-like structure replicates strongly**. **Bout-duration heterogeneity does not reproduce strongly**, and the DS-005 LLC signed net-turning result also weakens substantially. DS-006 instead shows a reproducible **turning-magnitude/intensity** axis.",
        "",
        "This should therefore be presented as a **mixed but informative replication**: several central claims reproduce, while some fine-grained DS-005 findings weaken or change form.",
        "",
        "## Frozen claim language",
        "",
        "> Self-supervised representations recover structured zebrafish behavioral variation that generalizes across held-out fish/recordings and across an independently acquired replication dataset. The organization is substantially related to locomotor speed but is not reducible to mean speed, fish-well identity, or experimental context. Exact eight-cluster boundaries remain moderately seed-sensitive. Across datasets, speed-change/acceleration-like and turning-magnitude structure are reproducible, whereas the DS-005-specific duration and signed net-turning effects do not reproduce as strongly.",
        "",
        "## Claim restrictions",
        "",
        "- Do not claim that all DS-005 fine-grained findings replicated.",
        "- Do not describe DS-006 as a direct `Long_CS` or `LLC` replication.",
        "- Do not claim that `k=8` represents eight universally fixed biological behaviors.",
        "- Do not claim that SSL contains information wholly absent from the handcrafted feature set.",
        "- Describe the DS-006 identity control as **fish-well identity leakage**.",
        "",
        "## Provenance",
        "",
    ]

    for key, digest in hashes.items():
        lines.append(f"- `{key}`: `{digest}`")

    lines += [
        "",
        "**TEST partition used: NO**",
        "",
    ]

    return "\n".join(lines)


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    doc_path = args.doc_path.expanduser().resolve()

    comparison_path = output_root / "comparison.json"

    if (comparison_path.exists() or doc_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "Comparison artifacts already exist. Use --overwrite for intentional rerun."
        )

    src = {key: load_json(path) for key, path in SOURCES.items()}
    ds006 = extract_ds006(src)
    rows = comparison_rows(ds006)
    hashes = {key: sha256(path) for key, path in SOURCES.items()}

    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    output_root.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = {
        "analysis": "ds005_ds006_directional_replication_comparison",
        "criterion": "direction and scientific interpretation, not individual p-values",
        "ds006_test_partition_used": False,
        "ds006_direct_long_cs_llc_class_mapping_available": False,
        "status_counts": status_counts,
        "rows": rows,
        "ds005_frozen_reference": DS005,
        "ds006_metrics": ds006,
        "source_sha256": hashes,
    }

    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    doc_path.write_text(
        render_markdown(rows, hashes),
        encoding="utf-8",
    )

    checksum_path = output_root / "DS005_DS006_DIRECTIONAL_COMPARISON_SHA256SUMS"
    checksum_path.write_text(
        f"{sha256(comparison_path)}  {comparison_path.name}\n"
        f"{sha256(doc_path)}  {doc_path.relative_to(REPO_ROOT)}\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("DS-005 vs DS-006 DIRECTIONAL REPLICATION COMPARISON")
    print("=" * 80)
    for row in rows:
        print(f"{row['status']:<24} {row['claim']}")
    print()
    print("Status counts:")
    for status in (
        "REPLICATED",
        "PARTIAL",
        "NOT_STRONGLY_REPLICATED",
        "NOT_DIRECTLY_TESTABLE",
    ):
        print(f"  {status:<24} {status_counts.get(status, 0)}")
    print()
    print("DS-006 TEST partition used: NO")
    print(f"JSON:      {comparison_path}")
    print(f"Markdown:  {doc_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()
