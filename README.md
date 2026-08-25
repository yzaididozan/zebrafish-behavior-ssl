# Zebrafish Behavior SSL

> Self-supervised temporal representation learning for reproducible discovery of zebrafish behavioral structure.

## Overview

This repository contains code, data-processing pipelines, preregistration materials, evaluation procedures, and reproducibility artifacts for a computational zebrafish behavior study.

The primary research question is:

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

The study compares two representations of the **same behavioral bouts**:

- **Input A — Hand-engineered baseline:** conventional timing, speed, acceleration, and turning features.
- **Input B — Learned representation:** a self-supervised temporal embedding learned from bout-level orientation and speed sequences.

The goal is not simply to generate clusters. Any SSL-specific structure must generalize to held-out fish, remain stable across seeds, survive nuisance controls, and receive independent replication support.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yzaididozan/zebrafish-behavior-ssl.git
cd zebrafish-behavior-ssl
```

### 2. Create the local environment and install dependencies

```bash
make setup
```

This creates:

```text
.venv/
```

and installs the project in editable mode with development dependencies.

### 3. Run the test suite

```bash
make test
```

The repository is intended to keep software tests independent of the large third-party raw datasets where possible.

### 4. Reproduce the frozen DS-005 baseline selection

After the required DS-005 data are present locally:

```bash
make baseline
```

This runs:

```bash
python src/discovery/baseline_clustering.py select
```

and reproduces the TRAIN/VALIDATION baseline clustering selection while preserving the held-out TEST partition.

---

## Reproduction Levels

### Level 1 — Software verification

Does not require the full raw research datasets:

```bash
make test
```

This verifies the installed codebase and automated tests.

### Level 2 — Primary baseline reproduction

Requires the locally prepared DS-005 dataset and frozen preprocessing artifacts.

```bash
make baseline
```

The primary frozen baseline discovery configuration is:

```text
PCA(6 components) -> GMM(k=2, seed=20260822)
```

Selection uses TRAIN and VALIDATION only.

### Level 3 — SSL reproduction

Requires DS-005 and the frozen SSL configuration.

The frozen method uses:

```yaml
objective: temporal_contrastive_learning
loss: NT-Xent
temperature: 0.10
embedding_dimension: 64
seeds:
  - 11
  - 23
  - 37
  - 51
  - 79
```

The implementation lives under:

```text
src/ssl/
configs/ssl/
```

### Level 4 — External replication

Requires DS-006 and follows the frozen replication governance documented in:

```text
docs/ds006-replication-protocol.md
docs/evaluation-protocol.md
```

That prerequisite was satisfied at commit `575ead5403d0b2f721d143366b4d2e0014bd67ee`.
DS-006 TEST was subsequently opened exactly once under the frozen procedure;
its final evaluation is complete. DS-005 TEST was subsequently opened exactly
once from freeze commit `d66aca763c76242edc719683a617c2511e8ec37b`; its
final evaluation is also complete.

---

## Data Availability

Raw third-party research datasets are **not** committed directly to this repository.

The project currently uses:

- **DS-005** — primary dataset
- **DS-006** — external replication dataset

Dataset provenance, licenses, versions, checksums, and project roles are documented in:

```text
docs/dataset-register.md
data/manifests/
```

Do not silently substitute dataset versions.

Material dataset changes require a documented amendment.

---

## Reproducibility Guarantees

The project records and freezes:

- random seeds
- fish-level and recording-level data splits
- TRAIN-only normalization statistics
- baseline feature definitions
- SSL input representation
- SSL objective and encoder configuration
- clustering configuration
- nuisance-control procedures
- evaluation metrics
- sensitivity-analysis categories
- artifact SHA-256 hashes
- decision-log freeze points
- protected held-out TEST partitions

Material methodological decisions are recorded in:

```text
docs/decision-log.md
```

---

## Research Question

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

---

## Target Claim

The project is governed by **Claim Level 2**:

> **Self-supervised temporal representations reveal reproducible behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose features.**

The study does **not** require SSL to outperform the baseline.

Equivalent, negative, nuisance-driven, unstable, or replication-failing outcomes remain valid scientific results.

---

## Novelty Scope

This project does **not** claim:

- the first use of SSL in zebrafish
- the first unsupervised discovery of zebrafish behavior
- the first ML analysis of zebrafish trajectories
- discovery of a complete zebrafish ethogram
- that every cluster is a biological state
- that `k=2` is the true number of zebrafish behaviors

The current novelty boundary is:

> **A preregistered, held-out-fish comparison of conventional hand-engineered zebrafish behavioral representations and self-supervised temporal representations using matched observations, explicit nuisance controls, stability analysis, and independent external replication.**

See:

```text
docs/novelty.md
```

---

# Primary Dataset — DS-005

DS-005 is the frozen primary dataset.

```yaml
dataset_role: PRIMARY
number_of_fish: 463
valid_bouts: 1203409
contexts: 14
frame_rate_hz: 700
samples_per_bout: 175
```

### Fish-level split

```yaml
split_seed: 20260822
train_fish: 323
validation_fish: 70
test_fish: 70
fish_overlap: 0
```

All bouts inherit the partition of their source fish.

The DS-005 TEST partition is protected from all method selection.

---

# External Replication — DS-006

DS-006 is reserved for independent external replication.

```yaml
dataset_role: EXTERNAL_REPLICATION
recordings: 32
fish_well_slots: 384
usable_fish_well_units: 374
accepted_bouts: 163065
authoritative_frame_rate_hz: 160
independence_from_ds005: CONFIRMED
direct_fish_or_recording_overlap: "no evidence found"
```

DS-006 is a separately sourced and acquired Reddy et al. (2022) dataset, not a
resplit or known reuse of the Marques et al. fish or recordings underlying
DS-005. The datasets have different assays, acquisition rates, durations,
stimulus conditions, and tracking pipelines. Some investigators overlap and the
datasets were later analyzed together in Sridhar et al. (2024), but this does
not create known fish- or recording-level overlap.

### Recording-level split

```yaml
split_seed: 20260822
train_recordings: 22
validation_recordings: 5
test_recordings: 5

train_bouts: 118100
validation_bouts: 18835
test_bouts: 26130

recording_overlap: 0
```

The DS-006 TRAIN/VALIDATION analysis was frozen before TEST access. DS-006 TEST
has now been opened exactly once and its final evaluation is complete. DS-005
TEST was likewise opened exactly once after its committed freeze, and its final
evaluation is complete.

DS-006 may not be used to change the primary DS-005 method.

---

# Unit of Analysis

The frozen primary unit is:

> **One valid behavioral bout belonging to one identifiable zebrafish.**

Each DS-005 bout contains 175 temporal samples.

The earlier provisional fixed-duration window design is superseded for the primary analysis.

---

# Input A — Hand-Engineered Baseline

Input A is frozen at **18 bout-level features** covering:

- bout timing
- inter-bout interval
- speed summaries
- acceleration / speed-change summaries
- orientation / turning summaries

`head_pos`-derived path/jump features are excluded from the primary baseline because of coordinate-semantic discontinuity concerns.

They remain secondary sensitivity features only.

---

# Input B — Self-Supervised Representation

Input B is frozen as:

```text
shape: (175, 3)

channel 0 = sin(orientation_smooth)
channel 1 = cos(orientation_smooth)
channel 2 = speed_head
```

The encoder receives no fish ID, context label, stimulus code, bout type, session label, partition label, or cluster label.

### TRAIN-only normalization

Only the speed channel is standardized.

```yaml
speed_mean: 0.858429032920
speed_std: 1.260544584910
train_bouts_used: 842841
temporal_speed_samples_used: 147497175
validation_used_for_fit: false
test_used_for_fit: false
```

---

# SSL Method

The primary SSL method is frozen.

### Objective

```yaml
family: temporal_contrastive_learning
loss: NT-Xent
temperature: 0.10
```

### Encoder

```text
Input: (175, 3)

Conv1d 3 -> 32, kernel 7
BatchNorm
GELU

Conv1d 32 -> 64, kernel 5
BatchNorm
GELU

Conv1d 64 -> 128, kernel 3
BatchNorm
GELU

Dropout 0.10
Global average pooling
Linear 128 -> 64
```

Projection head:

```text
64 -> 64 -> 64
```

The downstream representation is the **64-dimensional encoder embedding**, not the projection-head output.

### Frozen seed set

```yaml
seeds:
  - 11
  - 23
  - 37
  - 51
  - 79
```

---

# Baseline Discovery

The frozen primary baseline discovery configuration is:

```text
PCA(6 components) -> GMM(k=2, seed=20260822)
```

Selection used TRAIN and VALIDATION only.

```yaml
pca_components: 6
explained_variance_retained: 0.9579
method: GaussianMixture
k: 2
seed: 20260822
selection_score: 0.649252
validation_silhouette: 0.4158
stability: 0.9992
test_used: false
```

The selected `k=2` is the best evaluated baseline configuration under the frozen selection procedure.

It is **not** interpreted as the true biological number of zebrafish states.

---

# Confirmatory Evaluation

The frozen evaluation protocol is documented in:

```text
docs/evaluation-protocol.md
```

Primary confirmatory analyses include:

- fish-bootstrap ARI
- cross-seed ARI
- held-out cluster occupancy
- baseline-vs-SSL ARI and NMI
- Input A -> SSL-cluster prediction
- speed-only clustering
- embedding-to-speed regression
- fish-identity leakage
- context/session leakage
- tracking-artifact checks
- external replication

---

# Sensitivity Analyses

```yaml
ssl_seed_sensitivity: CONFIRMATORY
head_position_extended_baseline: SECONDARY
cluster_number_sensitivity: SECONDARY
visualization_dimensionality_reduction: EXPLORATORY
alternate_segmentation: NOT_PRIMARY
```

---

# Test-Set Governance

The DS-005 TEST partition must not influence feature definition, normalization, SSL design, clustering, nuisance models, evaluation metrics, decision thresholds, or interpretation criteria.

DS-006 followed the same governance: replication-side procedures were frozen
before its one-time final TEST evaluation. DS-006 TEST evaluation is complete;
DS-005 TEST evaluation is also complete after one authorized opening. Neither
TEST partition may be rerun or used for further selection.

Any accidental TEST inspection capable of influencing methodology must be documented as a deviation.

---
## Final Held-Out TEST Findings

DS-005 TEST contained 192,104 bouts. All frozen directly testable claims were
supported: moderate cross-seed k=8 organization (mean ARI `0.3606`), a strong
difference between the coarse baseline and SSL (mean ARI `0.0198`), nonlinear
recoverability from the 18 handcrafted features (balanced accuracy `0.9036`),
speed association without speed-only collapse, and low fish/context leakage.

The primary Long_CS finding was supported (mean eta-squared: duration `0.5539`,
acceleration RMS `0.5354`, acceleration absolute SD `0.5113`). The secondary
LLC finding was supported (turn-net eta-squared `0.1627`, mean TRAIN-to-TEST
profile Spearman `0.9714`, frozen cluster-0/cluster-6 directions retained in
5/5 seeds). Whether k=8 denotes eight distinct novel behaviors remains
`NOT_TESTABLE`.

The authoritative DS-005 final output checksum-manifest hash is:

```text
9695b4d0474f37ec1e380ad001684776bfc658c9d8b734433d7f4e95780c1305
```

## Current TRAIN / VALIDATION Findings

The five frozen SSL seeds produced complete TRAIN and VALIDATION embeddings and
downstream analyses without using the protected DS-005 TEST partition.

- SSL and baseline partitions differ strongly (mean VALIDATION ARI `0.0180`,
  NMI `0.0475`).
- SSL clustering is moderately reproducible across training seeds (mean
  VALIDATION pairwise ARI `0.3582`, NMI `0.4598`, aligned agreement `0.5655`).
- Fish-identity and context associations are low, but speed remains an important
  correlate of SSL cluster membership (mean VALIDATION eta-squared `0.4579`).
- A nonlinear probe reconstructs SSL labels from the 18 handcrafted features
  with mean VALIDATION balanced accuracy `0.9016`. The results therefore do not
  support a claim that SSL information is fundamentally absent from Input A.
- `Long_CS` is frozen as the primary within-class case study; `LLC` is secondary,
  and `BS` is supporting evidence.

The controlling pre-TEST interpretation and analysis plan are recorded in
[`docs/validation-freeze.md`](docs/validation-freeze.md). These are validation
findings, not final confirmatory or external-replication results.

---

# Repository Structure

```text
zebrafish-behavior-ssl/
├── README.md
├── LICENSE
├── CITATION.cff
├── Makefile
├── pyproject.toml
├── environment.yml
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
├── data/
├── docs/
├── external/
├── src/
├── tests/
└── results/
```

---

# Data and Third-Party Code

Raw third-party research data should generally **not** be committed to Git.

Dataset licenses remain independent of this repository's software license.

External repositories under `external/` retain their original licenses and Git history unless explicitly stated otherwise.

The project does not relicense third-party code or datasets.

---

# Documentation

Key project documents:

```text
docs/charter.md
docs/dataset-register.md
docs/decision-log.md
docs/evaluation-protocol.md
docs/literature.md
docs/novelty.md
docs/preregistration-draft.md
docs/research-question.md
```

---

# Citation

Citation metadata are provided in:

```text
CITATION.cff
```

---

# License

Original software and documentation in this repository are released under the **MIT License**, unless otherwise noted.

See:

```text
LICENSE
```

Third-party datasets, external code, pretrained models, and other external assets retain their own licenses and terms.

---

# Disclaimer

Behavioral clusters produced by this project must not automatically be interpreted as distinct biological behaviors.

The project identifies and evaluates **candidate reproducible behavioral structure**. Biological interpretation requires evidence beyond cluster formation alone.
