# Project Charter

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

**Repository:** `zebrafish-behavior-ssl`  
**Primary dataset:** `DS-005`  
**External replication dataset:** `DS-006`  
**Charter status:** `STABLE / FROZEN FOR PREREGISTRATION`  
**Last updated:** `2026-08-24`

---

# 1. Project Purpose

This project tests whether self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not fully captured by conventional hand-engineered locomotion and pose features.

The project is designed as a controlled comparison between:

```text
Input A
Hand-engineered locomotion / pose features

versus

Input B
Self-supervised temporal representations
```

The project does not assume that self-supervised learning must outperform the hand-engineered baseline.

Equivalent, negative, unstable, nuisance-driven, or replication-failing outcomes remain scientifically valid.

---

# 2. Primary Research Question

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

---

# 3. Target Claim

The project is governed by **Claim Level 2**:

> **Self-supervised temporal representations reveal reproducible behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose features.**

This claim may be supported only if SSL-derived structure:

- is reproducible;
- generalizes to held-out fish;
- is stable across the frozen seed set;
- is not predominantly fish identity;
- is not predominantly context/session;
- is not explained solely by locomotor speed;
- is not primarily a tracking artifact;
- survives the preregistered sensitivity analyses;
- is not fully reconstructed by Input A;
- and receives meaningful support in the external replication dataset.

---

# 4. Primary Dataset

## DS-005

**Role:** `PRIMARY`  
**Status:** `FROZEN`

Frozen dataset facts:

```yaml
number_of_fish: 463
valid_bouts: 1203409
contexts: 14
frame_rate_hz: 700
samples_per_bout: 175
```

The primary confirmatory unit is one valid bout belonging to one identifiable fish.

---

# 5. External Replication Dataset

## DS-006

**Role:** `EXTERNAL_REPLICATION`

Frozen replication facts include:

```yaml
recordings: 32
fish_well_slots: 384
usable_fish_well_units: 374
accepted_bouts: 163065
authoritative_frame_rate_hz: 160
independence_from_ds005: CONFIRMED
direct_fish_or_recording_overlap: "no evidence found"
```

DS-006 cannot be used to alter the frozen primary DS-005 method.
Its independence is based on its separate source, DOI, publication, acquisition,
assay protocol, frame rate, recording duration, and tracking pipeline. Some
investigators overlap and the datasets were later analyzed together, but DS-006
is not a resplit or known reuse of DS-005 fish or recordings.

---

# 6. Primary Unit of Analysis

The frozen primary unit is:

> **One valid behavioral bout belonging to one identifiable zebrafish.**

For DS-005, each valid bout contains 175 temporal samples.

The earlier provisional fixed-duration window plan is superseded for the primary analysis.

---

# 7. Train / Validation / Test Governance

DS-005 is split by fish:

```yaml
split_seed: 20260822
train_fish: 323
validation_fish: 70
test_fish: 70
fish_overlap: 0
```

All bouts inherit their source fish partition.

The TEST partition must not influence:

- feature definitions;
- normalization;
- SSL architecture;
- SSL objective;
- augmentation;
- embedding dimension;
- seed set;
- clustering method;
- cluster count;
- nuisance models;
- evaluation metrics;
- thresholds;
- or interpretation rules.

---

# 8. Input A — Hand-Engineered Baseline

Input A is frozen as an 18-feature core baseline containing:

## Timing

- bout duration
- inter-bout interval

## Speed

- mean
- standard deviation
- median
- maximum
- 95th percentile
- RMS

## Acceleration / speed change

- mean absolute change
- standard deviation
- maximum absolute change
- RMS change

## Orientation / turning

- total absolute turn
- net turn
- mean absolute turn
- standard deviation
- maximum absolute turn
- RMS turn

`head_pos`-derived path/jump features are excluded from the primary baseline because of coordinate-semantic discontinuity concerns.

They remain secondary sensitivity features only.

---

# 9. Input B — SSL Temporal Representation

Each valid bout is represented as:

```text
shape: (175, 3)

channel 0 = sin(orientation_smooth)
channel 1 = cos(orientation_smooth)
channel 2 = speed_head
```

No metadata such as fish ID, context, stimulus code, bout type, session label, partition, or cluster label may be included as encoder input.

---

# 10. SSL Method

The primary SSL method is frozen.

```yaml
objective: temporal_contrastive_learning
loss: NT-Xent
temperature: 0.10
embedding_dimension: 64
```

Encoder:

```text
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

Downstream discovery uses the encoder embedding, not the projection-head output.

Frozen seeds:

```text
11
23
37
51
79
```

---

# 11. Primary Baseline Discovery

The frozen handcrafted-feature discovery pipeline is:

```text
Input A
  ↓
TRAIN-only PCA
  ↓
6 components
  ↓
Gaussian Mixture Model
  ↓
k = 2
```

Frozen configuration:

```yaml
pca_components: 6
explained_variance_retained: 0.9579
clustering_method: GaussianMixture
k: 2
seed: 20260822
validation_silhouette: 0.4158
stability: 0.9992
selection_score: 0.649252
test_used: false
```

The selected `k=2` is the best evaluated baseline configuration under the predefined TRAIN/VALIDATION procedure.

It is not interpreted as evidence that zebrafish behavior has exactly two true biological states.

---

# 12. Confirmatory Evaluation Principles

The project evaluates:

- cross-fish reproducibility;
- cross-seed stability;
- Input A vs Input B structural overlap;
- predictability of SSL clusters from Input A;
- speed dependence;
- fish-identity leakage;
- context/session leakage;
- tracking-artifact dependence;
- external replication.

The frozen evaluation specification is documented in:

```text
docs/evaluation-protocol.md
```

---

# 13. Primary Validity Threats

## Identity leakage

The model may cluster individual fish rather than behavior.

## Context / session leakage

The model may encode recording context, stimulus context, or acquisition proxies instead of behavior.

## Speed-only solution

The embedding may primarily reconstruct locomotor speed.

## Tracking artifacts

Tracking failure or extreme coordinate behavior may create false clusters.

## Temporal-boundary artifacts

Discovered structure may depend on arbitrary bout boundary or padding effects.

## Hyperparameter fishing

Repeated methodological adaptation to improve downstream outcomes would invalidate confirmatory interpretation.

## Post-hoc storytelling

Clusters must not be assigned biological meaning solely because a qualitative narrative can be constructed after discovery.

---

# 14. Sensitivity Analysis Governance

Frozen categories:

```yaml
ssl_seed_sensitivity: CONFIRMATORY
head_position_extended_baseline: SECONDARY
cluster_number_sensitivity: SECONDARY
visualization_dimensionality_reduction: EXPLORATORY
alternate_segmentation: NOT_PRIMARY
```

No alternate primary segmentation is required because the unit of analysis is already a natural valid bout.

---

# 15. External Replication Governance

DS-006 may test whether the qualitative baseline-vs-SSL conclusion generalizes.

DS-006 may not be used to:

- alter Input A;
- alter Input B;
- alter the SSL objective;
- alter the encoder;
- alter the primary seed set;
- alter primary hyperparameters;
- alter the primary cluster count;
- or redefine the primary success criteria.

---

# 16. Scientific Success Criteria

The project is successful if it produces a rigorous, reproducible answer to the primary research question.

Scientific success does **not** require a positive SSL result.

Valid final outcomes include:

```text
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED_EQUIVALENT
NOT_SUPPORTED_NUISANCE
NOT_SUPPORTED_UNSTABLE
NOT_SUPPORTED_REPLICATION_FAILURE
```

---

# 17. Reproducibility Requirements

Material decisions must be recorded in:

```text
docs/decision-log.md
```

Dataset provenance, authorization, and role must be recorded in:

```text
docs/dataset-register.md
```

Confirmatory evaluation rules must be recorded in:

```text
docs/evaluation-protocol.md
```

Processed artifacts should be hashed where practical.

---

# 18. Scope Boundaries

This first study does not claim:

- the first use of SSL in zebrafish;
- the first unsupervised discovery of zebrafish behavior;
- a universal zebrafish ethogram;
- that clusters are automatically biological states;
- that `k=2` is the true number of behaviors;
- that SSL must outperform conventional features.

The novelty claim is narrower:

> A matched comparison of conventional hand-engineered zebrafish behavioral representations and self-supervised temporal representations under explicit held-out-fish, nuisance-control, reproducibility, and external-replication governance.

---

# 19. Current Project State

```yaml
primary_dataset: FROZEN
unit_of_analysis: FROZEN
fish_split: FROZEN
input_a: FROZEN
input_b: FROZEN
ssl_method: FROZEN
primary_baseline_discovery: FROZEN
evaluation_protocol: FROZEN
sensitivity_plan: FROZEN
claim_level: FROZEN

ssl_full_training: COMPLETE
ssl_train_validation_analysis: COMPLETE
validation_interpretation_freeze: PREPARED_PENDING_COMMIT

ds005_test_partition: PROTECTED
ds006_test_partition: SEALED

charter_status: STABLE
```

---

# 20. Charter Freeze Rule

This charter is considered stable for preregistration.

Any material change to:

- the research question;
- primary dataset;
- replication dataset;
- unit of analysis;
- primary representations;
- SSL method;
- clustering governance;
- evaluation framework;
- or target claim

requires a documented amendment in `docs/decision-log.md`.

No such change may be justified by protected TEST performance.
