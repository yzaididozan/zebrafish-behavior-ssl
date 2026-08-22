# Project Decisions Log

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

This file records material methodological decisions, freeze points, amendments, and rationale for the first confirmatory experiment.

---

## Decision Status Legend

- **FROZEN** — should not be changed based on downstream results without a documented amendment.
- **CANDIDATE** — selected for current development but not yet fully frozen.
- **OPEN** — not yet decided.
- **SUPERSEDED** — replaced by a later documented decision.

---

# DEC-001 — Primary Dataset

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

Use **DS-005** as the frozen primary dataset for the first experiment.

## Recorded dataset facts

- 463 zebrafish
- 1,203,409 valid bouts
- 14 contexts
- canonical fish identifiers available
- frozen fish-level train/validation/test split available
- bout-level temporal tracking/kinematic arrays available
- structural QC and archive verification completed

## Rationale

DS-005 supports large-scale temporal modeling, fish-level leakage control, baseline feature construction, and held-out-fish evaluation.

## Consequence

The first confirmatory study will be designed around the information available in DS-005. Conclusions must remain within the dataset's biological and recording scope.

---

# DEC-002 — Unit of Analysis

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

The primary unit of analysis is:

> **One valid behavioral bout belonging to one identifiable zebrafish.**

## Superseded plan

The earlier provisional plan used fixed-duration temporal windows such as 0.5 s, 1 s, 2 s, or 4 s.

That plan is **SUPERSEDED** for the primary DS-005 analysis.

## Rationale

DS-005 already contains a natural bout-level temporal representation with 175 temporal samples per valid bout.

## Consequence

All primary baseline and SSL representations must correspond to the same valid bouts.

---

# DEC-003 — Fish-Level Data Split

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

Use the frozen fish-level split:

```yaml
split_seed: 20260822
train_fish: 323
validation_fish: 70
test_fish: 70
fish_overlap: 0
```

All bouts inherit the partition of their source fish.

## Rule

The final test fish must not influence:

- baseline feature definition,
- normalization statistics,
- SSL architecture selection,
- SSL objective selection,
- augmentation selection,
- embedding dimension selection,
- clustering method selection,
- number-of-clusters selection,
- or primary metric selection.

---

# DEC-004 — Input A: Hand-Engineered Baseline

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

Use the generated **core** hand-engineered baseline as Input A.

The core baseline contains 18 bout-level features from:

- bout timing,
- inter-bout interval,
- speed summaries,
- acceleration/speed-change summaries,
- wrapped orientation-change summaries.

## Exclusion

Raw `head_pos` path/jump features are excluded from the primary baseline because preregistered QC identified coordinate-semantic discontinuities.

Head-position-derived features remain exploratory/sensitivity features only.

## Rule

Input A must not be expanded or altered based on downstream SSL or clustering results without a recorded protocol amendment.

---

# DEC-005 — Candidate Input B Temporal Representation

**Date:** 2026-08-22  
**Status:** CANDIDATE

## Decision

Represent each valid bout as a temporal SSL tensor with shape:

```text
(175, 3)
```

Channels:

```text
0 = sin(orientation_smooth)
1 = cos(orientation_smooth)
2 = speed_head
```

## Rationale

- `orientation_smooth` and `speed_head` are full temporal sequences rather than hand-engineered summary statistics.
- sine/cosine encoding avoids artificial angular wraparound discontinuities.
- `head_pos` is excluded from the primary candidate because of the previously documented coordinate-semantic discontinuity concern.
- metadata such as fish ID, context, stimulus code, bout type, and partition are not model input channels.

## Current status

The numerical Input B representation is defined for development, but the full Input B pipeline is not fully frozen until the primary SSL objective, encoder architecture, and embedding dimension are frozen.

---

# DEC-006 — Train-Only SSL Normalization

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

Normalize only the `speed_head` channel using global z-score statistics fitted on primary-QC-valid **training bouts only**.

The sine and cosine orientation channels remain unstandardized.

## Fitted statistics

```yaml
train_bouts_used: 842841
temporal_speed_samples_used: 147497175
speed_mean: 0.85842903292
speed_std: 1.26054458491
validation_used_for_fit: false
test_used_for_fit: false
```

## Rule

The same frozen training-derived statistics must be applied to train, validation, and test data.

Validation or test statistics must never be used to refit normalization.

---

# DEC-007 — SSL Augmentation Policy v1

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

Use conservative contiguous temporal masking as the primary SSL augmentation.

```yaml
temporal_mask:
  enabled: true
  probability: 0.75
  max_fraction: 0.10
  contiguous: true
  mask_value: 0.0

feature_mask:
  enabled: false
  probability: 0.0
```

## Explicitly disabled for primary v1

- whole-bout feature masking
- Gaussian noise
- temporal cropping
- temporal warping
- rotation augmentation
- translation augmentation

## Explicitly prohibited

- time reversal
- temporal shuffling
- cross-bout mixing
- cross-fish mixing
- cross-partition mixing

## Rationale

Initial feature masking removed the entire orientation representation from a bout and changed all 175 timesteps, which was considered too destructive for the first primary augmentation policy.

Temporal masking preserves temporal order and modifies only a small contiguous fraction of a bout.

---

# DEC-008 — Augmentation Validation

**Date:** 2026-08-22  
**Status:** FROZEN EVIDENCE

## Unit tests

```yaml
tests_passed: 12
tests_failed: 0
```

The augmentation implementation passed checks for:

- expected `(175, 3)` shape,
- finite outputs,
- deterministic same-seed behavior,
- different-seed variability,
- no in-place modification,
- correct grouped orientation masking behavior,
- speed-only masking behavior,
- and seed/RNG safety.

## Real-bout inspection

Using a real training bout after fitted normalization:

```yaml
fish_id: DS005-JM-F001
bout_index: 0
partition: train
view_a_changed_timesteps_fraction: 0.08
view_b_changed_timesteps_fraction: 0.0171
finite_outputs: true
shape_preserved: true
```

## 1,000-bout TRAIN QC

Final candidate configuration used temporal masking probability `0.75`.

```yaml
sample_bouts: 1000
partition: train
finite_failures: 0
shape_failures: 0
equal_view_pairs: 74
equal_view_pair_fraction: 0.074
max_changed_timestep_fraction: 0.0971
validation_used: false
test_used: false
```

## Interpretation

The v1 augmentation policy produced finite, shape-preserving, modest perturbations across the sampled training bouts while reducing identical positive-view pairs from 23.7% at probability 0.50 to 7.4% at probability 0.75.

---

# DEC-009 — Baseline Clustering Governance

**Date:** 2026-08-22  
**Status:** PARTIALLY FROZEN

## Current pipeline

Baseline discovery includes:

- train-only PCA fitting,
- clustering on the baseline representation,
- candidate methods including K-Means and Gaussian Mixture Models,
- model selection using training/validation data only,
- untouched test partition during selection.

## Frozen principle

The test partition must remain untouched until final evaluation.

## Still open

The final primary clustering method, final selected number of clusters, and any remaining confirmatory selection rule details must be frozen after training/validation selection and before final test evaluation.

---

# DEC-010 — Primary SSL Objective

**Date:** 2026-08-22  
**Status:** OPEN

## Current leading candidate

**Temporal contrastive learning**

## Rationale for candidacy

The current augmentation pipeline naturally creates two conservative views of the same bout, which is compatible with a contrastive objective.

## Rule

The primary SSL objective must be frozen before final test evaluation and must not be selected based on test performance.

---

# DEC-011 — Primary SSL Encoder

**Date:** 2026-08-22  
**Status:** OPEN

## Current leading candidate

A small **1D temporal convolutional encoder**.

## Rationale for candidacy

The primary sequence is short `(175, 3)`, and a compact 1D CNN provides:

- local temporal feature learning,
- lower computational cost,
- fewer hyperparameters,
- and a simpler confirmatory MVP than a Transformer.

## Still open

- exact architecture,
- number of layers,
- channel widths,
- pooling,
- projection head,
- embedding dimension,
- optimizer,
- learning rate,
- batch size,
- training epochs / stopping rule,
- random seed set.

---

# DEC-012 — Claim Threshold

**Date:** 2026-08-22  
**Status:** FROZEN

## Target claim

**Claim Level 2**

> Self-supervised temporal representations reveal reproducible behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose features.

## Required evidence

The claim requires converging evidence that SSL-derived structure:

- is nontrivial,
- is stable across repeated runs,
- reproduces in held-out fish,
- is not predominantly fish identity,
- is not predominantly session/context,
- cannot be explained solely by speed,
- is not primarily a tracking artifact,
- survives reasonable sensitivity analyses,
- and cannot be completely reconstructed from Input A.

The study does not require SSL to outperform the baseline.

Negative, equivalent, or nuisance-driven outcomes remain valid results.

---

# DEC-013 — Test-Set Protection Rule

**Date:** 2026-08-22  
**Status:** FROZEN

## Decision

The final DS-005 test partition remains untouched during:

- baseline clustering selection,
- SSL design,
- augmentation tuning,
- normalization fitting,
- encoder selection,
- embedding-dimension selection,
- clustering-method selection,
- and primary metric selection.

## Consequence

Any accidental test inspection that could influence a methodological choice must be documented as a deviation.

---

# Current Decision Summary

```yaml
primary_dataset: FROZEN
unit_of_analysis: FROZEN
fish_split: FROZEN
input_a: FROZEN
input_b_tensor: CANDIDATE
ssl_normalization: FROZEN
ssl_augmentation_v1: FROZEN
primary_ssl_objective: OPEN
primary_ssl_encoder: OPEN
primary_discovery_method: PARTIALLY_FROZEN
primary_metrics: PARTIALLY_FROZEN
nuisance_tests: DEFINED
claim_threshold: FROZEN
test_partition: PROTECTED
formal_preregistration: NOT_READY
```

---

# Next Decision Gate

Before formal preregistration, freeze:

1. the primary SSL objective,
2. the primary SSL encoder architecture,
3. the embedding dimension or validation-only selection rule,
4. the primary clustering method and state-number selection rule,
5. the complete primary confirmatory metric set and decision procedures.

No final test evaluation should occur before those decisions are recorded.
