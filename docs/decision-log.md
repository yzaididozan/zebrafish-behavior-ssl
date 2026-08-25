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
**Status:** SUPERSEDED

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

**Superseded by:** `DEC-024`, which freezes Input B and the complete primary SSL method.

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
**Status:** SUPERSEDED

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

**Superseded by:** `DEC-016`, which freezes `PCA(6) -> GMM(k=2, seed=20260822)`.

---

# DEC-010 — Primary SSL Objective

**Date:** 2026-08-22  
**Status:** SUPERSEDED

## Current leading candidate

**Temporal contrastive learning**

## Rationale for candidacy

The current augmentation pipeline naturally creates two conservative views of the same bout, which is compatible with a contrastive objective.

## Rule

The primary SSL objective must be frozen before final test evaluation and must not be selected based on test performance.

**Superseded by:** `DEC-024`, which freezes temporal contrastive learning with NT-Xent at temperature `0.10`.

---

# DEC-011 — Primary SSL Encoder

**Date:** 2026-08-22  
**Status:** SUPERSEDED

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

**Superseded by:** `DEC-024`, which freezes the exact encoder, embedding dimension, optimizer, training schedule, and seed set.

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

## DEC-014 — ML-03 smoke-test verification

**Date:** 2026-08-22  
**Status:** VERIFIED

A TRAIN-only smoke test of the SSL pipeline passed successfully using 2,048 behavioral bouts and training seed 11.

### Result

- Initial epoch loss: 4.739312
- Final epoch loss: 3.907831
- Epochs: 3
- Finite gradients: yes
- Model parameters updated: yes
- Checkpoint save/load: passed
- Validation partition used: no
- Test partition used: no

### Decision

The current SSL training pipeline is mechanically valid for progression to full TRAIN/VALIDATION training.

This smoke test is treated only as a pipeline-verification result and is not used as model-selection evidence.

## DEC-015 — SSL TRAIN/VALIDATION preflight passed

**Date:** 2026-08-22  
**Status:** VERIFIED

A capped TRAIN/VALIDATION preflight of the full SSL training pipeline passed successfully using seed 11.

### Result

- TRAIN bouts: 10,000
- VALIDATION bouts: 2,000
- Epochs: 2
- Batch size: 256
- Train loss: 4.275675 → 3.821423
- Validation loss: 3.762066 → 3.634544
- Best validation checkpoint selected: yes
- Last checkpoint saved: yes
- Checkpoint reload: passed
- Test partition used: no

### Decision

The SSL training pipeline is verified for progression to the preregistered full TRAIN/VALIDATION training runs.

This capped run is treated as pipeline verification only and is not used as final model-selection evidence.

## DEC-016 — Baseline clustering configuration selected

**Date:** 2026-08-22  
**Status:** FROZEN

Baseline clustering model selection was completed using TRAIN and VALIDATION only.

### Inputs

- TRAIN rows: 842,841
- VALIDATION rows: 168,464
- Hand-engineered input features: 18
- PCA fit partition: TRAIN only
- PCA components retained: 6
- Explained variance retained: 0.9579
- TEST partition used: no

### Selected configuration

- Method: Gaussian Mixture Model (GMM)
- Number of clusters: 2
- Seed: 20260822
- Selection score: 0.649252
- Validation silhouette: 0.4158
- Stability: 0.9992

### Comparison

The next-highest candidate was KMeans with k=2:

- Selection score: 0.5901
- Validation silhouette: 0.3315
- Stability: 0.9708

The selected GMM configuration therefore had the strongest predefined TRAIN/VALIDATION selection score among the evaluated candidates.

### Decision

The primary handcrafted-feature baseline discovery configuration is frozen as:

`PCA(6 components) -> GMM(k=2, seed=20260822)`

No further baseline clustering method or cluster-count tuning will be performed using the TEST partition.

The held-out TEST partition remains untouched until final evaluation.

### Artifacts

- `data/processed/DS-005/baseline_clustering/selection_results.json`
- `data/processed/DS-005/baseline_clustering/pca_diagnostics.json`
- `data/processed/DS-005/baseline_clustering/selected_configuration.json`
- `data/processed/DS-005/baseline_clustering/SELECTION_SHA256SUMS`

---

# DEC-017 — DS-006 External Replication Dataset

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Use **DS-006** as the external replication dataset.

DS-006 is restricted to external replication and cannot influence the frozen primary DS-005 method.

```yaml
replication_dataset_role: EXTERNAL_REPLICATION
allowed_to_change_primary_method: false
allowed_for_primary_hyperparameter_selection: false
allowed_for_ssl_architecture_selection: false
allowed_for_cluster_k_selection: false
independence_from_ds005:
  separate_source_dataset: true
  separate_dataset_doi: true
  separate_publication: true
  separately_acquired_recordings: true
  different_recording_protocol: true
  different_frame_rate: true
  different_recording_duration: true
  different_tracking_pipeline: true
  overlapping_authors: true
  direct_fish_or_recording_overlap: "no evidence found"
  independence_status: "CONFIRMED"
```

DS-006 is an independently acquired Reddy et al. (2022) dataset and is not a
resplit or known reuse of the Marques et al. fish or recordings underlying
DS-005. It differs in assay, acquisition rate, recording duration, stimulus
conditions, and tracking pipeline. Some investigators overlap, and the datasets
were later analyzed together in Sridhar et al. (2024); neither fact creates
direct fish- or recording-level overlap.

## Dataset provenance

- Repository: Dryad
- DOI: `10.5061/dryad.6t1g1jwwz`
- Local archive: `data/raw/DS-006/Data_all.zip`
- Archive size: 1,527,805,725 bytes
- SHA-256: `d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f`
- 32 MATLAB result files
- 32 text representations
- 32 observed experiments / recordings
- 12 wells per experiment
- 384 total fish-well slots

## Fish identity

`FishNumber` was observed to contain only `0`, consistent with one fish per well but not usable as a cross-recording identifier.

The canonical replication unit identifier is:

```text
DS006::<recording_id>::wellXX
```

where `recording_id = organization.videoName`.

Biological uniqueness of fish across separate recordings is not independently verified.

## Archive discrepancy

The author documentation indicates eight `pH_2b` experiments, while seven are present in the archive.

The apparent missing recording is:

```text
Catamaran_pH_2b_t7
```

This discrepancy remains documented and unresolved.

---

# DEC-018 — DS-006 Frame Rate Interpretation

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Use:

```yaml
authoritative_frame_rate_hz: 160
```

for DS-006 replication preprocessing.

## Evidence

The author analysis notebook explicitly sets:

```python
fps = 160.0
```

before processing all four experiment families and uses that value for bout duration, speed, frequency, angular speed, inter-bout interval, and trajectory timing.

The MATLAB metadata contains both `25` and `160` Hz values. The 25 Hz entries conflict with the author analysis pipeline and produce implausibly inflated bout durations.

## Consequence

Embedded `organization.fps = 25` values are preserved as a provenance discrepancy but are not used as authoritative timing metadata.

Corrected median bout durations at 160 Hz were approximately:

```yaml
pH_1a: 0.15625
pH_2a: 0.13750
pH_2b: 0.15000
pH_2c: 0.16875
```

---

# DEC-019 — DS-006 Condition Mapping and Well-Level QC

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Use author notebook well-type annotations to reconstruct DS-006 condition labels.

The mapping is stored at:

```text
data/manifests/DS-006/well-condition-map.csv
```

## Observed well-slot counts

```yaml
exploratory: 192
double_sharp_pH: 143
homogeneous_pH: 42
bad_data: 7
total: 384
```

## Well-level QC

Seven author-labeled `bad_data` wells and three empty wells are excluded.

```yaml
total_well_slots: 384
author_bad_wells: 7
empty_wells: 3
overlap_bad_and_empty: 0
combined_excluded_wells: 10
usable_wells: 374
```

Empty wells:

```text
191119_Catamaran_pH_2b_t8  well04
Catamaran_pH_1a_t2c        well04
Catamaran_pH_2a_t3         well11
```

## Consequence

The exact biological number of unique fish across recordings is not asserted.

The replication instead records:

- 384 total fish-well slots
- 374 usable fish-well units after well-level QC

---

# DEC-020 — DS-006 Deterministic Bout-Level QC

**Date:** 2026-08-23  
**Status:** FROZEN

## Author method

The author notebook rejects bouts when:

```text
time < 0.04 s
time > 1.2 s
distance > 25 mm
distance < 0 mm
speed > 50 mm/s
speed < 1 mm/s
abs(deltahead) > 180 degrees
```

with:

```yaml
px_to_mm: 0.071
fps_hz: 160
```

The author trajectory interpolation adds Gaussian noise before `splprep` smoothing:

```python
0.1 * np.random.randn(...)
```

No NumPy random seed was found in the notebook.

## Decision

Exact bitwise reproduction of the author QC is therefore not guaranteed.

For reproducible external replication, retain the author equations but fix:

```yaml
replication_qc_seed: 20260822
interpolation_noise_sd: 0.1
interpolation_smoothing_parameter: 10
```

## Frozen result

```yaml
raw_bouts: 165579
well_excluded_bouts: 1556
bout_qc_candidates: 164023
accepted_bouts: 163065
rejected_bouts: 958
acceptance_rate: 0.994159
```

Rejection reasons:

```yaml
speed_too_high: 532
speed_too_low: 279
distance_too_large: 206
time_too_long: 140
```

A bout may fail more than one criterion, so rejection-reason counts need not sum to the number of rejected bouts.

---

# DEC-021 — DS-006 Representation Feasibility

**Date:** 2026-08-23  
**Status:** VERIFIED

## Input A

The DS-006 feasibility audit confirmed that the same 18 target feature families used by the DS-005 handcrafted baseline can be computed.

All features are computable for 100% of accepted bouts except inter-bout interval:

```yaml
bout_duration: 100.000%
inter_bout_interval: 99.771%
remaining_16_features: 100.000%
```

The IBI reduction is expected because the first accepted bout in a usable well has no preceding accepted bout.

## Input B

All accepted bouts can form a valid 3-channel temporal representation:

```text
sin(Heading)
cos(Heading)
derived_head_speed_mm_s
```

Feasibility result:

```yaml
valid_ssl_bouts: 163065
accepted_bouts: 163065
input_b_feasibility: VERIFIED
```

## Comparability caveat

The DS-006 mappings are analogous rather than numerically identical to DS-005:

```text
DS-005 orientation_smooth  -> DS-006 Heading
DS-005 speed_head          -> DS-006 speed derived from HeadX/HeadY
```

This difference must remain explicit in reporting.

---

# DEC-022 — DS-006 Deterministic Preprocessing and Split

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Use the deterministic DS-006 preprocessing implementation in:

```text
src/data/prepare_ds006.py
```

The pipeline:

1. reproduces frozen well and bout QC;
2. creates the 18-feature Input A representation;
3. creates the 3-channel Input B temporal representation;
4. assigns leakage-safe replication partitions;
5. fits permitted normalization using replication TRAIN only;
6. writes processed artifacts and manifests;
7. keeps the replication TEST partition out of preprocessing fitting.

## Recording-level split

The split is conservative at the recording level using seed:

```yaml
split_seed: 20260822
```

Result:

```yaml
train_recordings: 22
validation_recordings: 5
test_recordings: 5

train_bouts: 118100
validation_bouts: 18835
test_bouts: 26130

recording_overlap: 0
```

By family:

```yaml
pH_1a:
  train: 6
  validation: 2
  test: 2

pH_2a:
  train: 5
  validation: 1
  test: 1

pH_2b:
  train: 5
  validation: 1
  test: 1

pH_2c:
  train: 6
  validation: 1
  test: 1
```

## Baseline preprocessing

The 18-feature baseline uses:

- TRAIN-only median imputation for undefined values such as first-bout IBI;
- TRAIN-only featurewise z-score scaling.

## SSL preprocessing

DS-006 variable-length bouts are deterministically resampled over normalized bout phase to:

```text
(175, 3)
```

so the frozen primary encoder geometry does not need to change.

Only the derived speed channel is standardized.

TRAIN-only fitted statistics:

```yaml
ssl_speed_mean: 9.773209740465
ssl_speed_std: 10.478936765051
```

The sine/cosine orientation channels remain unstandardized.

## Test protection

The DS-006 TEST arrays were created but were not used to fit:

- imputation,
- scaling,
- speed normalization,
- temporal transformation parameters,
- or split decisions.

The replication TEST partition remains sealed for final evaluation.

---

# DEC-023 — DS-006 Processed Artifact Freeze

**Date:** 2026-08-23  
**Status:** FROZEN EVIDENCE

## Decision

Record SHA-256 fingerprints for the DS-006 processed replication artifacts before clustering or representation evaluation.

Hash manifest:

```text
data/manifests/DS-006/processed-sha256.txt
```

Recorded artifacts include:

- baseline raw arrays,
- baseline scaled arrays,
- SSL arrays,
- feature manifest,
- normalization manifests,
- split assignments,
- bout metadata,
- QC summary.

## Key hashes

```text
5dcaf8447e969114cb7f1fa40ae24ed66194ec388fb02802583e2b716b14f315  data/processed/DS-006/baseline/feature_manifest.json
fe171fcc100bc63a576388cb7d195c098c79ba4009d3621bfd6c2856885a50ea  data/processed/DS-006/baseline/normalization.json
41a39dd0f2035520b5a2f07514e6cd90c2e9478cb4bc54ea5e52604a00a36406  data/processed/DS-006/baseline/train_core_raw.npz
78bd247585e06ce3f739b6c8cf1f15a21ee0e01d5313476d31162840fde895dc  data/processed/DS-006/baseline/train_core_scaled.npz
8d4ec58dc27647f373207822e8a49bf6dd1022f4e92cd8aafcbf325f1ef66888  data/processed/DS-006/baseline/validation_core_raw.npz
82099ee761ba03055f28e77b18f4d01ba056cc75cf11d5b5089d0fa83a6e2dfe  data/processed/DS-006/baseline/validation_core_scaled.npz
4c442917cf7e3549da712aff3cb25ef58be596bb0c86b639d3229611357032c6  data/processed/DS-006/baseline/test_core_raw.npz
5d9e576dee347f235efed51134c846aee01aee5e44483308d2bdb7a676c802d2  data/processed/DS-006/baseline/test_core_scaled.npz
30063fbce73b656c71a4ed38f54b4c46cfb68992b6f0f50089e2a761ca253fd3  data/processed/DS-006/ssl/train.npz
a3139a17cb037515ad2114959ed0ed1d9317a86647a68615648cac25d4028f7f  data/processed/DS-006/ssl/validation.npz
5b4291bd46ec06ddc0a5c03a7b4b595559d85f861dd882375f8bfee10ec81bd8  data/processed/DS-006/ssl/test.npz
7625b0f32731f0e8e67fbf09b32fac36a376128e0a2c8f85821e39ba71aad47e  data/processed/DS-006/metadata/bout_metadata.csv
4cb68d5c6cbe502460932205f43e3f5b4f9ee7d997165f9cf9a857b83d00d0eb  data/processed/DS-006/metadata/qc_summary.json
02110af57699a29c6ab2d2a795052195b103b422bb7891466089ea120d37155c  data/processed/DS-006/metadata/split_assignments.csv
6ddade95cc8c9a26843d2044c029159440b232ae9dcd1f9e451cd6348e823dbb  data/processed/DS-006/ssl/input_manifest.json
3fca90272b9a9bef52d5edfe5877c2df1bdc8a2c43d72b8a9472d0b3397f0de4  data/processed/DS-006/ssl/normalization.json
```

## Consequence

These processed artifacts constitute the frozen DS-006 replication input state.

Any later regeneration that changes a recorded hash must be treated as a preprocessing amendment and documented before replication evaluation continues.

---

# DEC-024 — Primary SSL Method Freeze

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Freeze the complete primary DS-005 SSL method before final TEST evaluation.

This decision supersedes the candidate/open states recorded in `DEC-005`, `DEC-010`, and `DEC-011`.

## Input B

Each valid bout is represented as:

```text
shape: (175, 3)

channel 0 = sin(orientation_smooth)
channel 1 = cos(orientation_smooth)
channel 2 = speed_head
```

Metadata such as fish ID, context, stimulus code, bout type, session, partition, or cluster label are not encoder inputs.

## SSL objective

```yaml
family: temporal_contrastive_learning
loss: NT-Xent
temperature: 0.10
```

Two conservative augmented views of the same bout form the positive pair.

## Encoder

Frozen encoder:

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

The downstream representation is the **64-dimensional encoder embedding**.

## Projection head

```text
64 -> 64 -> 64
```

The projection-head output is used only for the contrastive training objective.

It is not used for downstream clustering or confirmatory evaluation.

## Optimization

```yaml
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 256
maximum_epochs: 50
early_stopping_patience: 8
gradient_clip_norm: 1.0
mixed_precision: false
checkpoint_selection_metric: validation_loss
```

## Frozen seed set

```yaml
seeds:
  - 11
  - 23
  - 37
  - 51
  - 79
```

## Consequence

The active multi-seed SSL training runs execute this frozen method.

Differences among seed results must not be used to alter:

- Input B;
- objective;
- architecture;
- embedding dimension;
- projection head;
- augmentation policy;
- optimizer;
- learning rate;
- seed set;
- or stopping rule.

Full training completion is not required for the method itself to be considered frozen.

The DS-005 TEST partition remains protected.

---

# DEC-025 — Confirmatory Evaluation Protocol Freeze

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Adopt:

```text
docs/evaluation-protocol.md
```

as the confirmatory evaluation protocol for the primary DS-005 study and the governance framework for DS-006 external replication.

The protocol freezes the previously open evaluation and validity-control procedures before final TEST analysis.

## Cross-fish reproducibility

Primary procedures:

```yaml
bootstrap_unit: fish
bootstrap_replicates: 500
bootstrap_seed: 20260822
confidence_interval: percentile_95
```

Primary metrics:

- fish-bootstrap Adjusted Rand Index (ARI);
- held-out cluster occupancy by fish;
- cross-seed pairwise ARI.

For the five frozen SSL seeds, all 10 pairwise seed comparisons are reported.

Primary summaries:

- median;
- IQR;
- minimum;
- maximum where applicable.

Interpretive ARI bands:

```yaml
ARI_ge_0.75: strong
ARI_0.50_to_0.75: moderate
ARI_0.25_to_0.50: weak
ARI_lt_0.25: poor
```

Strong reproducibility evidence is defined as:

```yaml
median_fish_bootstrap_ARI: ">= 0.75"
lower_95pct_bootstrap_bound: ">= 0.50"
median_cross_seed_ARI: ">= 0.75"
```

These are interpretation / decision bands and do not alone establish biological validity.

## Baseline-vs-SSL structural comparison

Primary partition-overlap metrics:

```text
Adjusted Rand Index
Normalized Mutual Information
```

A fixed multinomial logistic-regression analysis predicts SSL cluster membership from Input A.

Primary metric:

```text
macro-averaged balanced accuracy
```

Secondary metric:

```text
macro F1
```

Near-perfect reconstruction of SSL clusters from Input A weakens the claim that SSL captures additional structure.

## Speed-control procedure

Frozen controls:

1. speed-only clustering using mean bout speed;
2. SSL embedding -> mean speed ridge regression;
3. cluster-level speed enrichment.

Frozen regression nuisance model:

```yaml
model: Ridge
alpha: 1.0
```

Primary speed-regression metric:

```text
R^2
```

Secondary:

```text
MAE
```

Descriptive speed-predictability bands:

```yaml
R2_lt_0.25: low
R2_0.25_to_0.50: moderate
R2_0.50_to_0.75: high
R2_ge_0.75: very_high
```

## Fish-identity leakage

Primary classifier:

```yaml
model: multinomial_logistic_regression
penalty: L2
C: 1.0
solver: saga
max_iter: 1000
class_weight: balanced
```

Primary metric:

```text
balanced accuracy
```

Secondary:

```text
macro F1
```

Also report:

```text
observed balanced accuracy / uniform chance
```

The matched Input A leakage result must be reported alongside SSL.

## Context / session leakage

Use the same frozen multinomial logistic-regression specification.

Primary target:

```text
DS-005 context label
```

Metrics:

```text
balanced accuracy
macro F1
```

If session/recording identity cannot be independently estimated from fish identity, that limitation must be documented rather than inventing a separate session leakage result.

## Tracking-quality rule

No new post-clustering DS-005 exclusion rule may be introduced.

```yaml
new_post_clustering_exclusion_rules: prohibited
```

Existing QC / artifact proxy distributions are compared across discovered clusters.

If a cluster is enriched for tracking or QC boundary cases, it is reported as potentially artifactual rather than silently removed.

## Primary confirmatory metric set

Frozen primary evaluation family:

```text
validation silhouette
fish-bootstrap ARI
cross-seed ARI
baseline-vs-SSL ARI
baseline-vs-SSL NMI
Input-A -> SSL-cluster balanced accuracy
Input-A -> SSL-cluster macro F1
speed-only vs SSL ARI
speed-only vs SSL NMI
embedding-to-speed R^2
embedding-to-speed MAE
cluster-level speed distributions
fish-ID balanced accuracy
fish-ID macro F1
fish-ID chance ratio
context balanced accuracy
context macro F1
cluster-wise tracking/QC proxy distributions
external replication stability / nuisance consistency
```

Because the bout count is very large, interpretation emphasizes:

- effect sizes;
- stability;
- confidence intervals;
- held-out generalization;
- nuisance-control comparisons;

rather than isolated p-values.

## SSL seed aggregation

For seed-dependent metrics, report:

```text
all five seed values
median
IQR
minimum
maximum
```

The primary across-seed summary is the **median**.

No seed may be removed because its result is unfavorable unless a documented technical failure invalidates the run.

## Claim Level 2 evaluation gates

A positive Claim Level 2 conclusion requires converging evidence across:

```text
A. reproducibility
B. structure beyond Input A
C. not speed-only
D. not identity-driven
E. not context/session-only
F. not artifact-driven
G. external replication support
```

Final outcome categories:

```yaml
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED_EQUIVALENT
NOT_SUPPORTED_NUISANCE
NOT_SUPPORTED_UNSTABLE
NOT_SUPPORTED_REPLICATION_FAILURE
```

Negative or equivalent outcomes remain valid confirmatory results.

---

# DEC-026 — Confirmatory Sensitivity Analysis Plan

**Date:** 2026-08-23  
**Status:** FROZEN

## Decision

Categorize planned sensitivity analyses before TEST evaluation.

### SSL seed sensitivity

```yaml
category: CONFIRMATORY
seeds: [11, 23, 37, 51, 79]
```

All frozen seeds are retained.

### Head-position extended baseline

```yaml
category: SECONDARY
```

Previously excluded `head_pos`-derived features may be reintroduced only as a sensitivity analysis.

They do not replace the frozen 18-feature primary baseline.

### Cluster-number sensitivity

```yaml
category: SECONDARY
```

Neighboring cluster counts may be evaluated to assess dependence on the frozen primary cluster-count choice.

The frozen primary clustering result remains primary.

### Visualization dimensionality reduction

```yaml
category: EXPLORATORY
```

PCA, UMAP, or t-SNE may be used for descriptive visualization.

They may not determine:

- confirmatory cluster labels;
- state number;
- thresholds;
- or representation selection.

### Temporal / segmentation sensitivity

```yaml
alternate_primary_segmentation: false
```

No alternate primary segmentation is required because DS-005 already provides a natural valid-bout unit with fixed 175-sample sequences.

Any alternate segmentation analysis is secondary or exploratory and cannot redefine the primary unit of analysis.

---

# DEC-027 — DS-002 / DS-003 / DS-004 Reference Governance

**Date:** 2026-08-23  
**Status:** FROZEN

## DS-002

The exact Zenodo dataset license was verified directly from record metadata:

```yaml
license_id: cc-by-4.0
license_name: Creative Commons Attribution 4.0 International
authorization_verified: true
verification_date: 2026-08-23
```

DS-002 remains a future social-behavior extension rather than part of the first single-fish confirmatory analysis.

## DS-003

Preserve DS-003 as:

```text
PRIOR_ART
BENCHMARK
```

Its principal role is conventional 3D tracking / feature-engineering / unsupervised-discovery precedent.

It is not used to tune the frozen DS-005 method or the official DS-006 replication.

## DS-004

Preserve DS-004 as:

```text
PRIOR_ART
BENCHMARK
TRACKING_CONVENTIONAL_ANALYSIS_REFERENCE
```

It supports tracking/QC and conventional behavioral-analysis context.

It is not used for primary or replication method selection.

## DS-001

DS-001 remains available for pose / tracking QC development **as needed**.

No additional DS-001 work is required unless a specific QC problem arises.

---

# Updated Current Decision Summary

```yaml
primary_dataset: FROZEN
unit_of_analysis: FROZEN
fish_split: FROZEN

input_a: FROZEN
input_a_normalization: FROZEN

input_b_tensor: FROZEN
ssl_normalization: FROZEN
ssl_augmentation_v1: FROZEN
primary_ssl_objective: FROZEN
primary_ssl_encoder: FROZEN
ssl_embedding_dimension: 64
ssl_seed_set: [11, 23, 37, 51, 79]
ssl_training_configuration: FROZEN
ssl_full_training: COMPLETE
ssl_train_validation_analysis: COMPLETE
validation_interpretation_freeze: PREPARED_PENDING_COMMIT

primary_discovery_method: FROZEN
primary_baseline_clustering: "PCA(6) -> GMM(k=2, seed=20260822)"

cross_fish_reproducibility_procedure: FROZEN
speed_control_procedure: FROZEN
tracking_quality_confirmatory_rule: FROZEN
nuisance_prediction_models: FROZEN
primary_confirmatory_metric_set: FROZEN
sensitivity_analysis_plan: FROZEN

claim_threshold: FROZEN
ds005_test_partition: PROTECTED

ds006:
  role: EXTERNAL_REPLICATION
  ingestion: COMPLETE
  preprocessing: COMPLETE
  deterministic_qc: FROZEN
  processed_artifacts_hashed: true
  test_partition: OPENED_ONCE_FINAL_EVALUATION_COMPLETE
  final_test_freeze_commit: 575ead5403d0b2f721d143366b4d2e0014bd67ee
  final_test_result_commit: 2e59bf0db0bd00230a5349cbf344c290da396f60

reference_governance:
  DS002_license: VERIFIED_CC_BY_4_0
  DS003: PRIOR_ART_BASELINE_REFERENCE
  DS004: TRACKING_CONVENTIONAL_ANALYSIS_REFERENCE
  DS001: QC_AS_NEEDED

formal_preregistration: INTERNALLY_READY_PENDING_IMMUTABLE_TIMESTAMP
```

---

# Updated Next Decision Gate

The primary methodological design is now substantially frozen.

Before formal preregistration is declared complete:

1. confirm `docs/charter.md` is stable;
2. ensure `docs/preregistration-draft.md` reflects `DEC-024` through `DEC-027`;
3. preserve the completed five-seed TRAIN / VALIDATION artifacts and hashes;
4. commit or otherwise immutably timestamp `docs/validation-freeze.md` before opening DS-005 TEST;
5. keep the DS-005 TEST partition protected until that freeze is recorded;
6. preserve the completed one-time DS-006 TEST result without rerunning or
   selecting new configurations from it.

DS-006 satisfied its replication-side freeze condition before its one-time
final evaluation. No DS-005 final TEST evaluation should occur before the
remaining DS-005 checks are complete.

---

# DEC-028 — DS-005 Validation Interpretation and Final TEST Analysis Freeze

**Date:** 2026-08-24
**Status:** FROZEN PENDING COMMIT / IMMUTABLE TIMESTAMP

## Decision

Adopt `docs/validation-freeze.md` as the controlling record for interpretation
of completed DS-005 TRAIN / VALIDATION findings and for the one-time final TEST
analysis.

The validation evidence requires the following claim restriction:

> SSL produces a richer organization than the frozen two-state handcrafted
> clustering, but much of the SSL cluster organization is recoverable from the
> 18 handcrafted variables by a nonlinear mapping.

The study therefore must not claim that the SSL representation contains
information fundamentally absent from Input A. Mean VALIDATION balanced
accuracy of the frozen nonlinear Input-A-to-SSL probe was `0.901642`.

`Long_CS` is frozen as the primary within-class TEST case study using
`bout_duration_s`, `accel_rms`, and `accel_abs_std`. `LLC` is frozen as the
secondary case study using `turn_net_rad`; `BS` remains supporting evidence.

## Governance

DS-005 TEST remains unopened. This decision becomes the operative pre-TEST
freeze only when committed or otherwise immutably timestamped with its source
TRAIN / VALIDATION artifacts. Any later departure must be recorded as a protocol
deviation.

---

# DEC-029 — One-Time DS-006 Final TEST Evaluation

**Date:** 2026-08-25
**Status:** COMPLETE / FROZEN RESULT

## Decision and execution

Open DS-006 TEST exactly once using the inference-only procedure committed at:

```text
575ead5403d0b2f721d143366b4d2e0014bd67ee
```

The runner verified frozen source, checkpoint, clustering-object, probe, label
mapping, and TEST-array hashes before loading TEST. It recorded:

```yaml
test_bouts: 26130
test_recordings: 5
ssl_seeds: [11, 23, 37, 51, 79]
test_used_for_fitting: false
no_configuration_changed: true
prohibited_operations_performed: []
```

## Result

```text
SUPPORTED       11
WEAKENED         2
CONTRADICTED     0
NOT_TESTABLE     1
```

The acceleration/speed-change and weak-duration interpretations were weakened.
No frozen claim was contradicted. Direct Long_CS/LLC replication remains not
testable because DS-006 lacks equivalent labels.

Final artifacts:

```text
data/processed/DS-006/final_test_evaluation/
```

Final checksum-manifest SHA-256:

```text
e80acf4a774650b71776ed24368b20e82a60aad7793560773ec917541859d189
```

## Consequence

DS-006 TEST is no longer sealed: it was opened once and final evaluation is
complete. It must not be rerun or used for further model, threshold, feature,
cluster, or interpretation selection. DS-005 TEST remains unopened and
protected under `DEC-028`.
