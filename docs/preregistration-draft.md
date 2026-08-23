# Preregistration Draft

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

**Repository:** `zebrafish-behavior-ssl`  
**Primary dataset:** `DS-005`  
**External replication dataset:** `DS-006`  
**Draft status:** `METHODS FROZEN — READY FOR FORMAL PREREGISTRATION REVIEW`  
**Last updated:** `2026-08-23`

---

# 1. Study Objective

The primary objective is to test whether self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not fully captured by conventional hand-engineered locomotion and pose features.

The confirmatory comparison is:

```text
Input A
Hand-engineered locomotion / pose features

versus

Input B
Self-supervised temporal representations
```

The study does not require SSL to outperform Input A.

Negative, equivalent, nuisance-driven, unstable, or non-replicating outcomes remain scientifically valid.

---

# 2. Primary Research Question

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

---

# 3. Target Claim

## Claim Level 2

> **Self-supervised temporal representations reveal reproducible behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose features.**

A positive Claim Level 2 result requires converging evidence that SSL-derived structure:

- is reproducible;
- generalizes to held-out fish;
- is stable across the frozen seed set;
- is not primarily explained by fish identity;
- is not primarily explained by context/session;
- is not explained solely by locomotor speed;
- is not primarily a tracking artifact;
- is not fully reconstructed by Input A;
- survives the frozen sensitivity analyses;
- and receives meaningful external replication support.

---

# 4. Primary Dataset

## DS-005

**Status:** `FROZEN`

```yaml
number_of_fish: 463
valid_bouts: 1203409
contexts: 14
frame_rate_hz: 700
samples_per_bout: 175
```

Authorization:

```yaml
license: CC BY 4.0
authorization_verified: true
```

The selected archive was downloaded, integrity checked, inventoried, structurally inspected, and frozen before final evaluation.

---

# 5. Unit of Analysis

The primary unit is:

> **One valid behavioral bout belonging to one identifiable zebrafish.**

Each valid DS-005 bout contains 175 temporal samples.

The earlier provisional fixed-window plan is superseded.

---

# 6. Fish-Level Split

```yaml
split_seed: 20260822
train_fish: 323
validation_fish: 70
test_fish: 70
fish_overlap: 0

train_bouts: 842841
validation_bouts: 168464
```

All bouts inherit the partition of their source fish.

The TEST partition remains protected until final confirmatory evaluation.

---

# 7. Input A — Hand-Engineered Baseline

Input A is frozen at 18 features.

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

Raw `head_pos` path/jump features are excluded from the primary baseline.

They may appear only in a secondary sensitivity analysis.

---

# 8. Input A Preprocessing

Input A normalization is fit on TRAIN only.

The frozen training-derived transformation is applied unchanged to VALIDATION and TEST.

---

# 9. Input B — SSL Temporal Tensor

Input B is frozen as:

```text
shape: (175, 3)

channel 0 = sin(orientation_smooth)
channel 1 = cos(orientation_smooth)
channel 2 = speed_head
```

The SSL encoder receives no:

- fish ID;
- context label;
- stimulus code;
- bout type;
- session label;
- partition label;
- cluster label.

---

# 10. SSL Normalization

Only the speed channel is standardized.

TRAIN-only statistics:

```yaml
train_bouts_used: 842841
temporal_speed_samples_used: 147497175
speed_mean: 0.858429032920
speed_std: 1.260544584910
validation_used_for_fit: false
test_used_for_fit: false
```

---

# 11. SSL Augmentation

Frozen augmentation:

```yaml
temporal_mask:
  enabled: true
  probability: 0.75
  max_fraction: 0.10
  contiguous: true
  mask_value: 0.0

feature_mask:
  enabled: false
```

Disabled:

- Gaussian noise;
- temporal cropping;
- temporal warping;
- rotation;
- translation;
- whole-feature masking.

Prohibited:

- time reversal;
- temporal shuffling;
- cross-bout mixing;
- cross-fish mixing;
- cross-partition mixing.

---

# 12. SSL Objective

```yaml
status: FROZEN
family: temporal_contrastive_learning
loss: NT-Xent
temperature: 0.10
```

---

# 13. SSL Encoder

```yaml
status: FROZEN
type: 1D_CNN
embedding_dimension: 64
```

Architecture:

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

Downstream discovery uses the encoder embedding.

---

# 14. SSL Optimization

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

---

# 15. SSL Seed Set

```yaml
seeds:
  - 11
  - 23
  - 37
  - 51
  - 79
```

No seed may be removed because its result is unfavorable unless a technical failure invalidates the run.

---

# 16. Pipeline Verification

TRAIN-only smoke test:

```yaml
bouts: 2048
seed: 11
epochs: 3
initial_loss: 4.739312
final_loss: 3.907831
finite_gradients: true
checkpoint_reload_passed: true
validation_used: false
test_used: false
```

TRAIN / VALIDATION preflight:

```yaml
train_bouts: 10000
validation_bouts: 2000
epochs: 2
seed: 11

train_loss:
  epoch_1: 4.275675
  epoch_2: 3.821423

validation_loss:
  epoch_1: 3.762066
  epoch_2: 3.634544

checkpoint_reload_passed: true
test_used: false
```

Both are pipeline-verification evidence only.

---

# 17. Full SSL Training

Full frozen multi-seed TRAIN / VALIDATION training is in progress.

Running results must not alter:

- Input B;
- augmentation;
- objective;
- architecture;
- embedding dimension;
- projection head;
- optimizer;
- learning rate;
- seed set;
- stopping rule.

---

# 18. Baseline Discovery

Frozen primary baseline discovery configuration:

```text
PCA(6 components) -> GMM(k=2, seed=20260822)
```

Selection used TRAIN / VALIDATION only.

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

The selected `k=2` is not interpreted as a biological claim about the true number of zebrafish states.

---

# 19. Dimensionality Reduction Governance

Input A uses frozen TRAIN-only PCA.

Input B uses the 64-dimensional encoder embedding as the primary representation.

PCA, UMAP, or t-SNE may be used descriptively, but visualization embeddings may not determine:

- confirmatory clusters;
- cluster count;
- representation selection;
- success thresholds.

---

# 20. Cross-Fish Reproducibility

Frozen fish-bootstrap procedure:

```yaml
bootstrap_unit: fish
bootstrap_replicates: 500
bootstrap_seed: 20260822
confidence_interval: percentile_95
```

Primary metrics:

- fish-bootstrap ARI;
- held-out cluster occupancy;
- cross-seed pairwise ARI.

For the five frozen SSL seeds, all 10 pairwise seed comparisons are reported.

Strong-evidence interpretation:

```yaml
median_fish_bootstrap_ARI: ">= 0.75"
lower_95pct_bootstrap_bound: ">= 0.50"
median_cross_seed_ARI: ">= 0.75"
```

Interpretive bands:

```yaml
ARI_ge_0.75: strong
ARI_0.50_to_0.75: moderate
ARI_0.25_to_0.50: weak
ARI_lt_0.25: poor
```

---

# 21. Baseline-vs-SSL Structural Comparison

Primary metrics:

```text
Adjusted Rand Index
Normalized Mutual Information
```

A fixed multinomial logistic regression predicts SSL cluster membership from Input A.

Primary metric:

```text
balanced accuracy
```

Secondary:

```text
macro F1
```

Near-perfect reconstruction of SSL cluster membership from Input A weakens Claim Level 2.

---

# 22. Speed Control

Frozen controls:

1. speed-only clustering using mean bout speed;
2. SSL embedding -> mean speed ridge regression;
3. cluster-level speed enrichment.

Ridge model:

```yaml
model: Ridge
alpha: 1.0
```

Primary regression metric:

```text
R^2
```

Secondary:

```text
MAE
```

Interpretive bands:

```yaml
R2_lt_0.25: low
R2_0.25_to_0.50: moderate
R2_0.50_to_0.75: high
R2_ge_0.75: very_high
```

---

# 23. Fish-Identity Leakage

Frozen classifier:

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

Input A and Input B leakage must be compared.

---

# 24. Context / Session Leakage

Use the same fixed multinomial logistic-regression model.

Primary target:

```text
DS-005 context label
```

Metrics:

```text
balanced accuracy
macro F1
```

If session/recording identity cannot be separately estimated from fish identity, this must be documented explicitly.

---

# 25. Tracking-Quality Rule

No new post-clustering DS-005 exclusion rules may be introduced.

```yaml
new_post_clustering_exclusion_rules: prohibited
```

Existing QC/proxy variables are compared across discovered clusters.

Potentially artifact-driven clusters are reported rather than silently removed.

---

# 26. Primary Confirmatory Metric Set

Frozen metric family:

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
external replication stability
external replication nuisance consistency
```

Because the bout count is very large, interpretation emphasizes effect size, stability, uncertainty, held-out generalization, and nuisance controls over isolated p-values.

---

# 27. Seed Aggregation

For seed-dependent metrics report:

```text
all individual seed values
median
IQR
minimum
maximum
```

The primary across-seed summary is the median.

---

# 28. Sensitivity Analyses

```yaml
ssl_seed_sensitivity: CONFIRMATORY
head_position_extended_baseline: SECONDARY
cluster_number_sensitivity: SECONDARY
visualization_dimensionality_reduction: EXPLORATORY
alternate_segmentation: NOT_PRIMARY
```

No alternate primary segmentation is required because DS-005 already provides a natural valid-bout unit.

---

# 29. Claim Decision Gates

A Claim Level 2 conclusion requires converging evidence across:

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

---

# 30. External Replication Dataset

## DS-006

Restrictions:

```yaml
role: EXTERNAL_REPLICATION
allowed_to_change_primary_method: false
allowed_for_primary_hyperparameter_selection: false
allowed_for_ssl_architecture_selection: false
allowed_for_cluster_k_selection: false
```

---

# 31. DS-006 Frozen Ingestion State

```yaml
recordings: 32
fish_well_slots: 384
usable_fish_well_units: 374
accepted_bouts: 163065
authoritative_frame_rate_hz: 160
```

Canonical replication unit:

```text
DS006::<recording_id>::wellXX
```

Biological uniqueness across recordings is not independently verified.

---

# 32. DS-006 Split

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

The DS-006 TEST partition remains sealed.

---

# 33. DS-006 Input Mapping

Input A:

- same 18 target feature families as DS-005;
- 17/18 computable for 100% of accepted bouts;
- IBI computable for 99.771%.

Input B:

```text
sin(Heading)
cos(Heading)
derived_head_speed_mm_s
```

All 163,065 accepted bouts form valid Input B tensors.

The mapping is analogous rather than numerically identical to DS-005.

---

# 34. DS-006 Preprocessing

DS-006 variable-length bouts are resampled to:

```text
(175, 3)
```

TRAIN-only speed normalization:

```yaml
speed_mean: 9.773209740465
speed_std: 10.478936765051
```

Processed artifacts are fingerprinted in:

```text
data/manifests/DS-006/processed-sha256.txt
```

---

# 35. External Replication Governance

DS-006 cannot alter the primary DS-005 method.

The replication question is:

> **Does the qualitative baseline-vs-SSL conclusion obtain independent support in DS-006?**

Exact cluster identity or occupancy need not match across datasets.

---

# 36. Pilot / Verification Disclosures

Disclosed non-confirmatory analyses:

- TRAIN-only SSL smoke test;
- capped TRAIN / VALIDATION SSL preflight;
- DS-006 feasibility audit;
- DS-006 ingestion and preprocessing verification.

These analyses are not treated as final confirmatory evidence.

---

# 37. Prohibited Post-Hoc Changes

After TEST inspection, do not:

- alter cluster count;
- change seed set;
- remove unfavorable seeds;
- invent new nuisance metrics;
- alter speed definition;
- introduce new QC exclusions;
- switch encoder layer;
- change dimensionality reduction;
- redefine claim thresholds;
- redefine success based on whichever metric looks strongest.

Any such analysis must be labeled post hoc.

---

# 38. Reproducibility

Material decisions are recorded in:

```text
docs/decision-log.md
```

Dataset provenance and authorization are recorded in:

```text
docs/dataset-register.md
```

Evaluation rules are recorded in:

```text
docs/evaluation-protocol.md
```

---

# 39. Current Freeze State

```yaml
charter: STABLE

primary_dataset: FROZEN
dataset_version: FROZEN
fish_split: FROZEN
unit_of_analysis: FROZEN

input_a: FROZEN
input_a_normalization: FROZEN

input_b_tensor: FROZEN
ssl_normalization: FROZEN
ssl_augmentation: FROZEN
ssl_objective: FROZEN
ssl_encoder: FROZEN
ssl_embedding_dimension: FROZEN
ssl_seed_set: FROZEN
ssl_training_configuration: FROZEN

primary_baseline_clustering: FROZEN
primary_evaluation_protocol: FROZEN
validity_controls: FROZEN
sensitivity_plan: FROZEN
claim_level: FROZEN

ssl_full_training: IN_PROGRESS

ds005_test_partition: PROTECTED
ds006_test_partition: SEALED
```

---

# 40. Updated Method Checklist

## Dataset

- [x] Select primary dataset — **DS-005**.
- [x] Verify license.
- [x] Record number of fish — **463**.
- [x] Freeze dataset version.
- [x] Freeze fish-level train/validation/test split.

## Unit of Analysis

- [x] Freeze primary unit — **valid bout**.
- [x] Define confirmatory segmentation sensitivity policy.

## Baseline

- [x] Freeze primary `core` hand-engineered baseline.
- [x] Define training-derived normalization policy.
- [x] Confirm exact baseline config/artifact identifiers.

## SSL

- [x] Freeze exact Input B tensor/modality.
- [x] Select SSL objective.
- [x] Select encoder architecture.
- [x] Freeze embedding dimension.
- [x] Freeze seed set.

## Discovery

- [x] Select primary clustering algorithm.
- [x] Define state-number selection method.
- [x] Define confirmatory dimensionality-reduction policy.

## Evaluation

- [x] Freeze cross-fish reproducibility metric/procedure.
- [x] Freeze speed-control metric/procedure.
- [x] Freeze exact tracking-quality rule.
- [x] Freeze nuisance prediction model specifications.
- [x] Freeze the complete primary confirmatory metric set and decision rules.

## Claim Governance

- [x] Freeze target claim level — **Claim Level 2**.
- [x] Define major nuisance / validity threats.

---

# 41. Registration Readiness Checklist

- [x] `docs/charter.md` is stable.
- [x] `docs/dataset-register.md` contains an approved primary dataset.
- [x] `docs/research-question.md` is frozen.
- [x] Primary dataset is downloaded, verified, and structurally inspected.
- [x] No final TEST analyses have been performed.
- [x] Unit of analysis is frozen.
- [x] Baseline features are frozen.
- [x] Primary SSL approach is frozen.
- [x] Primary clustering method is frozen.
- [x] Train/validation/test rules are frozen.
- [x] Primary evaluation metrics are frozen.
- [x] Validity controls are frozen.
- [x] Claim thresholds are frozen.
- [x] Planned sensitivity analyses are documented.
- [x] Known pilot / pipeline-verification analyses are disclosed.

---

# 42. Preregistration Status

The methodological preregistration package is now internally ready for formal review/freeze.

The ongoing SSL training does not reopen the frozen methods.

Before any DS-005 TEST evaluation:

1. preserve the current charter, decision log, dataset register, evaluation protocol, and preregistration draft;
2. complete the already-running SSL seed runs without methodological modification;
3. record final TRAIN / VALIDATION training artifacts and hashes;
4. verify that no protected TEST results have been inspected;
5. formally timestamp or register the preregistration package.

After that point, TEST evaluation may proceed exactly under the frozen protocol.
