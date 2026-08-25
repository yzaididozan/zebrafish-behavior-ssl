# DS-006 External Replication Protocol

## 1. Purpose

DS-006 is the external replication dataset for the Zebrafish SSL project.

Its role is to test whether the comparison between conventional hand-engineered behavioral features and self-supervised temporal representations generalizes beyond the primary DS-005 dataset.

DS-006 is **not** permitted to influence the frozen primary method.

```yaml
replication_dataset_role: EXTERNAL_REPLICATION
allowed_to_change_primary_method: false
allowed_for_primary_hyperparameter_selection: false
allowed_for_ssl_architecture_selection: false
allowed_for_cluster_k_selection: false
```

The external replication is intended to answer:

> Does the behavioral structure identified by the frozen primary analysis remain reproducible when the same comparison framework is transferred to an independent zebrafish dataset?

### Independence from DS-005

```yaml
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

DS-006 is an independently acquired experimental dataset from Reddy et al.
(2022), distinct from the Marques et al. recordings underlying DS-005. The two
datasets differ in assay design, acquisition rate, recording duration, stimulus
conditions, and tracking pipeline. They share some investigators and were later
analyzed together in Sridhar et al. (2024), but there is no indication that
DS-006 is a resplit or reuse of the DS-005 fish or recordings.

---

## 2. Dataset Identification

### Dataset

- Dataset ID: `DS-006`
- Dataset: Zebrafish larvae exploration and aversive chemotaxis dataset
- Repository: Dryad
- DOI: `10.5061/dryad.6t1g1jwwz`
- Role: External replication only

### Local archive

```text
data/raw/DS-006/Data_all.zip
```

### Archive integrity

```text
Size:   1,527,805,725 bytes
SHA256: d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f
```

### Extracted dataset

```text
data/raw/DS-006/extracted/Data_all
```

Observed scientific files:

- 32 MATLAB files
- 32 text files
- 64 scientific files total
- 32 apparent experiments / recordings

---

## 3. Experimental Unit and Identity

Each MATLAB results file represents one experiment.

Each experiment contains 12 independent wells.

The dataset documentation indicates one fish per well.

The `FishNumber` field was observed to contain only:

```text
0
```

Therefore, `FishNumber` is not used as a cross-recording biological identifier.

### Canonical replication fish-well ID

The provisional unit identifier is:

```text
DS006::<recording_id>::wellXX
```

where:

```text
recording_id = organization.videoName
```

Example:

```text
DS006::Catamaran_pH_2a_t3::well05
```

This identifier guarantees uniqueness within the replication dataset.

### Biological reuse caveat

Biological uniqueness of fish across separate recordings has not been independently verified.

Therefore, DS-006 replication analyses must not assume that fish-well units from different recordings represent known repeated or known unique biological individuals unless additional documentation establishes this.

---

## 4. Recording Structure

Observed experiment families:

| Family | Recordings | Raw bouts |
|---|---:|---:|
| pH_1a | 10 | 22,301 |
| pH_2a | 7 | 36,959 |
| pH_2b | 7 | 50,333 |
| pH_2c | 8 | 55,986 |
| **Total** | **32** | **165,579** |

### Archive discrepancy

The author documentation describes eight pH_2b experiments, while seven are observed in the archive.

The apparent missing recording is:

```text
Catamaran_pH_2b_t7
```

Status:

```yaml
pH_2b_t7_status: UNRESOLVED_ARCHIVE_DISCREPANCY
```

This discrepancy must remain documented but does not prevent use of the available recordings.

---

## 5. Frame Rate

### Frozen replication frame rate

```yaml
fps_hz: 160
```

The author analysis notebook explicitly sets:

```python
fps = 160.0
```

before loading all four dataset families.

The notebook then uses this value for:

- bout duration
- speed
- instantaneous speed
- oscillation frequency
- angular speed
- inter-bout interval
- trajectory timing

### Embedded MAT metadata discrepancy

The MATLAB files contain `organization.fps` values of both:

```text
25
160
```

The 25 Hz metadata values conflict with the author notebook analysis.

They are therefore preserved as a reproducibility note but are **not** treated as authoritative for replication preprocessing.

```yaml
authoritative_frame_rate_hz: 160
embedded_mat_fps_values:
  - 25
  - 160
embedded_mat_fps_status: INCONSISTENT_WITH_AUTHOR_ANALYSIS
```

---

## 6. Condition Annotation

Condition labels are reconstructed from the author analysis notebook.

The author well-type coding is:

| Code | Meaning |
|---:|---|
| -1 | acid on right |
| 0 | exploratory |
| 1 | acid on left |
| 2 | homogeneous pH |
| 3 | double sharp pH |
| 4 | bad data |
| 5 | paramecia |
| 6 | paramecia exploration |

The local mapping is stored at:

```text
data/manifests/DS-006/well-condition-map.csv
```

Observed well-slot counts:

| Condition | Wells |
|---|---:|
| exploratory | 192 |
| double_sharp_pH | 143 |
| homogeneous_pH | 42 |
| bad_data | 7 |
| **Total** | **384** |

---

## 7. Well-Level Quality Control

There are 384 total well slots.

### Author-labeled bad wells

Seven wells are marked as `bad_data`.

### Empty wells

Three additional wells contain no bouts:

```text
191119_Catamaran_pH_2b_t8  well04
Catamaran_pH_1a_t2c        well04
Catamaran_pH_2a_t3         well11
```

There is no overlap between the author-labeled bad wells and the empty wells.

### Frozen well-level exclusion

```yaml
total_well_slots: 384
author_bad_wells: 7
empty_wells: 3
combined_excluded_wells: 10
usable_wells: 374
```

The primary DS-006 replication dataset excludes:

1. author-labeled `bad_data` wells;
2. empty wells.

---

## 8. Author Bout-Level QC

The author notebook defines:

```python
if b.time < 0.04 or b.time > 1.2:
    reject

if b.dist > 25 or b.dist < 0.0:
    reject

if b.speed > 50 or b.speed < 1:
    reject

if abs(b.deltahead) > 180:
    reject
```

The author also defines:

```python
px_to_mm = 0.071
```

### Author trajectory interpolation

Before calculating distance and speed, the notebook applies:

```python
splprep(
    [
        b.posHeadX + 0.1 * np.random.randn(len(b.posHeadX)),
        b.posHeadY + 0.1 * np.random.randn(len(b.posHeadX))
    ],
    s=10
)
```

This means the original preprocessing contains a stochastic interpolation step.

No NumPy random seed was found in the notebook.

Therefore, exact bit-for-bit reproduction of the author's saved QC results is not guaranteed.

---

## 9. Deterministic Replication QC

To make the external replication reproducible while preserving the author method, DS-006 fixes:

```yaml
qc_random_seed: 20260822
fps_hz: 160
px_to_mm: 0.071
interpolation_noise_sd: 0.1
interpolation_smoothing_parameter: 10
```

The author interpolation and rejection logic are retained.

### Frozen QC result

```text
Raw bouts:                    165,579
Well-excluded bouts:            1,556
Bout-QC candidates:           164,023
Accepted bouts:               163,065
Rejected bouts:                   958
Acceptance rate:              99.4159%
```

### Rejection reasons

| Reason | Count |
|---|---:|
| speed too high | 532 |
| speed too low | 279 |
| distance too large | 206 |
| bout time too long | 140 |

A bout can fail more than one criterion, so rejection-reason counts are not required to sum to the number of rejected bouts.

### By-family accepted counts

| Family | Evaluated | Accepted | Rejected | Acceptance |
|---|---:|---:|---:|---:|
| pH_1a | 22,301 | 21,831 | 470 | 97.8925% |
| pH_2a | 36,289 | 36,200 | 89 | 99.7547% |
| pH_2b | 49,447 | 49,257 | 190 | 99.6158% |
| pH_2c | 55,986 | 55,777 | 209 | 99.6267% |

These counts are frozen for DS-006 preprocessing.

---

## 10. Unit of Analysis

The primary external-replication unit of analysis is:

> one accepted behavioral bout belonging to one usable fish-well unit.

Each bout retains:

- recording ID
- well ID
- canonical fish-well ID
- family
- condition label
- bout index
- bout timing metadata

Condition metadata must **not** be provided to the SSL encoder as an input feature.

---

# 11. Input A - Hand-Engineered Baseline

## Objective

Input A must reproduce the same **feature families and dimensionality** used for the frozen DS-005 baseline whenever the DS-006 data support a comparable calculation.

The target dimensionality is:

```text
18 features
```

The DS-006 feasibility audit confirmed that all 18 target features are computable.

### Feasibility result

All accepted bouts support every target feature at 100% computability except inter-bout interval.

```text
bout_duration             100.000%
inter_bout_interval        99.771%
speed_mean                100.000%
speed_std                 100.000%
speed_median              100.000%
speed_max                 100.000%
speed_p95                 100.000%
speed_rms                 100.000%
speed_change_abs_mean     100.000%
speed_change_std          100.000%
speed_change_max          100.000%
speed_change_rms          100.000%
turn_total_abs            100.000%
turn_net                  100.000%
turn_abs_mean             100.000%
turn_std                  100.000%
turn_max                  100.000%
turn_rms                  100.000%
```

The missing IBI values are expected for the first accepted bout in a usable well because no preceding accepted bout exists.

---

## 11.1 Timing Features

### 1. Bout duration

```text
bout_duration = number_of_temporal_samples / 160
```

Units:

```text
seconds
```

### 2. Inter-bout interval

For consecutive accepted bouts within the same recording and well:

```text
IBI = (BoutStart_current - BoutEnd_previous) / 160
```

Units:

```text
seconds
```

The first accepted bout in each usable fish-well unit has undefined IBI.

It must be represented consistently during preprocessing rather than silently filled using information from another fish or recording.

---

## 11.2 Speed Features

DS-006 does not expose the exact DS-005 `speed_head` variable.

A comparable head-speed time series is therefore derived from:

```text
HeadX
HeadY
```

using:

```text
step_distance_px =
sqrt(
    diff(HeadX)^2 +
    diff(HeadY)^2
)

speed_mm_s =
step_distance_px * 0.071 * 160
```

The temporal speed sequence is then used to compute:

1. mean speed
2. speed standard deviation
3. median speed
4. maximum speed
5. 95th percentile speed
6. RMS speed

### Important comparability note

This is a **mapped equivalent**, not an assertion that DS-006 speed is numerically identical to the DS-005 `speed_head` variable.

This distinction must be maintained in reporting.

---

## 11.3 Speed-Change Features

The temporal first difference of the speed sequence is used:

```text
delta_speed[t] = speed[t] - speed[t-1]
```

The four target features are:

1. mean absolute speed change
2. speed-change standard deviation
3. maximum absolute speed change
4. RMS speed change

The implementation used for replication must be fixed before final clustering and must not be tuned based on DS-006 downstream cluster quality.

---

## 11.4 Orientation / Turning Features

The feasibility audit indicates that `Heading` values are consistent with radians.

Frozen source variable:

```text
Heading
```

Heading must be converted to an unwrapped angular trajectory before temporal turn differences are calculated.

The six target turning features are:

1. total absolute turn
2. net turn
3. mean absolute turn
4. turn standard deviation
5. maximum absolute turn
6. RMS turn

### Comparability note

The DS-005 source variable is:

```text
orientation_smooth
```

The DS-006 source variable is:

```text
Heading
```

The replication therefore evaluates the same **orientation/turning feature family**, but not an identical upstream tracking signal.

---

# 12. Input B - Self-Supervised Temporal Representation

## Objective

DS-006 must support a temporal input structurally analogous to the frozen DS-005 SSL input.

DS-005 uses:

```text
sin(orientation_smooth)
cos(orientation_smooth)
speed_head
```

DS-006 maps these to:

```text
sin(Heading)
cos(Heading)
derived_head_speed_mm_s
```

Therefore, each DS-006 bout forms an array:

```text
(T, 3)
```

where `T` is the bout-specific temporal length.

Channels:

```text
0: sin(Heading)
1: cos(Heading)
2: derived head speed in mm/s
```

### Feasibility result

All frozen accepted bouts passed Input B shape and finite-value validation:

```text
163,065 valid / 163,065 accepted
```

Therefore:

```yaml
input_b_replication_feasibility: VERIFIED
```

---

## 13. Temporal Handling

DS-006 contains variable-length bouts.

The replication preprocessing must preserve the raw accepted bout sequences until the frozen model-input transformation is applied.

No behavior-adaptive segmentation may be introduced based on DS-006 results.

Any padding, cropping, resampling, or fixed-length transformation required by the frozen DS-005 encoder must be specified explicitly and applied deterministically.

### Rule

DS-006 temporal handling may adapt only what is technically required to apply the already-frozen primary representation method.

It may not be optimized according to:

- DS-006 validation loss
- DS-006 cluster silhouette
- DS-006 stability
- DS-006 condition separation
- visual interpretability of DS-006 embeddings

---

# 14. Normalization Policy

DS-006 is an external replication dataset.

Therefore, normalization must not modify or retroactively tune the primary DS-005 method.

The replication pipeline must explicitly distinguish between:

1. applying frozen DS-005 normalization parameters when variables are directly compatible;
2. replication-local normalization when source-variable semantics or physical units differ enough that direct transfer is invalid.

Any replication-local normalization must:

- be deterministic;
- use only the designated DS-006 replication training partition;
- never use the DS-006 replication test partition;
- not feed information back into DS-005 training or method selection.

The final normalization policy must be recorded in the preprocessing manifest before DS-006 model evaluation.

---

# 15. Replication Split Policy

DS-006 must use group-aware splitting.

### Required grouping unit

At minimum:

```text
canonical fish-well ID
```

The same fish-well unit must never appear in more than one partition.

Because biological identity across recordings is not independently established, recording-level grouping should also be evaluated as a conservative leakage control.

### Required partitions

```text
train
validation
test
```

### Test protection

The DS-006 test partition must remain untouched during:

- normalization fitting
- preprocessing choices
- model adaptation decisions
- cluster-number selection
- representation diagnostics used for decision-making
- threshold selection

The external test set is opened only after the replication configuration is frozen.

---

# 16. SSL Model Policy

The primary SSL architecture and training method are already frozen by DS-005.

DS-006 may not be used to:

- redesign the encoder;
- alter convolution widths;
- alter embedding dimensionality;
- alter projection-head architecture;
- alter augmentation strategy;
- alter temperature;
- alter optimizer family;
- alter primary learning rate;
- alter primary weight decay;
- select a new primary random-seed policy.

Where technical adaptation is unavoidable because DS-006 sequence geometry differs from DS-005, the adaptation must be minimal, documented, and cannot be selected according to DS-006 test performance.

---

# 17. Clustering Policy

DS-006 is not allowed to determine the primary number of clusters.

The frozen discovery strategy from DS-005 must be transferred as directly as technically possible.

DS-006 cannot be used for:

```yaml
primary_cluster_k_selection: false
primary_method_selection: false
primary_representation_selection: false
```

If DS-006 requires its own replication-side clustering selection for a secondary sensitivity analysis, that analysis must be clearly labeled as secondary and must use only DS-006 training/validation data.

The DS-006 test partition must not influence cluster selection.

---

# 18. Comparison Policy

The replication must preserve the central comparison:

```text
Input A: hand-engineered locomotion / pose representation

versus

Input B: self-supervised temporal representation
```

The comparison should examine whether the learned representation yields reproducible behavioral structure that is not fully captured by the evaluated hand-engineered baseline.

The analysis must not redefine success as simply obtaining a higher silhouette score.

Relevant comparison dimensions include:

- cluster reproducibility
- representation stability
- held-out fish-well generalization
- sensitivity to recording/context
- dependence on locomotor speed
- correspondence between baseline and SSL cluster structure
- behavioral structure uniquely captured by SSL
- failure to replicate primary structure

Negative or equivalent replication outcomes remain scientifically valid.

---

# 19. Replication Validation

The external replication should explicitly evaluate the major validity threats identified for the primary study.

## Identity leakage

Test whether representation or clustering structure predicts fish-well identity disproportionately.

## Recording / session leakage

Test whether embeddings or clusters are strongly determined by:

```text
recording_id
experiment family
well
condition
```

Condition labels are for validation and interpretation, not encoder input.

## Speed-only solution

Quantify the relationship between learned embeddings and locomotor speed.

The replication should test whether behavioral structure remains after accounting for speed.

## Tracking artifacts

Check whether clusters are enriched for:

- unusual bout duration
- extreme movement amplitude
- interpolation artifacts
- malformed tail coordinates
- boundary cases near QC thresholds

## Temporal-boundary dependence

If any fixed-length transformation is required, assess whether discovered structure is dominated by padding, truncation, or arbitrary bout boundaries.

## Representation stability

Assess stability across the already-defined random-seed policy wherever technically applicable.

---

# 20. Condition Labels and Interpretation

Condition labels must not be supplied as representation inputs.

They may be used after representation learning and clustering for:

- external interpretation;
- enrichment analysis;
- behavioral validation;
- checking whether obvious experimental context dominates representation structure.

A cluster that strongly separates experimental condition is not automatically evidence of a meaningful behavioral state.

Condition dependence must be distinguished from behaviorally reproducible structure.

---

# 21. Data Leakage Prohibitions

DS-006 replication must not use test-partition information for any of the following:

- normalization
- feature selection
- imputation policy selection
- embedding dimensionality selection
- hyperparameter selection
- augmentation selection
- early stopping
- cluster-number selection
- clustering-method selection
- threshold optimization
- manual representation redesign

The test set is reserved for final replication evaluation.

---

# 22. Frozen Feasibility Findings

The feasibility audit:

```text
src/data/ds006_feasibility.py
```

produced:

```text
MAT files:             32
Raw bouts:             165,579
Excluded-well bouts:   1,556
QC candidates:         164,023
Accepted:              163,065
Rejected:              958
Heading unit:          radians_likely

Frozen QC counts match: True

Input A: FEASIBLE_CANDIDATE
Input B: FEASIBLE_CANDIDATE
```

Input B validity:

```text
163,065 valid / 163,065 accepted
```

Therefore:

```yaml
ds006_ingestion: VERIFIED
ds006_qc: FROZEN
input_a_feasibility: VERIFIED
input_b_feasibility: VERIFIED
replication_preprocessing_ready: true
```

---

# 23. Remaining Preprocessing Decisions

Before final DS-006 arrays are generated, the following must be frozen:

1. exact Input A formulas and naming;
2. treatment of undefined first-bout IBI;
3. precise speed-sequence temporal alignment;
4. exact orientation unwrapping implementation;
5. DS-006 normalization strategy;
6. temporal padding / truncation / resampling required for SSL;
7. group-aware train / validation / test split;
8. random seed for the replication split;
9. artifact output schema;
10. metadata retained alongside model inputs.

None of these decisions may be selected based on DS-006 test performance.

---

# 24. Expected Processed Artifacts

The preprocessing pipeline should eventually write a structure similar to:

```text
data/processed/DS-006/
├── baseline/
│   ├── train_core_raw.npz
│   ├── validation_core_raw.npz
│   ├── test_core_raw.npz
│   └── feature_manifest.json
│
├── ssl/
│   ├── train.npz
│   ├── validation.npz
│   ├── test.npz
│   └── normalization.json
│
├── metadata/
│   ├── split_assignments.csv
│   ├── bout_metadata.csv
│   └── qc_summary.json
│
└── feasibility/
    └── ds006_feasibility.json
```

Exact filenames may be adapted to match the repository's existing DS-005 conventions.

---

# 25. Replication Claim Policy

DS-006 supports an **external replication test**, not re-optimization of the primary study.

The replication can strengthen the primary claim if:

1. the frozen comparison transfers successfully;
2. learned representations produce reproducible behavioral organization;
3. the organization is not explained solely by speed, identity, recording, or tracking artifacts;
4. the learned representation captures structure not fully reproduced by the matched hand-engineered baseline.

A failure to reproduce the primary effect must be reported as a valid outcome.

The replication result must not be hidden, discarded, or used as justification for post hoc redesign of the primary method.

---

# 26. Current Status

```yaml
dataset_id: DS-006

archive_verified: true
schema_verified: true
condition_mapping_verified: true

authoritative_fps_hz: 160

well_level_qc:
  status: FROZEN
  total_wells: 384
  excluded_wells: 10
  usable_wells: 374

bout_level_qc:
  status: FROZEN
  raw_bouts: 165579
  evaluated_after_well_qc: 164023
  accepted: 163065
  rejected: 958

input_a:
  feasibility: VERIFIED
  target_features: 18

input_b:
  feasibility: VERIFIED
  channels: 3
  valid_accepted_bouts: 163065

replication_constraints:
  may_change_primary_method: false
  may_select_primary_hyperparameters: false
  may_select_primary_ssl_architecture: false
  may_select_primary_cluster_k: false

next_phase: DS006_PREPROCESSING
```

---
