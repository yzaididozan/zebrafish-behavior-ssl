# Reproducibility Record

## Project

**Project:** Self-Supervised Discovery of Zebrafish Behavioral Structure  
**Repository:** `zebrafish-behavior-ssl`  
**Primary dataset:** DS-005  
**Primary analysis unit:** Valid behavioral bout  
**Split unit:** Fish  
**Primary comparison:** Hand-engineered behavioral features vs. self-supervised temporal representations  

This document records frozen data, analysis artifacts, hashes, and reproducibility decisions for the project.

The purpose of this file is to make it possible to determine exactly which dataset version, split, feature representation, scaler, and generated artifacts were used in the confirmatory analysis.

---

# 1. Reproducibility Principles

The following rules apply to the confirmatory analysis.

1. Fish, not bouts, are the train/validation/test split unit.
2. All bouts belonging to one fish remain in exactly one partition.
3. The frozen DS-005 split must not be regenerated because of downstream model or clustering performance.
4. `MetaData/lengths_data` is the authoritative definition of valid bouts.
5. Padding values must not be used to infer bout validity.
6. Normalization parameters for Input A must be estimated using training fish only.
7. Validation and test data must be transformed using the frozen training-set normalization parameters.
8. Test data must not be used for feature selection, model selection, cluster-number selection, hyperparameter tuning, or representation tuning.
9. QC thresholds must not be changed based on downstream clustering or SSL performance.
10. Changes to a frozen artifact require a new version and a documented protocol amendment.

---

# 2. DS-005 Primary Dataset Freeze

## 2.1 Dataset

**Dataset ID:** `DS-005`

**Dataset title:**  
*Dataset for Uncovering multiscale structure in the variability of larval zebrafish navigation V2*

**Repository:** Zenodo

**DOI:** `10.5281/zenodo.13605471`

**Associated paper:**  
*Uncovering multiscale structure in the variability of larval zebrafish navigation*

**License:** CC BY 4.0

**Primary cohort:** `JM_data`

**Developmental stage:** Larval zebrafish, approximately 6–7 dpf

**Sampling rate:** 700 Hz

**Number of fish:** 463

**Number of experimental contexts:** 14

**Total valid bouts:** 1,203,409

---

# 3. Raw Dataset Integrity

## 3.1 Archive

Primary archive:

```text
data/raw/DS-005/Datasets.tar.gz
```

Published MD5:

```text
b9a00fccda494bb49ea7c67d3b0f8d9e
```

Local archive SHA-256:

```text
5af065e44f3f00f1a975dfe6472eee985c9e73c2f4a98ba722b5233efb85cffd
```

## 3.2 Archive Inventory

Verified archive entries:

```text
2223
```

Verified extracted files:

```text
2218
```

Verified extracted directories excluding extraction root:

```text
5
```

Consistency check:

```text
2218 files + 5 directories = 2223 archive entries
```

Result:

```text
PASS
```

## 3.3 Raw Directory Lock

The DS-005 raw directory was made read-only after verification:

```bash
chmod -R a-w data/raw/DS-005
```

Frozen condition:

```text
raw_directory_read_only: true
```

The raw dataset must not be modified during confirmatory analysis.

---

# 4. Primary HDF5 Structure

Primary HDF5 file:

```text
data/raw/DS-005/DS-005-v1/Datasets/JM_data/filtered_jmpool_kin.h5
```

Verified datasets:

| Dataset | Shape | Type |
|---|---:|---|
| `bout_types` | `(463, 11651)` | `float32` |
| `converge_bouts` | `(463, 11651, 175)` | `float32` |
| `eye_convergence` | `(463, 11651)` | `float32` |
| `eye_convergence_state` | `(463, 11651)` | `float32` |
| `head_pos` | `(463, 11651, 175, 2)` | `float32` |
| `orientation_smooth` | `(463, 11651, 175)` | `float32` |
| `speed_head` | `(463, 11651, 175)` | `float32` |
| `stims` | `(463, 11651)` | `float32` |
| `times_bouts` | `(463, 11651, 2)` | `float32` |

Metadata:

| Dataset | Shape |
|---|---:|
| `MetaData/errmask` | `(463, 15116)` |
| `MetaData/frameRate` | `(1,)` |
| `MetaData/lengths_data` | `(463,)` |
| `MetaData/t0_bout` | `(1,)` |

Verified values:

```text
frameRate = 700 Hz
t0_bout = 0
fish = 463
total valid bouts = 1,203,409
```

`MetaData/lengths_data` is the authoritative valid-bout count for each fish.

---

# 5. Canonical Fish and Session Identity

Canonical fish IDs:

```text
DS005-JM-F000
...
DS005-JM-F462
```

Canonical session IDs:

```text
DS005-JM-S000
...
DS005-JM-S462
```

Each source fish index maps one-to-one to one canonical fish ID.

For DS-005, the project defines one canonical session per fish because independent author-provided session identifiers are not available in the HDF5 structure.

Therefore:

```text
fish identity == canonical session unit
```

This controls fish/session leakage jointly, but separate session leakage cannot be independently estimated for DS-005.

Fish metadata:

```text
data/metadata/DS-005/DS-005-fish-map.csv
```

---

# 6. Experimental Context Mapping

The 14 experimental contexts were reproduced from the authors' `Markov_Fish` analysis code.

Verified context count:

```text
14
```

All 463 fish have exactly one project-level experimental context assignment.

The context assignments are metadata only and must not be included as Input A baseline features or Input B SSL inputs unless explicitly required by a later supervised evaluation.

The `stims` field contains 73 distinct valid values and is not treated as the authoritative 14-context variable.

Complete `stims` semantics remain unresolved but are non-blocking because the experimental-context mapping was independently reproduced from the authors' analysis code.

---

# 7. Frozen Fish-Level Split

Split file:

```text
data/splits/DS-005-fish-split-v1.csv
```

Split seed:

```text
20260822
```

Split unit:

```text
fish
```

Partition counts:

| Partition | Fish |
|---|---:|
| Train | 323 |
| Validation | 70 |
| Test | 70 |
| **Total** | **463** |

Every experimental context appears in all three partitions.

No fish appears in more than one partition.

All bouts inherit the partition of their fish.

Bout-level random splitting is prohibited.

## 7.1 Frozen Split SHA-256

```text
19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935
```

The split must not be regenerated based on downstream results.

---

# 8. Frozen QC Rules

## 8.1 Structural QC

Across all 1,203,409 valid bouts:

```text
head_pos nonfinite: 0
head_pos all-zero: 0

orientation_smooth nonfinite: 0
orientation_smooth all-zero: 0

speed_head nonfinite: 0
speed_head all-zero: 34

times_bouts nonfinite: 0
times_bouts all-zero: 0
```

## 8.2 Primary Analysis Rules

Primary automatic exclusion:

```text
non-finite valid source values
```

All-zero-speed-only bouts:

```text
retain in primary analysis
exclude in sensitivity analysis
```

Extreme speed bouts with maximum speed greater than 100:

```text
retain in primary analysis
flag as likely tracking discontinuity
exclude in sensitivity analysis
```

Observed number of bouts with:

```text
max speed > 100 = 6
```

Raw head-position jumps are not used as an automatic exclusion criterion because the coordinate semantics and apparent wrapping/discontinuities are not sufficiently resolved.

Wrapped orientation-step magnitude is also not used as an automatic exclusion criterion.

Frozen condition:

```text
project_qc_thresholds_frozen: true
```

QC thresholds must not be changed based on clustering, SSL, or test-set results.

---

# 9. DS-005 Metadata Freeze

Manifest:

```text
data/metadata/DS-005/DS-005-manifest.yaml
```

SHA-256:

```text
62af545817ed48cc6918b4c5ad1448d878694f1816fe552099281e6469b5411e
```

Freeze record:

```text
data/metadata/DS-005/DS-005-freeze-record.yaml
```

SHA-256:

```text
ecf9a0bf45b34d1ec8c8b378c57860e0de20acc2e0aee5749a7bf6d442fd579c
```

Frozen state:

```yaml
status: FROZEN_PRIMARY
confirmatory_analysis_frozen: true
```

Freeze record state:

```yaml
freeze_status: FROZEN
confirmatory_analysis_frozen: true
pending_items: {}
```

---

# 10. Canonical DS-005 Loader

Implementation:

```text
src/data/ds005.py
```

The canonical loader:

- opens the primary HDF5 in read-only mode;
- uses `MetaData/lengths_data` to determine valid bouts;
- preserves canonical fish identity;
- preserves canonical session identity;
- attaches the 14-context mapping;
- attaches the frozen train/validation/test partition;
- prevents access to padded bout slots through the valid-bout interface;
- exposes QC flags without silently changing the primary bout universe;
- avoids loading the entire dataset into memory.

## 10.1 Loader Test Suite

Tests:

```text
tests/test_ds005.py
```

Verified result:

```text
16 passed
```

The loader tests validate:

- 463 fish;
- 1,203,409 valid bouts;
- 14 contexts;
- 323/70/70 fish partition counts;
- no fish overlap;
- canonical fish IDs;
- canonical session IDs;
- frozen split SHA-256;
- valid HDF5 shapes;
- frame rate;
- padded-bout protection;
- context representation across partitions.

---

# 11. Input A — Hand-Engineered Baseline

Implementation:

```text
src/features/baseline.py
```

Input A uses one engineered feature vector per valid behavioral bout.

The confirmatory feature profile is:

```text
core
```

## 11.1 Core Feature Schema

The frozen core baseline contains 18 features:

```text
bout_duration_s
inter_bout_interval_s

speed_mean
speed_std
speed_median
speed_max
speed_p95
speed_rms

accel_abs_mean
accel_abs_std
accel_abs_max
accel_rms

turn_abs_total_rad
turn_net_rad
turn_abs_mean_rad
turn_abs_std_rad
turn_abs_max_rad
turn_rms_rad
```

Context, stimulus code, and author-provided bout type are metadata only.

They are not part of the numeric feature vector.

## 11.2 Head-Position Features

The confirmatory `core` profile excludes raw `head_pos`-derived displacement and path features because coordinate discontinuities were observed during QC and the coordinate semantics remain insufficiently resolved.

An optional exploratory `extended` profile exists, but it is not the frozen confirmatory Input A representation.

## 11.3 Inter-Bout Interval Convention

For the first valid bout of each fish:

```text
inter_bout_interval_s = 0
```

because no preceding valid bout exists.

This representation rule must remain unchanged for the current artifact version.

---

# 12. Baseline Feature Tests

Tests:

```text
tests/test_baseline.py
```

Verified result:

```text
32 passed
```

The test suite verifies:

- the exact 18-feature core schema;
- one row per valid bout;
- finite engineered features;
- canonical bout indexing;
- timing calculations;
- speed summary relationships;
- wrapped-angle behavior;
- metadata preservation;
- QC sensitivity flags;
- optional extended-profile schema;
- context exclusion from numeric features;
- deterministic feature ordering;
- train-only scaler fitting;
- validation transformation using the training scaler;
- partition filtering;
- no fish-level leakage.

---

# 13. Input A Baseline Artifact Build

Build script:

```text
scripts/build_baseline_features.py
```

Output directory:

```text
data/processed/DS-005/baseline/
```

Generated artifacts:

```text
train_core_raw.npz
validation_core_raw.npz
test_core_raw.npz

train_core_scaled.npz
validation_core_scaled.npz
test_core_scaled.npz

scaler_core.json
feature_schema_core.json
build_audit_core.json
SHA256SUMS
```

Normalization protocol:

```text
fit scaler on training partition only
transform training with training scaler
transform validation with training scaler
transform test with training scaler
```

No validation or test observations are used to estimate normalization statistics.

The build verifies that all 1,203,409 valid DS-005 bouts are represented exactly once across the three frozen partitions.

---

# 14. DS-005 Input A Baseline Artifact Freeze

The following files constitute the current frozen Input A baseline artifact set.

## 14.1 SHA-256 Hashes

| Artifact | SHA-256 |
|---|---|
| `build_audit_core.json` | `b6c7b76797e582ac69d123f443d5486167371b2bf9bd3dea495cfa4512c8711e` |
| `feature_schema_core.json` | `a8b2fe73f3251f7788e99e6fb1fde2688256afbe34a39e40daa8431018a5e91a` |
| `scaler_core.json` | `85724491d36b4c7574663dfb7a74b263d3cfc2cf5151d488dddf28b7205084af` |
| `train_core_raw.npz` | `42e9d6575c81c88e13904100db690e7ded1ff153eed234b833364421c87b3c12` |
| `train_core_scaled.npz` | `b0f21568c7ef933f4d1341d9999afd09984c19523a7ea7f5da79ddbf742b2806` |
| `validation_core_raw.npz` | `abf487c32e5bf076f737c234da7d1d9f1b613edbd1d82d30d7157116d67e8a84` |
| `validation_core_scaled.npz` | `05696e864da5460e18b52bc3500263222ead4bd887580812bfb192c5bacff229` |
| `test_core_raw.npz` | `bbb5c53348720f873cc7be492a6a27e719c21586bd763b64b3b10d88c5f4f911` |
| `test_core_scaled.npz` | `bd9f9e4086fae94835409ca37c85163e22db607f3a76020c481dac12ab3474d6` |

The `SHA256SUMS` file in the same directory records these hashes locally.

---

# 15. Baseline Artifact Sizes

Observed generated artifact sizes:

| Artifact | Approximate Size |
|---|---:|
| `train_core_raw.npz` | 55.06 MiB |
| `validation_core_raw.npz` | 11.01 MiB |
| `test_core_raw.npz` | 12.57 MiB |
| `train_core_scaled.npz` | 53.42 MiB |
| `validation_core_scaled.npz` | 10.68 MiB |
| `test_core_scaled.npz` | 12.19 MiB |
| `scaler_core.json` | <0.01 MiB |
| `feature_schema_core.json` | <0.01 MiB |
| `build_audit_core.json` | ~0.01 MiB |
| `SHA256SUMS` | ~0.77 KiB |

File size is informational only.

SHA-256 is the authoritative artifact-integrity check.

---

# 16. Input A Freeze Decision

The generated `core` artifacts constitute the frozen hand-engineered baseline representation for DS-005.

They must not be regenerated, altered, replaced, or selectively filtered because of downstream:

- clustering performance;
- cluster interpretability;
- SSL performance;
- validation performance;
- test performance;
- desired biological narratives.

Any future change to the following requires a new artifact version:

- feature definitions;
- feature ordering;
- sampling or bout inclusion;
- first-bout interval convention;
- normalization procedure;
- QC handling;
- fish mapping;
- context mapping;
- train/validation/test assignment;
- source dataset;
- extraction code that changes numerical results.

---

# 17. Protocol Amendment Rule

If a legitimate methodological or integrity issue requires changing a frozen artifact:

1. Do not overwrite the existing artifact set.
2. Preserve the existing hashes.
3. Create a new versioned artifact set.
4. Record the reason for the change.
5. Record whether the issue was discovered before or after examining downstream results.
6. Record the exact files affected.
7. Record the new hashes.
8. Update the preregistration or protocol documentation where appropriate.

Example versioning:

```text
data/processed/DS-005/baseline-v2/
```

or:

```text
train_core_v2_raw.npz
train_core_v2_scaled.npz
```

The original version must remain recoverable.

---

# 18. Test-Set Protection

The DS-005 test partition is reserved for final evaluation.

Before the confirmatory evaluation, test data must not be used to choose:

- number of clusters;
- clustering algorithm;
- dimensionality;
- PCA component count;
- SSL architecture;
- SSL embedding dimension;
- temporal-window definition;
- augmentations;
- regularization;
- learning rate;
- stopping criteria;
- feature subset;
- speed-control design;
- clustering hyperparameters.

Training data are used for model fitting.

Validation data may be used for preregistered model and hyperparameter selection.

The test partition is used only after the analysis configuration is frozen.

---

# 19. Input A / Input B Comparability Requirement

Input A and Input B must be compared on equivalent behavioral units wherever technically possible.

For DS-005:

```text
primary unit = valid bout
```

Both representations must preserve:

- canonical fish identity;
- canonical session identity;
- frozen partition;
- bout index;
- experimental context metadata;
- QC sensitivity flags.

A downstream comparison must not allow one representation to benefit from additional test information, context labels, or different fish partitions.

---

# 20. Planned Input B Reproducibility Record

When the self-supervised representation is implemented, this document should be extended with:

- input tensor definition;
- temporal representation;
- preprocessing;
- augmentation definitions;
- architecture;
- embedding dimension;
- random seeds;
- optimizer;
- learning rate;
- batch size;
- training epochs;
- early stopping rule;
- checkpoint-selection rule;
- training fish;
- validation fish;
- model checkpoint SHA-256;
- embedding artifact SHA-256;
- software/environment versions;
- GPU/CPU information where relevant.

Input B must receive its own frozen artifact section before final comparison with Input A.

---

# 21. Planned Discovery Reproducibility Record

Before baseline clustering or SSL clustering becomes confirmatory, record:

- dimensionality-reduction method;
- dimensionality-reduction fitting partition;
- number of retained dimensions;
- clustering algorithms;
- cluster-number candidate range;
- cluster-number selection rule;
- clustering seeds;
- stability metrics;
- validation criteria;
- held-out-fish procedure;
- speed-only controls;
- identity-leakage tests;
- context/session checks;
- sensitivity analyses.

The final test partition must remain untouched until these decisions are frozen.

---

# 22. Current Reproducibility Status

As of the current freeze:

```text
DS-005 source dataset verified                 ✅
DS-005 raw archive integrity verified          ✅
DS-005 raw directory locked                    ✅
Fish identity mapping verified                 ✅
Canonical session definition verified          ✅
14-context mapping verified                    ✅
Fish-level split frozen                        ✅
QC rules frozen                                ✅
DS-005 manifest frozen                         ✅
DS-005 freeze record frozen                    ✅

Canonical DS-005 loader implemented            ✅
Canonical loader tests                         ✅ 16 passed

Input A core baseline implemented              ✅
Input A baseline tests                         ✅ 32 passed
Input A full feature extraction completed      ✅
Input A train-only normalization completed     ✅
Input A artifact hashes recorded               ✅
Input A baseline artifact set frozen           ✅

Baseline discovery/clustering                  ✅
Input B SSL implementation                     ✅
Input B representation freeze                  ✅
Five-seed SSL TRAIN / VALIDATION artifacts      ✅
SSL clustering/stability analysis              ✅
Input A vs Input B comparison                  ✅
Identity/context/speed validation              ✅
Known-class and substructure validation        ✅
Validation interpretation freeze               ✅ document prepared
DS-006 archive integrity verified              ✅
DS-006 extraction/inventory recorded           ✅ 64 scientific files
DS-006 recording/fish-well IDs frozen          ✅
DS-006 well-level QC frozen                    ✅ 374 usable units
DS-006 preprocessing and split                 ✅
Independent DS-006 replication analysis        ✅ final TEST complete
Final DS-005 held-out test evaluation          🔒 not opened
Final DS-006 held-out test evaluation          ✅ opened once; complete
```

---

# 23. Current Frozen References

## DS-005 Source

```text
Archive MD5:
b9a00fccda494bb49ea7c67d3b0f8d9e

Archive SHA-256:
5af065e44f3f00f1a975dfe6472eee985c9e73c2f4a98ba722b5233efb85cffd
```

## DS-005 Split

```text
19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935
```

## DS-005 Manifest

```text
62af545817ed48cc6918b4c5ad1448d878694f1816fe552099281e6469b5411e
```

## DS-005 Freeze Record

```text
ecf9a0bf45b34d1ec8c8b378c57860e0de20acc2e0aee5749a7bf6d442fd579c
```

## DS-006 Replication Source and Identity Freeze

```yaml
archive_filename: "Data_all.zip"
archive_sha256: "d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f"

scientific_files_extracted: 64
mat_files: 32
txt_files: 32

number_of_sessions: 32
fish_per_session: 12
number_of_fish: "384 potential fish-well units; 381 nonempty; 374 usable after frozen well-level QC"

recording_families:
  pH_1a: 10
  pH_2a: 7
  pH_2b: 7
  pH_2c: 8

canonical_recording_id: "organization.videoName"
canonical_fish_id: "DS006::<recording_id>::wellXX"
biological_identity_across_recordings_verified: false

archive_discrepancy:
  description: "Catamaran_pH_2b_t7 directory exists but no matching scientific .mat/.txt result pair is present"

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

The canonical fish-well identifier guarantees computational uniqueness within
DS-006. It does not establish whether a biological animal was reused across
recordings. Replication grouping and interpretation must retain this limitation.

The processed DS-006 TRAIN, VALIDATION, and originally sealed TEST artifacts
are recorded in `docs/ds006-replication-protocol.md`, with frozen source hashes
in `docs/decision-log.md` (`DEC-023`). Creating and checksum-verifying those
arrays did not constitute scientific inspection. DS-006 TEST was later opened
exactly once from freeze commit
`575ead5403d0b2f721d143366b4d2e0014bd67ee`; the final results and output hashes
are recorded in `docs/ds-006-replication.md` and
`data/processed/DS-006/final_test_evaluation/`. DS-005 TEST remains unopened.

DS-006 is an independently acquired Reddy et al. (2022) dataset, distinct from
the Marques et al. recordings underlying DS-005. Differences in assay design,
acquisition rate, duration, stimuli, and tracking pipeline support its external
replication role. Although the datasets share some investigators and were later
analyzed together in Sridhar et al. (2024), no evidence of direct fish or
recording overlap was found.

## Input A Build Audit

```text
b6c7b76797e582ac69d123f443d5486167371b2bf9bd3dea495cfa4512c8711e
```

## Input A Feature Schema

```text
a8b2fe73f3251f7788e99e6fb1fde2688256afbe34a39e40daa8431018a5e91a
```

## Input A Scaler

```text
85724491d36b4c7574663dfb7a74b263d3cfc2cf5151d488dddf28b7205084af
```

## Input A Matrices

```text
train_core_raw.npz
42e9d6575c81c88e13904100db690e7ded1ff153eed234b833364421c87b3c12

train_core_scaled.npz
b0f21568c7ef933f4d1341d9999afd09984c19523a7ea7f5da79ddbf742b2806

validation_core_raw.npz
abf487c32e5bf076f737c234da7d1d9f1b613edbd1d82d30d7157116d67e8a84

validation_core_scaled.npz
05696e864da5460e18b52bc3500263222ead4bd887580812bfb192c5bacff229

test_core_raw.npz
bbb5c53348720f873cc7be492a6a27e719c21586bd763b64b3b10d88c5f4f911

test_core_scaled.npz
bd9f9e4086fae94835409ca37c85163e22db607f3a76020c481dac12ab3474d6
```

---

# 24. Next Freeze Point

The TRAIN / VALIDATION analysis freeze is recorded in
`docs/validation-freeze.md`. Before opening DS-005 TEST, commit or otherwise
immutably timestamp that document together with the result artifacts and verify
their recorded hashes. The final evaluation must use the frozen `Long_CS`
primary case study, `LLC` secondary case study, and stated nuisance checks; TEST
must not be used to select replacements.

The baseline and SSL clustering methods, candidate selection, dimensionality
reduction, seeds, stability procedure, validation metrics, nuisance controls,
and sensitivity definitions are already frozen and evaluated on TRAIN /
VALIDATION. No additional method-selection freeze remains. Only after the
validation freeze is immutably recorded should the final held-out TEST partition
be evaluated.
