# DS-006 Replication Report

## Document Status

**Status:** FROZEN PRE-TEST REPLICATION REPORT
**Dataset:** DS-006
**Role:** External replication / transfer dataset
**Primary dataset:** DS-005
**Date frozen:** 2026-08-25
**TEST status:** **SEALED / NOT USED**

This document freezes the DS-006 replication findings obtained from TRAIN and
VALIDATION only. It must be committed before any scientific inspection of the
DS-006 TEST partition.

Any later deviation from this document must be documented explicitly rather
than silently replacing the frozen plan or interpretation.

---

# 1. Replication Objective

The purpose of DS-006 is to determine whether the major representation-level
findings from DS-005 generalize to a separately acquired zebrafish behavioral
dataset.

The replication tests whether:

1. the frozen DS-005 SSL encoders produce healthy representations on DS-006;
2. the frozen DS-005 clustering recipe produces reproducible behavioral
   organization in DS-006;
3. the resulting organization remains:
   - structured across encoder seeds;
   - substantially related to locomotor speed;
   - not reducible to mean speed alone;
   - not dominated by fish-well identity;
   - not dominated by recording or experimental context;
4. the recomputed DS-006 handcrafted representation relates to the transferred
   SSL partition in the same way observed in DS-005;
5. comparable kinematic axes appear in DS-006;
6. failures to reproduce DS-005 findings are recorded rather than hidden.

This is an external replication of the **method and broad scientific claims**,
not a requirement that every DS-005 fine-grained cluster boundary or
class-specific effect reappear identically.

---

# 2. Dataset Identity and Independence

## 2.1 DS-006 Source

**Dataset ID:** `DS-006`
**Dataset:** Reddy et al. larval exploration and aversive chemotaxis
**Paper:** *A lexical approach for identifying behavioural action sequences*
**Authors:** Gautam Reddy, Laura Desban, Hidenori Tanaka, Julian Roussel,
Olivier Mirat, Claire Wyart
**Repository:** Dryad
**DOI:** `10.5061/dryad.6t1g1jwwz`
**License:** CC0 1.0
**Access date:** 2026-08-21

DS-006 is treated as an **independently acquired external experimental
dataset/cohort**, not as a resplit of DS-005.

The wording **independent lab** is intentionally avoided because investigator
overlap exists.

## 2.2 Independence Statement

DS-006 is distinct from the Marques-derived recordings underlying DS-005.

The two datasets differ in source publication, repository record, acquisition
protocol, experimental conditions, recording structure, tracking pipeline, and
processing history.

There is no evidence that DS-006 is a resplit or reuse of DS-005 recordings.

**Independence status:** CONFIRMED FOR DATASET / ACQUISITION COHORT

---

# 3. Provenance and Archive Integrity

## 3.1 Archive

**Archive:** `data/raw/DS-006/Data_all.zip`

**SHA-256:**

```text
d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f
```

## 3.2 Scientific File Inventory

After excluding macOS metadata and archive junk:

- scientific files: **64**
- `.mat` files: **32**
- `.txt` files: **32**
- recording result pairs: **32**

Recording-family counts:

- `pH_1a`: 10
- `pH_2a`: 7
- `pH_2b`: 7
- `pH_2c`: 8

Total: **32 recordings**

## 3.3 Known Archive Discrepancy

The source documentation implies eight `pH_2b` experiments, but the extracted
archive contains no matching result pair for:

```text
pH_2b_t7
```

This discrepancy is retained as a provenance note and is not silently repaired.

---

# 4. DS-006 Schema

Each recording contains up to 12 wells.

Canonical fish-well identifier:

```text
DS006::<recording_id>::wellXX
```

Important limitation:

> Biological uniqueness of fish across separate recordings is not independently
> verified.

Therefore identity analyses are described as **fish-well identity leakage**.

Counts:

- recordings: **32**
- potential fish-well slots: **384**
- nonempty fish-well units: **381**
- usable fish-well units after frozen well-level QC: **374**

---

# 5. Frozen DS-006 QC

Frozen preprocessing/QC parameters:

```text
QC seed:        20260822
fps used:       160
px_to_mm:       0.071
interpolation noise SD: 0.1
smoothing:      10
```

Bout counts:

- raw bouts: **165,579**
- bouts removed by well exclusion: **1,556**
- post-well-exclusion candidates: **164,023**
- accepted bouts: **163,065**
- rejected bouts: **958**
- acceptance rate: **0.994159**

No TEST information was used to alter QC.

---

# 6. Frozen DS-006 Split

Split unit: **recording**
Split seed: `20260822`

| Partition | Recordings | Bouts |
|---|---:|---:|
| TRAIN | 22 | 118,100 |
| VALIDATION | 5 | 18,835 |
| TEST | 5 | 26,130 |

Recording overlap between partitions:

```text
0
```

**TEST remains sealed.**

---

# 7. Handcrafted Baseline Representation

The 18-feature handcrafted baseline was recomputed independently on DS-006.

Features:

1. `bout_duration`
2. `inter_bout_interval`
3. `speed_mean`
4. `speed_std`
5. `speed_median`
6. `speed_max`
7. `speed_p95`
8. `speed_rms`
9. `speed_change_abs_mean`
10. `speed_change_std`
11. `speed_change_max`
12. `speed_change_rms`
13. `turn_total_abs`
14. `turn_net`
15. `turn_abs_mean`
16. `turn_std`
17. `turn_max`
18. `turn_rms`

Relevant artifacts:

```text
data/processed/DS-006/baseline/train_core_raw.npz
data/processed/DS-006/baseline/validation_core_raw.npz
data/processed/DS-006/baseline/feature_manifest.json
```

---

# 8. Frozen DS-006 Handcrafted Baseline Clustering

The frozen DS-005 baseline recipe was applied to DS-006 without DS-006 model
selection:

```text
18 handcrafted features
→ TRAIN-only median imputation
→ TRAIN-only StandardScaler
→ PCA(6), fixed
→ GaussianMixture(k=2), fixed
→ TRAIN fit
→ VALIDATION prediction
```

Results:

```text
PCA retained variance     0.959686
TRAIN counts              [98,847, 19,253]
VALIDATION counts         [14,249, 4,586]
TRAIN silhouette          0.507564
VALIDATION silhouette     0.492493
GMM converged             YES
GMM iterations            6
```

No PCA selection, method selection, or `k` selection was performed.

**TEST partition used: NO**

---

# 9. SSL Transfer Setup

Primary SSL evaluation mode:

> **Frozen DS-005 encoder transfer**

Frozen encoder seeds:

```text
11
23
37
51
79
```

No projection head, fine-tuning, or DS-006 SSL retraining was used.

Frozen input geometry:

```text
(175, 3)
```

Channels:

```text
sin(Heading)
cos(Heading)
derived head speed
```

DS-006 TRAIN-only speed normalization:

```text
mean = 9.773209740465
std  = 10.478936765051
```

Per seed:

```text
TRAIN       (118100, 64)
VALIDATION  (18835, 64)
```

Checkpoint immutability checks passed for all seeds.

---

# 10. Transfer Embedding QC

Strict non-collapse gate:

```text
PASS
```

Across seeds:

- minimum TRAIN rank: **64/64**
- minimum VALIDATION rank: **64/64**
- near-zero dimensions: **0**
- 95% variance required roughly **11–14 PCs**

Mean cross-seed geometry Spearman correlation:

```text
TRAIN       0.710851
VALIDATION  0.766082
```

---

# 11. Frozen Transfer Clustering

Frozen recipe:

```text
StandardScaler: TRAIN only
PCA target:     95% variance, TRAIN only
Method:         KMeans
k:              8
random_state:   20260822
n_init:         10
KMeans fit:     TRAIN only
```

Aggregate:

```text
mean PCA components                 14.20
mean VALIDATION silhouette          0.161184
mean TRAIN repeated-fit ARI         0.911500
mean cross-seed VALIDATION ARI      0.361309
mean cross-seed VALIDATION NMI      0.501519
mean pairwise aligned agreement     0.537372
mean VAL agreement to seed11        0.529015
```

Interpretation:

> Within-encoder clustering is stable, while exact `k=8` boundaries are only
> moderately reproducible across independently trained encoders.

---

# 12. Speed Dependence and Speed-Only Collapse Control

Across seeds:

```text
mean TRAIN eta²        0.446975
mean VALIDATION eta²   0.545421
```

Mean-speed-only TRAIN→VALIDATION probe:

```text
balanced accuracy      0.276377
macro F1               0.217589
accuracy               0.312822
chance balanced acc    0.125000
```

Interpretation:

> DS-006 transfer clusters are substantially speed-related but are not reducible
> to mean speed alone.

Replication status:

```text
REPLICATED
```

---

# 13. Fish-Well Identity Leakage

VALIDATION aggregate:

```text
mean NMI          ~0.0315
mean AMI          ~0.0279
mean Cramer's V   ~0.1631
mean cluster entropy ~0.875
```

Interpretation:

> Transfer clusters are not primarily explained by canonical fish-well
> identity.

Replication status:

```text
REPLICATED
```

---

# 14. Recording and Experimental-Context Leakage

Fields tested:

```text
recording_id
family
well
condition_code
condition_label
```

Approximate VALIDATION NMI values:

```text
recording_id     0.0183
family           0.0125
well             0.0079
condition        0.0076
```

Interpretation:

> Transfer clusters are not primarily explained by recording or experimental
> context.

Replication status:

```text
REPLICATED
```

---

# 15. Direct DS-006 Baseline-versus-SSL Comparison

This analysis directly compares the frozen DS-006 handcrafted baseline with the
transferred SSL `k=8` organization.

## 15.1 Baseline Cluster Labels versus SSL Cluster Labels

Across five SSL seeds:

```text
Mean VALIDATION ARI                              0.058651
Mean VALIDATION NMI                              0.120044
Mean VALIDATION AMI                              0.119918
Mean H(SSL|baseline) / H(SSL)                    0.923641
Mean H(baseline|SSL) / H(baseline)               0.719446
```

Interpretation:

> The coarse frozen handcrafted `k=2` clustering is substantially different
> from the transferred SSL `k=8` partition.

This direction is compatible with DS-005.

## 15.2 Linear 18-Feature → SSL Probe

TRAIN fit → VALIDATION evaluation:

```text
Mean VALIDATION balanced accuracy     0.377456
Mean VALIDATION macro F1              0.344832
Chance balanced accuracy              0.125000
```

The full handcrafted feature representation contains useful information about
SSL cluster membership, but linear recoverability is limited.

## 15.3 Nonlinear 18-Feature → SSL Probe

Frozen nonlinear sensitivity model:

```text
HistGradientBoostingClassifier
```

TRAIN fit → VALIDATION evaluation:

```text
Mean VALIDATION balanced accuracy     0.412155
Mean VALIDATION macro F1              0.388654
Chance balanced accuracy              0.125000
```

The nonlinear improvement over the linear probe is modest:

```text
0.412155 - 0.377456 = 0.034699
```

## 15.4 Cross-Dataset Interpretation

DS-005 nonlinear recovery:

```text
VALIDATION balanced accuracy ≈ 0.901642
```

DS-006 nonlinear recovery:

```text
VALIDATION balanced accuracy ≈ 0.412155
```

Therefore the DS-005 conclusion:

> Most SSL cluster structure is nonlinearly recoverable from the 18 handcrafted
> features.

does **not** strongly reproduce in DS-006.

Replication status:

```text
NOT_STRONGLY_REPLICATED
```

The correct interpretation is:

> In both datasets, the coarse handcrafted clustering differs substantially from
> the SSL organization. However, whereas DS-005 SSL cluster assignments were
> highly recoverable from the full 18-feature handcrafted representation using
> a nonlinear classifier, DS-006 transferred SSL clusters were only moderately
> recoverable. Thus, the conclusion that SSL primarily reorganizes information
> already captured by the handcrafted representation is strongly supported for
> DS-005 but does not fully generalize to DS-006.

Possible explanations include:

- SSL may encode temporal structure not summarized by the 18 bout-level features;
- DS-005 and DS-006 acquisition/tracking differences may alter the informativeness
  of the handcrafted features;
- domain shift may affect transferred SSL organization;
- handcrafted feature definitions may not be equivalently informative across
  datasets;
- the unresolved DS-006 frame-rate metadata discrepancy could influence some
  derived kinematic features.

No single explanation is selected at this stage.

---

# 16. Behavioral / Kinematic Substructure

DS-006 does not provide direct `Long_CS` / `LLC` labels, so these comparisons
are kinematic analogues rather than direct class-label replications.

Strongest mean VALIDATION eta² features:

| Rank | Feature | Mean eta² |
|---:|---|---:|
| 1 | `speed_rms` | 0.563485 |
| 2 | `speed_std` | 0.546856 |
| 3 | `speed_mean` | 0.545421 |
| 4 | `speed_p95` | 0.526291 |
| 5 | `speed_max` | 0.522172 |
| 6 | `speed_change_abs_mean` | 0.478660 |
| 7 | `speed_change_max` | 0.478099 |
| 8 | `speed_change_rms` | 0.475963 |
| 9 | `speed_change_std` | 0.475873 |
| 10 | `turn_total_abs` | 0.291605 |

---

# 17. Acceleration / Speed-Change Analogue

`speed_change_rms` and `speed_change_std` showed substantial VALIDATION eta² and
very strong TRAIN→VALIDATION profile reproducibility, typically around
`0.93–1.00`.

Interpretation:

> Acceleration / speed-change heterogeneity reproduces as a broad kinematic axis.

Replication status:

```text
REPLICATED
```

---

# 18. Bout Duration

VALIDATION `bout_duration` eta² ranged approximately:

```text
0.021 – 0.156
```

This is much weaker than the DS-005 Long_CS duration effect.

Replication status:

```text
NOT_STRONGLY_REPLICATED
```

---

# 19. Turning Structure

## 19.1 Signed Net Turning

DS-006 `turn_net` VALIDATION eta²:

```text
approximately 0.003 – 0.014
```

Replication status:

```text
NOT_STRONGLY_REPLICATED
```

## 19.2 Turning Magnitude

`turn_total_abs`:

```text
mean VALIDATION eta²                  ~0.292
cross-seed VALIDATION mean-profile rho ~0.748
```

Replication status:

```text
PARTIAL
```

---

# 20. Updated Cross-Dataset Directional Comparison

| Claim | Replication status |
|---|---|
| Moderately reproducible `k=8` organization across encoder seeds | **REPLICATED** |
| Speed-related but not reducible to mean speed | **REPLICATED** |
| Low subject/fish identity leakage | **REPLICATED** |
| Low recording/context leakage | **REPLICATED** |
| Acceleration / speed-change heterogeneity | **REPLICATED** |
| Most SSL structure recoverable nonlinearly from 18 handcrafted features | **NOT_STRONGLY_REPLICATED** |
| Long_CS-like duration heterogeneity | **NOT_STRONGLY_REPLICATED** |
| LLC-like signed net-turning structure | **NOT_STRONGLY_REPLICATED** |
| Turning magnitude/intensity as a reproducible axis | **PARTIAL** |
| Direct within-Long_CS / within-LLC subdivision replication | **NOT_DIRECTLY_TESTABLE** |

Updated status counts:

```text
REPLICATED                 5
PARTIAL                    1
NOT_STRONGLY_REPLICATED    3
NOT_DIRECTLY_TESTABLE      1
```

---

# 21. Overall Replication Interpretation

DS-006 supports several major representation-level conclusions from DS-005:

- frozen SSL representations transfer without collapse;
- `k=8` organization remains moderately reproducible across encoder seeds;
- clusters are substantially speed-related but not reducible to mean speed;
- fish-well identity leakage is low;
- recording/context leakage is low;
- speed-change/acceleration-like structure reproduces.

However, several finer conclusions weaken:

- DS-005-style strong bout-duration structure does not reproduce;
- DS-005 LLC signed net-turning does not reproduce strongly;
- most importantly, the strong DS-005 nonlinear recoverability of SSL labels
  from the 18 handcrafted features does **not** reproduce in DS-006.

Therefore DS-006 is a **mixed but informative replication**.

The broad SSL organization generalizes better than some of the exact
fine-grained explanatory relationships observed in DS-005.

---

# 22. Updated Frozen Cross-Dataset Claim

> Self-supervised representations recover structured zebrafish behavioral
> variation that generalizes across held-out fish/recordings and across an
> independently acquired replication dataset. The organization is substantially
> related to locomotor speed but is not reducible to mean speed, fish-well
> identity, or experimental context. Exact eight-cluster boundaries remain
> moderately seed-sensitive. Across datasets, speed-change/acceleration-like and
> turning-magnitude structure are reproducible, whereas the DS-005-specific
> duration, signed net-turning, and strong nonlinear recoverability from the
> 18-feature handcrafted representation do not reproduce as strongly.

---

# 23. Claim Restrictions

Do **not** claim:

- that all DS-005 findings replicated;
- that DS-006 directly replicated `Long_CS` or `LLC`;
- that `k=8` represents eight universally fixed biological behaviors;
- that exact cluster identities are perfectly reproducible across seeds;
- that SSL contains information wholly absent from handcrafted features;
- that SSL structure is wholly recoverable from handcrafted features across
  datasets;
- that DS-006 validates eight "new behaviors";
- that DS-006 fish-well IDs prove longitudinal individual-fish identity.

The correct baseline-versus-SSL claim is:

> Strong nonlinear recoverability from handcrafted features is a DS-005 result
> that did not strongly generalize to DS-006.

---

# 24. TEST Protection

As of this freeze:

```text
DS-006 TEST recordings: 5
DS-006 TEST bouts:       26,130
Scientific TEST access:  NO
```

TEST has not been used for:

- representation QC;
- PCA fitting;
- clustering fitting;
- model or `k` selection;
- baseline-vs-SSL comparison;
- linear/nonlinear probe evaluation;
- speed analysis;
- identity leakage;
- context leakage;
- substructure characterization;
- cross-dataset interpretation.

---

# Exact One-Time DS-006 TEST Procedure

## Authorization boundary

The only project program authorized to read the sealed DS-006 TEST arrays is:

```text
src/evaluation/ds006_final_test.py
```

The protected inputs are:

```text
data/processed/DS-006/ssl/test.npz
data/processed/DS-006/baseline/test_core_raw.npz
```

The final mode requires the explicit `--confirm-open-test` flag and the full
40-character SHA of the committed pre-TEST freeze. Without confirmation, the
program exits before loading TEST. It also exits before TEST loading if the
working tree is dirty, repository `HEAD` differs from `--freeze-commit`, a
frozen input hash differs, frozen probe objects are absent, or the final output
directory already exists. The final command must not be run until the runner,
this procedure, TRAIN-only probe objects, and their manifest have been
committed and pushed as the final pre-TEST freeze.

## Pre-freeze TRAIN-only probe serialization

The existing comparison artifacts contain probe results but not fitted probe
objects. Before the final freeze commit, serialize the unchanged, preregistered
TRAIN-only linear, nonlinear, and mean-speed-only models with:

```bash
PYTHONPATH=. python3 src/evaluation/ds006_final_test.py --prepare-probes
```

This mode is not authorized to open TEST. It reads the frozen 18-feature DS-006
TRAIN matrix and TRAIN-derived aligned SSL labels only, writes fitted objects
under `data/processed/DS-006/frozen_test_probes/`, and records their hashes in
`probe_manifest.json`. These objects and the manifest must be included in the
pre-TEST freeze commit. Probe fitting or deterministic reconstruction after
TEST is opened is prohibited.

## Frozen inputs and transformations

For SSL seeds `11`, `23`, `37`, `51`, and `79`, the runner verifies and uses:

```text
results/ssl/checkpoints/ssl_seed{seed}_best.pt
data/processed/DS-006/transfer_clustering/seed{seed}/scaler.joblib
data/processed/DS-006/transfer_clustering/seed{seed}/pca.joblib
data/processed/DS-006/transfer_clustering/seed{seed}/kmeans.joblib
```

The inference-only SSL path is:

```text
DS-006 TEST tensor [26,130, 175, 3]
  -> frozen DS-005 encoder
  -> TEST embedding [26,130, 64]
  -> saved DS-006 TRAIN StandardScaler.transform
  -> saved DS-006 TRAIN PCA.transform
  -> saved DS-006 TRAIN KMeans.predict
  -> saved TRAIN-derived seed-to-seed11 label mapping
```

The baseline path verifies and uses:

```text
data/processed/DS-006/baseline_clustering/imputer.joblib
data/processed/DS-006/baseline_clustering/scaler.joblib
data/processed/DS-006/baseline_clustering/pca.joblib
data/processed/DS-006/baseline_clustering/gmm.joblib
```

and applies only:

```text
TEST 18-feature matrix
  -> saved imputer.transform
  -> saved scaler.transform
  -> saved PCA.transform
  -> saved GMM.predict
```

Before processing, the runner verifies the frozen SHA-256 values embedded in
the runner from `DEC-023` and the committed clustering manifests. It asserts:

```yaml
test_bouts: 26130
recordings: 5
input_shape: [26130, 175, 3]
embedding_shape: [26130, 64]
ssl_seeds: [11, 23, 37, 51, 79]
unique_nonempty_bout_ids: true
```

The following operations are prohibited in final mode:

- `fit` or `fit_transform`;
- encoder fine-tuning;
- checkpoint or model selection;
- a new PCA, KMeans, GMM, label alignment, or value of `k`;
- probe fitting or reconstruction;
- any configuration change based on TEST.

## Frozen TEST metrics

For each SSL seed, calculate:

- cluster occupancy;
- contributing fish-wells and recordings per cluster;
- silhouette on a deterministic seed-`20260822` sample of 20,000 TEST bouts;
- distance-margin confidence;
- pairwise cross-seed ARI and NMI;
- agreement after the TRAIN-derived alignment.

Compare the frozen baseline and each aligned SSL partition using:

- ARI, NMI, and AMI;
- normalized conditional entropy in both directions;
- the frozen linear 18-feature-to-SSL probe;
- the frozen nonlinear 18-feature-to-SSL probe.

Nuisance controls are:

- mean-speed eta-squared;
- frozen speed-only accuracy, balanced accuracy, and macro-F1;
- fish-well identity NMI, AMI, Cramér's V, entropy, and maximum concentration;
- the same association metrics for `recording_id`, `family`,
  `condition_label`, `condition_code`, and `well`.

The frozen kinematic axes are `speed_change_rms`, `speed_change_std`,
`bout_duration`, `turn_net`, and `turn_total_abs`. For each axis, report TEST
eta-squared, TRAIN-to-TEST aligned-profile Spearman correlation, and cross-seed
TEST profile reproducibility.

Every frozen pre-TEST claim assessed by the runner must receive exactly one of:

```text
SUPPORTED
WEAKENED
CONTRADICTED
NOT_TESTABLE
```

No claim may be silently omitted or reworded after TEST is opened. Assessment
rules must be committed in the runner before the freeze commit is recorded.
The frozen rules cover all fourteen interpretations in the freeze declaration:
cross-seed structure, speed dependence, speed-only collapse, identity and
context leakage, baseline/SSL disagreement, linear/nonlinear recoverability,
the speed-change, duration, signed-turning and turning-magnitude axes, broad
transfer success, and the direct Long_CS/LLC limitation. Thresholds and
three-way `SUPPORTED`/`WEAKENED`/`CONTRADICTED` branches are encoded directly
in `claim_assessment.json` generation; direct Long_CS/LLC label replication is
frozen as `NOT_TESTABLE` because DS-006 has no corresponding class labels.

## Exact one-time command

After committing and pushing the complete pre-TEST freeze, substitute that
commit's full SHA and run exactly once from the repository root:

```bash
PYTHONPATH=. python3 src/evaluation/ds006_final_test.py \
  --confirm-open-test \
  --freeze-commit <FULL_40_CHARACTER_FREEZE_COMMIT_SHA>
```

## Exclusive output location

All TEST-derived outputs go only under:

```text
data/processed/DS-006/final_test_evaluation/
```

Required outputs are:

```text
run_manifest.json
seed11/test_embeddings.npz
seed11/test_labels.npy
seed11/metrics.json
seed23/test_embeddings.npz
seed23/test_labels.npy
seed23/metrics.json
seed37/test_embeddings.npz
seed37/test_labels.npy
seed37/metrics.json
seed51/test_embeddings.npz
seed51/test_labels.npy
seed51/metrics.json
seed79/test_embeddings.npz
seed79/test_labels.npy
seed79/metrics.json
baseline_test_labels.npy
baseline_vs_ssl_summary.json
cross_seed_summary.json
nuisance_summary.json
kinematic_axes_summary.json
claim_assessment.json
FINAL_TEST_SHA256SUMS
```

`run_manifest.json` records the freeze commit, exact command, UTC timestamp,
Python and package versions, verified input hashes, output hashes, and explicit
assertions that TEST was not used for fitting and no configuration changed.

---

# 25. Frame-Rate Interpretation and Source-Metadata Discrepancy

The analytical interpretation is formally resolved and frozen:

```yaml
authoritative_frame_rate_hz: 160
analytical_interpretation_status: RESOLVED_AND_FROZEN
source_metadata_discrepancy_status: UNRESOLVED_ORIGIN
```

All DS-006 preprocessing and derived timing quantities use **160 Hz**. This
decision follows the author analysis notebook, which explicitly sets
`fps = 160.0` before processing all four recording families and uses that
value for bout duration, speed, oscillation frequency, angular speed,
inter-bout interval, and trajectory timing. The decision and supporting
evidence are frozen in `docs/decision-log.md` (`DEC-018`).

Project records nevertheless retain a source-metadata discrepancy:

- some raw/source metadata report **25 Hz**;
- other records report **160 Hz**;
- the reason some source files contain **25 Hz** remains unknown.

The embedded 25 Hz values are preserved for provenance but are not treated as
authoritative timing metadata. Substituting 25 Hz would materially change
derived quantities such as:

- speed;
- acceleration / speed change;
- bout duration;
- turning rates

The unknown origin of the 25 Hz entries does not leave the analysis frame rate
undecided: **160 Hz is the sole frozen value for replication and final TEST
evaluation**. The unresolved source-level origin must remain visible in the
provenance record and be disclosed in publication materials.

---

# 26. Pre-TEST Replication Completion Status

```text
[x] Dataset independence confirmed
[x] Authorization and provenance recorded
[x] Schema mapped
[x] QC completed
[x] No DS-006 tuning rule frozen
[x] Handcrafted features recomputed independently
[x] Frozen DS-006 baseline clustering completed
[x] Frozen DS-005 SSL encoder transfer completed
[x] Transfer embedding QC passed
[x] Frozen k=8 clustering applied
[x] Cross-seed stability quantified
[x] Speed dependence retested
[x] Speed-only collapse control completed
[x] Fish-well identity leakage retested
[x] Recording/context leakage retested
[x] Direct baseline-vs-SSL ARI/NMI/AMI computed
[x] Conditional entropy computed
[x] Linear 18-feature -> SSL probe completed
[x] Nonlinear 18-feature -> SSL probe completed
[x] Comparable kinematic substructure retested
[x] Cross-dataset directional comparison produced
[x] Replication failures explicitly reported
[x] Pre-TEST replication report updated
[x] Frame-rate analytical interpretation resolved and formally documented
[x] Unknown origin of embedded 25 Hz metadata retained as a provenance limitation
[ ] DS-006 TEST opened
[ ] Final TEST confirmation completed
```

---

# 27. Freeze Declaration

As of **2026-08-25**, the DS-006 TRAIN/VALIDATION replication analysis is frozen
with the following interpretation:

1. broad SSL representation transfer is successful;
2. frozen `k=8` clustering reproduces moderate cross-seed structure;
3. speed dependence reproduces;
4. mean-speed-only collapse is rejected;
5. fish-well identity leakage is low;
6. recording/context leakage is low;
7. coarse handcrafted baseline clustering differs substantially from SSL;
8. linear and nonlinear handcrafted-feature probes recover DS-006 SSL labels
   only moderately;
9. the strong DS-005 nonlinear-recoverability result does not reproduce;
10. acceleration/speed-change heterogeneity reproduces;
11. turning magnitude is a partial analogue;
12. strong duration heterogeneity does not reproduce;
13. signed net-turning does not reproduce strongly;
14. direct Long_CS/LLC replication is not testable in DS-006.

Any TEST result that weakens these conclusions must be reported.

**DS-006 TEST remains sealed at freeze time.**
