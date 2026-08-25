# Validation Freeze Before Final Test Evaluation

Date frozen: 2026-08-24

Freeze status: operative at commit
`d66aca763c76242edc719683a617c2511e8ec37b`; one-time final TEST evaluation
completed 2026-08-25.

Primary source artifacts:

- `data/processed/DS-005/baseline_vs_ssl/aggregate_summary.json`
- `data/processed/DS-005/baseline_vs_ssl_nonlinear/aggregate_summary.json`
- `data/processed/DS-005/ssl_cluster_stability/stability_summary.json`
- `data/processed/DS-005/ssl_identity_leakage/aggregate_summary.json`
- `data/processed/DS-005/ssl_context_leakage/aggregate_summary.json`
- `data/processed/DS-005/ssl_speed_dependence/aggregate_summary.json`
- `data/processed/DS-005/ssl_within_class_substructure/summary.json`
- `data/processed/DS-005/ssl_subcluster_characterization/summary.json`
- `data/processed/DS-005/ssl_long_cs_kinematic_reproducibility/cross_seed_summary.json`
- `data/processed/DS-005/ssl_llc_turn_reproducibility/cross_seed_summary.json`

All cited summaries report `test_partition_used: false`.

## 1. Test partition status

Historical pre-TEST state: the DS-005 TEST partition was unopened and unused
for all TRAIN/VALIDATION analyses described below. It was later opened exactly
once under the committed procedure in Section 11; final evaluation is complete.

No TEST-derived result has been used for:

- SSL training
- hyperparameter selection
- clustering selection
- cluster-count selection
- feature selection
- biological interpretation
- candidate-class selection
- threshold selection

The TEST partition was opened once after this document and its guarded runner
were committed at `d66aca763c76242edc719683a617c2511e8ec37b`.

---

## 2. Frozen SSL configuration

The final SSL clustering configuration is:

- Representation: SSL temporal encoder embeddings
- SSL training seeds: 11, 23, 37, 51, 79
- Clustering method: KMeans
- Number of clusters: k = 8
- Cluster-label alignment: Hungarian mapping estimated on TRAIN only
- Reference seed: 11

No changes to the SSL encoder, clustering method, k, or seed set will be made after TEST evaluation begins.

---

## 3. Frozen baseline comparison

Input A consists of the 18 preregistered handcrafted features.

Frozen baseline clustering:

- TRAIN-only scaling
- PCA: 6 components
- GaussianMixture
- k = 2
- seed = 20260822

The nonlinear Input-A sensitivity probe showed that SSL cluster membership is highly recoverable from the handcrafted feature set.

Mean VALIDATION balanced accuracy:

- nonlinear Input A -> SSL cluster: 0.901642

Therefore, the final claim will NOT state that SSL contains information fundamentally absent from the handcrafted representation.

Instead, the comparison claim is:

> SSL produces a richer behavioral organization than the frozen handcrafted clustering, while much of that organization remains recoverable from the underlying handcrafted features.

---

## 4. Primary biological case study

### Long_CS

Long_CS is frozen as the primary within-class case study.

The three characterization variables were selected before TEST evaluation:

1. bout_duration_s
2. accel_rms
3. accel_abs_std

VALIDATION results:

| Variable | Mean eta² | Range across SSL seeds |
|---|---:|---:|
| bout_duration_s | 0.525723 | 0.496875–0.550624 |
| accel_rms | 0.538565 | 0.494719–0.573067 |
| accel_abs_std | 0.532868 | 0.485768–0.568807 |

TRAIN-to-VALIDATION mean-profile Spearman:

- bout_duration_s: 0.861905
- accel_rms: 0.819048
- accel_abs_std: 0.733333

Cross-seed VALIDATION median-profile Spearman:

- bout_duration_s: 0.722098
- accel_rms: 0.721429
- accel_abs_std: 0.697619

### Frozen Long_CS interpretation

> Long_CS contains reproducible SSL-defined kinematic substructure characterized by substantial differences in bout duration and acceleration dynamics.

This is the primary biological finding to evaluate on TEST.

---

## 5. Secondary biological case study

### LLC

LLC is frozen as a secondary within-class case study.

Primary LLC characterization variable:

- turn_net_rad

VALIDATION results:

- mean eta² = 0.164450
- eta² range across seeds = 0.058726–0.267405
- mean TRAIN-to-VALIDATION profile Spearman = 0.952381
- mean cross-seed VALIDATION mean-profile Spearman = 0.423810
- mean cross-seed VALIDATION median-profile Spearman = 0.300000

Directional patterns of particular interest:

- aligned cluster 0: positive turning in 5/5 SSL seeds
- aligned cluster 6: negative turning in 5/5 SSL seeds

### Frozen LLC interpretation

> LLC contains reproducible turning-related kinematic structure, although its exact fine-grained cluster boundaries are seed-sensitive.

This is a secondary biological finding.

---

## 6. Supporting result

### BS

BS shows reproducible within-class subdivision primarily associated with speed and movement intensity.

BS will be treated as supporting evidence rather than a primary biological discovery.

No new BS-specific hypotheses will be introduced after TEST is opened.

---

## 7. Final TEST evaluation

Once TEST is opened, the following analyses will be performed exactly once.

### 7.1 General SSL clustering

For all five frozen SSL seeds:

- apply the frozen preprocessing
- use the frozen trained encoder checkpoints
- extract TEST embeddings
- apply TRAIN-fitted scaler/PCA
- apply the frozen selected KMeans k=8 model
- report cluster occupancy

No clustering will be refit using TEST.

### 7.2 Long_CS

For TEST Long_CS bouts, calculate for each frozen SSL seed:

- eta² for bout_duration_s
- eta² for accel_rms
- eta² for accel_abs_std
- aligned 8-cluster mean profiles
- aligned 8-cluster median profiles

Compare TEST profiles against the already-frozen TRAIN/VALIDATION patterns.

### 7.3 LLC

For TEST LLC bouts, calculate:

- eta² for turn_net_rad
- aligned subcluster mean and median turn profiles
- directional status using the already-frozen ±0.10 rad threshold
- whether cluster 0 remains positive
- whether cluster 6 remains negative

### 7.4 Leakage and nuisance checks

Run the already-defined final checks for:

- fish identity
- experimental context
- speed dependence

No new nuisance variables will be selected based on TEST results.

---

## 8. Confirmation criteria

TEST is treated as supporting the Long_CS finding if:

1. all three frozen variables continue to show meaningful subcluster-associated variation;
2. the direction/ranking of the subcluster profiles remains broadly consistent with VALIDATION;
3. the result is not attributable to a TEST-specific context or fish-identity artifact.

No single arbitrary p-value threshold will be used to declare a novel biological behavior.

TEST is treated as supporting the LLC finding if:

1. turn_net_rad remains associated with SSL subcluster membership;
2. TRAIN/VALIDATION-derived turning structure remains visible;
3. aligned cluster 0 remains positively biased and cluster 6 negatively biased, or any deviations are reported transparently.

---

## 9. Claim restrictions

Even if TEST confirms the findings, the study will not claim that:

- the eight SSL clusters are eight distinct biological behaviors;
- Long_CS subclusters are definitively novel behavior categories;
- SSL captures information completely absent from handcrafted features;
- every seed produces identical behavioral boundaries.

The strongest permitted interpretation is:

> SSL reveals reproducible fine-grained kinematic organization within conventional zebrafish behavioral categories, particularly Long_CS, while the exact cluster boundaries remain partly seed-dependent and much of the SSL organization is recoverable from conventional movement features.

---

## 10. Freeze rule

After this file is committed:

- TEST may be opened once;
- no change may be made to the selected SSL model family;
- no change may be made to k = 8;
- no new candidate behavioral class may replace Long_CS based on TEST results;
- no new feature may replace the three frozen Long_CS variables based on TEST results;
- no new LLC variable may replace turn_net_rad based on TEST results;
- unexpected TEST findings may be reported only as exploratory findings.

Any deviation from this plan must be documented explicitly as a deviation.

---

## 11. Exact One-Time DS-005 TEST Procedure

### 11.1 Authorized program and protected TEST inputs

The only program authorized to open the DS-005 TEST partition is:

```text
src/evaluation/ds005_final_test.py
```

It protects and verifies these exact TEST sources before analysis:

| Input | Frozen SHA-256 |
|---|---|
| `data/raw/DS-005/DS-005-v1/Datasets/JM_data/filtered_jmpool_kin.h5` | `7aa22dad1005d4a7d7929d590899e04ea7337a0d3db134587704c30be17ab4a3` |
| `data/processed/DS-005/baseline/test_core_raw.npz` | `bbb5c53348720f873cc7be492a6a27e719c21586bd763b64b3b10d88c5f4f911` |
| `data/processed/DS-005/baseline/test_core_scaled.npz` | `bd9f9e4086fae94835409ca37c85163e22db607f3a76020c481dac12ab3474d6` |
| `data/splits/DS-005-fish-split-v1.csv` | `19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935` |

Expected TEST constraints are 192,104 valid bouts, one unique
`fish_id`/`bout_index` pair per row, temporal SSL input shape
`[192104, 175, 3]`, embedding shape `[192104, 64]`, and seeds
`[11, 23, 37, 51, 79]`.

### 11.2 TRAIN/VALIDATION-only object freeze

Before the final freeze commit, run exactly once:

```bash
PYTHONPATH=. python3 scripts/prepare_ds005_final_test_objects.py
```

This preparation program is prohibited from opening TEST. It reconstructs the
already-selected models from unchanged TRAIN data, verifies their TRAIN and
VALIDATION predictions exactly against the frozen label arrays, and writes the
serialized objects and their hashes under:

```text
data/processed/DS-005/frozen_final_test_objects/
```

For each SSL seed it freezes a TRAIN-fitted StandardScaler, 95%-variance PCA,
k=8 KMeans, linear 18-feature probe, nonlinear 18-feature probe, and speed-only
probe. It also freezes the baseline PCA(6) and GMM(k=2). The preparation step
uses the existing TRAIN-derived Hungarian mappings to seed 11; it does not
derive a new mapping.

The final freeze commit must contain `object_manifest.json` and every serialized
object (Git LFS may be used). The final runner verifies every hash in that
manifest before opening TEST.

### 11.3 Frozen inference paths

For each seed, the SSL path is:

```text
canonical DS-005 TEST bout
  -> frozen sin(orientation), cos(orientation), speed tensor construction
  -> frozen TRAIN speed normalization
  -> results/ssl/checkpoints/ssl_seed{seed}_best.pt encoder
  -> 64-dimensional embedding
  -> saved TRAIN StandardScaler.transform
  -> saved TRAIN PCA.transform
  -> saved TRAIN KMeans.predict
  -> saved TRAIN-derived mapping to seed 11 labels
```

The baseline path is:

```text
frozen TEST 18-feature scaled matrix
  -> saved baseline PCA.transform
  -> saved baseline GMM.predict
```

The runner calculates cluster occupancy and contributing fish/contexts,
deterministic silhouette on at most 20,000 bouts, distance-margin confidence,
cross-seed ARI/NMI/aligned agreement, baseline-versus-SSL ARI/NMI/AMI and both
normalized conditional entropies, linear and nonlinear feature-probe metrics,
and speed-only accuracy/balanced accuracy/macro-F1.

Frozen nuisance outputs comprise mean-speed eta-squared and fish, session, and
experimental-context NMI, AMI, Cramer's V, normalized entropy, and maximum
concentration.

The primary Long_CS analysis is restricted to `bout_duration_s`, `accel_rms`,
and `accel_abs_std`. The secondary LLC analysis is restricted to
`turn_net_rad`, including the frozen +/-0.10 rad direction rule for aligned
clusters 0 and 6. Each reports TEST eta-squared, aligned mean/median profiles,
TRAIN-to-TEST profile Spearman correlation, and cross-seed TEST-profile
reproducibility.

### 11.4 Claim-assessment rules frozen before TEST

Every preregistered interpretation receives exactly one of `SUPPORTED`,
`WEAKENED`, `CONTRADICTED`, or `NOT_TESTABLE`.

- Long_CS is `SUPPORTED` only when all three frozen variables have mean TEST
  eta-squared at least 0.25 and mean TRAIN-to-TEST profile Spearman at least
  0.50. It is `WEAKENED` when all three reach 0.10 and 0.25, respectively;
  otherwise it is `CONTRADICTED`.
- LLC is `SUPPORTED` when mean TEST eta-squared is at least 0.10, mean
  TRAIN-to-TEST Spearman is at least 0.50, and clusters 0/6 retain the frozen
  positive/negative directions in at least four of five seeds. It is
  `WEAKENED` when eta-squared is at least 0.03 and Spearman at least 0.25;
  otherwise it is `CONTRADICTED`.
- General structure, baseline difference, handcrafted-feature recoverability,
  identity/context leakage, and speed-control thresholds are encoded directly
  in the committed runner and may not be altered after TEST is opened.
- Whether eight clusters are eight distinct novel biological behaviors remains
  `NOT_TESTABLE` by this evaluation.

### 11.5 Guard, command, restrictions, and outputs

Without `--confirm-open-test`, the runner exits before opening TEST. It also
requires the full 40-character freeze commit, exact equality between that
commit and `HEAD`, a clean worktree, all frozen hashes to match, and a
nonexistent output directory.

After committing and pushing the complete freeze, run once:

```bash
PYTHONPATH=. python3 src/evaluation/ds005_final_test.py \
  --confirm-open-test \
  --freeze-commit <full-40-character-freeze-commit>
```

The following are prohibited: `fit`, `fit_transform`, encoder fine-tuning,
checkpoint selection, new PCA, new KMeans/GMM, new label alignment, new k
selection, feature selection, candidate-class selection, tuning, and any
method change. TEST observations may only pass through saved `transform` and
`predict` operations and the frozen encoder.

All new outputs go exclusively under:

```text
data/processed/DS-005/final_test_evaluation/
```

Required outputs are per-seed TEST embeddings, aligned labels and metrics;
`baseline_test_labels.npy`; `baseline_vs_ssl_summary.json`;
`cross_seed_summary.json`; `nuisance_summary.json`;
`long_cs_primary_summary.json`; `llc_secondary_summary.json`;
`claim_assessment.json`; `run_manifest.json`; and
`FINAL_TEST_SHA256SUMS`. The run manifest records the freeze commit, exact
command, UTC time, environment versions, input/output hashes, expected shapes,
and explicit assertions that TEST was not used for fitting, configuration,
alignment, or selection.

---

## 12. Final Held-Out DS-005 TEST Result

### 12.1 Execution and integrity

DS-005 TEST was opened exactly once on 2026-08-25 using freeze commit:

```text
d66aca763c76242edc719683a617c2511e8ec37b
```

The completed run processed 192,104 bouts for seeds 11, 23, 37, 51, and 79.
All entries in `FINAL_TEST_SHA256SUMS` verify successfully. The run manifest
records no fitting, configuration change, new label alignment, method
selection, or prohibited operation.

Final artifacts:

```text
data/processed/DS-005/final_test_evaluation/
```

Authoritative checksum-manifest SHA-256:

```text
9695b4d0474f37ec1e380ad001684776bfc658c9d8b734433d7f4e95780c1305
```

Principal result hashes:

| Artifact | SHA-256 |
|---|---|
| `run_manifest.json` | `fa941420d06d65d83957d039864ccda173b4927ce24ff7e33257c5c1ddc64767` |
| `claim_assessment.json` | `d845dfc985d02d42d27c4f428b73199e2d678103cd4e1889e6ecf0c918ca1937` |
| `cross_seed_summary.json` | `f8d1a3308822ce843a1ce0b9e74a9fe92245e095854f8593d7ffffdb9835874d` |
| `baseline_vs_ssl_summary.json` | `53a01dd5f524400549685ff826ca18f19cc2cc76259970b4ad99b9a856d4045e` |
| `nuisance_summary.json` | `40b684ec9f58002c1ceaadf9548a1bb4761ce4bfa631c82a9f0ebecaeb6f8632` |
| `long_cs_primary_summary.json` | `997381bb85182c228f2e50075a5f7f341d4a2f222e910d3f076333505902319e` |
| `llc_secondary_summary.json` | `2e0c2e9d1c3161adba13d5bc71b4fbecb23be2c783ef8217f5b0fd384c1be09a` |

### 12.2 General clustering and representation comparison

- Mean cross-seed TEST ARI: `0.360634`
- Mean cross-seed TEST NMI: `0.462654`
- Mean aligned agreement: `0.539512`
- Mean baseline-versus-SSL ARI: `0.019832`
- Mean linear 18-feature probe balanced accuracy: `0.481834`
- Mean nonlinear 18-feature probe balanced accuracy: `0.903573`

The frozen two-state baseline and SSL organizations remain substantially
different, while the nonlinear probe confirms that much of the DS-005 SSL
organization is recoverable from the 18 handcrafted variables.

### 12.3 Long_CS primary result

| Frozen variable | Mean TEST eta-squared | Mean TRAIN-to-TEST Spearman |
|---|---:|---:|
| `bout_duration_s` | `0.553873` | `0.719048` |
| `accel_rms` | `0.535440` | `0.580952` |
| `accel_abs_std` | `0.511263` | `0.500000` |

The preregistered Long_CS primary interpretation is `SUPPORTED`. Although some
individual-seed profile correlations are weaker, all three variables meet the
frozen aggregate support thresholds.

### 12.4 LLC secondary result

- Mean TEST `turn_net_rad` eta-squared: `0.162747`
- Mean TRAIN-to-TEST profile Spearman: `0.971429`
- Aligned cluster 0 positive and cluster 6 negative: `5/5` seeds

The preregistered LLC secondary interpretation is `SUPPORTED`.

### 12.5 Nuisance controls and claim assessment

- Mean-speed eta-squared: `0.481895`
- Speed-only balanced accuracy: `0.217024`
- Fish-identity Cramer's V: `0.180825`
- Context Cramer's V: `0.119920`

Final frozen assessments:

```text
SUPPORTED       8
WEAKENED        0
CONTRADICTED    0
NOT_TESTABLE    1
```

General k=8 structure, baseline/SSL difference, nonlinear handcrafted-feature
recoverability, low fish identity leakage, low context leakage, speed-related
but not speed-only structure, Long_CS, and LLC are `SUPPORTED`. Whether the
eight clusters constitute eight distinct novel biological behaviors remains
`NOT_TESTABLE`.

### 12.6 Final TEST status

```text
DS-005 TEST: opened once; final evaluation complete
DS-006 TEST: opened once; final evaluation complete
```

Neither TEST partition may be rerun or used for additional method, feature,
threshold, cluster, alignment, or interpretation selection.
