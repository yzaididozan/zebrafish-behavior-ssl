# Evaluation Protocol

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

**Primary dataset:** `DS-005`  
**External replication dataset:** `DS-006`  
**Protocol status:** `FROZEN — RECORDED IN DEC-025; VALIDATION INTERPRETATION FROZEN SEPARATELY`
**Last updated:** `2026-08-24`

---

# 1. Purpose

This protocol defines the confirmatory evaluation rules for determining whether the frozen self-supervised representation reveals reproducible zebrafish behavioral structure that is not fully captured by the frozen hand-engineered baseline.

The evaluation is designed to guard against the following failure modes:

- fish-identity leakage;
- context/session leakage;
- speed-only representations;
- tracking artifacts;
- unstable clustering;
- seed-specific results;
- post hoc cluster interpretation;
- and improvements that fail to reproduce in an independent dataset.

No metric in this protocol may be changed based on DS-005 TEST performance.

---

# 2. Protected Data Rule

The DS-005 TEST partition remains protected until:

- SSL multi-seed training is complete;
- all confirmatory metrics in this protocol are frozen;
- all nuisance-model specifications are frozen;
- all sensitivity analyses are categorized;
- and any remaining methodological ambiguity is resolved.

The DS-006 TEST partition was required to remain sealed until the
replication-side TRAIN/VALIDATION procedure was frozen. That condition was met
at commit `575ead5403d0b2f721d143366b4d2e0014bd67ee`; DS-006 TEST was then opened
exactly once and its final evaluation is complete. DS-005 TEST remains
protected and unopened.

The prerequisite DS-005 multi-seed TRAIN / VALIDATION analyses are complete.
Their evidence-driven interpretation, selected within-class case studies, and
exact one-time TEST analysis are frozen in `docs/validation-freeze.md`. That
document narrows the permitted claim in response to validation evidence without
changing the protected-TEST rules in this protocol.

---

# 3. Primary Representations

## Input A

Frozen hand-engineered baseline:

```text
18 features
```

Primary baseline clustering:

```text
PCA(6) -> GaussianMixture(k=2, seed=20260822)
```

## Input B

Frozen SSL encoder embedding:

```text
64-dimensional encoder representation
```

The projection-head representation is not used for downstream confirmatory analysis.

---

# 4. Evaluation Units

Unless otherwise specified:

- one observation = one valid bout;
- each bout belongs to exactly one fish;
- fish are the grouping unit for cross-fish evaluation;
- all group-aware resampling must preserve fish boundaries.

Metrics may be computed at bout level and then aggregated by fish where specified.

---

# 5. Cross-Fish Reproducibility

## Objective

Determine whether discovered behavioral structure learned from TRAIN generalizes to previously unseen fish.

## Primary procedure

For each frozen representation:

1. Fit the representation-side clustering model using TRAIN only.
2. Assign cluster labels to VALIDATION using the frozen fitted model.
3. Evaluate whether the same cluster structure is recoverable across held-out fish.

For GMM-based clustering, held-out labels are assigned using the fitted component posterior / prediction rule.

For SSL clustering, the same frozen clustering-selection governance must be applied using TRAIN / VALIDATION only before TEST is opened.

## Primary reproducibility metrics

### A. Held-out cluster occupancy

For each fish, calculate the proportion of its bouts assigned to each cluster.

A cluster is considered supported across fish if it is represented in more than a small number of individual fish rather than being driven by one or a few animals.

Report:

```text
number of fish contributing to each cluster
median per-fish occupancy
IQR per-fish occupancy
minimum / maximum occupancy
```

### B. Fish-bootstrap cluster stability

Bootstrap fish rather than bouts.

Procedure:

```yaml
bootstrap_unit: fish
bootstrap_replicates: 500
sampling: with_replacement
```

For each replicate:

1. sample TRAIN fish with replacement;
2. refit the clustering model under the frozen configuration;
3. assign labels to the fixed VALIDATION representation;
4. align replicate cluster labels to the reference clustering;
5. calculate adjusted Rand index.

Primary statistic:

```text
median fish-bootstrap ARI
```

Also report:

```text
IQR
2.5th percentile
97.5th percentile
```

## Label alignment

Cluster labels are arbitrary.

When explicit alignment is required, use the Hungarian assignment algorithm maximizing overlap with the reference clustering.

Metrics that are label-permutation invariant, such as ARI, do not require label relabeling before calculation.

## Cross-seed SSL reproducibility

For each pair of frozen SSL seeds:

1. obtain embeddings for the same validation bouts;
2. apply the frozen clustering procedure separately;
3. compute pairwise ARI between the resulting partitions.

For five seeds, report all 10 pairwise values and summarize with:

```text
median pairwise ARI
minimum pairwise ARI
IQR pairwise ARI
```

## Decision interpretation

Cross-fish reproducibility is considered **strong evidence** when:

```yaml
median_fish_bootstrap_ARI: ">= 0.75"
lower_95pct_bootstrap_bound: ">= 0.50"
median_cross_seed_ARI: ">= 0.75"
```

Interpretation bands:

```yaml
ARI_ge_0.75: strong
ARI_0.50_to_0.75: moderate
ARI_0.25_to_0.50: weak
ARI_lt_0.25: poor
```

These bands are used for interpretation, not as a binary guarantee of biological validity.

A Claim Level 2 conclusion should not rely on structure with poor cross-fish reproducibility.

---

# 6. Baseline-vs-SSL Structural Comparison

## Objective

Determine whether SSL-derived organization contains reproducible structure not fully reconstructed by Input A.

## Primary procedure

Using the same bouts in the same evaluation partition:

1. obtain Input A cluster assignments;
2. obtain Input B cluster assignments;
3. compare the two partitions.

Primary metrics:

```text
Adjusted Rand Index (ARI)
Normalized Mutual Information (NMI)
```

## Interpretation

High ARI/NMI indicates substantial overlap between baseline and SSL structure.

Low or moderate overlap is not automatically evidence that SSL is superior.

Additional validation is required to establish whether the differing SSL structure is:

- reproducible;
- behaviorally interpretable;
- not speed-driven;
- not identity-driven;
- not context-driven;
- not artifact-driven.

## Conditional predictability analysis

Fit a simple multinomial logistic regression using Input A features to predict SSL cluster membership.

Train on TRAIN and evaluate on VALIDATION.

Primary metric:

```text
macro-averaged balanced accuracy
```

Also report:

```text
macro F1
```

If Input A predicts SSL clusters nearly perfectly, the claim that SSL captures structure not represented by Input A is weakened.

---

# 7. Speed-Only Control

## Objective

Determine whether the SSL representation or its clusters are largely explained by locomotor speed.

Use a predefined bout-level speed summary:

```text
mean speed per bout
```

The same definition must be used across all relevant analyses.

## Control A — Speed-only clustering

Construct a one-dimensional representation containing only mean bout speed.

Using TRAIN / VALIDATION only:

1. fit the same primary clustering family where technically valid;
2. use the same selected primary cluster count for the direct confirmatory comparison;
3. assign held-out validation labels.

Compare speed-only partitions to SSL partitions using:

```text
ARI
NMI
```

If SSL clustering is nearly identical to speed-only clustering, this weakens the claim of additional behavioral structure.

## Control B — Embedding-to-speed regression

Fit ridge regression from the SSL embedding to mean bout speed.

```yaml
model: Ridge
alpha: 1.0
fit_partition: TRAIN
evaluation_partition: VALIDATION
```

Primary metric:

```text
R^2
```

Also report:

```text
MAE
```

No hyperparameter tuning is performed for this nuisance model.

## Control C — Cluster speed enrichment

For each cluster, report:

```text
median speed
IQR speed
distribution overlap
```

Calculate Kruskal-Wallis H across clusters as a descriptive omnibus statistic.

Effect size should be emphasized over p-value because of the very large number of bouts.

## Speed-dependence interpretation

Use the following descriptive bands for embedding speed predictability:

```yaml
R2_lt_0.25: low
R2_0.25_to_0.50: moderate
R2_0.50_to_0.75: high
R2_ge_0.75: very_high
```

A Claim Level 2 result is not supported if the purported additional SSL structure is both:

1. highly reproducible by the speed-only representation; and
2. associated with very high embedding-to-speed predictability;

without evidence of remaining behavioral organization beyond speed.

---

# 8. Fish-Identity Leakage

## Objective

Test whether the SSL embedding contains excessive fish-specific information.

Because DS-005 contains hundreds of fish, direct multiclass fish-ID prediction is the primary leakage test.

## Model

```yaml
model: multinomial_logistic_regression
regularization: L2
C: 1.0
solver: saga
max_iter: 1000
class_weight: balanced
```

No hyperparameter tuning is performed.

## Data procedure

To avoid trivial bout-level leakage:

- TRAIN nuisance model on bouts from a predefined subset of TRAIN fish only when the target classes occur in both nuisance-model train and evaluation data;
- within each fish, split bouts into nuisance-train and nuisance-validation subsets;
- all preprocessing must be fit using nuisance-train only.

This analysis assesses whether fish identity is encoded within repeated bouts of known fish.

It is not a substitute for the primary held-out-fish split.

## Metrics

Primary:

```text
top-1 balanced accuracy
```

Secondary:

```text
macro F1
```

Compare against:

```text
uniform chance = 1 / number_of_fish_classes
majority-class baseline
```

Because class count is large, report the ratio:

```text
observed balanced accuracy / uniform chance
```

## Interpretation

Identity leakage is concerning when fish ID is predicted far above chance and substantially more strongly from SSL than from Input A.

No single universal numerical threshold is treated as definitive.

The principal comparison is:

```text
SSL fish-ID leakage vs Input A fish-ID leakage
```

If SSL disproportionately encodes fish identity and its discovered clusters align with fish-specific structure, Claim Level 2 is weakened.

---

# 9. Context / Session Leakage

## Objective

Determine whether embeddings predominantly encode experimental context or recording/session proxies.

## Context model

```yaml
model: multinomial_logistic_regression
regularization: L2
C: 1.0
solver: saga
max_iter: 1000
class_weight: balanced
```

Input:

```text
representation under evaluation
```

Target:

```text
DS-005 context label
```

Fit on TRAIN and evaluate on VALIDATION.

## Metrics

Primary:

```text
balanced accuracy
```

Secondary:

```text
macro F1
```

Report the same nuisance prediction analysis for:

- Input A;
- Input B.

## Session proxy

If a canonical session / recording identifier supports repeated observations across sufficiently many classes, run the same fixed logistic-regression procedure.

If session identifiers are effectively unique per fish and therefore not separately identifiable from fish identity, document that the session leakage test is not independently estimable and rely on:

- fish leakage;
- context leakage;
- known acquisition metadata.

## Interpretation

High context predictability alone does not invalidate a representation because behavioral context can legitimately alter behavior.

Concern arises when:

- cluster structure is almost completely explained by context;
- within-context behavioral structure is weak;
- or context predictability dominates SSL substantially more than Input A without independent behavioral evidence.

---

# 10. Tracking-Quality Confirmatory Rule

## Decision

No new DS-005 bout exclusions will be introduced during confirmatory evaluation unless a preregistered structural QC flag already exists.

The primary dataset has already undergone structural and tracking QC.

Therefore:

```yaml
new_post_clustering_exclusion_rules: prohibited
```

## Confirmatory artifact check

For each discovered cluster, compare distributions of available tracking-quality indicators or proxy variables.

At minimum report, where available:

```text
bout duration
fraction nonfinite before primary QC
boundary / padding indicators
extreme speed
extreme orientation change
other existing QC flags
```

No bout may be removed because it makes a cluster look artifact-driven after cluster inspection.

If a cluster is strongly enriched for tracking/QC boundary cases, it must be reported as potentially artifactual rather than silently excluded.

---

# 11. Nuisance Prediction Model Governance

All nuisance prediction models are deliberately simple and fixed.

Primary model family:

```text
linear / multinomial logistic regression
```

Continuous nuisance target:

```text
ridge regression
```

Reasons:

- low flexibility;
- easy reproducibility;
- reduced hyperparameter fishing;
- interpretable comparison between Input A and Input B.

## Fixed parameters

Classification:

```yaml
C: 1.0
penalty: L2
class_weight: balanced
max_iter: 1000
```

Regression:

```yaml
model: Ridge
alpha: 1.0
```

No TEST-based tuning is permitted.

---

# 12. Primary Confirmatory Metric Set

The following are the primary confirmatory metrics.

## Discovery quality

```text
validation silhouette
```

Used only under the frozen discovery-selection procedure.

## Cluster stability

```text
fish-bootstrap ARI
cross-seed ARI
```

## Baseline-vs-SSL structural overlap

```text
ARI
NMI
```

## Baseline predictability of SSL structure

```text
balanced accuracy
macro F1
```

from multinomial logistic regression predicting SSL cluster label from Input A.

## Speed dependence

```text
speed-only vs SSL ARI
speed-only vs SSL NMI
embedding-to-speed R^2
embedding-to-speed MAE
cluster-level speed distributions
```

## Fish-identity leakage

```text
balanced accuracy
macro F1
chance ratio
```

## Context leakage

```text
balanced accuracy
macro F1
```

## Tracking-artifact dependence

```text
cluster-wise QC/proxy distributions
```

## External replication

```text
replication cluster stability
primary-vs-replication qualitative structural agreement
nuisance-control consistency
```

Exact cross-dataset cluster-label equality is not required because DS-006 is independently acquired and uses mapped rather than numerically identical source variables.

DS-006 independence from DS-005 is confirmed at the dataset and acquisition
levels: it has a separate source, DOI, publication, recordings, assay protocol,
frame rate, recording duration, and tracking pipeline. Some investigators
overlap and both datasets were later analyzed together, but no evidence of
direct fish or recording overlap was found. This does not remove the separate
within-DS-006 uncertainty about biological reuse across DS-006 recordings.

---

# 13. Multiple Comparisons and Statistical Emphasis

The primary study is representation/discovery focused rather than a null-hypothesis significance-testing study.

Because bout counts are extremely large, p-values can become trivially small.

Therefore:

- effect sizes;
- stability;
- held-out generalization;
- confidence intervals;
- and nuisance-control comparisons

take precedence over isolated p-values.

Where hypothesis tests are reported, they are treated as supporting descriptive evidence unless explicitly designated primary.

---

# 14. Confidence Intervals

For fish-aggregated metrics, use fish-level bootstrap confidence intervals.

Frozen bootstrap policy:

```yaml
bootstrap_unit: fish
replicates: 500
confidence_interval: percentile_95
seed: 20260822
```

Bouts must not be independently bootstrapped when the estimand is intended to generalize across fish.

---

# 15. SSL Seed Aggregation

The five frozen SSL seeds are:

```text
11
23
37
51
79
```

For seed-dependent confirmatory metrics report:

```text
all individual seed values
median
IQR
minimum
maximum
```

The primary summary is the **median across seeds**.

No seed may be discarded because it produces an unfavorable result unless a documented technical failure invalidates the run.

---

# 16. Primary Claim Decision Procedure

Claim Level 2 is supported only if the overall evidence satisfies all of the following qualitative gates.

## Gate A — Reproducibility

SSL structure must show at least moderate and preferably strong:

- fish-bootstrap stability;
- cross-seed agreement;
- held-out-fish support.

## Gate B — Beyond baseline

SSL structure must not be almost completely reconstructed by Input A.

Evidence includes:

- nontrivial baseline-vs-SSL partition difference;
- less-than-near-perfect prediction of SSL labels from Input A;
- reproducible structure specific to SSL.

## Gate C — Not speed-only

The distinguishing SSL structure must not be adequately explained by mean speed alone.

## Gate D — Not identity-driven

SSL must not owe its primary cluster structure to fish identity.

## Gate E — Not context/session-only

The additional structure must not merely reproduce experimental context or session proxies.

## Gate F — Not artifact-driven

The distinguishing structure must not be concentrated in obvious tracking/QC artifacts.

## Gate G — External support

At least part of the principal qualitative conclusion should remain supported in DS-006 under the frozen external-replication procedure.

## Outcome categories

The final confirmatory outcome is classified as one of:

```yaml
SUPPORTED:
  description: "Converging evidence supports Claim Level 2."

PARTIALLY_SUPPORTED:
  description: "Some reproducible SSL-specific structure exists, but one or more validity or replication gates are weak."

NOT_SUPPORTED_EQUIVALENT:
  description: "SSL and baseline recover substantially equivalent reproducible structure."

NOT_SUPPORTED_NUISANCE:
  description: "Apparent SSL-specific structure is predominantly explained by speed, identity, context/session, or artifact."

NOT_SUPPORTED_UNSTABLE:
  description: "SSL-specific structure is not sufficiently stable or reproducible."

NOT_SUPPORTED_REPLICATION_FAILURE:
  description: "Primary structure does not obtain meaningful support in external replication."
```

The study does not require a `SUPPORTED` outcome to be considered successful research.

---

# 17. Sensitivity Analyses

## 17.1 Head-position extended baseline

**Category:** secondary sensitivity analysis

Reintroduce previously excluded head-position-derived features.

Purpose:

> Test whether the primary conclusion depends on excluding coordinate-sensitive position features.

This analysis must not replace the frozen 18-feature primary baseline.

---

## 17.2 SSL seed sensitivity

**Category:** confirmatory sensitivity analysis

Use all five frozen SSL seeds.

No new seeds are selected based on results.

---

## 17.3 Cluster-number sensitivity

**Category:** secondary sensitivity analysis

Evaluate neighboring cluster counts around the frozen primary selection where technically feasible.

Purpose:

> Determine whether the qualitative conclusion depends exclusively on one cluster-count choice.

The frozen primary cluster result remains primary.

---

## 17.4 Dimensionality-reduction visualization sensitivity

**Category:** exploratory / descriptive

PCA, UMAP, or t-SNE may be used for visualization.

These visualizations may not determine confirmatory clusters or thresholds.

---

## 17.5 Temporal / segmentation sensitivity

**Decision:** no alternate primary segmentation is required.

Rationale:

DS-005 already supplies a natural valid-bout unit with fixed 175-sample temporal sequences.

Optional alternate segmentation analyses may be performed only as exploratory or secondary analyses and must not redefine the primary unit.

---

# 18. DS-006 External Replication Evaluation

DS-006 remains external replication only.

## Replication TRAIN / VALIDATION phase

Allowed:

- apply frozen representation definitions;
- apply frozen model family;
- evaluate stability;
- evaluate nuisance dependence;
- perform replication-side analysis using TRAIN / VALIDATION only.

Not allowed:

- change the primary DS-005 architecture;
- change the primary objective;
- change the primary seed set;
- change the primary Input A definition;
- change the primary metric definitions based on replication results.

## Replication TEST

The DS-006 TEST partition was opened exactly once after the replication-side
procedure was frozen. Its final evaluation is complete; this historical rule
remains the governing justification for that access.

Primary replication emphasis:

```text
Does the qualitative baseline-vs-SSL conclusion reproduce?
```

Exact cluster identity or exact cluster occupancy need not match DS-005.

---

# 19. Reporting Requirements

For every primary metric report:

- partition used;
- representation;
- seed where applicable;
- number of fish;
- number of bouts;
- point estimate;
- uncertainty interval where applicable.

For nuisance analyses report both Input A and Input B whenever feasible.

This prevents judging SSL leakage without a matched baseline reference.

---

# 20. Prohibited Post-Hoc Actions

After TEST inspection, do not:

- alter cluster count;
- change SSL seed set;
- remove unfavorable seeds;
- invent a new nuisance metric;
- alter speed definition;
- introduce new QC exclusions;
- switch embedding layer;
- change dimensionality reduction;
- redefine the claim threshold;
- redefine success based on whichever metric looks strongest.

Any such analysis must be labeled post hoc.

---

# 21. Implementation Targets

Recommended implementation files:

```text
src/evaluation/reproducibility.py
src/evaluation/speed_controls.py
src/evaluation/nuisance_prediction.py
src/evaluation/tracking_artifacts.py
src/evaluation/compare_representations.py
src/evaluation/bootstrap.py
```

Recommended orchestration entry point:

```text
src/evaluation/run_confirmatory_evaluation.py
```

The orchestration script should refuse to load TEST unless explicitly run in the final-evaluation mode.

---

# 22. Current Evaluation Freeze Summary

```yaml
cross_fish_reproducibility:
  status: FROZEN
  metrics:
    - fish_bootstrap_ARI
    - held_out_cluster_occupancy
    - cross_seed_ARI

speed_control:
  status: FROZEN
  procedures:
    - speed_only_clustering
    - embedding_to_speed_ridge
    - cluster_speed_enrichment

tracking_quality:
  status: FROZEN
  new_post_clustering_exclusions: false
  policy: "report artifact enrichment rather than removing clusters post hoc"

nuisance_models:
  status: FROZEN
  classification: "L2 multinomial logistic regression, C=1.0"
  continuous: "Ridge(alpha=1.0)"

primary_metric_set:
  status: FROZEN

sensitivity_analyses:
  head_position_extended_baseline: SECONDARY
  ssl_seed_sensitivity: CONFIRMATORY
  cluster_number_sensitivity: SECONDARY
  visualization_dimensionality_reduction: EXPLORATORY
  alternate_segmentation: NOT_PRIMARY

test_partitions:
  DS005: PROTECTED_UNOPENED
  DS006: OPENED_ONCE_FINAL_EVALUATION_COMPLETE
```

---

# 23. Preregistration Consequence

With this protocol frozen, the following previously open preregistration items can be marked complete:

```markdown
- [x] Freeze cross-fish reproducibility metric/procedure.
- [x] Freeze speed-control metric/procedure.
- [x] Freeze exact tracking-quality rule.
- [x] Freeze nuisance prediction model specifications.
- [x] Freeze the complete primary confirmatory metric set and associated decision rules.
- [x] Document planned sensitivity analyses.
```

The remaining preregistration readiness checks should focus on:

```text
1. confirming docs/charter.md is stable;
2. preserving the completed five-seed TRAIN / VALIDATION artifacts;
3. committing or otherwise immutably timestamping docs/validation-freeze.md;
4. keeping DS-005 TEST protected until that record is fixed.
```
