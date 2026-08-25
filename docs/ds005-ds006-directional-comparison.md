# DS-005 vs DS-006 Directional Replication Comparison

## Scope

This comparison evaluates whether the **direction and scientific interpretation**
of the frozen DS-005 findings reproduce in DS-006.

It does **not** use individual p-values as the replication criterion.

**Current TEST status:** DS-005 and DS-006 TEST were each opened exactly once
under their respective frozen procedures; both final evaluations are complete.

DS-006 does not provide the same conventional bout-class labels as DS-005
(`Long_CS`, `LLC`, etc.), so those class-specific comparisons are treated as
kinematic analogues rather than direct within-class replications.

---

## Frozen TRAIN/VALIDATION Comparison

| Claim | DS-005 | DS-006 | Status | Interpretation |
|---|---|---|---|---|
| Moderately reproducible `k=8` organization across encoder seeds | VAL ARI = 0.358; NMI = 0.460; aligned agreement = 0.565; silhouette = 0.127 | VAL ARI = 0.361; NMI = 0.502; aligned agreement = 0.537; silhouette = 0.161 | **REPLICATED** | Cross-seed clustering stability is extremely similar in direction and magnitude. |
| Speed-related but not reducible to mean speed | VAL eta² = 0.458; speed-only balanced accuracy = 0.295; macro-F1 = 0.236 | VAL eta² = 0.545; speed-only balanced accuracy = 0.276; macro-F1 = 0.218 | **REPLICATED** | Both datasets show strong speed association but weak mean-speed-only recovery. |
| Low subject/fish identity leakage | NMI = 0.032; AMI = 0.032; Cramer's V = 0.184 | fish-well NMI ≈ 0.032; AMI ≈ 0.028; Cramer's V ≈ 0.163 | **REPLICATED** | DS-006 fish-well associations remain low and comparable to DS-005. |
| Low recording/context leakage | NMI = 0.032; AMI = 0.031; Cramer's V = 0.151 | maximum tested context NMI ≈ 0.018 | **REPLICATED** | All tested DS-006 context fields show very low held-out associations. |
| Acceleration / speed-change heterogeneity | Long_CS accel_rms eta² = 0.539; accel_abs_std eta² = 0.533 | speed_change_rms eta² ≈ 0.476; speed_change_std eta² ≈ 0.476 with strong TRAIN→VAL profile reproducibility | **REPLICATED** | The exact Long_CS label is unavailable, but the acceleration/speed-change axis generalizes strongly. |
| Coarse handcrafted baseline clustering differs from SSL | VAL ARI = 0.018; NMI = 0.047 | VAL ARI = 0.059; NMI = 0.120; AMI = 0.120 | **REPLICATED** | In both datasets, the frozen coarse handcrafted clustering is substantially different from the SSL organization. |
| Most SSL structure is nonlinearly recoverable from the 18 handcrafted features | nonlinear VAL balanced accuracy = 0.902; macro-F1 = 0.902 | nonlinear VAL balanced accuracy = 0.412; macro-F1 = 0.389 | **NOT_STRONGLY_REPLICATED** | DS-006 recovery is above chance but far below DS-005, so strong nonlinear recoverability is not a cross-dataset result. |
| Linear recoverability from the 18 handcrafted features | VAL balanced accuracy = 0.482; macro-F1 = 0.458 | VAL balanced accuracy = 0.377; macro-F1 = 0.345 | **PARTIAL** | Handcrafted features retain predictive information in both datasets, but the DS-006 relationship is weaker. |
| Long_CS-like duration heterogeneity | duration eta² = 0.526; TRAIN→VAL rho = 0.862 | duration eta² is much smaller, roughly 0.021–0.156 | **NOT_STRONGLY_REPLICATED** | DS-006 duration association is much weaker, and no direct Long_CS label exists. |
| LLC-like signed net-turning structure | turn_net eta² = 0.164; TRAIN→VAL rho = 0.952 | turn_net eta² ≈ 0.003–0.014 | **NOT_STRONGLY_REPLICATED** | Signed net turning is very weak in DS-006. |
| Turning magnitude/intensity as a reproducible axis | related turning structure present, but signed `turn_net` was the primary LLC result | turn_total_abs eta² ≈ 0.292; cross-seed profile rho ≈ 0.748 | **PARTIAL** | DS-006 shows robust turning magnitude, but this is not the same as reproducing signed turn direction. |
| Direct within-Long_CS / within-LLC subdivision replication | Available | Unavailable | **NOT_DIRECTLY_TESTABLE** | DS-006 lacks DS-005-equivalent conventional bout-class labels. |

---

## Updated Status Counts

Using the claim rows above:

```text
REPLICATED                 6
PARTIAL                    2
NOT_STRONGLY_REPLICATED    3
NOT_DIRECTLY_TESTABLE      1
```

The exact count is less important than preserving the status of each individual
claim.

---

## Baseline-versus-SSL Result

The new direct DS-006 comparison is important because it separates two
questions:

1. Does the frozen handcrafted **clustering** resemble SSL?
2. Does the full handcrafted **feature representation** contain enough
   information to recover SSL labels?

### Cluster-level comparison

DS-006:

```text
Mean VAL ARI                              0.058651
Mean VAL NMI                              0.120044
Mean VAL AMI                              0.119918
Mean H(SSL|baseline)/H(SSL)               0.923641
Mean H(baseline|SSL)/H(baseline)          0.719446
```

This supports the same broad result as DS-005:

> The coarse handcrafted clustering and SSL organization are substantially
> different.

### Feature-to-SSL probes

DS-006 linear probe:

```text
balanced accuracy   0.377456
macro F1            0.344832
```

DS-006 nonlinear probe:

```text
balanced accuracy   0.412155
macro F1            0.388654
```

Chance balanced accuracy:

```text
0.125000
```

For comparison, DS-005 nonlinear balanced accuracy was approximately:

```text
0.901642
```

Therefore:

> The strong DS-005 conclusion that most SSL cluster structure is recoverable
> nonlinearly from the 18 handcrafted features does not strongly generalize to
> DS-006.

The nonlinear improvement in DS-006 is modest:

```text
0.412155 - 0.377456 = 0.034699
```

Thus DS-006 does not support the simpler interpretation that the handcrafted
information is essentially all present and merely requires a nonlinear decoder.

---

## Overall Interpretation

DS-006 continues to support the broad representation-level conclusions:

- transfer embeddings remain healthy;
- frozen `k=8` organization is moderately reproducible across encoder seeds;
- speed dependence reproduces;
- mean-speed-only collapse is rejected;
- fish-well identity leakage remains low;
- recording/context leakage remains low;
- acceleration/speed-change structure reproduces.

The new baseline-versus-SSL comparison adds an important limitation:

> Strong nonlinear recoverability of SSL labels from the 18 handcrafted
> features is **dataset-specific rather than universally replicated**.

Fine-grained effects also remain mixed:

- duration does not reproduce strongly;
- signed net-turning does not reproduce strongly;
- turning magnitude provides a partial analogue.

The overall outcome is therefore a **mixed but informative replication**.

---

## Updated Frozen Claim Language

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

## Claim Restrictions

- Do not claim that all DS-005 fine-grained findings replicated.
- Do not describe DS-006 as a direct `Long_CS` or `LLC` replication.
- Do not claim that `k=8` represents eight universally fixed biological behaviors.
- Do not claim that SSL contains information wholly absent from handcrafted features.
- Do not claim that most SSL structure is recoverable from handcrafted features
  **across datasets**.
- The strong nonlinear-recoverability statement is limited to DS-005.
- Describe DS-006 identity analysis specifically as **fish-well identity leakage**.
- Preserve the weak duration and signed-turning replication results.

---

## TEST Status

```text
DS-006 TEST partition: OPENED ONCE / FINAL EVALUATION COMPLETE
DS-005 TEST partition: OPENED ONCE / FINAL EVALUATION COMPLETE
```

The comparison table above preserves the TRAIN/VALIDATION evidence frozen
before TEST access. The one-time DS-006 TEST evaluation used freeze commit
`575ead5403d0b2f721d143366b4d2e0014bd67ee`, performed no fitting or
configuration changes, and produced the following final claim assessment:

```text
SUPPORTED       11
WEAKENED         2
CONTRADICTED     0
NOT_TESTABLE     1
```

The speed-change/acceleration interpretation and weak-duration interpretation
were **WEAKENED**. All other directly testable frozen interpretations were
supported, including moderate cross-seed organization, speed dependence without
mean-speed-only collapse, low identity/context association, baseline-versus-SSL
disagreement, moderate handcrafted-feature recoverability, weak signed turning,
and the partial turning-magnitude analogue. Direct Long_CS/LLC label replication
remains **NOT_TESTABLE**. Full TEST metrics and hashes are recorded in
`docs/ds-006-replication.md` and
`data/processed/DS-006/final_test_evaluation/`.

DS-005 TEST was subsequently evaluated once from freeze commit
`d66aca763c76242edc719683a617c2511e8ec37b`. All eight directly testable frozen
DS-005 interpretations were `SUPPORTED`; the proposition that eight clusters
are eight distinct novel biological behaviors remains `NOT_TESTABLE`. Key TEST
results were mean cross-seed ARI `0.3606`, nonlinear handcrafted-feature probe
balanced accuracy `0.9036`, Long_CS eta-squared `0.5539`/`0.5354`/`0.5113` for
duration/acceleration RMS/acceleration absolute SD, and LLC turn-net
eta-squared `0.1627` with TRAIN-to-TEST Spearman `0.9714`. The authoritative
DS-005 output checksum-manifest SHA-256 is
`9695b4d0474f37ec1e380ad001684776bfc658c9d8b734433d7f4e95780c1305`.
