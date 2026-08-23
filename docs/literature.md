# Literature Review

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

## Document Status

```yaml
document_status: ACTIVE
review_type: structured_scoping_synthesis
last_updated: 2026-08-23
research_gap_status: FROZEN
baseline_status: FROZEN
ssl_method_status: FROZEN
primary_dataset_status: FROZEN
evaluation_protocol_status: FROZEN
evidence_matrix_status: EXPANDED
```

---

# 1. Purpose

This document synthesizes the literature relevant to the frozen research question:

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

The review now serves four purposes:

1. establish the zebrafish-specific novelty boundary;
2. justify the conventional Input A baseline;
3. justify the temporal self-supervised Input B family;
4. trace the project's validation and discovery choices to prior methodological evidence.

This is a **structured scoping synthesis**, not an exhaustive systematic review. The paper-level screening audit is maintained separately so the project does not overstate search completeness.

Related files:

```text
docs/literature/search-log.csv
docs/literature/screening-ledger.csv
docs/literature/screening-protocol.md
docs/literature/literature-matrix.csv
docs/literature-protocol.md
docs/dataset-register.md
docs/research-question.md
docs/preregistration-draft.md
docs/evaluation-protocol.md
```

---

# 2. Executive Summary

The literature establishes that all of the following are already prior art:

- automated zebrafish tracking;
- conventional locomotor quantification;
- zebrafish pose estimation;
- machine-learning-based zebrafish behavioral feature analysis;
- unsupervised zebrafish behavioral-state/repertoire discovery;
- deep unsupervised temporal behavioral representation learning in animals;
- self-supervised animal pose/video representation learning;
- direct zebrafish self-supervised sequence representation learning.

The project therefore does **not** claim novelty from SSL, clustering, zebrafish behavior analysis, pose tracking, or unsupervised discovery individually.

The frozen contribution is narrower:

> **Compare conventional hand-engineered zebrafish behavioral representations against self-supervised temporal representations on matched observations, then test whether SSL-specific structure is reproducible across held-out fish and survives explicit controls for speed, fish identity, context/session, tracking artifacts, seed instability, and independent replication.**

The expanded evidence matrix now maps **18 studies** spanning zebrafish prior art, direct zebrafish SSL, general animal-behavior SSL, unsupervised behavioral discovery, pose tracking, temporal modeling, and discovery-method comparisons.

---

# 3. Evidence-Map Scope

The evidence matrix is intentionally broader than the original approximately 10-study key-prior-art table.

It includes studies that materially inform one or more of:

```text
zebrafish conventional behavioral measurement
zebrafish unsupervised discovery
zebrafish temporal sequence modeling
zebrafish SSL
pose/trajectory SSL
animal behavioral video SSL
unsupervised pose-state discovery
tracking and pose infrastructure
identity/tracking-QC concerns
temporal segmentation
state-number selection
cross-subject generalization
method sensitivity
```

The matrix should not be described as exhaustive until the screening ledger contains the complete set of screened title/abstract and full-text decisions.

---

# 4. Zebrafish Conventional Behavioral Analysis

## Barreiros et al. (2021)

Barreiros et al. developed an automated zebrafish conditioning and behavioral-analysis system combining fish detection, tracking, experimental control, and interpretable movement measurements.

Relevant conventional descriptors include:

- distance traveled;
- swimming speed;
- movement trajectory;
- spatial response;
- group polarization.

### Design consequence

A fair Input A cannot be a speed-only straw baseline. Conventional zebrafish analysis already supports multiple locomotor and directional descriptors.

---

## Yang et al. (2021)

Yang et al. used three-dimensional trajectory information, dimensionality reduction, and Fuzzy Adaptive Resonance Theory clustering to identify zebrafish behavioral features associated with experimental stimulation.

### Design consequence

The project cannot claim that machine learning or unsupervised behavioral feature recognition is new in zebrafish.

Yang et al. also provides a precedent for:

```text
trajectory-derived features
→ dimensionality reduction
→ unsupervised discovery
```

which closely motivates the conventional baseline side of the current comparison.

---

# 5. Zebrafish Unsupervised Behavioral Discovery

## Marques et al. (2018)

Marques et al. developed an unsupervised clustering framework for larval zebrafish swim bouts and identified **13 basic swim types** from millions of naturally segmented bouts.

This study is a major novelty boundary because it establishes:

- naturally segmented zebrafish bouts as a meaningful behavioral unit;
- high-throughput unsupervised repertoire discovery;
- hierarchical organization of movement;
- repeated use of bout types across different behavioral contexts.

### Design consequence

The present project must not claim that unsupervised discovery of zebrafish behavioral states is new.

The scientifically relevant question is whether **representation strategy** changes the reproducible structure recovered from the same observations.

---

## Ghosh & Rihel (2020)

Ghosh & Rihel used unsupervised learning to identify active and inactive behavioral modules and then examined hierarchical temporal structure across scales extending from sub-second events toward much longer behavioral organization.

### Design consequence

There is no single universally correct temporal scale for zebrafish behavior.

This supports using a biologically natural bout representation when the dataset already supplies well-defined bouts rather than imposing arbitrary fixed windows.

---

## Johnson et al. (2020)

Johnson et al. modeled naturalistic larval zebrafish behavioral sequences across exploration and hunting. Their pipeline categorized naturally segmented bouts and modeled action selection using probabilistic sequence models.

A bout was represented using a short postural sequence, providing direct precedent for sub-second temporal representations.

### Design consequence

Bout-level temporal structure is scientifically defensible, and behavioral history can contain information not reducible to static kinematic summaries.

---

# 6. Zebrafish Pose and Tracking Resources

## Scholz et al. (2025)

Scholz et al. provide annotated larval zebrafish data and pretrained DeepLabCut/SLEAP networks using a **15-keypoint pose schema**.

This establishes that detailed zebrafish pose variables such as orientation, tail curvature, and body geometry can be extracted automatically.

### Design consequence

SSL must be compared with a serious conventional movement representation rather than speed alone.

For DS-005, the frozen baseline uses the variables that can be computed reliably from that dataset rather than copying the exact Scholz keypoint schema.

---

## Deligkaris et al. (2026)

Deligkaris et al. provide large-scale adult dyadic zebrafish recordings with:

- three-dimensional tracking;
- persistent fish identity;
- anatomical landmarks;
- long recordings;
- explicit discussion of residual identity swaps and landmark errors.

### Design consequence

Tracking errors and identity information are plausible sources of false behavioral structure and therefore must be treated as nuisance variables.

The dataset remains a future social extension rather than the current primary dataset.

---

## DeepLabCut and SLEAP

Mathis et al. (2018) and Pereira et al. (2022) establish mature markerless pose-estimation infrastructure.

Their role in this project is methodological rather than a novelty claim:

```text
video
→ pose / identity tracking
→ behavioral representation
```

Tracking output is measurement data, not automatically a behavioral state.

---

# 7. Direct Zebrafish SSL

## Xu & Wang (2024)

The targeted search identified direct zebrafish SSL precedent in a technical report on **masked skeleton-sequence modeling for larval zebrafish behavioral embeddings**.

The study uses a spatial-temporal sequence architecture to learn latent representations from zebrafish skeleton sequences.

### Novelty consequence

The project must not claim:

> "This is the first application of self-supervised learning to zebrafish behavior."

Direct zebrafish SSL precedent exists.

The gap must therefore depend on the matched comparison and validation framework.

---

## McKenzie-Smith et al. (2024)

Zebrafish-adjacent temporal contrastive precedent also exists in work involving biological and artificial fish, where temporal contrastive signals are used in the artificial-agent learning framework.

### Design consequence

Temporal contrastive learning is not being presented as a new objective. It is used as a simple established family suitable for the frozen Input B experiment.

---

# 8. Animal-Behavior SSL Precedent

## ContrastivePose — Zhou et al. (2023)

ContrastivePose learns self-supervised pose representations using contrastive learning and compares learned features with handcrafted features for downstream behavioral classification.

### Novelty consequence

A broad claim that "SSL has never been compared with handcrafted animal behavioral features" would be incorrect.

### Remaining distinction

ContrastivePose is primarily a **downstream classification** precedent.

The present study instead asks whether learned representations produce additional **unsupervised behavioral organization** under explicit reproducibility and nuisance controls.

---

## BEAST

BEAST provides modern behavioral-video SSL precedent using masked autoencoding and temporal contrastive learning.

Its importance to the current project is methodological:

- temporal SSL is established in behavioral analysis;
- masked and contrastive objectives are credible choices;
- video SSL itself is not a novelty claim.

The current project intentionally uses a simpler sequence-based approach on matched behavioral inputs.

---

# 9. Broader Unsupervised Computational Ethology

## MotionMapper — Berman et al. (2014)

Berman et al. mapped stereotyped behavior from postural dynamics in freely moving flies using unsupervised behavioral-space construction.

### Relevance

This is foundational evidence that unsupervised behavioral maps from movement/posture predate modern SSL.

UMAP/t-SNE-like visual islands alone are therefore not sufficient evidence of behavioral discovery.

---

## B-SOiD — Hsu & Yttri (2021)

B-SOiD combines pose-derived spatiotemporal features, nonlinear dimensionality reduction/clustering, and fast supervised assignment of discovered states.

It also demonstrates cross-subject/cross-setup generalization analyses.

### Relevance

This supports:

- evaluating discovery beyond internal clustering scores;
- examining generalization across subjects;
- considering how pose variability affects clustering.

---

## VAME — Luxem et al. (2022)

VAME learns a deep temporal latent representation from pose time series and then discovers behavioral structure from that latent space.

### Novelty consequence

Deep temporal representation learning followed by unsupervised state discovery is established in computational ethology.

The current contribution therefore depends on the controlled **Input A versus Input B** comparison in zebrafish, not on the existence of a learned latent space.

---

## Keypoint-MoSeq — Weinreb et al. (2024)

Keypoint-MoSeq links pose dynamics to discrete behavioral syllables using a temporal generative model.

Its validation includes comparison with VAME, B-SOiD, and MotionMapper and attention to behavioral boundaries, kinematic changes, temporal scale, and keypoint noise.

### Design consequence

Different discovery algorithms can yield materially different temporal segmentations.

For the current study:

- the primary clustering family remains frozen;
- alternate state-number/discovery choices remain sensitivity analyses;
- temporal-boundary and artifact interpretation should not rely on visualization alone.

---

## Mlost et al. (2025)

A comparative evaluation of B-SOiD, BFA, VAME, and Keypoint-MoSeq found that unsupervised pose-analysis tools differ substantially in architecture and output characteristics.

### Design consequence

There is no universally privileged clustering/state-discovery algorithm.

This strengthens the rationale for preregistering one primary discovery procedure and limiting alternative algorithms to declared sensitivities rather than selecting whichever produces the most attractive result.

---

# 10. What Input A Should Represent

The literature supports a multi-family conventional baseline.

The **frozen DS-005 primary Input A** is an 18-feature bout representation:

```yaml
timing:
  - bout_duration_s
  - inter_bout_interval_s

speed:
  - speed_mean
  - speed_std
  - speed_median
  - speed_max
  - speed_p95
  - speed_rms

acceleration_speed_change:
  - accel_abs_mean
  - accel_abs_std
  - accel_abs_max
  - accel_rms

orientation_turning:
  - turn_abs_total_rad
  - turn_net_rad
  - turn_abs_mean_rad
  - turn_abs_std_rad
  - turn_abs_max_rad
  - turn_rms_rad
```

`head_pos` path/jump features are excluded from the primary representation because of observed coordinate-semantic discontinuities and are reserved for secondary sensitivity analysis.

This feature set is narrower than every possible conventional zebrafish descriptor, but it is a serious, reproducible conventional baseline supported by the available DS-005 measurements.

---

# 11. Frozen Input B

The primary SSL input is frozen as:

```text
shape: (175, 3)

channel 0 = sin(orientation_smooth)
channel 1 = cos(orientation_smooth)
channel 2 = speed_head
```

The downstream representation is a **64-dimensional encoder embedding** learned using a temporal contrastive / NT-Xent objective.

Frozen seed set:

```text
11, 23, 37, 51, 79
```

The model receives no fish ID, context label, stimulus code, bout label, or partition label.

### Literature rationale

This matched-input design makes the comparison cleaner:

```text
same underlying movement signal
        │
        ├── hand-engineered summaries
        └── learned temporal representation
```

Any difference is therefore less easily attributed to simply giving SSL richer sensors or raw appearance information.

---

# 12. Discovery Literature and Frozen Project Choice

The broader literature supports multiple discovery families:

- k-means;
- Gaussian mixtures;
- density-based clustering;
- hierarchical clustering;
- HMM/AR-HMM state models;
- nonparametric state models;
- latent-space clustering.

The project is **not** attempting to identify the universally best discovery algorithm.

For the frozen conventional baseline, TRAIN/VALIDATION model selection selected:

```text
PCA(6) → GMM(k=2, seed=20260822)
```

The selected `k=2` is interpreted only as the best evaluated configuration under the preregistered selection procedure.

It is **not** interpreted as the true number of biological zebrafish behaviors.

---

# 13. Validation Evidence and Threats to Validity

The literature map reinforces that unsupervised structure can reflect nuisance variables rather than behavior.

The frozen project battery therefore explicitly evaluates:

```text
cross-fish reproducibility
fish identity
context/session information
locomotor speed
tracking/QC proxies
seed stability
baseline recoverability
external replication
```

---

## 13.1 Fish Identity

Animals differ in morphology, baseline activity, and movement style.

Therefore the primary data split is by fish rather than by randomly sampled bouts.

The project also includes an explicit fish-ID nuisance probe.

---

## 13.2 Context / Session

Behavioral features may encode experimental context.

Context/session prediction is therefore treated as a nuisance analysis rather than automatically interpreted as behavior.

---

## 13.3 Speed

Speed is a dominant conventional zebrafish descriptor and a plausible shortcut for learned embeddings.

Frozen controls include:

- speed-only clustering;
- speed-only versus SSL ARI/NMI;
- ridge regression from SSL embeddings to mean speed;
- cluster speed distributions.

---

## 13.4 Tracking Artifacts

Tracking errors can themselves form clusters.

The project therefore reports artifact/QC-proxy enrichment and prohibits inventing new post-clustering exclusion rules after seeing cluster structure.

---

## 13.5 Seed and Method Instability

Modern discovery methods can vary across initialization and state-number choice.

The project therefore reports all five frozen SSL seeds and treats cluster-number alternatives as declared sensitivity analyses.

---

# 14. Expanded Evidence Matrix Summary

The full machine-readable matrix is:

```text
docs/literature/literature-matrix.csv
```

Current expanded map:

| ID | Study | Main evidence role |
|---|---|---|
| ZF-001 | Barreiros et al. 2021 | Conventional zebrafish tracking/behavioral features |
| ZF-002 | Yang et al. 2021 | Zebrafish ML + unsupervised trajectory prior art |
| POSE-001 | Scholz et al. 2025 | Zebrafish pose resource |
| DATA-001 | Deligkaris et al. 2026 | Long-form 3D identity-aware zebrafish data/QC |
| SSL-001 | ContrastivePose 2023 | Pose SSL + handcrafted-feature comparison |
| SSL-002 | Xu & Wang 2024 | Direct zebrafish SSL |
| SSL-003 | BEAST | Behavioral-video SSL + temporal contrastive precedent |
| UBD-001 | Marques et al. 2018 | Zebrafish unsupervised swim repertoire |
| UBD-002 | Ghosh & Rihel 2020 | Zebrafish temporal hierarchy / bout modules |
| SSL-004 | McKenzie-Smith et al. 2024 | Zebrafish-adjacent temporal contrastive precedent |
| UBD-003 | Johnson et al. 2020 | Bout sequence / probabilistic temporal modeling |
| METHOD-001 | Berman et al. 2014 | MotionMapper / foundational unsupervised ethology |
| METHOD-002 | Hsu & Yttri 2021 | B-SOiD / pose-state discovery |
| METHOD-003 | Luxem et al. 2022 | VAME / deep temporal latent-state discovery |
| METHOD-004 | Weinreb et al. 2024 | Keypoint-MoSeq / temporal state discovery and validation |
| TRACK-001 | Mathis et al. 2018 | DeepLabCut markerless pose infrastructure |
| TRACK-002 | Pereira et al. 2022 | SLEAP multi-animal pose/identity infrastructure |
| REVIEW-001 | Mlost et al. 2025 | Comparative evaluation of unsupervised pose methods |

This matrix is now broad enough to function as a structured evidence map of the project's major methodological streams, while the screening ledger continues to determine whether the review can later be described as exhaustive.

---

# 15. Literature-to-Design Traceability

| Literature finding | Project consequence |
|---|---|
| Zebrafish conventional analyses use locomotor and directional measures | Input A must be a serious multi-feature baseline |
| Zebrafish unsupervised repertoire discovery already exists | Do not claim novelty from clustering/discovery alone |
| Natural zebrafish bouts are established behavioral units | DS-005 natural bout is defensible as the primary unit |
| Direct zebrafish SSL exists | Do not claim first zebrafish SSL |
| Animal pose SSL has been compared with handcrafted features | Novelty cannot be generic SSL-vs-handcrafted comparison |
| Deep temporal latent-state discovery exists in animal behavior | Novelty depends on matched zebrafish comparison + validation |
| Discovery methods produce different state structures | Freeze primary discovery method; limit sensitivities |
| Tracking/identity errors can survive preprocessing | Explicit QC and identity nuisance controls are required |
| Temporal behavior spans multiple scales | Avoid claiming one universal biological window |
| Subject/cross-setup generalization is methodologically important | Use held-out fish and explicit nuisance tests |

---

# 16. Frozen Research Gap

The literature search now supports the following bounded gap:

> **Prior work establishes unsupervised zebrafish behavioral discovery, direct zebrafish self-supervised representation learning, and broader self-supervised and deep unsupervised representation learning for animal behavior. The targeted literature review did not identify a study directly comparing a conventional hand-engineered zebrafish behavioral representation against a self-supervised temporal representation on matched observations under the same unsupervised discovery framework while simultaneously evaluating held-out-fish reproducibility, fish-identity leakage, context/session effects, locomotor speed dependence, tracking artifacts, representation stability, and independent replication.**

This is intentionally narrower than:

> "No previous study has used SSL for zebrafish."

That broader statement is unsupported.

---

# 17. Claim Boundary

The target claim remains:

> **Self-supervised temporal representations reveal reproducible behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose features.**

The phrase **evaluated hand-engineered feature set** matters.

The study cannot establish that no conceivable manually designed feature could recover the same information.

---

# 18. Valid Scientific Outcomes

The literature supports treating all of the following as legitimate outcomes.

### SSL adds reproducible structure

Supported only if it survives the frozen reproducibility and nuisance controls.

### SSL reconstructs the baseline

This would indicate that the conventional features already capture most of the dominant organization.

### SSL is largely speed-driven

This would weaken an interpretation of additional behavioral complexity.

### SSL is identity/context/artifact driven

This would indicate nuisance structure rather than robust behavioral discovery.

### SSL structure is unstable

This would indicate that the representation/discovery combination is not sufficiently reproducible.

### External replication fails

This would narrow or reject generalization of the primary finding.

---

# 19. Dataset Literature Roles

```yaml
DS-005:
  role: PRIMARY
  status: FROZEN

DS-006:
  role: EXTERNAL_REPLICATION
  status: PREPROCESSING_COMPLETE

DS-002:
  role: FUTURE_SOCIAL_EXTENSION
  license: CC_BY_4.0

DS-003:
  role:
    - PRIOR_ART
    - BASELINE_REFERENCE

DS-004:
  role:
    - PRIOR_ART
    - CONVENTIONAL_ANALYSIS_REFERENCE

DS-001:
  role:
    - POSE_RESOURCE
  status: DEFERRED
```

---

# 20. Current Method Status

```yaml
primary_unit:
  value: one_valid_bout_per_identifiable_fish
  status: FROZEN

input_a:
  features: 18
  status: FROZEN

input_b:
  shape: [175, 3]
  embedding_dim: 64
  status: FROZEN

ssl:
  family: temporal_contrastive
  loss: NT_Xent
  temperature: 0.10
  seeds: [11, 23, 37, 51, 79]
  status: FROZEN

baseline_discovery:
  dimensionality_reduction: PCA_6
  clustering: GMM_k2
  seed: 20260822
  status: FROZEN

evaluation:
  status: FROZEN
```

---

# 21. Literature Phase Status

The major design-changing literature questions are now resolved sufficiently for preregistration.

```yaml
direct_zebrafish_ssl:
  resolved: true
  finding: direct precedent exists

zebrafish_unsupervised_discovery:
  resolved: true
  finding: strong prior art exists

temporal_scale:
  resolved: true
  finding: both natural bouts and fixed windows have precedent

handcrafted_baseline:
  resolved: true
  finding: multi-family locomotion/turning/bout descriptors justified

animal_behavior_ssl:
  resolved: true
  finding: strong contrastive/masked/deep representation precedent exists

discovery_method_sensitivity:
  resolved: true
  finding: methods can yield different motif/state organizations

research_gap:
  status: FROZEN
```

The remaining literature work is maintenance rather than open-ended method selection:

- add newly screened papers to the ledger and matrix;
- keep citations synchronized with the bibliography;
- update the novelty statement only if genuinely closer prior art is found;
- preserve retrospective versus prospective screening status.

---

# 22. Core References

- Barreiros, M. O., et al. (2021). *Zebrafish automatic monitoring system for conditioning and behavioral analysis.* Scientific Reports. `10.1038/s41598-021-87502-6`.
- Yang, P., et al. (2021). *Zebrafish behavior feature recognition using three-dimensional tracking and machine learning.* Scientific Reports. `10.1038/s41598-021-92854-0`.
- Marques, J. C., Lackner, S., Félix, R., & Orger, M. B. (2018). *Structure of the Zebrafish Locomotor Repertoire Revealed with Unsupervised Behavioral Clustering.* Current Biology. `10.1016/j.cub.2017.12.002`.
- Ghosh, M., & Rihel, J. (2020). *Hierarchical Compression Reveals Sub-Second to Day-Long Structure in Larval Zebrafish Behavior.* eNeuro. `10.1523/ENEURO.0408-19.2020`.
- Johnson, R. E., et al. (2020). *Probabilistic Models of Larval Zebrafish Behavior Reveal Structure on Many Scales.* Current Biology. `10.1016/j.cub.2019.11.026`.
- Scholz, L. A., et al. (2025). *Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses.* bioRxiv. `10.1101/2025.06.04.657938`.
- Deligkaris, K., et al. (2026). *A dataset of fine-grained zebrafish interactions in health and disease.* Scientific Data. `10.1038/s41597-026-06953-6`.
- Zhou, T., et al. (2023). *ContrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavioral classification of interacting animals.* Computers in Biology and Medicine. `10.1016/j.compbiomed.2023.107416`.
- Berman, G. J., et al. (2014). *Mapping the stereotyped behaviour of freely moving fruit flies.* Journal of the Royal Society Interface. `10.1098/rsif.2014.0672`.
- Hsu, A. I., & Yttri, E. A. (2021). *B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors.* Nature Communications. `10.1038/s41467-021-25420-x`.
- Luxem, K., et al. (2022). *Identifying behavioral structure from deep variational embeddings of animal motion.* Communications Biology. `10.1038/s42003-022-04080-7`.
- Weinreb, C., et al. (2024). *Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics.* Nature Methods. `10.1038/s41592-024-02318-2`.
- Mathis, A., et al. (2018). *DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.* Nature Neuroscience. `10.1038/s41593-018-0209-y`.
- Pereira, T. D., et al. (2022). *SLEAP: A deep learning system for multi-animal pose tracking.* Nature Methods. `10.1038/s41592-022-01426-1`.
- Mlost, J., et al. (2025). *Evaluation of unsupervised learning algorithms for the classification of behavior from pose estimation data.* Patterns. `10.1016/j.patter.2025.101237`.

Direct zebrafish SSL, BEAST, and McKenzie-Smith entries are retained in the evidence matrix with the project bibliography identifiers currently used by the repository and should remain synchronized with `papers.bib` or the eventual formal reference manager export.

---

# 23. Final Literature Conclusion

The expanded literature map strengthens rather than weakens the project's current scope.

There is substantial prior art for:

```text
zebrafish behavioral clustering
zebrafish temporal sequence modeling
pose tracking
deep latent behavioral modeling
self-supervised representation learning
pose-based behavioral discovery
```

The remaining contribution is therefore methodological and comparative:

> **Under a preregistered matched-input design, determine whether self-supervised temporal zebrafish representations contain reproducible behavioral organization beyond a strong conventional representation, and attempt to falsify that interpretation through held-out-animal, speed, identity, context, artifact, stability, and replication controls.**

A negative result remains scientifically informative.

---

# Control-Extraction Completion

The evidence matrix now uses standardized values for the five nuisance/validation fields:

```text
YES = explicitly reported or implemented in the paper
NR  = not reported / not identifiable from the reported methods
NA  = not applicable to the study design
NO  = explicitly reported as not performed or explicitly absent
```

`NR` is preferred over guessing that a control was absent.

The previously ambiguous fields were re-checked for six high-priority studies:

| Study | Held-out animals | Identity control | Session control | Speed control | Tracking QC |
|---|---:|---:|---:|---:|---:|
| Yang et al. 2021 | NR | NR | NR | NR | YES |
| ContrastivePose 2023 | NR | NR | NR | NR | NR |
| Xu & Wang 2024 | NR | NR | NR | NR | NR |
| BEAST | YES | NR | YES | NR | YES |
| Marques et al. 2018 | NR | NR | NR | NR | NR |
| Ghosh & Rihel 2020 | NR | NR | NR | NR | YES |

Interpretive notes:

- **Yang et al.** explicitly performs tracking cleanup for missing samples, reflection clusters, rogue points, smoothing, and interpolation, but does not report the project-style held-out-fish or nuisance-leakage controls.
- **ContrastivePose** establishes self-supervised pose-feature learning and handcrafted-feature comparison, but the paper does not report the present project's identity/session/speed/QC control battery.
- **Xu & Wang** reports validation-set reconstruction, but does not specify a fish-level held-out split and does not report the nuisance controls above.
- **BEAST** has the strongest positive validation precedent among these rows: it evaluates held-out animals for action segmentation and entirely held-out videos for pose estimation. It also evaluates pose/keypoint quality. It does not report an explicit identity-prediction or speed-dependence test.
- **Marques et al.** establishes robust unsupervised zebrafish repertoire discovery, but these project-style nuisance controls were not reported.
- **Ghosh & Rihel** documents extensive acquisition/data-quality exclusions and artifact handling, but does not report a fish-identity, held-out-animal, session-leakage, or speed-matching control.

This reinforces the research-gap framing: several component practices exist in prior work, but the full combined held-out-fish + identity + context/session + speed + artifact-control battery remains a distinctive part of the present study design.

