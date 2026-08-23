# Literature Review Protocol

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

## Document Status

```yaml
document_status: UPDATED_ACTIVE_PROTOCOL
review_type: structured_scoping_review
project_phase: preregistration_and_model_development
primary_question_status: FROZEN
primary_dataset_status: FROZEN
primary_dataset: DS-005
research_gap_status: FROZEN
ssl_method_status: FROZEN_V1
literature_search_status: SUFFICIENT_FOR_PREREGISTRATION
```

---

# 1. Purpose

This document defines the protocol for identifying, screening, reading, recording, and synthesizing literature relevant to the **Self-Supervised Discovery of Zebrafish Behavioral Structure** project.

The literature review is intended to answer four practical questions:

1. **What has already been done?**
2. **What methods should form the project's conventional baseline?**
3. **Which self-supervised and unsupervised approaches are appropriate for temporal animal behavior?**
4. **What evidence would be required to claim that learned representations recover behavioral structure beyond existing methods?**

The review is not intended to maximize the number of papers collected.

Its purpose is to establish:

- the novelty boundary,
- methodological precedent,
- baseline features,
- dataset candidates,
- representation-learning options,
- validation standards,
- and defensible research claims.

---

# 2. Relationship to Other Project Documents

This protocol informs:

```text
docs/charter.md
docs/dataset-register.md
docs/research-question.md
docs/preregistration-draft.md
```

Findings from the literature review should eventually be synthesized into:

```text
docs/literature.md
```

Individual papers may additionally be tracked in a structured bibliography or spreadsheet.

---

# 3. Primary Literature Question

The primary literature question is:

> **To what extent have self-supervised or unsupervised temporal representation-learning methods already been used to discover behavioral structure in zebrafish, and how have these methods been validated against conventional behavioral features?**

The review should identify whether the proposed contribution has already been performed directly or whether closely related work limits the novelty claim.

---

# 4. Secondary Literature Questions

## LQ1 — Conventional Zebrafish Behavior

> What locomotion, trajectory, pose, posture, and temporal features are commonly used to characterize zebrafish behavior?

---

## LQ2 — Zebrafish Unsupervised Discovery

> What unsupervised methods have previously been used to identify zebrafish behavioral states, motifs, clusters, or phenotypes?

---

## LQ3 — Self-Supervised Learning

> What self-supervised methods have been used to learn behavioral representations from zebrafish or other animal video/pose sequences?

---

## LQ4 — Temporal Representation Learning

> Which representation-learning objectives are most appropriate for short temporal windows of animal behavior?

---

## LQ5 — Validation

> How have previous studies demonstrated that discovered behavioral states are reproducible, biologically meaningful, or robust to nuisance variables?

---

## LQ6 — Identity Leakage

> How do animal-behavior representation-learning studies prevent models from encoding animal identity instead of behavior?

---

## LQ7 — Session Leakage

> How do relevant studies address camera, session, laboratory, background, or recording-condition confounds?

---

## LQ8 — Speed Dependence

> How do behavioral discovery studies demonstrate that learned states contain information beyond locomotor speed or movement intensity?

---

## LQ9 — Tracking Artifacts

> How are pose-estimation and tracking errors identified and prevented from becoming apparent behavioral states?

---

## LQ10 — Temporal Segmentation

> How do prior studies select behavioral-window durations, segmentation rules, or temporal scales?

---

# 5. Review Type

This project will use a:

> **Structured scoping review with targeted methodological deep dives.**

It is not currently intended to be a formal systematic review or meta-analysis.

The protocol nevertheless uses explicit:

- search concepts,
- eligibility rules,
- screening decisions,
- extraction fields,
- and stopping criteria

to reduce selective citation.

---

# 6. Literature Domains

The review is divided into six major domains.

```text
DOMAIN A
Zebrafish behavioral analysis

DOMAIN B
Zebrafish tracking and pose estimation

DOMAIN C
Unsupervised animal behavior discovery

DOMAIN D
Self-supervised animal behavior representation learning

DOMAIN E
General temporal/video self-supervised learning

DOMAIN F
Validation, reproducibility, and confound control
```

Each domain answers a different part of the research design.

---

# 7. Domain A — Zebrafish Behavioral Analysis

## Objective

Determine what constitutes a defensible conventional behavioral baseline.

The review should identify commonly used measures such as:

- speed,
- distance traveled,
- acceleration,
- immobility,
- movement bouts,
- turning,
- angular velocity,
- trajectory curvature,
- tank occupancy,
- edge preference,
- vertical position,
- posture,
- body curvature,
- tail movement.

---

## Questions to Extract

For each relevant study:

- What behavioral measurements were used?
- Were measurements calculated frame-by-frame or summarized over intervals?
- What interval/window durations were used?
- Were behaviors manually labeled?
- Were features used for classification or clustering?
- Were movement features normalized?
- Were spatial features included?
- Were pose features included?
- Were temporal dynamics represented explicitly?

---

# 8. Domain B — Zebrafish Tracking and Pose

## Objective

Determine which tracking or pose methods are sufficiently reliable for Input A and potentially Input B.

Important resources already considered include:

- Scholz et al. larval zebrafish pose/tracking work,
- DeepLabCut,
- SLEAP,
- StrIPETrack,
- automated zebrafish monitoring systems,
- three-dimensional tracking approaches.

---

## Questions to Extract

For each tracking paper:

- species/developmental stage,
- single-fish or multi-fish,
- 2D or 3D,
- number of keypoints,
- frame rate,
- tracking accuracy,
- confidence outputs,
- failure modes,
- handling of occlusion,
- identity tracking,
- public code,
- pretrained models,
- public training data.

---

# 9. Domain C — Unsupervised Animal Behavior Discovery

## Objective

Determine how behavioral structure has previously been discovered without predefined labels.

The search should include both zebrafish and other animals.

Relevant terms may include:

```text
behavioral clustering
behavioral states
behavioral motifs
behavioral syllables
behavioral segmentation
unsupervised behavior
behavior embedding
behavioral repertoire
computational ethology
```

---

## Important Method Families

Search for studies using:

- k-means,
- Gaussian mixture models,
- hierarchical clustering,
- HDBSCAN,
- spectral clustering,
- hidden Markov models,
- autoregressive state models,
- density estimation,
- manifold learning,
- behavioral motif discovery.

---

## Cross-Species Literature

Non-zebrafish studies should be included when they provide transferable methodological precedent.

Relevant animal domains may include:

- mice,
- flies,
- worms,
- primates,
- rodents,
- birds,
- other fish.

These papers are especially useful for representation learning and unsupervised behavioral discovery methodology.

---

# 10. Domain D — Self-Supervised Animal Behavior

## Objective

Identify methods that learn temporal behavioral representations without requiring human behavior labels.

Search concepts include:

```text
self-supervised animal behavior
self-supervised pose representation
self-supervised behavior representation
contrastive animal behavior
unsupervised pose embedding
temporal contrastive behavior
representation learning animal movement
computational ethology self-supervised
```

---

## Methods of Interest

Prioritize papers using:

- temporal contrastive learning,
- contrastive predictive coding,
- masked sequence modeling,
- future prediction,
- sequence reconstruction,
- temporal order prediction,
- augmentation invariance,
- teacher-student learning,
- multimodal self-supervision.

---

# 11. Domain E — General Temporal SSL

## Objective

Identify general self-supervised techniques that may transfer well to zebrafish behavioral data.

This domain may include:

- video SSL,
- time-series SSL,
- pose-sequence SSL,
- trajectory representation learning.

The goal is not to review all of self-supervised learning.

Papers should be included only when the method is plausibly applicable to the project's behavioral-window representation.

---

# 12. Domain F — Validation and Reproducibility

## Objective

Identify standards for determining whether unsupervised behavioral structure is meaningful rather than an artifact.

Priority topics:

- held-out subject validation,
- cross-session validation,
- cluster stability,
- representation stability,
- nuisance prediction,
- behavioral interpretation,
- independent replication,
- negative controls.

---

# 13. Search Sources

Search should prioritize scholarly databases and primary sources.

Preferred sources include:

- Google Scholar,
- PubMed,
- Semantic Scholar,
- Web of Science where accessible,
- Crossref,
- bioRxiv,
- arXiv,
- publisher databases,
- institutional repositories,
- Zenodo,
- Dryad,
- Figshare,
- OSF.

GitHub may be used to locate:

- official source code,
- pretrained models,
- dataset utilities.

GitHub repositories should not replace the associated scientific publication when one exists.

---

# 14. Search Date Recording

Every search session should record:

```yaml
search_date:
source:
query:
filters:
results_reviewed:
papers_retained:
notes:
```

This creates a reproducible trail of how the literature base developed.

---

# 15. Core Search Strings

The following searches should be run or adapted across databases.

## Search A — Zebrafish Unsupervised Behavior

```text
zebrafish AND
("unsupervised learning" OR clustering OR "behavioral states"
OR "behavioral motifs" OR "behavioral repertoire")
```

---

## Search B — Zebrafish Representation Learning

```text
zebrafish AND
("representation learning" OR "self-supervised learning"
OR embedding OR "contrastive learning")
AND behavior
```

---

## Search C — Zebrafish Pose

```text
zebrafish AND
("pose estimation" OR tracking OR keypoints)
AND behavior
```

---

## Search D — Animal Self-Supervised Behavior

```text
("animal behavior" OR "computational ethology")
AND
("self-supervised learning" OR "representation learning"
OR "contrastive learning")
```

---

## Search E — Animal Unsupervised Discovery

```text
("animal behavior" OR "computational ethology")
AND
(unsupervised OR clustering)
AND
("behavioral states" OR motifs OR repertoire)
```

---

## Search F — Temporal Pose Representation

```text
("pose sequence" OR "pose trajectories")
AND
("self-supervised" OR contrastive OR "representation learning")
```

---

## Search G — Identity Leakage

```text
("animal behavior" OR video OR pose)
AND
("identity leakage" OR "subject identity" OR "individual identity")
AND
("representation learning" OR clustering)
```

---

## Search H — Session Confounds

```text
("representation learning" OR clustering)
AND
("session effects" OR "camera effects" OR "domain shift")
AND behavior
```

---

## Search I — Behavioral Speed Confound

```text
("behavioral clustering" OR "behavior representation")
AND
(speed OR locomotion)
AND animal
```

---

# 16. Snowball Search Strategy

For highly relevant papers, perform:

## Backward Search

Review references for:

- foundational methods,
- prior zebrafish behavioral studies,
- previous unsupervised discovery work.

## Forward Search

Review papers that cite the study.

This is especially important for:

- landmark computational ethology papers,
- zebrafish 3D tracking papers,
- animal pose representation methods.

---

# 17. Seed Papers

The review currently includes the following high-priority starting points.

## Zebrafish-Specific

### Scholz et al. (2025)

Topic:

> Automated behavioral tracking of zebrafish larvae using DeepLabCut and SLEAP.

Primary relevance:

- pose estimation,
- tracking infrastructure,
- public resources,
- possible dataset candidate.

---

### Deligkaris et al. (2026)

Topic:

> Fine-grained zebrafish interactions in health and disease.

Primary relevance:

- high-resolution behavior,
- 3D interaction data,
- identity,
- social behavior,
- possible future replication.

---

### Yang et al. (2021)

Topic:

> Zebrafish behavior feature recognition using three-dimensional tracking and machine learning.

Primary relevance:

- prior art,
- 3D behavior analysis,
- hand-engineered features,
- novelty boundary.

---

### Barreiros et al. (2021)

Topic:

> Zebrafish automatic monitoring system for conditioning and behavioral analysis.

Primary relevance:

- automated tracking,
- conventional behavioral analysis,
- pipeline design.

---

## Tool / Method References

### DeepLabCut

Primary relevance:

- pose estimation.

### SLEAP

Primary relevance:

- pose estimation,
- multi-animal tracking.

### StrIPETrack

Primary relevance:

- zebrafish/fish tracking.

### AquaMaze

Primary relevance:

- conventional zebrafish behavioral measures.

---

# 18. Eligibility Criteria

A paper should be included in the core literature set when it contributes directly to at least one of the following:

- zebrafish behavior measurement,
- zebrafish tracking,
- animal pose analysis,
- temporal animal representation learning,
- unsupervised behavioral discovery,
- self-supervised behavioral learning,
- behavioral state validation,
- leakage/confound control,
- dataset selection,
- experimental design.

---

# 19. Exclusion Criteria

Exclude papers from the core set when they:

- mention zebrafish only incidentally,
- contain no relevant behavioral methodology,
- use only static biological images,
- focus entirely on molecular mechanisms without behavioral analysis,
- provide no transferable methodological insight,
- are duplicate publications,
- are secondary summaries when the primary study is available.

Reviews may still be retained as navigation resources.

---

# 20. Publication Type Classification

Each source should be classified as:

```text
PRIMARY_RESEARCH
REVIEW
PREPRINT
DATASET_PAPER
METHOD_PAPER
SOFTWARE_PAPER
PROTOCOL
THESIS
OTHER
```

Primary scientific claims should preferably rely on primary research papers.

---

# 21. Evidence Priority

When multiple sources support the same point, prioritize:

1. primary research,
2. recent peer-reviewed work,
3. well-established methodological papers,
4. dataset documentation,
5. high-quality review articles.

Preprints may be included but should be clearly identified.

---

# 22. Screening Process

Screening occurs in three stages.

```text
Search Results
      ↓
Title Screening
      ↓
Abstract Screening
      ↓
Full-Text Screening
      ↓
Included Literature
```

---

# 23. Stage 1 — Title Screening

Retain papers whose titles suggest relevance to:

- zebrafish behavior,
- animal behavioral discovery,
- tracking,
- pose,
- SSL,
- temporal representation learning,
- computational ethology.

When uncertain, retain for abstract screening.

---

# 24. Stage 2 — Abstract Screening

Ask:

- Does the paper analyze temporal behavior?
- Does it introduce a relevant representation?
- Does it use unsupervised discovery?
- Does it provide a useful behavioral baseline?
- Does it discuss validation?
- Does it provide reusable data or code?

If at least one answer is strongly yes, retain for full-text screening.

---

# 25. Stage 3 — Full-Text Screening

During full-text review, determine the paper's actual role.

Possible roles:

```text
NOVELTY
BASELINE
DATASET
TRACKING
SSL_METHOD
DISCOVERY
VALIDATION
NEGATIVE_CONTROL
INTERPRETATION
BACKGROUND
```

A paper may have multiple roles.

---

# 26. Reading Priority

Use three reading levels.

## Level 1 — Scan

Read:

- abstract,
- figures,
- conclusion,
- methods overview.

Purpose:

Determine relevance.

---

## Level 2 — Targeted Read

Read:

- methods,
- experiments,
- validation,
- limitations.

Purpose:

Extract information directly relevant to the project.

---

## Level 3 — Deep Read

Reserved for papers that materially determine:

- novelty,
- baseline,
- SSL method,
- validation,
- or experimental design.

These papers should receive detailed notes.

---

# 27. Literature Extraction Template

For every core paper, record:

```yaml
paper_id:
title:
authors:
year:
venue:
doi:
url:

publication_type:
peer_reviewed:

species:
developmental_stage:
number_of_animals:

behavioral_setting:
recording_modality:
frame_rate:
tracking_method:
pose_method:

representation:
features:
temporal_window:

learning_type:
supervised:
unsupervised:
self_supervised:

model:
ssl_objective:
clustering_method:

validation:
held_out_animals:
cross_session:
cluster_stability:
identity_control:
session_control:
speed_control:
artifact_control:

dataset_public:
code_public:

main_findings:
limitations:
project_relevance:
novelty_implication:
```

Unknown values should be recorded as:

```text
unknown
```

rather than inferred.

---

# 28. Minimal Paper Note

Each core paper should also receive a human-readable summary.

Recommended format:

```markdown
## PAPER-ID — Citation

### Why I Read It

### Research Question

### Data

### Method

### Behavioral Representation

### Discovery Method

### Validation

### Main Result

### Limitations

### What This Means for Zebrafish SSL

### Action for This Project
```

---

# 29. Novelty Extraction

For every directly related paper, answer:

1. Did the study use zebrafish?
2. Did it analyze temporal behavior?
3. Did it learn its representation?
4. Was learning self-supervised?
5. Was behavioral discovery unsupervised?
6. Was a hand-engineered baseline included?
7. Were the same observations compared across representations?
8. Were clusters validated on held-out animals?
9. Were speed and identity confounds tested?
10. Did the authors claim novel behavioral states?

---

# 30. Novelty Matrix

The novelty matrix is now populated from the targeted literature review.

Legend:

```text
YES      directly supported
PARTIAL  related component exists, but not the same design/claim
NO       clearly not part of the study
NR       not identified/reported in reviewed material
NA       not applicable
```

| Study | Zebrafish | Temporal | Learned Representation | SSL | Unsupervised Discovery | Direct Handcrafted-vs-Learned Comparison | Held-Out Fish | Identity / Session / Speed Controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Marques et al. (2018) | YES | YES | NO — engineered bout kinematics | NO | **YES** | NO | NR | PARTIAL — context/kinematic validation; no full leakage battery identified |
| Ghosh & Rihel (2020) | YES | **YES — multi-scale** | NO — engineered activity/modules | NO | **YES** | NO | NR | PARTIAL — activity-level confound considered; no complete identity/session battery identified |
| Yang et al. (2021) | YES | YES | NO — engineered 3D trajectory features | NO | **YES** | NO | NR | PARTIAL — stimulus association/tracking validation; explicit identity/speed controls NR |
| Barreiros et al. (2021) | YES | YES | NO — conventional tracking metrics | NO | NO primary discovery | NO | NR | PARTIAL — tracking/condition controls; speed is an outcome rather than a controlled confound |
| Scholz et al. (2025) | YES | YES | PARTIAL — learned pose-estimation models, not behavioral SSL | NO behavioral SSL | NO | NO | PARTIAL/NR | **YES for tracking QC**; behavior-representation leakage controls NA/NR |
| Deligkaris et al. (2026) | YES | YES | PARTIAL — tracking/identity representation | NO | NO primary discovery | NO | NA | **YES/PARTIAL** — explicit persistent identity and tracking-QC limitations |
| ContrastivePose (2023) | NO zebrafish | YES | **YES** | **YES** | NO primary unsupervised discovery | **YES** | NR | PARTIAL/NR |
| Xu & Wang (2024) | **YES** | **YES** | **YES** | **YES** | PARTIAL — latent representation learning, not matched discovery framework | NO matched comparison identified | NR | NR |
| BEAST (ICLR 2026) | NO direct zebrafish focus | **YES** | **YES** | **YES** | NO primary matched zebrafish discovery | NO matched zebrafish baseline comparison | PARTIAL/NR | PARTIAL/NR |
| Proposed study | **YES** | **YES** | **YES** | **YES** | **YES** | **YES — primary design** | **YES** | **YES — explicit identity, context/session, speed, tracking and stability controls** |

## Novelty Matrix Conclusion

The matrix rejects two overly broad novelty claims:

```text
"Unsupervised zebrafish behavioral discovery does not exist."  -> FALSE
"Self-supervised zebrafish behavioral representation learning does not exist." -> FALSE
```

The remaining defensible gap is the **matched controlled comparison**:

> conventional handcrafted zebrafish representation versus self-supervised temporal representation, on the same observations, under matched unsupervised discovery and explicit cross-animal/nuisance/reproducibility validation.

This matrix should be refreshed immediately before manuscript submission, especially for preprints and rapidly evolving animal-behavior SSL work.

---

# 31. Baseline Feature Extraction

Maintain a cumulative list of features encountered in zebrafish literature.

## Locomotion

```text
speed
distance
acceleration
immobility
movement duration
bout duration
bout frequency
inter-bout interval
```

## Turning

```text
heading
angular velocity
turning angle
turn frequency
trajectory curvature
```

## Spatial

```text
tank position
distance to wall
center occupancy
edge occupancy
vertical position
```

## Pose

```text
orientation
body curvature
tail curvature
tail beat
keypoint geometry
```

For each feature, record citations showing its use.

---

# 32. Baseline Selection Rule

A feature should be considered for Input A when:

- it appears repeatedly in zebrafish literature,
- it captures a meaningful established behavioral quantity,
- and it can be calculated reliably from the selected dataset.

Input A should not become an exhaustive collection of every feature ever reported.

The goal is a **defensible conventional baseline**, not a maximal feature engineering competition.

---

# 33. SSL Method Extraction

For each relevant SSL method, record:

```yaml
method:
input_modality:
architecture:
objective:
positive_pair_definition:
negative_pair_definition:
augmentations:
temporal_context:
embedding_dimension:
training_scale:
downstream_evaluation:
code_available:
```

---

# 34. SSL Candidate Evaluation

Each method should be judged on:

| Criterion | Question |
|---|---|
| Temporal sensitivity | Does the method learn temporal dynamics? |
| Dataset compatibility | Can it operate on available data? |
| Scale | Does it require more data than available? |
| Complexity | Can it be implemented reliably? |
| Reproducibility | Is code available? |
| Interpretability | Can embeddings be analyzed? |
| Confound risk | Could augmentations create unwanted invariances? |
| Compute | Is training computationally feasible? |

---

# 35. SSL Method Selection Rule

The first experiment should favor:

> **the simplest self-supervised temporal approach with strong literature precedent that is appropriate for the selected input modality.**

The project does not require inventing a new SSL architecture.

The scientific contribution is primarily the controlled behavioral comparison.

---

# 36. Discovery Method Extraction

For each unsupervised behavior study, record:

```yaml
representation:
preprocessing:
dimensionality_reduction:
clustering_method:
number_of_states:
state_selection_method:
temporal_model:
stability_analysis:
interpretation_method:
```

---

# 37. Clustering Selection Evidence

The literature review should specifically investigate how prior studies choose:

- number of clusters,
- density thresholds,
- state count,
- dimensionality,
- clustering parameters.

This will inform the preregistration and reduce arbitrary parameter selection.

---

# 38. Validation Extraction

For every important behavioral discovery or representation-learning paper, record whether the study performed the following validation procedures.

Legend:

```text
YES      explicitly reported
PARTIAL  related control/validation was performed, but not the exact requested test
NO       explicitly absent or clearly not part of the design
NR       not identified/reported in the reviewed source
NA       not applicable to the study's purpose
```

The current extraction is intentionally conservative: absence of a reported method is recorded as `NR`, not inferred as `NO`.

| Study | Repeated clustering | Multiple seeds | Bootstrap / resampling | Held-out animals | Held-out sessions | External replication | Manual video validation | Known behavior recovery | Stimulus association | Condition association | Identity control | Speed control | Tracking-quality control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Marques et al. (2018) | YES/PARTIAL | NR | PARTIAL | NR | NR | NR | YES/PARTIAL | **YES** — recovered seven previously recognized swim types and identified six additional clusters | **YES** | **YES** | NR | PARTIAL — kinematic structure includes movement magnitude but no explicit speed-removal test identified | YES/PARTIAL |
| Ghosh & Rihel (2020) | **YES** — evidence-accumulation clustering repeatedly samples/fits mixture models | NR | **YES/PARTIAL** — random subsampling is part of evidence accumulation and cluster-size normalization | NR | PARTIAL — repeated experiments/day-night structure, but not a formal held-out-session test | NR | PARTIAL — module interpretations are linked to observed seizure-like/inactive behavior | **YES/PARTIAL** | PARTIAL — day/night and pharmacological manipulations | **YES** — genotype, melatonin, PTZ, day/night | NR | **PARTIAL** — authors explicitly show compressibility changes are not simply explained by overall activity level | NA/PARTIAL — pixel-change activity representation rather than anatomical pose tracking |
| Yang et al. (2021) | PARTIAL — grid search over segmentation/PCA/FuzzyART settings | NR | NR | NR | NR | PARTIAL — previously identified patterns can be evaluated on new user-supplied data, but independent external replication was not demonstrated | **YES** — behavior was manually observed while videos were recorded | PARTIAL | **YES** — electric-stimulus association tested statistically | **YES** | NR | NR | PARTIAL — automated EthoVision tracking plus 3D reconstruction; no dedicated artifact-ablation test identified |
| Barreiros et al. (2021) | NA | NR | NR | NR | NR | NR | PARTIAL | **YES/PARTIAL** — expected conditioning responses were recovered | **YES** — visual, vibration, food-reward conditioning | **YES** — stimulated versus control comparisons | NR | NO explicit speed-confound control; speed itself is an outcome | PARTIAL — automated detection/tracking system validation |
| Scholz et al. (2025 preprint) | NA | NR | NR | PARTIAL — model performance evaluated across diverse videos/imaging settings, but this is tracking validation rather than behavioral discovery | PARTIAL | NR | **YES** — annotated pose ground truth | NA | NA | **YES/PARTIAL** — performance across imaging conditions | NA | NA | **YES** — direct pose-estimation benchmarking, annotated ground truth, imaging-condition evaluation |
| Deligkaris et al. (2026) | NA | NA | NA | NA | NA | NA | PARTIAL | NA | NA | **YES** — sex/genotype/disease-model conditions represented | **YES/PARTIAL** — persistent fish identity tracked; residual swaps explicitly acknowledged | NA | **YES** — explicit QC workflow; residual identity/body-part errors documented |
| ContrastivePose (Zhou et al., 2023) | NA | NR | NR | NR in reviewed source | NR | **PARTIAL** — evaluated on more than one animal-interaction dataset | **YES/PARTIAL** — downstream labeled behaviors provide human-scored reference | **YES** — known behavior classification | NA | **YES** — downstream behavior-classification conditions | NR | NR | PARTIAL — relies on pose-estimation inputs; augmentation invariance tested |
| Xu & Wang (2024 technical report) | NA | NR | NR | NR | NR | NR | PARTIAL/NR | PARTIAL — latent behavior discrimination is evaluated, but not as a full unsupervised ethogram-validation battery | NR | NR | NR | NR | NR |
| BEAST (Wang et al., 2025 preprint) | NA | NR | NR | PARTIAL/NR — cross-animal/multi-dataset evaluation is relevant, but an explicit held-out-subject leakage battery was not confirmed from the reviewed source | PARTIAL/NR | **YES/PARTIAL** — multiple species/datasets/tasks | **YES/PARTIAL** — pose/action labels used in downstream evaluation | **YES** — action segmentation/pose tasks | NR | **YES/PARTIAL** — multiple datasets/species/tasks | NR | NR | PARTIAL — robustness evaluated through downstream video tasks |

## Validation Extraction Synthesis

The literature does **not** reveal one universal validation standard for unsupervised animal-behavior discovery. Instead, different studies validate structure through combinations of:

- recovery of known behavior,
- stimulus or condition association,
- repeated/resampled clustering,
- manual or ground-truth comparison,
- pharmacological/genetic perturbation,
- cross-dataset or cross-condition evaluation,
- explicit tracking-quality checks.

The strongest transferable lesson for this project is that biological plausibility alone is insufficient. The present study therefore adopts a stricter validation battery than any single reviewed zebrafish paper:

```text
fish-level TRAIN / VALIDATION / TEST separation
+
multi-seed SSL training
+
cluster stability
+
fish-identity prediction
+
context/session prediction
+
speed-dependence analysis
+
tracking-QC exclusions/sensitivity flags
+
baseline-vs-SSL recoverability
+
held-out TEST evaluation only after method freeze
```

This validation design is methodological synthesis rather than a claim that every precedent paper used every control.

Primary sources supporting this extraction include:

- Marques et al. (2018), *Current Biology*, DOI: 10.1016/j.cub.2017.12.002
- Ghosh & Rihel (2020), *eNeuro*, DOI: 10.1523/ENEURO.0408-19.2020
- Yang et al. (2021), *Scientific Reports*, DOI: 10.1038/s41598-021-92854-0
- Barreiros et al. (2021), *Scientific Reports*, DOI: 10.1038/s41598-021-87502-6
- Zhou et al. (2023), *Computers in Biology and Medicine*, DOI: 10.1016/j.compbiomed.2023.107416
- Scholz et al. (2025), bioRxiv DOI: 10.1101/2025.06.04.657938
- Deligkaris et al. (2026), *Scientific Data*, DOI: 10.1038/s41597-026-06953-6
- Xu & Wang (2024), arXiv:2403.15693
- Wang et al. / BEAST (2025), arXiv:2507.09513

---

# 39. Identity Leakage Literature

Search specifically for methods that determine whether learned representations encode subject identity.

Possible techniques to look for:

- subject-ID classifiers,
- leave-one-subject-out validation,
- subject-balanced batches,
- identity-invariant augmentation,
- cross-subject retrieval,
- cluster enrichment analysis.

Any useful method should be considered for the preregistration.

---

# 40. Session Leakage Literature

Extract methods addressing:

- different cameras,
- recording days,
- environments,
- tank configurations,
- laboratories,
- backgrounds.

Possible approaches:

- leave-one-session-out evaluation,
- domain-adversarial learning,
- session prediction,
- domain normalization,
- augmentation,
- cross-session replication.

---

# 41. Speed-Control Literature

Search for studies demonstrating whether behavioral embeddings capture more than overall movement magnitude.

Possible analyses include:

- regressing speed from embeddings,
- conditioning on speed,
- speed-matched sampling,
- clustering after removing speed,
- comparing against speed-only representations.

Relevant methods should inform the project's primary speed control.

---

# 42. Tracking Artifact Literature

Extract common tracking-failure detection methods.

Examples:

- confidence thresholds,
- missing-keypoint rates,
- velocity outlier detection,
- anatomical consistency,
- temporal smoothing,
- coordinate-jump thresholds.

Record which methods can reasonably transfer to the selected zebrafish dataset.

---

# 43. Temporal Window Literature

For each study using temporal clips or pose sequences, record:

```yaml
window_duration:
frame_count:
frame_rate:
stride:
overlap:
reason_for_window_size:
sensitivity_analysis:
```

The goal is to establish a defensible range for the project's own behavioral windows.

---

# 44. Dataset Literature

For each dataset paper, extract:

```yaml
dataset_name:
species:
age:
number_of_animals:
number_of_sessions:
duration:
frame_rate:
video_available:
tracking_available:
pose_available:
identity_available:
conditions:
license:
repository:
```

Then update:

```text
docs/dataset-register.md
```

---

# 45. Public Data Verification

A paper saying that data are "available" is not sufficient.

For promising datasets:

1. follow the official data link,
2. verify that the repository still exists,
3. inspect available files,
4. inspect license/terms,
5. verify that required modalities are actually downloadable.

Dataset findings should then be transferred to the dataset register.

---

# 46. Citation Chaining

When a paper contains a statement central to the project's novelty, locate the original study cited for that statement rather than relying entirely on the secondary citation.

For example:

```text
Review says:
"Unsupervised methods have been used to identify zebrafish behavioral states."

        ↓

Find original study.

        ↓

Read original study.

        ↓

Record exact method and evidence.
```

---

# 47. Review Paper Policy

Review papers are useful for:

- terminology,
- identifying foundational studies,
- understanding field history,
- finding datasets.

They should not replace primary papers when making claims such as:

- "first use,"
- "no prior work,"
- "previous studies did not validate X."

---

# 48. Novelty Claim Policy

Avoid claims such as:

> "Nobody has ever used self-supervised learning for zebrafish behavior."

unless the literature review provides unusually strong evidence.

Prefer bounded claims such as:

> "We did not identify prior work directly comparing hand-engineered zebrafish behavioral features against self-supervised temporal embeddings under a shared unsupervised discovery and cross-animal validation framework."

The final wording must reflect the actual search results.

---

# 49. Literature Evidence Categories

Assign relevant claims one of the following evidence levels.

## Strong

Supported by multiple directly relevant primary studies.

## Moderate

Supported by one direct study or several related studies.

## Weak

Supported indirectly or by adjacent fields.

## Unknown

Insufficient literature found.

---

# 50. Contradictory Findings

Do not discard literature because it conflicts with the proposed research direction.

Record:

```markdown
### Contradictory Evidence

**Paper:**

**Finding:**

**Why it conflicts:**

**Possible explanation:**

**Implication for project:**
```

Contradictory findings may be particularly important when assessing whether SSL actually adds value.

---

# 51. Negative Results

Searches should not focus only on papers reporting successful behavioral discovery.

Where available, retain evidence showing:

- poor generalization,
- weak representation transfer,
- unstable clustering,
- strong identity effects,
- limitations of pose representations.

These findings inform the project's controls.

---

# 52. Reading Order

Recommended reading order:

```text
1. Zebrafish behavioral feature literature
        ↓
2. Zebrafish tracking / pose literature
        ↓
3. Zebrafish unsupervised / ML behavior literature
        ↓
4. Computational ethology behavioral discovery
        ↓
5. Animal SSL
        ↓
6. General temporal SSL
        ↓
7. Validation / leakage literature
```

This prevents selecting a sophisticated SSL model before understanding the biological baseline.

---

# 53. First-Pass Reading Goal

The initial review should aim to identify approximately:

```yaml
zebrafish_behavior_core: 10-20 papers
zebrafish_tracking_core: 5-10 papers
unsupervised_behavior_core: 10-15 papers
animal_ssl_core: 10-15 papers
temporal_ssl_methods: 5-10 papers
validation_and_confounds: 5-10 papers
```

These are targets, not quotas.

Quality and relevance take priority.

---

# 54. Literature Saturation Rule

A literature domain may be considered provisionally saturated when:

- repeated searches primarily return already-reviewed studies,
- new papers no longer materially change the research design,
- major terminology and methods are established,
- citation chaining returns mostly known work.

Saturation does not mean no additional literature exists.

## Current Saturation Decision

```yaml
final_targeted_search_date: 2026-08-22
broad_search_status: COMPLETE_FOR_PREREGISTRATION
saturation_basis:
  - repeated targeted searches returned core studies already represented in the review
  - direct zebrafish SSL precedent was identified and incorporated
  - direct zebrafish unsupervised-discovery precedent was identified and incorporated
  - new literature no longer changed the primary dataset, Input A, SSL family, unit of analysis, or validation framework
  - no reviewed study matched the complete handcrafted-vs-SSL plus held-out-fish/nuisance-control design
remaining_search_mode: TARGETED_ONLY
preprint_refresh_required_before_manuscript: true
```

The search log is stored at:

```text
references/search-log.csv
```

Counts that were not recorded during the original search sessions are intentionally left blank rather than reconstructed retrospectively.

---

# 55. Search Stopping Rule

Initial literature search may pause when the project can confidently answer:

- What is the conventional baseline?
- What closely related zebrafish ML work exists?
- Has the proposed direct comparison already been performed?
- Which SSL method will be used?
- Which discovery method will be used?
- Which validation controls are necessary?
- Which dataset is suitable?

Additional searches should continue when specific methodological questions arise.

---

# 56. Literature Review Deliverables

The literature phase now produces:

```text
docs/literature-protocol.md
docs/literature.md
docs/dataset-register.md
references/papers.bib
references/search-log.csv
references/paper-notes/
```

Structured bibliography and paper notes are now part of the reproducibility trail rather than optional future artifacts.

The bibliography should be treated as a living file. Preprints and technical reports must be refreshed before manuscript submission.

---

# 57. Suggested Paper Notes Directory

```text
references/
├── papers.bib
│
└── paper-notes/
    ├── ZF-001.md
    ├── ZF-002.md
    ├── SSL-001.md
    ├── SSL-002.md
    └── ...
```

---

# 58. Paper Identifier Convention

Use categories such as:

```text
ZF-###       zebrafish behavior
POSE-###     tracking / pose
SSL-###      self-supervised learning
UBD-###      unsupervised behavioral discovery
VAL-###      validation / leakage
DATA-###     dataset papers
```

Example:

```text
ZF-001
POSE-001
SSL-001
```

---

# 59. Literature Synthesis Structure

The final:

```text
docs/literature.md
```

should synthesize findings rather than summarize papers sequentially.

The current literature synthesis is sufficiently mature to populate the following sections.

# Literature Review

## Zebrafish Behavioral Measurement

Conventional zebrafish behavioral analysis is dominated by interpretable kinematic, temporal, and spatial quantities. Repeatedly encountered measures include:

- speed,
- distance traveled,
- activity/immobility,
- bout duration and frequency,
- inter-bout interval,
- turning and heading,
- trajectory/spatial occupancy,
- posture or tail/body geometry when pose tracking is available.

Barreiros et al. (2021) provide direct precedent for distance traveled, speed, route/spatial behavior, and polarization in automated adult-zebrafish conditioning experiments. AquaMaze (Ayık et al., 2026) similarly exposes swim distance, speed, quadrant occupancy, rest/activity, and assay-specific spatial metrics. StrIPETrack (Cummings et al., 2026) validates activity measures and spatial/Y-maze navigation metrics.

For the present project, this literature supports a compact conventional Input A rather than exhaustive feature engineering. The frozen DS-005 core baseline of timing, speed, acceleration/speed-change, and orientation/turning features is consistent with this evidence.

## Zebrafish Tracking and Pose Estimation

Zebrafish tracking ranges from centroid/activity systems to multi-keypoint 2D/3D pose estimation.

Scholz et al. (2025) provide a larval resource using a 15-keypoint pose schema, annotated free-swimming and head-embedded videos, and pretrained DeepLabCut and SLEAP networks. The study explicitly evaluates performance across imaging conditions, making it especially relevant to tracking-quality and generalization concerns.

Deligkaris et al. (2026) provide 3D multi-animal tracking of three anatomical landmarks with persistent individual identity. Their pipeline combines SLEAP body-point detection with identity tracking and explicitly acknowledges residual identity swaps/body-part misidentification after QC, especially during close interactions.

StrIPETrack (2026) provides an open-source ROI-flexible tracking system and validates tracking against an earlier LabVIEW system and manual Y-maze scoring.

The methodological implication is that tracking errors can themselves generate apparent behavioral structure. Tracking QC must therefore be treated as a validity condition, not merely a preprocessing convenience.

## Existing Machine Learning for Zebrafish Behavior

Machine learning is already established in zebrafish behavior analysis.

Yang et al. (2021) reconstruct 3D trajectories, segment time series, reduce dimensionality with PCA, and use unsupervised FuzzyART to identify electric-stimulus-associated behavior features. Parameters were explored using grid search over temporal segmentation, explained variance, and FuzzyART vigilance. The work demonstrates that unsupervised ML can identify treatment-linked structure in zebrafish trajectories.

Marques et al. (2018) use unsupervised clustering of millions of naturally segmented larval swim bouts to recover a 13-type locomotor repertoire. Ghosh & Rihel (2020) use evidence-accumulation clustering to derive active/inactive modules and hierarchical compression to identify longer behavioral motifs.

Therefore, neither “machine learning for zebrafish behavior” nor “unsupervised discovery of zebrafish behavior” is novel by itself.

## Unsupervised Behavioral Discovery

Genuinely unsupervised behavioral discovery has strong zebrafish precedent.

Marques et al. (2018) identified thirteen basic swimming patterns from millions of high-speed bouts across behavioral contexts. Importantly, seven corresponded to previously recognized movements, providing known-behavior recovery, while six expanded the repertoire.

Ghosh & Rihel (2020) identified ten active/inactive modules and nearly 50,000 recurrent motifs across sub-second to minute scales, then related module/motif usage to day/night, pharmacological, and genotype conditions. Their evidence-accumulation approach and hierarchical compression show that behavioral structure exists at multiple temporal scales.

Yang et al. (2021) provide a separate adult-zebrafish example in which PCA + FuzzyART identifies treatment-associated temporal trajectory patterns.

No single clustering family dominates the field. Methods include custom robust clustering, mixture-model/evidence-accumulation methods, hierarchical compression, PCA-assisted clustering, and adaptive-resonance methods. Likewise, state count is chosen in method-specific ways rather than by one field-wide rule.

## Self-Supervised Behavioral Representation Learning

Self-supervised behavioral representation learning is established in broader computational ethology and now has direct zebrafish precedent.

ContrastivePose (Zhou et al., 2023) learns features from unlabeled animal pose data using contrastive learning. Positive pairs are behavior-preserving augmentations of the same pose observation, while other batch samples act as negatives. Learned features outperform handcrafted features in downstream supervised behavior classification, providing particularly relevant precedent for a handcrafted-versus-learned representation comparison.

Xu & Wang (2024) directly apply self-supervised masked skeleton-sequence modeling to larval zebrafish. Their SSTFormer combines transformer and CNN components to encode inter-frame and inter-joint structure and produces latent behavioral embeddings.

BEAST (Wang et al., 2025) combines masked autoencoding with temporal contrastive learning for animal behavioral video and evaluates the pretrained representation across species and downstream tasks.

Thus, the defensible novelty claim is **not** that SSL has never been applied to zebrafish.

## Temporal Representation Learning

Animal behavior is intrinsically multi-scale.

Marques et al. use naturally segmented larval swim bouts as meaningful short behavioral units. Ghosh & Rihel demonstrate structure extending from sub-second bouts to minute-scale motifs and day-long behavioral organization. Yang et al. evaluate predefined 1 s and 10 s time segments, providing fixed-window precedent.

The literature therefore does not justify a single canonical window duration. Both fixed windows and behavior-adaptive segmentation are defensible. For DS-005, one valid behavioral bout is a particularly strong unit because the source data already provide natural bout segmentation at high temporal resolution.

Temporal contrastive learning is a reasonable SSL family because the research question concerns temporal behavioral structure, while a small 1D CNN is computationally appropriate for short fixed-length bout sequences.

## Validation and Reproducibility

Validation practices are heterogeneous.

Strong forms of evidence found in the literature include:

- known-behavior recovery (Marques et al.),
- repeated/evidence-accumulation clustering (Ghosh & Rihel),
- stimulus association (Yang et al.),
- pharmacological/genetic condition association (Ghosh & Rihel),
- manual/ground-truth tracking validation (Scholz; StrIPETrack),
- explicit identity/tracking QC (Deligkaris),
- cross-dataset downstream evaluation (ContrastivePose; BEAST).

However, explicit combinations of held-out animals, identity prediction, session prediction, speed dependence, cluster stability, and final untouched test evaluation are not consistently reported together.

The present project therefore adopts a stronger composite validation framework:

- fish-disjoint TRAIN/VALIDATION/TEST split,
- TEST protection until final evaluation,
- predefined SSL seeds,
- clustering stability,
- fish-identity leakage analysis,
- fish-aware context/session leakage analysis,
- speed prediction/correlation and speed-matched controls where needed,
- tracking-QC exclusions and sensitivity flags,
- baseline-vs-SSL recoverability,
- cross-seed SSL reproducibility.

## Public Datasets

The literature search identified multiple reusable zebrafish resources, including pose/tracking datasets and high-resolution behavior datasets.

Scholz et al. provide annotated larval pose videos and pretrained DeepLabCut/SLEAP models. Deligkaris et al. release tracked adult dyadic social-interaction experiments, metadata, and sample videos through Zenodo with accompanying code. Ghosh & Rihel release processed behavioral data and analysis code.

For the primary experiment, DS-005 has now passed project-specific suitability checks and is frozen as the primary dataset. Its authorization, fish-level identity structure, temporal bout data, and reproducibility artifacts are tracked separately in `docs/dataset-register.md` and project decisions.

## Research Gap

Targeted search does **not** support either of the broad claims:

> “No unsupervised zebrafish behavioral discovery exists.”

or

> “No zebrafish behavioral SSL exists.”

Both are false or too broad.

The literature-supported gap is narrower:

> **Prior work establishes unsupervised zebrafish behavioral discovery and self-supervised animal-behavior representation learning, and direct zebrafish SSL precedent exists. However, the targeted review did not identify a study directly comparing conventional hand-engineered zebrafish bout representations against self-supervised temporal representations on the same observations under matched unsupervised discovery and explicit controls for held-out fish, identity leakage, context/session effects, speed dependence, tracking artifacts, and representation stability.**

This is the frozen novelty boundary for the preregistration. It is a bounded search-supported statement, not an absolute claim that no future or unindexed overlapping work exists.

## Implications for Experimental Design

The literature directly informs the experimental design:

| Literature finding | Project decision |
|---|---|
| Conventional zebrafish work repeatedly uses speed, movement, timing and turning metrics | Use a compact 18-feature handcrafted Input A |
| Larval behavior is naturally organized into bouts | Use one valid DS-005 bout as the primary unit |
| Pose/tracking errors can persist and create false structure | Apply explicit QC and sensitivity flags |
| Unsupervised zebrafish discovery already exists | Frame contribution as representation comparison, not first-ever clustering |
| Direct zebrafish SSL exists | Do not claim first zebrafish SSL |
| ContrastivePose shows contrastive pose SSL can outperform handcrafted features downstream | Use temporal contrastive learning as a practical candidate family |
| Behavior occurs across multiple timescales | Treat bout-based analysis as primary and retain temporal-scale sensitivity as a limitation/control |
| Prior validation often relies on stimulus/condition association rather than complete leakage batteries | Add fish identity, context/session, speed and stability controls |
| Repeated/resampled clustering can improve robustness | Evaluate cluster stability and multiple seeds |
| Subject-specific nuisance information is a credible threat | Split at fish level and quantify identity predictability |
| Movement magnitude is a dominant behavioral variable | Explicitly test speed dependence rather than assuming embeddings capture behavior beyond speed |

The experimental contribution is therefore best described as:

```text
CONTROLLED REPRESENTATION COMPARISON
+
UNSUPERVISED DISCOVERY
+
STRONG NUISANCE / REPRODUCIBILITY VALIDATION
```

rather than a new SSL architecture.

---

# 60. Research Gap Test

Before stating the final research gap, explicitly test the following possibilities.

## Gap Candidate A

No zebrafish SSL behavior work exists.

**Risk:** Too broad and easy to falsify.

---

## Gap Candidate B

No unsupervised zebrafish behavior discovery exists.

**Risk:** Very likely false given existing ML and behavioral clustering work.

---

## Gap Candidate C

No prior study has directly compared conventional hand-engineered zebrafish behavioral representations against temporal SSL representations under matched unsupervised discovery and strong cross-animal validity controls.

**Current preferred candidate.**

This must still be verified by the literature review.

---

# 61. Contribution Boundary

The literature review should determine whether the project's defensible contribution is primarily:

```text
METHOD COMPARISON
```

rather than:

```text
NEW MODEL
```

or:

```text
NEW BIOLOGICAL BEHAVIOR
```

The current working assumption is:

> The primary contribution is a controlled comparison of behavioral representations and an evaluation of whether SSL reveals additional reproducible behavioral structure.

---

# 62. Literature-to-Experiment Traceability

Every major experimental decision should ideally have a literature basis.

Example:

```text
Literature Finding
        ↓
Experimental Decision
```

Examples:

```text
Prior zebrafish studies commonly use speed
        ↓
Include speed in Input A
```

```text
Behavior studies validate across animals
        ↓
Use fish-level train/test split
```

```text
Temporal behavior occurs at multiple scales
        ↓
Run window-length sensitivity analysis
```

---

## Current Literature-to-Decision Traceability

| Literature finding | Core evidence | Project consequence |
|---|---|---|
| Unsupervised zebrafish behavior discovery already exists | Marques 2018; Ghosh & Rihel 2020; Yang 2021 | Do not claim first zebrafish behavioral clustering/discovery |
| Direct zebrafish SSL exists | Xu & Wang 2024 | Do not claim first zebrafish SSL |
| Learned-vs-handcrafted animal pose comparison has precedent | ContrastivePose 2023 | Frame contribution as matched discovery/validation comparison rather than generic learned-vs-handcrafted novelty |
| Temporal contrastive behavioral SSL has strong animal precedent | ContrastivePose; BEAST | Supports temporal contrastive objective |
| Zebrafish behavior is naturally organized into bouts and multiple temporal scales | Marques 2018; Ghosh & Rihel 2020 | Supports DS-005 bout as primary unit; retain temporal-scale limitations/sensitivity |
| Conventional zebrafish analysis uses speed, movement and spatial/turning metrics | Barreiros 2021; Yang 2021; AquaMaze 2026 | Supports compact handcrafted Input A |
| Tracking and identity errors can survive automated pipelines | Scholz 2025; Deligkaris 2026 | Apply QC, fish-ID leakage tests, and cautious cluster interpretation |
| Biological association alone is not a complete validity test | Validation synthesis across discovery papers | Add held-out fish, stability, identity, context/session and speed controls |


# 63. Decision Recording

When literature materially changes the planned research design, record the decision in:

```text
docs/decisions.md
```

Example:

```markdown
## Decision: Use Pose Sequences as SSL Input

**Date:**

**Evidence:**

**Alternatives:**

**Decision:**

**Reason:**
```

---

# 64. Study Quality Questions

When deeply reviewing a paper, ask:

- Is the number of independent animals adequate?
- Are windows incorrectly treated as independent subjects?
- Are train/test sets separated by animal?
- Are clustering parameters justified?
- Is biological interpretation independent of clustering?
- Are nuisance variables tested?
- Is code available?
- Is the dataset available?
- Are results replicated?

These considerations matter when deciding how much methodological weight to give the study.

---

# 65. Avoiding Literature Confirmation Bias

The literature review must actively search for evidence that could weaken the proposed study.

Search for:

```text
zebrafish behavioral clustering
zebrafish self-supervised behavior
zebrafish representation learning
zebrafish unsupervised pose
fish behavioral embeddings
zebrafish behavioral motifs
```

The objective is to discover overlapping work before conducting the experiment, not afterward.

---

# 66. Evidence Against Novelty

Any paper that appears to perform:

```text
zebrafish
+
self-supervised temporal representation
+
unsupervised behavior discovery
+
hand-engineered comparison
+
held-out fish validation
```

should immediately receive **highest reading priority**.

If such a paper exists, the research question must be refined.

---

# 67. Dataset Candidate Escalation

When literature identifies a promising dataset:

```text
Paper
  ↓
Repository
  ↓
License
  ↓
Data Inspection
  ↓
docs/dataset-register.md
```

Do not select a dataset directly from a paper abstract.

---

# 68. Citation Management

Use a consistent citation manager or BibTeX file.

Preferred repository file:

```text
references/papers.bib
```

Suggested key format:

```text
lastnameYEARkeyword
```

Example:

```text
yang2021zebrafish
```

---

# 69. Duplicate Management

When the same study appears through multiple sources:

- retain one canonical citation,
- link preprint and published version if useful,
- prioritize the final peer-reviewed version.

Dataset publications and method publications may be retained separately if they are genuinely distinct.

---

# 70. Search Log

The reproducible search log is stored at:

```text
references/search-log.csv
```

Required fields:

```csv
date,database,query,results_screened,papers_retained,notes
```

Rules:

- do not invent retrospective result counts,
- leave unknown counts blank,
- add new targeted searches as they occur,
- record searches that weaken novelty as carefully as searches that support it,
- refresh fast-moving SSL/preprint searches before manuscript submission.

Current log coverage includes:

- zebrafish unsupervised behavioral clustering,
- zebrafish representation learning,
- direct zebrafish self-supervised learning,
- zebrafish temporal contrastive learning,
- pose/tracking resources,
- conventional behavioral metrics,
- animal pose SSL,
- BEAST / temporal contrastive behavioral SSL,
- identity leakage,
- session/domain leakage,
- speed confounds,
- behavioral temporal-scale selection,
- bibliographic verification of core papers.

---

# 71. Literature Backlog

Initial literature tasks:

- [x] Deep-read Scholz et al. (2025).
  - Confirmed 15-keypoint larval pose schema, annotated free-swimming/head-embedded videos, DeepLabCut/SLEAP pretrained networks, and cross-imaging-condition pose evaluation.

- [x] Deep-read Deligkaris et al. (2026).
  - Confirmed 3D tracking of three anatomical landmarks plus persistent identity, 173 five-hour dyadic recordings at 140 Hz, public Zenodo data/code, and explicit residual identity/body-part error warnings.

- [x] Deep-read Yang et al. (2021).
  - Confirmed 3D trajectory reconstruction, temporal segmentation, PCA, FuzzyART clustering, grid-search parameter selection, and electric-stimulus association testing.

- [x] Deep-read Barreiros et al. (2021).
  - Confirmed automated detection/tracking and conventional metrics including distance traveled, speed, route/spatial behavior, and polarization under conditioning/stimulus manipulations.

- [x] Review conventional zebrafish locomotion features.
  - Speed, distance, activity/immobility, timing, turning, orientation, spatial occupancy, and pose/postural metrics are well supported.

- [x] Review AquaMaze behavioral metrics.
  - Reviewed 2026 AquaMaze framework; supports swim distance, speed, quadrant occupancy, rest/activity and assay-specific spatial analyses.

- [x] Review StrIPETrack.
  - Reviewed 2026 StrIPETrack; validated activity tracking against earlier software and manual Y-maze tracking; supports spatial preference, transitions and turn/navigation metrics.

- [x] Review DeepLabCut zebrafish use.
  - Direct larval zebrafish precedent verified through Scholz et al.

- [x] Review SLEAP zebrafish use.
  - Direct larval precedent verified through Scholz et al.; adult multi-animal/3D workflow precedent verified through Deligkaris et al.

- [x] Search zebrafish unsupervised behavioral clustering.
  - Strong direct precedent identified: Marques et al. (2018), Ghosh & Rihel (2020), Yang et al. (2021).

- [x] Search zebrafish representation learning.
  - Direct latent-sequence representation precedent identified.

- [x] Search zebrafish self-supervised learning.
  - Xu & Wang (2024) identified as direct zebrafish masked skeleton-sequence SSL precedent.

- [x] Search computational ethology behavioral discovery.
  - Cross-species literature confirms clustering, motifs/syllables, latent behavior representations, and reproducibility concerns.

- [x] Search animal pose SSL.
  - ContrastivePose identified and deep-read as direct self-supervised pose-feature precedent.

- [x] Search temporal contrastive learning for behavior.
  - ContrastivePose and BEAST provide strong adjacent precedent; BEAST explicitly combines masked autoencoding with temporal contrastive learning.

- [x] Search identity leakage controls.
  - Identity prediction and subject-disjoint evaluation retained as appropriate explicit nuisance diagnostics; zebrafish tracking literature also documents identity-swap risk.

- [x] Search session/domain leakage controls.
  - Cross-session/domain evaluation, nuisance prediction, and fish-aware splitting retained as best-practice methodological controls. No single zebrafish paper was found that supplies the complete intended session-leakage battery.

- [x] Search speed-confound controls.
  - Speed/activity is a dominant conventional behavioral variable; Ghosh & Rihel explicitly distinguish some higher-order structure from overall activity. Project will use stronger direct speed-prediction/correlation and speed-matched controls.

- [x] Search behavioral-window selection methods.
  - Natural bout segmentation and fixed windows both have precedent; no universal duration exists.

- [x] Build baseline-feature evidence table.
  - Completed in this protocol and `docs/literature.md`; frozen Input A is supported by conventional zebrafish measures.

- [x] Build novelty matrix.
  - Novelty boundary now distinguishes existing zebrafish unsupervised discovery, existing direct zebrafish SSL, and the still-unmatched controlled representation-comparison framework.

- [x] Build SSL candidate-method comparison.
  - Contrastive pose learning, masked sequence modeling, and masked-video + temporal contrastive learning reviewed. Temporal contrastive learning with a small 1D CNN selected for the first experiment.

- [x] Update dataset register from verified repositories.
  - Dataset review informed the register; DS-005 is now frozen as primary.

## Literature Backlog Status

```yaml
initial_backlog_status: COMPLETE
remaining_search_mode: TARGETED_ONLY
blocking_literature_unknowns: NONE
research_gap_status: FROZEN
```

Future literature searches should be triggered by specific methodological issues or newly discovered overlapping papers rather than continuing broad collection indefinitely.

---

# 72. Exit Criteria for Literature Phase

The literature phase is sufficiently mature when:

- [x] At least one defensible primary dataset candidate exists.
  - DS-005 is frozen as the primary dataset.

- [x] Reuse authorization has been investigated.
  - Dataset authorization/licensing was investigated during dataset verification and is tracked in `docs/dataset-register.md`.

- [x] The conventional Input A feature families are justified.
  - Timing, speed, acceleration/speed-change, and orientation/turning are literature-supported and computable reliably from DS-005.

- [x] At least one practical SSL method is identified.
  - Temporal contrastive learning is selected; masked sequence modeling remains direct zebrafish precedent/alternative.

- [x] A primary unsupervised discovery approach can be selected.
  - Baseline discovery selection has been completed using TRAIN/VALIDATION only; PCA + GMM with k=2 is frozen for the handcrafted baseline.
  - SSL discovery uses the same governed candidate family and will be selected using TRAIN/VALIDATION embeddings only after full SSL training.

- [x] Prior zebrafish ML work has been mapped.
  - Yang, Marques, Ghosh & Rihel, Barreiros, tracking/pose resources, and direct SSL precedent are incorporated.

- [x] The novelty boundary is documented.
  - The contribution is a controlled handcrafted-vs-SSL representation comparison under matched discovery and explicit nuisance/reproducibility controls.

- [x] Major validity threats are supported by literature or methodological rationale.
  - Identity leakage, session/context leakage, speed dependence, tracking artifacts, segmentation effects, and clustering instability are all explicitly represented in the protocol.

- [x] Window-duration choices have literature precedent or clear justification.
  - Both fixed windows and natural bouts have precedent; DS-005 naturally supplies behavioral bouts.

- [x] The research question can be frozen.
  - Frozen question: whether self-supervised temporal representations reveal reproducible behavioral structure not captured by conventional handcrafted locomotion/pose features.

- [x] The preregistration draft can be completed without major unknown methodological decisions.
  - Dataset, split, Input A, candidate/frozen Input B, normalization, augmentation, encoder family, contrastive objective, clustering governance, validation framework and claim threshold are defined.

## Literature Phase Decision

```yaml
literature_phase_exit_criteria: MET
literature_phase_status: SUFFICIENT_FOR_PREREGISTRATION
broad_searches_required: false
targeted_followup_searches_allowed: true
```

The literature review remains a living scholarly document, but additional broad searching is no longer a blocker for model development or preregistration.

---

# 73. Literature Phase Definition of Done

The literature review is ready to support preregistration when the project can complete the following statement with citations:

> Existing zebrafish behavioral research commonly represents behavior using **[FEATURES]**, while related computational ethology work has used **[METHODS]** to learn or discover behavioral structure. Prior work most closely overlapping this project includes **[STUDIES]**. However, **[VERIFIED RESEARCH GAP]** remains insufficiently studied. Therefore, this project will compare **[INPUT A]** with **[INPUT B]** using **[DISCOVERY METHOD]**, and evaluate the difference using **[VALIDATION FRAMEWORK]**.

If those blanks cannot yet be filled confidently, literature review should continue.

---

# 74. Current Working Gap

The research gap is now frozen for preregistration at the following bounded scope:

> **Prior work establishes unsupervised zebrafish behavioral discovery and self-supervised behavioral representation learning, including direct zebrafish SSL precedent. However, the targeted literature review did not identify a study directly comparing conventional hand-engineered zebrafish behavioral representations against self-supervised temporal representations on the same observations under matched unsupervised discovery and explicit controls for held-out fish, identity leakage, context/session effects, locomotor speed, tracking artifacts, and representation stability.**

This gap does **not** claim:

- that zebrafish behavioral clustering is new,
- that zebrafish SSL is new,
- that temporal representation learning is new,
- or that any resulting cluster is automatically a novel biological behavior.

The intended contribution is a controlled representation comparison and validation framework.

---

# 75. Current Protocol Status

```yaml
zebrafish_behavior_search: SUFFICIENT_FOR_PREREGISTRATION
zebrafish_tracking_search: SUFFICIENT_FOR_PREREGISTRATION
zebrafish_unsupervised_search: SUFFICIENT_FOR_PREREGISTRATION
animal_ssl_search: SUFFICIENT_FOR_PREREGISTRATION
temporal_ssl_search: SUFFICIENT_FOR_PREREGISTRATION
validation_search: SUFFICIENT_FOR_PREREGISTRATION_WITH_TARGETED_FOLLOWUP
dataset_search: COMPLETE_FOR_PRIMARY_SELECTION
novelty_assessment: FROZEN_BOUNDED_CLAIM
research_gap_status: FROZEN
primary_dataset_status: FROZEN
primary_dataset: DS-005
baseline_feature_set: FROZEN
ssl_method_status: FROZEN_V1
ssl_objective: TEMPORAL_CONTRASTIVE_NT_XENT
ssl_encoder_family: SMALL_1D_TEMPORAL_CNN
discovery_governance: FROZEN_TRAIN_VALIDATION_ONLY
baseline_discovery_method_status: FROZEN
baseline_discovery_selection: PCA_6_COMPONENTS_PLUS_GMM_K2
ssl_discovery_governance_status: FROZEN
ssl_discovery_selected_configuration_status: PENDING_FULL_TRAIN_VALIDATION_EMBEDDINGS
test_partition_status: PROTECTED_NOT_USED_FOR_SELECTION
literature_phase_exit_criteria: MET
bibliography_status: CREATED
search_log_status: CREATED
paper_notes_status: CREATED
preprint_refresh_required_before_manuscript: true
```

Broad literature search is now considered complete for preregistration purposes. New literature should still be added when it:

- directly overlaps the frozen research gap,
- changes a methodological assumption,
- supplies a stronger validation control,
- or appears before manuscript submission and materially affects novelty wording.

---


# 76. Verified Core References for Current Protocol

The following references were verified during the current update and should be prioritized in `references/papers.bib`.

1. Marques JC, Lackner S, Félix R, Orger MB. **Structure of the Zebrafish Locomotor Repertoire Revealed with Unsupervised Behavioral Clustering.** *Current Biology*. 2018;28(2):181-195.e5. DOI: 10.1016/j.cub.2017.12.002.

2. Ghosh M, Rihel J. **Hierarchical Compression Reveals Sub-Second to Day-Long Structure in Larval Zebrafish Behavior.** *eNeuro*. 2020;7(4):ENEURO.0408-19.2020. DOI: 10.1523/ENEURO.0408-19.2020.

3. Yang P, Takahashi H, Murase M, Itoh M. **Zebrafish behavior feature recognition using three-dimensional tracking and machine learning.** *Scientific Reports*. 2021;11:13492. DOI: 10.1038/s41598-021-92854-0.

4. Barreiros MO, Barbosa FG, Dantas DO, et al. **Zebrafish automatic monitoring system for conditioning and behavioral analysis.** *Scientific Reports*. 2021;11:9330. DOI: 10.1038/s41598-021-87502-6.

5. Zhou T, Cheah CCH, Chin EWM, et al. **ContrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavorial classification of interacting animals.** *Computers in Biology and Medicine*. 2023;165:107416. DOI: 10.1016/j.compbiomed.2023.107416.

6. Xu L, Wang S. **Masked Skeleton Sequence Modeling for Learning Larval Zebrafish Behavior Latent Embeddings.** Technical report. arXiv:2403.15693 (2024).

7. Scholz LA, Mancienne T, Stednitz SJ, Scott EK, Lee CCY. **Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses.** bioRxiv. 2025. DOI: 10.1101/2025.06.04.657938.

8. Wang Y, Yu H, Blau A, et al. **Animal behavioral analysis and neural encoding with transformer-based self-supervised pretraining (BEAST).** arXiv:2507.09513 (2025).

9. Deligkaris K, Neiman R, Hiroi M, et al. **A dataset of fine-grained zebrafish interactions in health and disease.** *Scientific Data*. 2026;13:583. DOI: 10.1038/s41597-026-06953-6.

10. Cummings CE, Bastien BL, Martinez JA, Luo J, Thyme SB. **StrIPETrack: a real-time, ROI-flexible tracking platform for high-throughput zebrafish behavior.** *Biology Open*. 2026;15(4):bio062503. DOI: 10.1242/bio.062503.

11. Ayık AS, Aydoğan C, Yılmaz BD, Arslan A. **AquaMaze: A Computer Vision-Based Framework for Automated Behavioral Analysis of Zebrafish in Controlled Environments.** *Zebrafish*. 2026. DOI: 10.1177/15458547251408038.

## Evidence-Certainty Note

The current literature search supports a **bounded novelty statement**, not proof of universal absence. Fields such as computational ethology and self-supervised learning evolve quickly; a final novelty refresh should be performed immediately before manuscript submission.

---


# 77. Literature Reproducibility Artifacts

The following structured literature artifacts are now maintained:

```text
references/
├── papers.bib
├── search-log.csv
└── paper-notes/
    ├── README.md
    ├── UBD-001-marques2018.md
    ├── UBD-002-ghosh2020.md
    ├── ZF-001-yang2021.md
    ├── ZF-002-barreiros2021.md
    ├── ZF-003-aquamaze2026.md
    ├── POSE-001-scholz2025.md
    ├── POSE-002-stripetrack2026.md
    ├── DATA-001-deligkaris2026.md
    ├── SSL-001-contrastivepose2023.md
    ├── SSL-002-xu2024.md
    └── SSL-003-beast2026.md
```

## Maintenance Rules

- Add a paper note when a source materially affects novelty, baseline, SSL method, validation, or dataset decisions.
- Use `NR` when a validation element was not identified; do not silently convert absence of evidence into `NO`.
- Update `references/search-log.csv` for new targeted searches.
- Refresh Xu & Wang, Scholz et al., and any other preprint/technical-report status before manuscript submission.
- Re-run a narrow novelty search immediately before manuscript submission.
- If a newly published study matches the complete frozen research gap, update the gap wording and record the change in `docs/decisions.md`.

---
