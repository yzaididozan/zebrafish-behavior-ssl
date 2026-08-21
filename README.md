# Zebrafish Behavior SSL

> Self-supervised temporal representation learning for reproducible discovery of zebrafish behavioral motifs.

## Overview

This repository contains code, experiments, and documentation for a research project investigating whether **self-supervised temporal representations learned from zebrafish video or pose sequences reveal stable behavioral structure beyond conventional hand-engineered locomotion and pose features**.

The project directly compares two representations derived from the same zebrafish behavioral samples:

- **Input A — Baseline:** hand-engineered locomotion, pose, and tail-dynamics features.
- **Input B — Learned:** self-supervised temporal representations learned from zebrafish video or pose sequences.

The goal is not simply to generate clusters.

The goal is to determine whether learned representations reveal **reproducible behavioral structure that is not adequately captured by conventional features**, while explicitly testing alternative explanations such as fish identity, recording conditions, locomotor speed, tracking errors, and clustering instability.

---

## Research Question

**Can self-supervised temporal representation learning from zebrafish video or pose support the reproducible discovery of behavioral motifs beyond those captured by conventional human-defined behavioral categories?**

---

## Research Hypothesis

A self-supervised temporal representation learned from unlabeled zebrafish video or pose sequences will preserve established behavioral structure while exposing additional latent states or motifs that recur across independent animals or recordings.

Compared with handcrafted kinematic or pose features, learned representations are hypothesized to produce behavioral structure with:

- greater cross-animal reproducibility;
- greater temporal coherence;
- alignment with independent kinematic changes;
- resistance to fish-identity and recording-session confounds;
- information not reducible to locomotor speed;
- behavioral structure not fully represented by conventional engineered features.

---

## Novelty Scope

This project does **not** claim that:

- unsupervised zebrafish behavior discovery is new;
- learned zebrafish behavioral representations are new;
- self-supervised learning for zebrafish representations is entirely new;
- data-driven methods have never identified zebrafish structure beyond human-defined categories.

Previous work has separately demonstrated these capabilities.

Instead, this project investigates their comparatively underexplored intersection:

> **Whether self-supervised temporal representations learned from unlabeled zebrafish behavior provide a better substrate for reproducible open-ended behavioral motif discovery than conventional hand-engineered locomotion and pose representations.**

Any newly identified state will initially be described as a **candidate behavioral motif**, not a newly discovered biological behavior.

---

## Experimental Design

```text
                Zebrafish Recording
                        |
                        v
                Temporal Samples
                        |
               +--------+--------+
               |                 |
               v                 v
          INPUT A             INPUT B
      Hand-Engineered     Self-Supervised
         Features          Representation
               |                 |
               +--------+--------+
                        |
                        v
              Same Discovery Methods
                        |
               +--------+--------+
               |                 |
               v                 v
          A Structure       B Structure
               |                 |
               +--------+--------+
                        |
                        v
                   Comparison
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
      Stability      Temporal      Confound
                    Validity       Testing
          |             |             |
          +-------------+-------------+
                        |
                        v
                Held-Out Validation
                        |
                        v
             Candidate Behavioral
                    Motifs
```

---

## Unit of Analysis

The primary representation unit will be a **temporally contiguous behavioral window or naturally occurring locomotor bout from a single zebrafish**.

The exact primary temporal duration will be preregistered before final experiments.

### Important Units

| Purpose | Unit |
|---|---|
| Representation | Behavioral window / bout |
| Clustering | Behavioral window / bout |
| Statistical inference | Fish or independent recording |
| Train/validation/test split | Fish or recording |

Frames from the same fish must **never be randomly distributed across training and test sets**.

This prevents models from exploiting fish identity rather than learning generalizable behavior.

---

# Input A — Hand-Engineered Baseline

Input A represents conventional behavioral quantification.

The baseline should be intentionally strong rather than artificially simplified.

## Locomotion Features

Candidate variables include:

- centroid speed;
- acceleration;
- jerk;
- distance traveled;
- displacement;
- angular velocity;
- turn angle;
- path curvature;
- bout duration;
- immobility fraction.

## Pose Features

Where keypoint tracking is available:

- body orientation;
- head-tail angle;
- body curvature;
- inter-keypoint distances;
- keypoint velocity;
- keypoint acceleration;
- relative keypoint angles.

## Tail Dynamics

Especially for larval zebrafish:

- tail curvature;
- maximum curvature;
- tail-beat frequency;
- tail-beat amplitude;
- curvature propagation;
- left-right asymmetry;
- tail angular velocity.

## Temporal Summaries

Window-level statistics may include:

- mean;
- standard deviation;
- minimum;
- maximum;
- quantiles;
- spectral power;
- autocorrelation;
- temporal change.

Final feature definitions will be frozen before evaluation on the held-out test set.

---

# Input B — Self-Supervised Representation

Input B will be generated from the **same temporal behavioral samples** used to generate Input A.

Behavior labels will not be used during representation learning.

## Pose-Based SSL

Example input:

```text
T frames × K keypoints × coordinates
```

Potential model families include:

- masked skeleton transformers;
- temporal transformers;
- masked sequence autoencoders;
- temporal convolutional encoders;
- predictive sequence encoders;
- temporal contrastive models.

Pose-based SSL is a strong primary experiment because Input A and Input B can receive essentially the same underlying information.

---

## Raw-Video SSL

A secondary experiment may operate directly on video:

```text
T frames × height × width × channels
```

Potential approaches include:

- masked video modeling;
- masked patch reconstruction;
- temporal contrastive learning;
- future-state prediction;
- spatiotemporal transformers.

This experiment can test whether retaining visual information provides useful behavioral structure that is discarded by pose extraction.

---

# Behavioral Discovery

Representation learning and behavioral discovery are treated as **separate stages**.

Whenever possible, Input A and Input B will be passed through the same discovery pipeline.

This prevents downstream clustering choices from unfairly favoring one representation.

## Primary Discovery Method

### HDBSCAN

HDBSCAN is a candidate primary clustering method because it:

- does not require a fixed number of clusters;
- supports irregular cluster geometry;
- explicitly identifies noise;
- avoids forcing every observation into a behavioral state.

---

## Temporal Discovery

A temporal state model may also be evaluated, such as:

- Hidden Markov Model (HMM);
- Hidden Semi-Markov Model (HSMM);
- another preregistered sequential segmentation method.

This allows comparison between:

```text
Representation
      |
      v
Geometric Clustering
      |
      v
Behavioral Clusters
```

and:

```text
Representation Sequence
        |
        v
Temporal State Model
        |
        v
Behavioral Motifs
```

---

# Comparison

The primary scientific question is:

> **Does the self-supervised representation contain reproducible behavioral structure that is not adequately captured by the hand-engineered baseline?**

Several complementary comparisons will be used.

---

## 1. Known-Behavior Recovery

Where independent behavioral labels are available, evaluate whether discovered structure recovers established zebrafish behaviors.

Possible metrics include:

- Adjusted Mutual Information (AMI);
- Normalized Mutual Information (NMI);
- Adjusted Rand Index (ARI);
- homogeneity;
- completeness;
- cluster purity.

Labels are used for **evaluation only**.

They are not used to train the self-supervised representation.

---

## 2. Cross-Fish Stability

Candidate behavioral states should recur across independent fish.

The analysis will determine whether cluster structure remains similar across:

- held-out animals;
- bootstrap samples;
- model seeds;
- clustering seeds.

---

## 3. Cross-Session Stability

Where possible, states will also be evaluated across:

- recording sessions;
- experimental days;
- cameras;
- lighting conditions;
- recording environments.

---

## 4. Temporal Validity

Inferred behavioral transitions will be compared with independent kinematic changes.

Candidate signals include:

- acceleration peaks;
- angular-velocity changes;
- tail-curvature transitions;
- optical-flow changes;
- stimulus-aligned behavioral changes.

Possible metrics include boundary precision/recall and temporal distance to independently detected changepoints.

---

## 5. Added Information Beyond Input A

Additional clusters produced by Input B do not automatically represent additional behavioral information.

A probe model may therefore attempt to predict Input B cluster assignments using only Input A features.

```text
Hand-Engineered Features
          |
          v
     Probe Classifier
          |
          v
Predict SSL-Derived State
```

If Input A predicts a supposedly new Input B state almost perfectly, that state provides limited evidence for additional representational structure.

---

# Validation Framework

Validation will occur across several independent axes.

## Technical Validity

Candidate structure should survive reasonable changes in:

- model initialization;
- random seed;
- data resampling;
- clustering seed;
- clustering parameters.

---

## Confound Validity

Representations will be tested for information about variables that should not define behavior.

Probe models may attempt to predict:

- fish identity;
- recording session;
- camera;
- lighting condition;
- average speed;
- distance traveled;
- tracking confidence.

A useful behavioral representation should not produce clusters primarily explained by these nuisance variables.

---

## Behavioral Validity

Candidate motifs should demonstrate coherent:

- pose dynamics;
- locomotor dynamics;
- temporal duration;
- within-cluster similarity;
- between-cluster differences.

---

## External Validity

Where metadata permit, states may be compared with independent experimental variables such as:

- prey strikes;
- sensory stimuli;
- social conditions;
- light/dark transitions;
- pharmacological conditions;
- experimental manipulations.

These variables should not be used to create the clusters.

---

## Expert Review

Potentially novel motifs may undergo blinded human review.

Reviewers should not be shown:

- cluster IDs;
- experimental condition;
- fish identity;
- representation type.

This reduces post-hoc storytelling.

---

# Threats to Validity

## Identity Leakage

**Threat:** The model learns individual fish rather than behavior.

### Controls

- split datasets by fish;
- never randomly split frames across train/test;
- train probes to predict fish identity;
- require candidate motifs to occur across multiple animals.

---

## Session / Camera Leakage

**Threat:** Lighting, tank appearance, camera angle, compression, or recording setup determines representation structure.

### Controls

- split by recording session;
- test session prediction from embeddings;
- standardize preprocessing;
- evaluate cross-session generalization;
- evaluate camera/session composition within clusters.

---

## Speed-Only Solution

**Threat:** The learned representation primarily encodes locomotor speed.

### Controls

- calculate embedding-speed relationships;
- train speed probes;
- include speed-only baselines;
- speed-match observations;
- residualize speed and repeat discovery;
- determine whether behavioral structure persists after controlling for speed.

---

## Tracking Artifacts

**Threat:** Occlusion, incorrect keypoints, or tracking jitter creates false behavioral states.

### Controls

- retain tracking-confidence values;
- filter low-confidence frames;
- inspect artifact-enriched clusters;
- repeat discovery after quality filtering;
- compare results across tracking pipelines where possible.

---

## Window-Boundary Artifacts

**Threat:** Cluster membership depends on arbitrary clip boundaries.

### Controls

- preregister a primary window length;
- evaluate alternative window lengths;
- shift temporal starting positions;
- compare overlapping and non-overlapping windows;
- measure assignment consistency.

---

## Hyperparameter Fishing

**Threat:** Interesting clusters occur only under a convenient seed or clustering configuration.

### Controls

- preregister primary algorithms;
- preregister primary hyperparameter ranges;
- evaluate multiple seeds;
- report robustness across configurations;
- avoid selecting results solely because they appear biologically interesting.

---

## Post-Hoc Storytelling

**Threat:** Researchers assign biological interpretations after seeing visually attractive clusters.

### Controls

- preregister interpretation criteria;
- use blinded expert review;
- separate discovery from biological interpretation;
- preserve a held-out validation set;
- classify additional states as candidate motifs until independently validated.

---

# Claim Thresholds

Finding a cluster is not sufficient evidence for discovering a behavior.

Claims will therefore be made at different evidence levels.

## Level 1 — Additional Cluster

The learned representation produces a distinguishable cluster not produced by the baseline.

This is a computational result only.

---

## Level 2 — Reproducible Latent Behavioral State

A state may receive this designation if it:

- appears in held-out animals;
- appears across recordings;
- survives repeated model runs;
- survives reasonable clustering configurations;
- is not dominated by a single fish;
- is not dominated by a single session;
- is not explained primarily by locomotor speed;
- remains after tracking-quality controls.

---

## Level 3 — Candidate Novel Behavioral Motif

A stronger claim additionally requires:

- reproducible temporal structure;
- coherent observable movement;
- distinction from established behavioral categories;
- evidence that hand-engineered features do not fully account for the state;
- replication in held-out animals or recordings;
- independent expert or experimental validation.

The term **candidate novel behavioral motif** will be preferred over **new zebrafish behavior** unless substantially stronger biological evidence becomes available.

---

# Preregistration

Before final experiments, the following will be frozen.

## Research Questions

- Primary research question
- Secondary research questions
- Primary hypothesis
- Secondary hypotheses

## Dataset Rules

- dataset inclusion criteria
- dataset exclusion criteria
- recording-quality requirements
- tracking-quality requirements

## Data Splits

Train, validation, and test partitions will occur at the:

- fish level; or
- independent recording level.

Random-frame splitting across partitions is prohibited.

## Models

Preregister:

- hand-engineered baseline;
- primary SSL model family;
- secondary SSL models;
- clustering algorithms;
- temporal discovery models.

## Evaluation

Preregister:

- primary metric;
- secondary metrics;
- number of random seeds;
- number of repeated experiments;
- ablations;
- confound tests;
- stability threshold;
- candidate-motif criteria.

## Statistics

Preregister:

- statistical tests;
- effect-size measures;
- confidence intervals;
- multiple-comparison correction where applicable.

---

# Planned Ablations

Candidate ablations include:

- hand-engineered features only;
- speed only;
- pose without temporal information;
- masked reconstruction only;
- temporal objective only;
- combined reconstruction + temporal learning;
- different embedding dimensionalities;
- different temporal window lengths;
- alternative clustering methods;
- tracking-quality filtering;
- speed residualization.

---

# Candidate Datasets

Potential authorized/open datasets include:

- **Scholz et al. zebrafish larvae pose/video dataset**
- **Marques et al. zebrafish locomotor repertoire dataset**
- **PoseR / Mullen et al. larval zebrafish dataset**
- **Larval zebrafish prey-capture datasets**
- **Zebrafish social-experience datasets**

Dataset licenses and reuse conditions must be verified before inclusion.

Raw research data should generally not be committed directly to this repository.

---

# Planned Repository Structure

```text
zebrafish-behavior-ssl/
|
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
|
├── configs/
|   ├── baseline/
|   ├── ssl/
|   └── clustering/
|
├── data/
|   ├── README.md
|   ├── raw/
|   ├── interim/
|   └── processed/
|
├── docs/
|   ├── literature/
|   ├── preregistration/
|   └── methodology/
|
├── notebooks/
|   ├── exploratory/
|   └── validation/
|
├── src/
|   ├── data/
|   ├── tracking/
|   ├── features/
|   ├── models/
|   ├── clustering/
|   ├── evaluation/
|   └── visualization/
|
├── scripts/
|   ├── preprocess/
|   ├── train/
|   ├── discover/
|   └── evaluate/
|
├── tests/
|
└── results/
    ├── embeddings/
    ├── clusters/
    ├── validation/
    ├── figures/
    └── tables/
```

---

# Reproducibility

The project aims to make experimental results reproducible through:

- version-controlled configuration files;
- fixed random seeds;
- recorded train/validation/test splits;
- saved model configurations;
- documented preprocessing;
- automated evaluation scripts;
- full reporting of negative and null results;
- explicit dataset provenance.

Final test data should remain untouched until model and analysis decisions are frozen.

---

# Project Status

> **Status: Research design / literature review / pipeline development**

Current work includes:

- [x] Define working research question
- [x] Define working hypothesis
- [x] Conduct initial literature landscape review
- [x] Search for novelty-threatening prior work
- [x] Identify candidate public zebrafish datasets
- [x] Define baseline vs learned comparison
- [x] Identify major threats to validity
- [ ] Select primary dataset
- [ ] Define dataset inclusion/exclusion criteria
- [ ] Finalize unit of analysis
- [ ] Implement preprocessing pipeline
- [ ] Implement hand-engineered baseline
- [ ] Select primary SSL architecture
- [ ] Implement self-supervised representation pipeline
- [ ] Freeze preregistration
- [ ] Run development experiments
- [ ] Freeze final analysis pipeline
- [ ] Run held-out evaluation
- [ ] Validate candidate motifs
- [ ] Prepare manuscript

---

# Related Methods

The project is informed by work involving:

- zebrafish locomotor repertoire discovery;
- DeepLabCut;
- SLEAP;
- MotionMapper;
- B-SOiD;
- VAME;
- MoSeq;
- Keypoint-MoSeq;
- BehaveNet;
- TREBA;
- BEAST;
- masked skeleton modeling;
- self-supervised video representation learning.

A formal bibliography will be maintained separately.

---

# Ethical and Data-Use Considerations

This project is designed primarily around **previously collected and authorized research datasets**.

No new animal experiments are required for the initial computational study.

Each dataset will be used according to its:

- license;
- repository terms;
- citation requirements;
- original study documentation.

The repository will not redistribute third-party data unless its license explicitly permits redistribution.

---

# Contributing

This repository is currently maintained as a research project.

Issues documenting:

- reproducibility problems;
- dataset inconsistencies;
- implementation bugs;
- methodological concerns;
- relevant prior work

are welcome.

---

# Citation

Citation information will be added if this work results in a preprint, publication, or public software release.

A `CITATION.cff` file will be included before the first research release.

---

# License

A software license will be selected before public release.

Dataset licenses remain independent of the license applied to this repository.

---

## Disclaimer

Behavioral clusters produced by this project should not automatically be interpreted as distinct biological behaviors.

The purpose of the project is to identify **candidate reproducible behavioral structure** and subject that structure to rigorous computational and biological validation.
