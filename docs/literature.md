# Literature Review

## Project

**Self-Supervised Discovery of Zebrafish Behavioral Structure**

## Document Status

```yaml
document_status: ACTIVE
review_type: structured_scoping_synthesis
last_updated: 2026-08-21
research_gap_status:  FROZEN
baseline_status: PROVISIONAL
ssl_method_status: FROZEN_V1
primary_dataset_status: FROZEN
```

---

# 1. Purpose

This document synthesizes the literature relevant to the central research question:

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

The literature review is organized around three questions that directly determine the research design:

1. **What has already been done in zebrafish behavioral machine learning?**
2. **What should Input A contain to constitute a fair conventional behavioral baseline?**
3. **What methodological gap remains after accounting for the strongest overlapping work?**

This document is a synthesis rather than a paper-by-paper reading log.

Detailed search procedures are defined in:

```text
docs/literature-protocol.md
```

Dataset-specific information is maintained in:

```text
docs/dataset-register.md
```

The experimental consequences of the literature are reflected in:

```text
docs/research-question.md
docs/preregistration-draft.md
```

---

# 2. Executive Summary

The literature establishes that computational analysis of zebrafish behavior is already well developed enough that this project cannot claim novelty merely from using:

- machine learning,
- automated tracking,
- clustering,
- three-dimensional trajectories,
- pose estimation,
- or computational behavioral features.

Yang et al. (2021), for example, used three-dimensional zebrafish trajectories with a machine-learning method called Fuzzy Adaptive Resonance Theory to identify behavioral features associated with electric shock. This directly demonstrates that machine learning has already been used to extract behavioral structure from zebrafish trajectories.

Barreiros et al. (2021) developed an automated zebrafish conditioning and behavioral-analysis system combining fish detection, tracking, experimental automation, and behavioral measures such as distance traveled, swimming speed, and group polarization. This establishes strong precedent for conventional computational zebrafish analysis based on interpretable locomotor measurements.

Recent zebrafish resources also make substantially richer representations possible. Scholz et al. (2025) released annotated larval zebrafish pose data and pretrained DeepLabCut and SLEAP networks based on a 15-keypoint pose representation, demonstrating that detailed zebrafish posture and tail kinematics can now be recovered without building an entire tracking system from scratch. The work is currently a bioRxiv preprint rather than a peer-reviewed journal article.

Deligkaris et al. (2026) greatly extends the available behavioral data scale for adult social zebrafish, providing 173 approximately five-hour dyadic experiments, three-dimensional tracking of three anatomical landmarks, identity tracking, and recordings sampled at 140 Hz. This makes sophisticated temporal representation learning feasible, although its dyadic structure introduces additional identity and social-interaction confounds that make it less attractive for the simplest first experiment.

Self-supervised representation learning is also no longer novel by itself in animal behavioral analysis. ContrastivePose demonstrated self-supervised contrastive feature learning from pose data and showed that learned features could outperform handcrafted features for downstream animal behavioral classification. More recently, BEAST combined masked autoencoding with temporal contrastive learning for behavioral video and evaluated learned representations across multiple species and behavioral-analysis tasks.

Therefore, the strongest currently defensible contribution is not:

> "Use machine learning to discover zebrafish behavior."

Nor is it:

> "Use self-supervised learning for animal behavior."

Instead, the working contribution is:

> **Perform a controlled comparison between conventional hand-engineered zebrafish behavioral representations and self-supervised temporal representations on the same observations, and determine whether SSL reveals reproducible behavioral structure that remains after explicit controls for individual identity, recording session, locomotor speed, tracking artifacts, temporal segmentation, and analytical instability.**
---

# 3. Zebrafish Behavioral Measurement

## 3.1 Conventional Behavioral Analysis

Zebrafish behavior has traditionally been quantified through interpretable measurements of movement and spatial organization.

The most basic representation of zebrafish behavior begins with trajectory:

```text
time
  ↓
x/y position
  ↓
movement
  ↓
behavioral summaries
```

From these trajectories, commonly useful quantities include:

- swimming speed,
- distance traveled,
- acceleration,
- movement duration,
- immobility,
- movement bouts,
- heading,
- turning,
- angular velocity,
- trajectory curvature,
- tank occupancy,
- distance from boundaries,
- and vertical or horizontal location when the experimental setup permits it.

These variables remain important because they represent what a conventional analyst could reasonably extract without representation learning.

The SSL model therefore must be compared against them rather than against an artificially weak baseline.

---

# 4. Evidence From Automated Zebrafish Analysis

## Barreiros et al. (2021)

**Citation**

Marta de Oliveira Barreiros, Felipe Gomes Barbosa, Diego de Oliveira Dantas, Daniel de Matos Luna dos Santos, Sidarta Ribeiro, Giselle Cutrim de Oliveira Santos, and Allan Kardec Barros.

**Title**

*Zebrafish automatic monitoring system for conditioning and behavioral analysis*

**Venue**

Scientific Reports, 2021.

**DOI**

`10.1038/s41598-021-87502-6`

---

## 4.1 What They Did

Barreiros et al. developed an automated system that combined:

```text
Experimental Stimulus Control
            +
Video Capture
            +
Fish Detection
            +
Tracking
            +
Behavioral Measurement
```

Their fish-detection system used a YOLOv2-based convolutional architecture.

The resulting behavioral analysis quantified variables including:

- distance traveled,
- speed,
- spatial response to stimuli,
- and polarization in groups.

The system could analyze both individual fish and fish schools.

---

## 4.2 Why This Matters

This paper demonstrates that conventional computational zebrafish analysis already provides a strong baseline based on:

```text
Detection
   ↓
Tracking
   ↓
Explicit behavioral variables
   ↓
Statistical / experimental comparison
```

Therefore, simply replacing manual scoring with computer vision would not be a meaningful contribution for this project.

---

## 4.3 Implication for Input A

At minimum, Input A should contain strong locomotion variables such as:

```text
speed
distance
acceleration
movement / immobility
turning
```

where they can be calculated reliably.

If the final dataset contains multiple animals simultaneously, social variables such as polarization or inter-animal distance may become relevant, but these are not necessary for the initial single-animal design.

---

# 5. Existing Machine Learning for Zebrafish Behavioral Structure

## Yang et al. (2021)

**Citation**

Peng Yang, Hiro Takahashi, Masataka Murase, and Motoyuki Itoh.

**Title**

*Zebrafish behavior feature recognition using three-dimensional tracking and machine learning*

**Venue**

Scientific Reports, 2021.

**DOI**

`10.1038/s41598-021-92854-0`

---

## 5.1 What They Did

Yang et al. recorded zebrafish using two cameras to obtain three-dimensional trajectory information.

They then used:

```text
3D Tracking
     ↓
Trajectory Features
     ↓
Fuzzy Adaptive Resonance Theory
     ↓
Behavioral Feature Recognition
```

The study examined fish exposed to electric shock and identified behavioral features statistically associated with the stimulus.

---

## 5.2 Novelty Consequence

This paper establishes an important boundary.

The present project cannot reasonably claim:

> "Previous zebrafish research only used manually labeled behavior."

It also cannot claim:

> "Machine learning has never been used to identify zebrafish behavioral features."

Both would conflict with existing literature.

---

# 6. What Yang et al. Does Not Eliminate

Yang et al. nevertheless does not appear to answer the exact question proposed here.

The proposed project differs conceptually by explicitly comparing:

```text
HAND-ENGINEERED REPRESENTATION
             vs.
SELF-SUPERVISED REPRESENTATION
```

using the same underlying temporal observations.

The focus is therefore not merely whether ML can recognize behavioral structure.

The focus is whether **representation learning contains reproducible information beyond an explicit conventional baseline**.

This distinction is central to the project's novelty.

---

# 7. Zebrafish Pose Estimation

Trajectory-only measurements do not capture all zebrafish movement.

Two fish may have similar:

- positions,
- speeds,
- acceleration,
- or trajectory directions

while exhibiting substantially different body configurations.

Pose therefore provides a potentially richer description.

---

# 8. Scholz et al. (2025)

**Citation**

Leandro A. Scholz, Tessa Mancienne, Sarah J. Stednitz, Ethan K. Scott, and Conrad C. Y. Lee.

**Title**

*Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses*

**Publication**

bioRxiv preprint, 2025.

**DOI**

`10.1101/2025.06.04.657938`

---

## 8.1 Contribution

Scholz et al. provide:

- annotated larval zebrafish behavioral data,
- free-swimming recordings,
- head-embedded recordings,
- pretrained DeepLabCut models,
- pretrained SLEAP models,
- and a 15-keypoint zebrafish pose representation.

The authors explicitly designed the resource to reduce the substantial annotation and model-training burden associated with zebrafish pose tracking.

---

# 9. Importance of Pose for This Project

Pose information makes it possible to measure aspects of behavior that center-of-mass trajectories alone may miss.

Candidate pose variables include:

```text
body orientation
body curvature
tail curvature
tail movement
tail-beat amplitude
tail-beat frequency
relative keypoint geometry
pose velocity
```

This matters because SSL should not receive an unfair advantage by being compared only against speed when conventional pose estimation already exposes richer behavioral information.

---

# 10. Input A Must Be a Serious Baseline

The literature therefore suggests that Input A should consist of multiple behavioral feature families.

## Tier 1 — Locomotion

Recommended core variables:

- instantaneous speed,
- mean speed,
- median speed,
- maximum speed,
- speed variance,
- acceleration,
- distance traveled,
- immobility,
- movement duration.

---

## Tier 2 — Turning and Trajectory

Recommended when tracking permits:

- heading change,
- angular velocity,
- turning angle,
- turning frequency,
- trajectory curvature.

---

## Tier 3 — Movement Bout Dynamics

Recommended when temporal resolution permits:

- bout duration,
- bout frequency,
- inter-bout interval,
- peak bout velocity,
- bout displacement.

---

## Tier 4 — Pose

Recommended when reliable keypoints are available:

- body orientation,
- body curvature,
- tail curvature,
- tail movement,
- keypoint angles,
- pose velocity.

---

## Tier 5 — Spatial Context

Potential features include:

- tank coordinates,
- distance from boundary,
- center occupancy,
- edge occupancy,
- vertical location.

These require additional caution because spatial position can encode experimental context rather than behavior.

---

# 11. Proposed Baseline Structure

The working Input A representation is therefore:

```text
Behavioral Window
       │
       ├── Locomotion
       │
       ├── Turning
       │
       ├── Bout Dynamics
       │
       ├── Pose
       │
       └── Selected Spatial Variables
                │
                ▼
        Hand-Engineered Vector
```

This should be treated as the conventional reference against which Input B is evaluated.

---

# 12. Why Speed Alone Is Not a Fair Baseline

A speed-only baseline is scientifically useful but intentionally weak.

It answers:

> Does SSL learn anything besides movement intensity?

It does **not** answer:

> Does SSL outperform conventional behavioral analysis?

Therefore the project requires two distinct comparisons:

```text
CONTROL
SSL
vs.
Speed-only baseline
```

and:

```text
PRIMARY COMPARISON
SSL
vs.
Full hand-engineered baseline
```

The distinction must be maintained throughout the project.

---

# 13. Fine-Grained Adult Zebrafish Data

## Deligkaris et al. (2026)

**Citation**

Kosmas Deligkaris, Radmila Neiman, Makoto Hiroi, Tatsuo Izawa, Liam O'Shaughnessy, Luis Carretero Rodriguez, Ichiro Masai, and Greg J. Stephens.

**Title**

*A dataset of fine-grained zebrafish interactions in health and disease*

**Venue**

Scientific Data, 2026.

**DOI**

`10.1038/s41597-026-06953-6`

Dataset DOI:

`10.5281/zenodo.17190142`

---

# 14. What the Dataset Contains

Deligkaris et al. provide an unusually rich adult zebrafish behavioral resource.

The published dataset consists of:

- `173` experiments,
- approximately five-hour recordings,
- adult zebrafish dyads,
- male/male and female/female pairings,
- wild-type and disease-model animals,
- three-dimensional tracking,
- three anatomical landmarks per fish,
- fish identity,
- temporal sampling at `140 Hz`,
- different arena geometries,
- and metadata describing experiments.

The authors also explicitly note that residual identity swaps or landmark misidentifications may remain, especially during close interactions.

---

# 15. Why Deligkaris et al. Is Important

This resource demonstrates that zebrafish datasets now exist at a temporal scale sufficient for serious representation-learning experiments.

It also demonstrates why artifact controls matter.

A learned representation might identify:

```text
tracking failure
```

as easily as:

```text
behavior
```

unless the pipeline explicitly tests for this possibility.

---

# 16. Why It May Not Be the Simplest Primary Dataset

The Deligkaris dataset consists of **dyads** rather than independent single animals.

That changes the representation problem from:

```text
What is Fish A doing?
```

to potentially:

```text
What is Fish A doing?
What is Fish B doing?
What is their relative configuration?
What interaction is occurring?
```

The model could encode:

- Fish A identity,
- Fish B identity,
- pair identity,
- social distance,
- arena geometry,
- genotype,
- interaction outcome,
- or tracking swaps.

For an initial methodological study, these factors substantially increase validation complexity.

---

# 17. Potential Future Social Extension

The dataset nevertheless represents an excellent future extension.

A later project could define the unit of analysis as:

> **a fixed-duration dyadic behavioral window**

and compare handcrafted social features against SSL.

Candidate conventional social features might include:

- inter-fish distance,
- relative heading,
- relative velocity,
- approach rate,
- following distance,
- orientation alignment,
- contact frequency,
- displacement asymmetry.

This is intentionally outside the simplest first experiment.

---

# 18. Self-Supervised Behavioral Representation Learning

A major literature finding is that SSL itself cannot constitute the entire novelty claim.

Self-supervised learning has already been applied to animal pose and behavioral representation.

---

# 19. ContrastivePose

## Zhou et al. (2023)

**Title**

*ContrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavioral classification of interacting animals*

**Venue**

Computers in Biology and Medicine, 2023.

**DOI**

`10.1016/j.compbiomed.2023.107416`

---

## 19.1 Contribution

ContrastivePose directly addresses one of the motivations behind the present project.

The authors identify limitations of:

- manual behavioral labeling,
- and manually designed higher-level pose features.

They use contrastive learning on unlabeled pose information to learn representations and report improved downstream behavioral classification compared with handcrafted features alone.

---

# 20. Implication of ContrastivePose

The project therefore cannot claim:

> "No one has compared self-supervised animal representations against handcrafted features."

Related work already exists.

The remaining question must be narrower and zebrafish-specific.

ContrastivePose is especially important because it validates the conceptual premise that:

```text
Pose Coordinates
        ↓
Self-Supervised Learning
        ↓
Representation
```

can replace or augment manual feature engineering.

---

# 21. Important Difference From the Proposed Study

The proposed research is centered on:

> **unsupervised behavioral structure discovery**

rather than primarily supervised classification performance.

Conceptually:

```text
ContrastivePose-style question:

Learn representation
        ↓
Can it improve labeled behavior classification?
```

versus:

```text
Proposed question:

Learn representation
        ↓
Discover behavioral structure without behavior labels
        ↓
Does that structure exceed what handcrafted features reveal?
```

That distinction should be preserved.

---

# 22. Recent Self-Supervised Behavioral Video Work

## BEAST

Recent work introduces BEAST:

> **BEhavioral Analysis via Self-supervised pretraining of Transformers**

The approach combines:

- masked autoencoding,
- temporal contrastive learning,
- experiment-specific behavioral video pretraining.

It has been evaluated across multiple species and behavioral-analysis tasks including:

- neural encoding,
- pose estimation,
- action segmentation.

The work demonstrates that domain-specific unlabeled behavioral video can support rich self-supervised representations.

---

# 23. Implication of BEAST

The existence of BEAST further narrows the novelty claim.

The project should not claim novelty merely because it applies:

```text
video
+
temporal contrastive learning
+
animal behavior
```

Instead, the contribution must come from the **scientific comparison and validation framework**.

---

# 24. Why Temporal Learning Remains Relevant

Behavior is not merely a pose.

Consider two windows containing identical body configurations but in different order.

```text
Window A

Pose 1 → Pose 2 → Pose 3
```

versus:

```text
Window B

Pose 3 → Pose 2 → Pose 1
```

A static pose representation may view the two sequences as similar.

A temporal model can distinguish their dynamics.

This provides a strong theoretical reason to use a temporal SSL objective rather than only frame-level feature extraction.

---

# 25. Working SSL Direction

The literature currently supports several possible approaches:

```text
Temporal Contrastive Learning
Masked Temporal Reconstruction
Future Prediction
Sequence Autoencoding
Predictive Coding
```

For the first project, complexity should be minimized.

The intended contribution does not require developing a new SSL architecture.

---

# 26. Recommended SSL Selection Principle

Use:

> **the simplest established temporal SSL method that can reliably operate on the selected dataset and produce one embedding per behavioral window.**

The representation should have the form:

```text
Temporal Behavioral Window
            ↓
       SSL Encoder
            ↓
        Embedding
```

The clustering/discovery pipeline can then operate on those embeddings.

---

# 27. Pose vs Raw Video

The literature makes two realistic Input B strategies possible.

## Option A — Pose/Trajectory SSL

```text
Video
  ↓
Tracking / Pose
  ↓
Coordinate Sequence
  ↓
Temporal SSL
```

### Advantages

- computationally simpler,
- easier to interpret,
- reduced appearance/camera leakage,
- directly comparable with handcrafted movement features.

### Disadvantages

- cannot learn information removed by the tracking system,
- vulnerable to pose-estimation errors.

---

## Option B — Video SSL

```text
Video Window
     ↓
Video Encoder
     ↓
Temporal Embedding
```

### Advantages

- preserves visual information,
- potentially captures movement or posture missed by keypoints.

### Disadvantages

- higher computational cost,
- greater risk of identity leakage,
- greater camera/background leakage,
- harder interpretation.

---

# 28. Current Preferred First-Study Direction

If the selected dataset contains reliable tracking or pose information, the literature currently favors beginning with:

> **pose or trajectory sequence SSL**

rather than full raw-video SSL.

This creates a cleaner comparison:

```text
Same tracking / pose information
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼

Manual Features   Temporal SSL

      │           │
      └─────┬─────┘
            ▼

     Compare Structure
```

This controls the source information available to both representations.

---

# 29. Why the Matched-Input Design Is Strong

Suppose Input A uses trajectories while Input B receives raw high-resolution video.

If SSL performs better, it would be difficult to distinguish:

```text
better representation learning
```

from:

```text
more information provided to the model
```

A matched-input first experiment avoids that problem.

Ideally:

```text
Underlying Behavioral Signal
             │
             ├───────────────┐
             ▼               ▼
     Hand Engineering     Temporal SSL
             │               │
             ▼               ▼
        Baseline z_A       Learned z_B
```

---

# 30. Unsupervised Behavioral Discovery

The proposed analysis requires a second distinction:

> Representation learning and behavioral discovery are separate processes.

The pipeline is:

```text
Behavioral Data
      ↓
Representation
      ↓
Embedding / Feature Vector
      ↓
Discovery Algorithm
      ↓
Candidate Behavioral States
```

A model can produce an excellent representation without naturally producing discrete behavioral categories.

---

# 31. Discovery Methods

Reasonable candidate methods include:

- k-means,
- Gaussian mixture models,
- hierarchical clustering,
- HDBSCAN,
- hidden-state models.

The first experiment should use one predefined primary method with limited sensitivity analyses.

The purpose is to compare representations rather than optimize dozens of clustering algorithms.

---

# 32. Why UMAP Is Not Discovery Evidence

A common analytical danger is:

```text
Representation
     ↓
UMAP
     ↓
Pretty Islands
     ↓
"Behaviors discovered"
```

This is insufficient.

Low-dimensional visualization can exaggerate apparent separation.

UMAP or t-SNE should therefore primarily be used for:

```text
exploration
+
visualization
```

rather than primary validation.

---

# 33. The Core Validation Problem

Unsupervised algorithms will produce structure even when the structure reflects nuisance variables.

Potential explanations for a cluster include:

```text
actual behavior
fish identity
speed
camera
session
tank location
tracking failure
window boundaries
random instability
```

Therefore:

> **cluster existence is not evidence of behavioral discovery.**

The scientific contribution depends primarily on eliminating plausible alternative explanations.

---

# 34. Threat 1 — Identity Leakage

Fish differ consistently in:

- size,
- appearance,
- morphology,
- baseline activity,
- movement style,
- pose geometry.

A representation can therefore learn:

```text
Fish 1
Fish 2
Fish 3
```

rather than:

```text
Behavior A
Behavior B
Behavior C
```

---

# 35. Literature-Derived Design Consequence: Split by Fish

Randomly splitting windows is unsafe.

Do not use:

```text
Fish 01
├── training windows
└── testing windows
```

as the primary generalization test.

Prefer:

```text
TRAIN
Fish 01
Fish 02
Fish 03

TEST
Fish 04
Fish 05
```

This is one of the most important design requirements.

---

# 36. Identity Leakage Test

The learned representation should be evaluated for fish identity information.

Conceptually:

```text
SSL Embedding
     ↓
Fish-ID Classifier
     ↓
Prediction Accuracy
```

Additionally, each discovered cluster should be inspected for fish composition.

A cluster dominated by one animal is weaker evidence of a behavioral state.

---

# 37. Threat 2 — Session Leakage

Recordings may differ in:

- lighting,
- camera geometry,
- compression,
- background,
- tank,
- recording date,
- experimental setup.

A representation may cluster these variables.

This risk is especially severe for raw-video SSL.

---

# 38. Session-Control Consequence

Where metadata permit, test:

```text
Embedding
    ↓
Session-ID Classifier
```

and inspect cluster membership by session.

Cross-session replication is stronger evidence than within-session separation.

---

# 39. Threat 3 — Speed-Only Representations

Speed is likely to explain substantial zebrafish behavioral variance.

A learned representation may therefore look complex while merely encoding:

```text
stationary
slow
medium
fast
```

This possibility must be tested directly.

---

# 40. Required Speed Controls

The literature synthesis supports three increasingly strong controls.

## Control A — Speed Correlation

Measure whether embedding coordinates correlate strongly with speed.

---

## Control B — Speed Prediction

Test:

```text
SSL Embedding
     ↓
Regressor
     ↓
Swimming Speed
```

---

## Control C — Speed-Matched Behavior

Compare learned structure among windows with approximately similar speed.

This is the most important conceptual test.

If SSL structure remains within speed-matched windows, the representation is stronger evidence for behavior beyond locomotor intensity.

---

# 41. Threat 4 — Tracking Artifacts

The Deligkaris dataset explicitly acknowledges possible residual:

- identity swaps,
- body-part errors

during difficult close-proximity interactions.

The Scholz work also demonstrates that pose-estimation error varies with imaging conditions and keypoint configuration, reinforcing the need to quantify tracking quality rather than assume tracking is ground truth.

---

# 42. Artifact-Control Consequence

Where possible, derive tracking QC variables such as:

```text
confidence
missing detections
coordinate jumps
impossible body geometry
extreme instantaneous displacement
```

Then test whether cluster membership is related to these variables.

Any cluster consisting primarily of tracking failure should be labeled:

```text
ARTIFACT
```

not behavior.

---

# 43. Threat 5 — Window-Boundary Artifacts

Temporal SSL requires segmentation.

A sequence might be divided into:

```text
0–2 seconds
2–4 seconds
4–6 seconds
```

But a behavioral event can cross those boundaries.

If clusters disappear when the segmentation shifts slightly, the state may be an artifact of window construction.

---

# 44. Window-Control Consequence

The project should compare the primary window against nearby durations and offsets.

For example:

```text
1-second windows
2-second windows
4-second windows
```

or:

```text
Window alignment A
vs.
Window alignment B
```

The goal is robustness, not identical results.

---

# 45. Threat 6 — Hyperparameter Fishing

A flexible pipeline can produce attractive results through repeated adjustment of:

- window size,
- latent dimension,
- clustering method,
- number of clusters,
- random seed,
- UMAP settings,
- architecture.

This makes preregistration particularly important.

---

# 46. Validation Principle

The literature synthesis supports the following hierarchy of evidence:

```text
Weak
│
├── visually separated embedding
├── good internal clustering score
│
├── stable clustering
├── interpretable examples
│
├── replication across fish
├── nuisance controls
│
├── baseline comparison
│
└── independent replication
Strong
```

The target contribution should rely on evidence toward the bottom of this hierarchy.

---

# 47. What Has Already Been Done?

The answer to the first central literature question is now clearer.

## Already Established

### Automated zebrafish tracking

Yes.

---

### Automated zebrafish behavioral measurement

Yes.

Barreiros et al. provide one direct example.

---

### 3D zebrafish behavioral tracking

Yes.

Yang et al. and Deligkaris et al. provide clear examples.

---

### Machine learning applied to zebrafish behavioral features

Yes.

Yang et al. explicitly used FuzzyART on 3D zebrafish trajectory information.

---

### Detailed zebrafish pose estimation resources

Yes.

Scholz et al. provide annotated data and pretrained DeepLabCut/SLEAP models.

---

### Self-supervised animal pose representations

Yes.

ContrastivePose is direct precedent.

---

### Self-supervised behavioral video representations

Yes.

BEAST provides recent multi-species precedent combining masked and temporally contrastive learning.

---

# 48. What Should Input A Contain?

The answer to the second central literature question is:

> **A multi-family behavioral representation rather than a speed-only baseline.**

The provisional baseline should contain:

```yaml
locomotion:
  - speed
  - acceleration
  - distance_traveled
  - movement_duration
  - immobility

turning:
  - heading_change
  - angular_velocity
  - turning_angle
  - trajectory_curvature

bout_dynamics:
  - bout_duration
  - bout_frequency
  - inter_bout_interval

pose_if_available:
  - orientation
  - body_curvature
  - tail_curvature
  - pose_velocity
```

Window-level summary statistics may include:

```yaml
statistics:
  - mean
  - standard_deviation
  - median
  - minimum
  - maximum
  - quantiles
```

---

# 49. Why This Baseline Is Fair

The purpose of Input A is not to make SSL look good.

It should instead answer:

> If a zebrafish researcher carefully engineered conventional locomotion and pose descriptors, would SSL still reveal additional structure?

Only then does an SSL advantage become scientifically interesting.

---

# 50. What Gap Remains?

The literature reviewed so far considerably narrows the research gap.

The gap is **not**:

> Machine learning has not been used for zebrafish behavior.

False.

---

The gap is **not**:

> Zebrafish behavior has not been analyzed automatically.

False.

---

The gap is **not**:

> Animal behavior has not been represented with self-supervised learning.

False.

---

The gap is **not**:

> Handcrafted features have never been compared with self-supervised features in animal behavior.

Too broad; ContrastivePose already provides related precedent.

---

# 51. Provisional Research Gap

The strongest working gap is:

> **It remains insufficiently established whether self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral organization beyond a strong conventional locomotion/pose representation when both representations are evaluated on matched behavioral observations and subjected to explicit cross-animal and nuisance-variable validation.**

The nuisance variables of particular interest are:

- fish identity,
- recording session,
- locomotor speed,
- tracking quality,
- temporal windowing,
- analytical instability.

---

# 52. More Conservative Gap Wording

An even safer formulation is:

> **Our literature search has not yet identified a study directly comparing hand-engineered zebrafish locomotion/pose features with self-supervised temporal representations under a matched unsupervised behavioral-discovery pipeline that additionally tests cross-animal reproducibility, speed dependence, identity leakage, recording-session leakage, tracking artifacts, and temporal-window robustness.**

This wording is preferred over an absolute:

> "No previous study has..."

until the literature search is saturated.

---

# 53. Proposed Contribution

The contribution can therefore be represented as:

```text
             ZEBRAFISH BEHAVIOR
                    │
                    ▼
          Same Temporal Windows
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
 Hand-Engineered          Temporal SSL
     Features             Embeddings
          │                   │
          ▼                   ▼
      Discovery            Discovery
          │                   │
          └─────────┬─────────┘
                    ▼
                  Compare
                    │
                    ▼
             Validity Controls
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
     Fish ID       Speed        Session
       │            │             │
       └───────┬────┴─────┬───────┘
               ▼          ▼
            Tracking    Windows
               │          │
               └────┬─────┘
                    ▼
             Reproducibility
                    │
                    ▼
               Claim Level
```

---

# 54. Why This Contribution Matters

Hand-engineered features impose assumptions about which aspects of behavior matter.

For example:

```text
Researcher decides:
speed matters
turning matters
curvature matters
```

SSL instead asks whether useful structure can be learned directly from temporal patterns.

If both methods recover essentially the same organization, that is informative:

> conventional features may already capture the dominant structure.

If SSL consistently reveals additional organization, that is also informative:

> important temporal structure may be lost by conventional summaries.

Therefore both positive and negative results answer a meaningful scientific question.

---

# 55. Expected Outcome Categories

## Outcome A — SSL Adds Reproducible Structure

```text
SSL structure
    +
cross-fish replication
    +
nuisance controls
    +
incomplete baseline recoverability
```

Interpretation:

> Evidence supports additional behavioral structure.

---

## Outcome B — SSL Reconstructs the Baseline

```text
SSL clusters
≈
speed / pose / locomotion features
```

Interpretation:

> SSL provides little evidence of additional behavioral organization.

---

## Outcome C — SSL Encodes Nuisance Variables

```text
clusters
≈
Fish ID
or
Session
or
Tracking Errors
```

Interpretation:

> Apparent behavioral discovery is likely confounded.

---

## Outcome D — No Stable Structure

Interpretation:

> The tested representation/discovery combination does not yield reproducible states.

All are acceptable research outcomes.

---

# 56. Dataset Implications

The literature also clarifies the dataset decision.

The ideal first dataset should maximize:

```text
simple behavioral interpretation
+
fish identity
+
temporal resolution
+
pose/tracking quality
+
sufficient duration
```

while minimizing:

```text
social confounds
+
camera variation
+
identity ambiguity
+
unnecessary complexity
```

---

# 57. Current Dataset Literature Roles

## Scholz et al. Resource

Best current role:

```text
POSE_RESOURCE
+
PILOT_CANDIDATE
```

because it provides strong pose infrastructure but must still be evaluated for temporal SSL scale and experimental structure.

---

## Deligkaris et al.

Best current role:

```text
FUTURE_SOCIAL_EXTENSION
+
POSSIBLE_REPLICATION_RESOURCE
```

because of its exceptional temporal scale and tracking richness but substantially greater multi-animal complexity.

---

## Yang et al.

Best current role:

```text
PRIOR_ART
+
BASELINE_REFERENCE
+
NOVELTY_BOUNDARY
```

because it directly demonstrates machine-learning-based behavioral feature discovery from 3D zebrafish movement.

---

## Barreiros et al.

Best current role:

```text
CONVENTIONAL_ANALYSIS_REFERENCE
+
BASELINE_REFERENCE
```

because it demonstrates automated tracking and interpretable movement metrics in zebrafish behavioral experiments.

---

# 58. Preliminary Method Recommendation

Based on the current literature, the cleanest first experiment is:

## Data Representation

Use:

```text
tracked trajectory / pose sequences
```

if a sufficiently strong dataset is available.

---

## Input A

Use:

```text
locomotion
+
turning
+
bout dynamics
+
pose features
```

---

## Input B

Use:

```text
self-supervised temporal embedding
```

learned from the same underlying tracking/pose sequences.

---

## Discovery

Use one predefined clustering method.

---

## Comparison

Measure:

- stability,
- baseline recoverability,
- cross-fish reproducibility.

---

## Controls

Test:

- speed,
- fish identity,
- session,
- tracking quality,
- window duration.

---

# 59. Why Pose-Based SSL Is Currently Attractive

Pose-based SSL gives the experiment an especially clean causal interpretation.

Both pipelines begin from:

```text
same coordinates
```

Then:

```text
Pipeline A
Coordinates
    ↓
Human-defined transformations
    ↓
Behavioral features
```

and:

```text
Pipeline B
Coordinates
    ↓
Self-supervised learning
    ↓
Learned features
```

If the learned representation adds information, the difference can more confidently be attributed to the **representation strategy**, not unequal sensor information.

---

# 60. Literature-Informed Primary Research Question

The literature supports retaining:

> **Do self-supervised temporal representations of zebrafish behavior reveal reproducible behavioral structure that is not captured by conventional hand-engineered locomotion and pose features?**

However, the term:

```text
"not captured"
```

must be operationalized through quantitative comparisons rather than visual judgment.

---

# 61. Operational Meaning of "Not Captured"

Evidence that Input A does not fully capture SSL structure may include:

- poor or incomplete prediction of SSL cluster membership,
- distinct neighborhood structure,
- within-speed behavioral separation,
- differences in temporal dynamics,
- stable SSL states with overlapping conventional measurements.

No single criterion will be sufficient.

---

# 62. Literature-Informed Claim Threshold

The strongest initial target claim remains:

> **Self-supervised temporal representations reveal reproducible zebrafish behavioral structure not fully captured by the evaluated hand-engineered locomotion and pose feature set.**

The wording:

```text
"evaluated hand-engineered feature set"
```

is important.

The study cannot prove that **no conceivable handcrafted feature** could recover the same information.

---

# 63. Claims to Avoid

Do not write:

> "SSL discovered previously unknown zebrafish behavior."

without additional biological validation.

Do not write:

> "SSL understands zebrafish behavior."

Do not write:

> "Traditional methods cannot detect these behaviors."

unless the evidence actually supports such a strong statement.

Do not write:

> "This is the first use of machine learning for zebrafish behavioral discovery."

Existing research contradicts that claim.

Do not currently write:

> "This is the first use of SSL in zebrafish behavior."

without a much more exhaustive search.

---

# 64. Preferred Claim Language

Use language such as:

> "The learned representation contained reproducible behavioral structure that was incompletely recoverable from the predefined hand-engineered representation."

or:

> "Under the evaluated dataset and analysis conditions, SSL-derived behavioral organization persisted after controlling for locomotor speed and major nuisance variables."

This is narrower and scientifically defensible.

---

# 65. Remaining Literature Questions

Several questions remain open before preregistration can be frozen.

## Zebrafish-Specific SSL

- [ ] Has SSL been directly applied to zebrafish behavioral sequences?
- [ ] If yes, was it used for representation learning, classification, or discovery?
- [ ] Were handcrafted features compared directly?

---

## Discovery

- [ ] Which zebrafish studies perform genuinely unsupervised behavioral-state discovery?
- [ ] Which clustering methods dominate?
- [ ] How is the number of states chosen?

---

## Validation

- [ ] Which animal-behavior SSL papers explicitly use held-out subjects?
- [ ] Which quantify subject identity leakage?
- [ ] Which test session leakage?
- [ ] Which test speed dependence?

---

## Temporal Scale

- [ ] What window durations are typical for comparable zebrafish movement?
- [ ] Does the literature favor fixed clips or behavior-adaptive segmentation?

---

# 66. Literature Search Priorities

The next literature-search wave should prioritize:

```text
HIGH PRIORITY

1. zebrafish self-supervised behavior
2. zebrafish contrastive behavioral representation
3. zebrafish unsupervised behavioral states
4. zebrafish behavioral motifs machine learning
5. pose-based animal behavioral discovery
6. subject identity leakage animal SSL
7. behavioral embeddings locomotor speed confound
8. cross-animal representation validation
```

These searches have the greatest potential to change the research question.

---

# 67. Novelty Kill Test

Any new paper should receive immediate priority if it combines all or most of:

```text
Zebrafish
        +
Temporal Behavior
        +
Self-Supervised Learning
        +
Unsupervised Discovery
        +
Handcrafted Baseline
        +
Held-Out Fish
        +
Nuisance Controls
```

If such a study is found, the contribution must be refined before substantial implementation.

---

# 68. Current Literature Matrix

| Study | Zebrafish | Temporal Behavior | Tracking / Pose | ML | SSL | Unsupervised Behavioral Discovery | Handcrafted Comparison | Main Relevance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Barreiros et al. 2021 | Yes | Yes | Tracking | Yes | No | No/limited | Conventional metrics | Baseline |
| Yang et al. 2021 | Yes | Yes | 3D tracking | Yes | No | Yes/relevant | Feature-based | Prior art |
| Scholz et al. 2025 | Yes | Yes | **15-keypoint pose** | Yes | No | No | Tracking benchmark | Pose resource |
| Deligkaris et al. 2026 | Yes | **Yes** | **3D landmarks + identity** | Tracking workflow | No primary SSL | Dataset enables it | Not primary aim | Dataset |
| ContrastivePose 2023 | Other animals | Yes | Pose | Yes | **Yes** | No primary discovery | **Yes** | SSL precedent |
| BEAST | Multi-species | Yes | Video | Yes | **Yes** | Downstream behavioral analysis | Multiple baselines | SSL precedent |
| Proposed Study | **Yes** | **Yes** | Yes | Yes | **Yes** | **Yes** | **Primary comparison** | Proposed contribution |

Cells describing methodological interpretation should be updated after complete deep reads.

---

# 69. Research Gap Matrix

| Potential Gap | Current Assessment |
|---|---|
| Automated zebrafish behavior analysis | **Already studied** |
| Zebrafish tracking | **Already studied extensively** |
| Zebrafish pose estimation | **Already studied** |
| Machine learning on zebrafish behavior | **Already studied** |
| 3D zebrafish behavioral analysis | **Already studied** |
| SSL for animal behavior | **Already studied** |
| SSL versus handcrafted animal features generally | **Related work exists** |
| SSL for zebrafish temporal behavioral discovery | **Requires deeper verification** |
| Matched handcrafted-vs-SSL zebrafish discovery comparison | **No direct match identified so far** |
| Same comparison with strong leakage/speed/artifact controls | **Strongest provisional gap** |

---

# 70. Literature-to-Design Traceability

## Literature Finding

Conventional zebrafish analyses rely heavily on interpretable locomotion variables.

### Design Decision

Include a serious locomotion baseline.

---

## Literature Finding

Modern pose estimation exposes detailed body kinematics.

### Design Decision

Include pose-derived features when supported by the dataset.

---

## Literature Finding

ML behavioral feature recognition already exists in zebrafish.

### Design Decision

Do not claim novelty from ML or clustering alone.

---

## Literature Finding

Self-supervised animal behavioral representations already exist.

### Design Decision

Do not claim novelty from SSL alone.

---

## Literature Finding

SSL can outperform handcrafted features in downstream animal behavioral classification.

### Design Decision

Focus the project on **unsupervised structure and validation**, not merely downstream accuracy.

---

## Literature Finding

Large zebrafish datasets contain tracking and identity errors.

### Design Decision

Tracking QC becomes an explicit validity analysis.

---

## Literature Finding

Rich social zebrafish datasets now exist.

### Design Decision

Reserve multi-animal interaction SSL as a future extension unless needed.

---

# 71. Current Experimental Architecture

The literature currently supports:

```text
AUTHORIZED ZEBRAFISH DATA
          │
          ▼
TRACKING / POSE
          │
          ▼
FIXED TEMPORAL WINDOWS
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼

INPUT A       INPUT B

Handcrafted   Temporal
Features      SSL
    │           │
    ▼           ▼
Representation Vectors
    │           │
    ▼           ▼
    Behavioral Discovery
          │
          ▼
       Comparison
          │
          ▼
    VALIDITY BATTERY
          │
 ┌────────┼────────┐
 ▼        ▼        ▼
Fish     Speed   Session
 ID
 │
 ├── Tracking QC
 ├── Window sensitivity
 └── Seed stability
          │
          ▼
     Held-Out Fish
          │
          ▼
      Final Claim
```

---

# 72. Working Baseline Specification

The literature synthesis currently recommends freezing a baseline broadly equivalent to:

```yaml
input_a:

  locomotion:
    - mean_speed
    - median_speed
    - max_speed
    - speed_sd
    - mean_acceleration
    - distance_traveled
    - immobility_fraction

  turning:
    - mean_angular_velocity
    - angular_velocity_sd
    - cumulative_heading_change
    - trajectory_curvature

  bouts:
    - bout_count
    - mean_bout_duration
    - mean_interbout_interval
    - peak_bout_speed

  pose:
    - mean_body_orientation
    - body_curvature
    - curvature_variability
    - tail_curvature
    - pose_velocity
```

Exact features depend on the selected dataset.

---

# 73. Working Input B Specification

The literature currently supports:

```yaml
input_b:

  modality:
    preferred: pose_or_trajectory_sequence

  learning:
    type: self_supervised
    temporal: true

  candidate_objectives:
    - temporal_contrastive_learning
    - masked_temporal_modeling

  output:
    one_embedding_per_behavioral_window
```

No final SSL method has yet been frozen.

---

# 74. Working Discovery Specification

```yaml
discovery:

  representation_a:
    hand_engineered_features

  representation_b:
    ssl_embeddings

  primary_method:
    TBD

  visualization:
    PCA_UMAP_optional

  validation:
    - repeated_seed_stability
    - held_out_fish
    - baseline_recoverability
    - speed_control
    - fish_identity_control
    - session_control
    - tracking_qc
    - window_sensitivity
```

---

# 75. Current Novelty Statement

The preferred current wording is:

> **Existing work demonstrates automated tracking, conventional behavioral quantification, three-dimensional machine-learning-based behavioral analysis in zebrafish, and self-supervised representation learning for animal behavior more broadly. However, the literature reviewed to date has not established whether temporal self-supervised zebrafish representations provide reproducible behavioral organization beyond a strong hand-engineered locomotion/pose baseline under matched inputs and explicit controls for animal identity, recording conditions, locomotor speed, tracking artifacts, segmentation choices, and analytical instability.**

This statement remains provisional.

---

# 76. Literature-Supported Candidate Contribution

The candidate contribution should therefore remain:

> **Compare hand-engineered locomotion and pose features with self-supervised temporal representations, then test whether the learned representation yields reproducible behavioral structure not captured by the hand-engineered baseline.**

The critical addition is:

> **and actively attempt to falsify that interpretation using nuisance-variable and reproducibility controls.**

---

# 77. Scientific Value of a Negative Result

The literature makes a negative result particularly meaningful.

If:

```text
SSL
≈
hand-engineered locomotion + pose
```

then the study would provide evidence that the additional complexity of SSL may not be necessary for the evaluated zebrafish behavioral setting.

Likewise, if:

```text
SSL
≈
speed
```

the study would demonstrate the danger of interpreting attractive embedding structure as complex behavioral discovery.

Both are valuable methodological findings.

---

# 78. Implication for Preregistration

Before formal preregistration, the literature should be sufficiently developed to freeze:

- the baseline feature families,
- the SSL family,
- the input modality,
- the discovery algorithm,
- the window scale,
- the validation metrics.

The exact implementation must be justified from:

```text
literature
+
dataset characteristics
+
computational feasibility
```

rather than from final test performance.

---

# 79. Current Confidence Assessment

```yaml
automated_zebrafish_behavior_prior_art:
  confidence: HIGH

machine_learning_zebrafish_prior_art:
  confidence: HIGH

pose_tracking_prior_art:
  confidence: HIGH

animal_behavior_ssl_prior_art:
  confidence: HIGH

direct_zebrafish_ssl_discovery_comparison:
  confidence: MODERATE_LOW
  interpretation: no direct match identified yet

matched_ssl_vs_handcrafted_with_full_validity_battery:
  confidence: MODERATE
  interpretation: strongest current candidate gap
```

The lowest-confidence claims require continued searching.

---

# 80. Core References

## Barreiros et al. (2021)

Barreiros, M. O., Barbosa, F. G., Dantas, D. O., et al.  
**Zebrafish automatic monitoring system for conditioning and behavioral analysis.**  
*Scientific Reports*, 11, 9330.

DOI:

`10.1038/s41598-021-87502-6`

Primary project relevance:

```text
conventional tracking
behavioral baseline
automated analysis
```

---

## Yang et al. (2021)

Yang, P., Takahashi, H., Murase, M., & Itoh, M.  
**Zebrafish behavior feature recognition using three-dimensional tracking and machine learning.**  
*Scientific Reports*, 11, 13492.

DOI:

`10.1038/s41598-021-92854-0`

Primary project relevance:

```text
3D tracking
machine learning
behavioral feature discovery
novelty boundary
```

---

## Scholz et al. (2025)

Scholz, L. A., Mancienne, T., Stednitz, S. J., Scott, E. K., & Lee, C. C. Y.  
**Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses.**

bioRxiv preprint.

DOI:

`10.1101/2025.06.04.657938`

Primary project relevance:

```text
pose estimation
15-keypoint zebrafish representation
pretrained tracking
annotated data
```

---

## Deligkaris et al. (2026)

Deligkaris, K., Neiman, R., Hiroi, M., et al.  
**A dataset of fine-grained zebrafish interactions in health and disease.**  
*Scientific Data*, 13, 583.

DOI:

`10.1038/s41597-026-06953-6`

Dataset DOI:

`10.5281/zenodo.17190142`

Primary project relevance:

```text
large-scale temporal data
3D tracking
adult zebrafish
fish identity
social behavior
future replication
```

---

## Zhou et al. (2023)

Zhou, T., Cheah, C. C. H., Chin, E. W. M., et al.  
**ContrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavioral classification of interacting animals.**  
*Computers in Biology and Medicine*, 165, 107416.

DOI:

`10.1016/j.compbiomed.2023.107416`

Primary project relevance:

```text
self-supervised pose representation
contrastive learning
handcrafted-feature comparison
animal behavior
```

---

# 81. Literature Conclusions

The literature currently supports five major conclusions.

## Conclusion 1

**The conventional baseline must be strong.**

Zebrafish behavior is already routinely represented computationally through interpretable movement variables.

---

## Conclusion 2

**Machine learning itself is not the novelty.**

Existing zebrafish studies already use machine-learning algorithms to identify behavioral features.

---

## Conclusion 3

**Self-supervised behavioral representation learning itself is not sufficient novelty.**

Related animal-behavior systems already use contrastive and masked self-supervised learning.

---

## Conclusion 4

**The strongest contribution is the comparison and validation framework.**

The project should determine whether SSL captures additional behavioral structure rather than merely demonstrating that SSL can produce embeddings.

---

## Conclusion 5

**Reproducibility and falsification are central.**

The strongest version of this project is one that actively attempts to demonstrate that apparent SSL structure is **not**:

```text
fish identity
session
speed
tracking noise
window construction
random instability
```

before interpreting it as additional behavior.

---

# 82. Final Working Research Gap

> **Current zebrafish literature establishes automated tracking, hand-engineered behavioral analysis, pose estimation, and machine-learning-based behavioral feature recognition, while broader computational-ethology literature establishes self-supervised behavioral representation learning. What remains insufficiently established is whether self-supervised temporal representations of zebrafish behavior recover reproducible behavioral organization beyond a strong conventional locomotion/pose representation when evaluated on matched behavioral observations with explicit cross-animal, speed, session, tracking-quality, segmentation, and stability controls.**

---

# 83. Immediate Literature Actions

- [x] Continue targeted search for direct zebrafish SSL studies.
  - Direct zebrafish SSL was identified in Xu & Wang (2024), which uses masked skeleton-sequence modeling to learn latent behavioral embeddings from larval zebrafish sequences. :contentReference[oaicite:0]{index=0}
  - The targeted search did not identify a closer published study combining zebrafish SSL, unsupervised behavioral-state discovery, direct handcrafted-feature comparison, and the full nuisance-validation battery used in this project.

- [x] Search specifically for zebrafish temporal contrastive learning.
  - No direct zebrafish behavioral study matching the present temporal-contrastive discovery design was identified in the targeted search.
  - Zebrafish-specific SSL precedent currently identified is primarily masked sequence modeling rather than contrastive learning. :contentReference[oaicite:1]{index=1}
  - Temporal contrastive learning has clear precedent in broader animal-behavior SSL, including BEAST, which combines masked autoencoding with temporal contrastive learning. :contentReference[oaicite:2]{index=2}

- [x] Search for zebrafish representation-learning studies using pose sequences.
  - Xu & Wang (2024) directly learns latent representations from zebrafish skeletal sequences using a spatial-temporal Transformer/CNN architecture. :contentReference[oaicite:3]{index=3}
  - Scholz et al. provide a complementary zebrafish pose-estimation resource using a 15-keypoint pose schema with pretrained DeepLabCut and SLEAP models, supporting pose-sequence analysis even though the study itself is focused on tracking rather than SSL discovery. :contentReference[oaicite:4]{index=4}

- [x] Search zebrafish unsupervised behavioral-state discovery.
  - Unsupervised behavioral discovery is well established in zebrafish.
  - Marques et al. (2018) identified 13 larval swim types using unsupervised behavioral clustering of millions of naturally segmented bouts. :contentReference[oaicite:5]{index=5}
  - Ghosh & Rihel (2020) used unsupervised learning to identify behavioral modules and hierarchical compression to identify recurrent motifs across multiple timescales. :contentReference[oaicite:6]{index=6}
  - Yang et al. (2021) used PCA-related dimensionality reduction and FuzzyART clustering on temporal behavioral segments. :contentReference[oaicite:7]{index=7}
  - Adult zebrafish exploratory behavior has also been separated into multiple unsupervised behavioral clusters rather than a simple bold/shy dichotomy. :contentReference[oaicite:8]{index=8}

- [x] Deep-read Yang et al. methods and feature definitions.
  - Yang et al. use 3D tracking-derived behavioral features and segment time-series data before dimensionality reduction and FuzzyART clustering.
  - Cluster-analysis parameters, including explained-variance and vigilance settings, were selected through grid search across time segments. :contentReference[oaicite:9]{index=9}
  - This provides precedent for trajectory/kinematic handcrafted baselines, while also illustrating why parameter selection must be separated from held-out final evaluation.

- [x] Deep-read Barreiros et al. behavioral metrics.
  - Barreiros et al. quantify individual and group behavior using tracking-derived measures including distance traveled, speed, route/tracking information, polarization, and group dynamics. :contentReference[oaicite:10]{index=10}
  - Their monitoring pipeline uses automated fish detection/tracking and explicitly defines polarization as a group-coordination metric based on aligned velocity direction. :contentReference[oaicite:11]{index=11}
  - These results support speed, distance/movement, orientation-related, and group-organization measures as conventional zebrafish behavioral descriptors.

- [x] Deep-read Scholz et al. dataset structure and pose schema.
  - The resource provides annotated larval zebrafish behavioral videos and pretrained DeepLabCut/SLEAP networks.
  - Its pose representation uses 15 keypoints and includes both free-swimming and head-embedded behavioral recordings. :contentReference[oaicite:12]{index=12}
  - This establishes detailed multi-keypoint larval pose tracking as a viable zebrafish representation, though DS-005 uses its own bout-level representation and does not require adoption of this exact schema.

- [x] Deep-read Deligkaris et al. tracking/QC representation.
  - Deligkaris et al. track three anatomical landmarks in 3D together with persistent fish identity in dyadic adult zebrafish recordings. :contentReference[oaicite:13]{index=13}
  - Their workflow combines SLEAP body-point detections with idtracker-based identity tracking across experiments. :contentReference[oaicite:14]{index=14}
  - The authors explicitly warn that residual identity swaps and body-part misidentifications can remain after QC, especially during close interactions. :contentReference[oaicite:15]{index=15}
  - This directly supports treating tracking quality and identity errors as potential sources of false behavioral structure.

- [x] Deep-read ContrastivePose methodology.
  - ContrastivePose performs self-supervised contrastive representation learning directly from pose-estimation data.
  - Positive examples are generated through behavior-preserving augmentations of the same pose data, while other batch examples provide negatives.
  - Reported augmentations include flipping, rotation, and translation.
  - The learned representation is then used for downstream supervised behavioral classification and compared directly with handcrafted features. :contentReference[oaicite:16]{index=16}
  - This provides strong methodological precedent for the present handcrafted-vs-SSL comparison, while differing because the present project evaluates unsupervised discovery rather than primarily supervised classification.

- [x] Deep-read BEAST methodology and subject-split procedures.
  - BEAST uses self-supervised behavioral-video pretraining based on masked autoencoding plus temporal contrastive learning.
  - It evaluates learned representations across multiple species and downstream neurobehavioral tasks including pose estimation, action segmentation, and neural encoding. :contentReference[oaicite:17]{index=17}
  - Its methodology supports temporal contrastive learning as a credible animal-behavior SSL family.
  - For this project, fish-level held-out splitting remains the stronger confirmatory requirement regardless of the exact split policy of any single precedent paper.

- [x] Build citation-backed baseline feature list.
  - Supported baseline feature families include:
    - speed and movement magnitude,
    - distance/path-related movement,
    - orientation/turning,
    - acceleration or speed-change descriptors,
    - bout timing/duration,
    - pose/posture-derived descriptors where tracking quality permits.
  - Barreiros et al. support speed, distance, route, orientation/group-motion measures. :contentReference[oaicite:18]{index=18}
  - Yang et al. support trajectory/time-series handcrafted behavioral descriptors before unsupervised clustering. :contentReference[oaicite:19]{index=19}
  - ContrastivePose explicitly identifies pose-derived handcrafted quantities such as animal orientation and length as conventional behavioral-analysis features. :contentReference[oaicite:20]{index=20}
  - The frozen DS-005 primary baseline therefore remains consistent with literature precedent.

- [x] Identify literature precedent for behavioral-window duration.
  - Zebrafish behavior is analyzed over multiple temporal scales rather than one universal window length.
  - Natural swim bouts provide a biologically meaningful sub-second unit in larval zebrafish. :contentReference[oaicite:21]{index=21}
  - Ghosh & Rihel explicitly identify structure extending from sub-second bouts through minute-scale motifs and longer circadian organization. :contentReference[oaicite:22]{index=22}
  - Yang et al. also analyze predefined temporal segments, including 10-second examples, demonstrating fixed-window precedent. :contentReference[oaicite:23]{index=23}
  - Therefore both fixed segmentation and behavior-adaptive/bout segmentation have precedent; the DS-005 natural bout unit is defensible.

- [x] Identify best-practice cross-animal evaluation.
  - Confirmatory behavioral representation evaluation should separate animals across fitting/model-selection and held-out evaluation whenever fish identity is available.
  - This prevents individual-specific morphology, movement style, or recording idiosyncrasies from appearing as behavioral generalization.
  - The project's fish-level TRAIN/VALIDATION/TEST split therefore remains an appropriate and conservative design choice.

- [x] Identify explicit identity-leakage tests in computational ethology.
  - Subject identity should be treated as a nuisance variable that can be directly predicted from learned representations.
  - The appropriate project-level diagnostic is therefore an explicit fish-ID prediction analysis, compared against chance/baseline performance, using many fish and balanced sampling.
  - Deligkaris et al.'s explicit attention to persistent identity and residual identity errors further reinforces identity as a critical tracking/representation variable in zebrafish analysis. :contentReference[oaicite:24]{index=24}

- [x] Identify speed-controlled behavioral embedding analyses.
  - Locomotor speed is repeatedly treated as a major behavioral variable in zebrafish studies, making it a particularly important nuisance explanation for learned embeddings. :contentReference[oaicite:25]{index=25}
  - The validation battery should therefore include:
    - embedding-to-speed correlation,
    - prediction of speed from embeddings,
    - examination of cluster speed distributions,
    - and, where needed, speed-matched comparisons.
  - These controls are now implemented/planned explicitly in the present project.

- [x] Freeze research gap only after these searches.
  - Targeted search saturation is now sufficient to freeze the preregistered research gap at the following scope:

    > Prior work establishes both unsupervised zebrafish behavioral discovery and self-supervised animal-behavior representation learning, and direct zebrafish SSL precedent now exists. However, the targeted literature search did not identify a matched study that tests whether self-supervised temporal zebrafish representations reveal reproducible behavioral structure beyond a conventional handcrafted baseline while simultaneously controlling for held-out fish, identity leakage, context/session effects, speed dependence, tracking artifacts, and representation stability.

  - This is a bounded literature-supported gap, not a claim that SSL, clustering, or zebrafish behavior analysis individually are novel.

---

# 84. Literature Phase Exit Condition

The literature review can transition from exploratory design to preregistration when the project can confidently fill:

```yaml
conventional_baseline:
  locomotion_features: FROZEN
  pose_features: FROZEN

primary_ssl:
  modality: FROZEN
  objective: FROZEN
  architecture_family: FROZEN

discovery:
  method: FROZEN
  state_selection: FROZEN

validation:
  fish_identity: FROZEN
  session: FROZEN
  speed: FROZEN
  tracking_qc: FROZEN
  window_sensitivity: FROZEN
  seed_stability: FROZEN

novelty_statement:
  status: VERIFIED_TO_REASONABLE_SEARCH_SATURATION
```
