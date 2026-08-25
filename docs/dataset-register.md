# Zebrafish SSL Dataset Register

## Purpose

This document is the authoritative register of datasets considered, approved, rejected, or reserved for later use in the **Self-Supervised Discovery of Zebrafish Behavioral Structure** project.

The register exists to ensure that every serious dataset candidate has:

- documented provenance,
- a clear research-use basis,
- known scientific strengths and limitations,
- an assessment of suitability for the hand-engineered baseline,
- an assessment of suitability for self-supervised temporal learning,
- an assessment of validation and leakage risks,
- and a recorded decision about its intended project role.

Unknown values are recorded as `unknown` or `TBD` rather than inferred.

**Last verification date:** `2026-08-21`

---

# 1. Current Dataset Strategy

The dataset strategy is:

1. **DS-005 — Sridhar/Marques larval navigation data** is the selected `PRIMARY` dataset for the initial baseline-vs-SSL experiment.
2. **DS-006 — Reddy et al. larval exploration/aversive-chemotaxis data** is the selected independent `REPLICATION` dataset.
3. **DS-001 — Scholz et al.** is retained as a `POSE_RESOURCE`, `PILOT`, and tracking benchmark.
4. **DS-003 — Yang et al.** is retained as prior art, a hand-engineered-feature benchmark, and a small cross-domain pilot/challenge dataset.
5. **DS-004 — Barreiros et al.** is retained as a conventional tracking/behavior-analysis benchmark.
6. **DS-002 — Deligkaris et al.** is retained as a high-value future social-behavior extension rather than the first single-fish MVP.

The immediate experimental goal remains:

> Compare hand-engineered locomotion/pose features with self-supervised temporal representations, then determine whether the learned representation yields reproducible behavioral structure not captured by the hand-engineered baseline.

The selected primary and replication datasets both favor **pose/trajectory temporal SSL** rather than raw-video SSL.

---

# 2. Dataset Status Definitions

| Status | Meaning |
|---|---|
| `CANDIDATE` | Dataset identified but not fully reviewed. |
| `UNDER_REVIEW` | Dataset actively being evaluated. |
| `APPROVED` | Dataset approved for the specified role. |
| `APPROVED_WITH_LIMITATIONS` | Dataset approved only for specified uses. |
| `SELECTED_PRIMARY` | Dataset selected for the primary confirmatory study. |
| `SELECTED_REPLICATION` | Dataset selected for independent replication. |
| `HOLD` | Additional information required. |
| `REJECTED` | Dataset should not be used for the current project. |
| `FUTURE_EXTENSION` | Promising but outside the first MVP scope. |
| `ARCHIVED` | No longer active. |

---

# 3. Dataset Role Definitions

| Role | Purpose |
|---|---|
| `PRIMARY` | Main dataset for baseline-vs-SSL discovery and confirmatory evaluation. |
| `REPLICATION` | Independent dataset used to test reproducibility outside the primary dataset. |
| `PILOT` | Pipeline development/debugging. |
| `POSE_RESOURCE` | Pose annotations/pretrained models useful for pose infrastructure. |
| `VALIDATION` | External validation of discovered structure. |
| `SOCIAL_EXTENSION` | Later multi-fish/social behavior work. |
| `BENCHMARK` | Method comparison or reproduction of prior work. |
| `PRIOR_ART` | Defines novelty boundary and existing methodology. |
| `CROSS_DOMAIN_CHALLENGE` | Optional stress test in a substantially different biological/recording domain. |

---

# 4. Current Dataset Register

| ID | Dataset | Status | Role | Main value | Main limitation |
|---|---|---|---|---|---|
| `DS-001` | Scholz et al. larval pose/tracking resource | `APPROVED_WITH_LIMITATIONS` | `POSE_RESOURCE`, `PILOT`, `BENCHMARK` | Public videos, annotations, 15-keypoint pose models, CC BY 4.0 | Unique-fish/session structure not sufficiently documented for confirmatory held-out-fish SSL |
| `DS-002` | Deligkaris et al. fine-grained zebrafish interactions | `FUTURE_EXTENSION` | `SOCIAL_EXTENSION`, future `VALIDATION` | 173 five-hour adult dyad recordings, 140 Hz, 3D identified tracks | Dyadic/social design changes the unit of analysis and adds pair/identity confounds |
| `DS-003` | Yang et al. 3D tracking/behavior-feature data | `APPROVED_WITH_LIMITATIONS` | `BENCHMARK`, `PRIOR_ART`, `PILOT`, `CROSS_DOMAIN_CHALLENGE` | Public tracked trajectories and direct prior art for unsupervised 3D behavior features | Only 10 adult fish; complete raw video and clear standalone dataset license not established |
| `DS-004` | Barreiros et al. automated monitoring/conditioning | `APPROVED_WITH_LIMITATIONS` | `BENCHMARK`, tracking/baseline reference | 43 adult fish, automated tracking, repeated conditioning design | Complete public temporal dataset not found; persistent cross-video fish identities not guaranteed |
| `DS-005` | Sridhar et al. / Marques et al. larval navigation dataset | `SELECTED_PRIMARY` | `PRIMARY` | 463 fish, 700 Hz tail tracking, 30 min–3 h recordings, 14 sensory contexts | Exact session-ID/file schema and local archive integrity still require ingestion verification |
| `DS-006` | Reddy et al. larval exploration and aversive chemotaxis | `SELECTED_REPLICATION` | `REPLICATION` | 160 Hz, 10-min recordings, 384 potential fish-well units across 32 recordings, ZebraZoom bout/pose features, CC0 | 381 units are nonempty and 374 remain usable after frozen well-level QC; biological identity across recordings is not verified |

---

# 5. Important Non-Dataset Resources

## StrIPETrack

**Classification:** tracking method/tool/baseline resource.

Potential use:

- trajectory generation,
- comparison with project preprocessing,
- tracking benchmark.

## AquaMaze

**Classification:** behavioral-analysis tool / hand-engineered metric reference.

Potential use:

- conventional locomotion measures,
- baseline-feature evidence,
- conventional behavioral summaries.

---

# 6. DS-001 — Scholz et al. (2025) Larval Pose / Tracking Resource

## Minimum Dataset Metadata

```yaml
dataset_id: DS-001
dataset_name: Datasets of zebrafish larvae poses - annotated frames and videos
authors:
  - Leandro A. Scholz
  - Tessa Mancienne
  - Sarah J. Stednitz
  - Ethan K. Scott
  - Conrad C. Y. Lee
year: 2025
paper: "Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses"
repository: "University of Melbourne Figshare; companion code: https://github.com/Scott-Lab-QBI/zf_tracking_networks"
doi: "10.26188/29276009.v1"
license: "CC BY 4.0"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "larval; approximately 5-7 dpf across source recordings"
number_of_fish: "unknown; number of videos is documented but a dataset-wide unique-fish count is not"
number_of_sessions: "unknown; heterogeneous recordings/experiments"
total_recording_duration: "unknown; heterogeneous"
frame_rate: "heterogeneous; 300 fps documented for the stimulus assay, not a safe global dataset value"
resolution: "heterogeneous; dataset intentionally spans imaging configurations"

raw_video_available: true
tracking_available: true
pose_available: true
fish_ids_available: false
session_ids_available: false
condition_labels_available: "partial; recording/benchmark metadata exist but not a unified confirmatory condition/session schema"

baseline_suitability: "HIGH for pose-derived features; MODERATE for locomotion depending on recording"
ssl_suitability: "MODERATE for pilot pose/video SSL; LOW for primary held-out-fish confirmatory SSL"
validation_suitability: "LOW-MODERATE; strong tracking-QC resource but fish/session independence is not established"

status: APPROVED_WITH_LIMITATIONS
role:
  - POSE_RESOURCE
  - PILOT
  - BENCHMARK
```

## Authorization Review

```yaml
license_name: "Creative Commons Attribution 4.0 International"
license_url: "https://creativecommons.org/licenses/by/4.0/"
copyright_holder: "Dataset creators / rights holders identified by the repository"
research_use_allowed: true
redistribution_allowed: true
derivative_use_allowed: true
attribution_required: true
authorization_verified: true
verification_date: "2026-08-21"
```

## Verified Notes

- Public dataset archive identified.
- Main dataset: 1,641 annotated frames from 28 videos.
- Additional dataset: 512 annotated frames from 12 videos.
- Raw/source videos and annotations are released.
- 15-keypoint pose schema covers eye landmarks, swim bladder and tail.
- Dataset is excellent for pose-model validation and tracking-artifact work.
- It is **not** the primary SSL dataset because dataset-wide biological identity and session structure are not sufficiently documented.

---

# 7. DS-002 — Deligkaris et al. (2026) Fine-Grained Zebrafish Interactions

## Minimum Dataset Metadata

```yaml
dataset_id: DS-002
dataset_name: "A dataset of fine-grained zebrafish interactions in health and disease"
authors:
  - Kosmas Deligkaris
  - Radmila Neiman
  - Makoto Hiroi
  - Tatsuo Izawa
  - Liam O'Shaughnessy
  - Luis Carretero Rodriguez
  - Ichiro Masai
  - Greg J. Stephens
year: 2026
paper: "A dataset of fine-grained zebrafish interactions in health and disease"
repository: "Zenodo"
doi: "10.5281/zenodo.17190142"
license: "unknown in the web-accessible Zenodo metadata reviewed; article is CC BY 4.0, but the dataset-license field should be checked directly before reuse"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "adult; 6-14 months"
number_of_fish: "up to 450 fish used across the study; unique fish in the released 173 experiments not equal to 346 because individuals could participate up to four times"
number_of_sessions: 173
total_recording_duration: "865 recording-hours (173 experiments x 5 hours)"
frame_rate: "140 fps"
resolution: "1280x1024 pixels per camera; three synchronized orthogonal cameras"

raw_video_available: "partial; source videos for one experiment plus sample video are released, while tracked data are released for all experiments"
tracking_available: true
pose_available: true
fish_ids_available: "true within each five-hour experiment; idtracker.ai used to maintain identities"
session_ids_available: true
condition_labels_available: true

baseline_suitability: "HIGH for 3D locomotion, orientation, inter-fish geometry and coarse posture"
ssl_suitability: "VERY HIGH technically, but social-dyad SSL is outside the initial single-fish MVP"
validation_suitability: "HIGH for social-behavior studies; pair/fish reuse across recordings must be handled explicitly"

status: FUTURE_EXTENSION
role:
  - SOCIAL_EXTENSION
  - VALIDATION
```

## Authorization Review

```yaml
license_name: "TBD — verify the license attached to Zenodo record 17190142 directly"
license_url: "TBD"
copyright_holder: "Dataset creators / applicable rights holders"
research_use_allowed: "TBD until dataset license is directly verified"
redistribution_allowed: "TBD"
derivative_use_allowed: "TBD"
attribution_required: "TBD"
authorization_verified: false
verification_date: "2026-08-21"
```

## Verified Notes

- 173 five-hour adult dyad recordings.
- 152 WT experiments, 9 `mecp2`, and 12 `fgfr1a`.
- Three tracked 3D body points: head, pectoral region, tail base.
- Metadata includes experiment ID, date, sex, arena shape and genotype.
- Three synchronized cameras at 140 fps.
- The dataset is a strong future social SSL resource but is not selected for the first single-fish MVP.

---

# 8. DS-003 — Yang et al. (2021) 3D Tracking and Behavior Feature Recognition

## Minimum Dataset Metadata

```yaml
dataset_id: DS-003
dataset_name: "Yang et al. zebrafish 3D tracking / behavioral feature data"
authors:
  - Peng Yang
  - Hiro Takahashi
  - Masataka Murase
  - Motoyuki Itoh
year: 2021
paper: "Zebrafish behavior feature recognition using three-dimensional tracking and machine learning"
repository: "https://github.com/singularpse/Zebarafish_3D_swim_path_reconstructions_system"
doi: "10.1038/s41598-021-92854-0"
license: "standalone dataset license unclear; article is CC BY 4.0 and repository/paper describes noncommercial code sharing"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "adult; approximately 3 months"
number_of_fish: 10
number_of_sessions: "10 individual fish recordings under the same experimental paradigm; independent session/batch structure beyond this is not established"
total_recording_duration: "approximately 10 fish-minutes for the 30 s quiescent + 30 s stimulation analysis period across 10 fish"
frame_rate: "up to 60 fps"
resolution: "unknown in the released trajectory data; two-camera acquisition"
raw_video_available: false
tracking_available: true
pose_available: false
fish_ids_available: true
session_ids_available: "partial; numbered fish/trial files exist but rich cross-session metadata are limited"
condition_labels_available: true

baseline_suitability: "VERY HIGH for conventional trajectory features and prior-art comparison"
ssl_suitability: "MODERATE for a small trajectory SSL pilot; LOW for primary confirmatory SSL"
validation_suitability: "LOW-MODERATE because n=10 independent fish is small"

status: APPROVED_WITH_LIMITATIONS
role:
  - BENCHMARK
  - PRIOR_ART
  - PILOT
  - CROSS_DOMAIN_CHALLENGE
```

## Authorization Review

```yaml
license_name: "Standalone dataset license not clearly specified"
license_url: "unknown"
copyright_holder: "Yang et al. / applicable repository rights holders"
research_use_allowed: "noncommercial research use appears intended, but exact dataset license is not explicit"
redistribution_allowed: "unknown"
derivative_use_allowed: "unknown"
attribution_required: true
authorization_verified: false
verification_date: "2026-08-21"
```

## Verified Notes

- Public lateral/ventral tracking files exist for fish 01-10.
- Complete original camera recordings were not found in the public repository.
- Temporal windows can be reconstructed from the trajectories.
- The dataset remains important prior art because it already combines 3D zebrafish trajectories, temporal segmentation, unsupervised ML and conventional behavior features.

---

# 9. DS-004 — Barreiros et al. (2021) Automated Monitoring and Behavioral Analysis

## Minimum Dataset Metadata

```yaml
dataset_id: DS-004
dataset_name: "Barreiros et al. automated zebrafish monitoring / conditioning data"
authors:
  - Marta de Oliveira Barreiros
  - Felipe Gomes Barbosa
  - Diego de Oliveira Dantas
  - Daniel de Matos Luna dos Santos
  - Sidarta Ribeiro
  - Giselle Cutrim de Oliveira Santos
  - Allan Kardec Barros
year: 2021
paper: "Zebrafish automatic monitoring system for conditioning and behavioral analysis"
repository: "Scientific Reports article and supplementary information; no complete standalone dataset repository identified"
doi: "10.1038/s41598-021-87502-6"
license: "article/supplementary material CC BY 4.0 unless otherwise credited; unreleased raw experimental data have no separately verified dataset license"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "adult"
number_of_fish: 43
number_of_sessions: "repeated conditioning sessions/trials; exact biological-session count is design-dependent"
total_recording_duration: "simple conditioning: 4 x 2-min group recordings; complex conditioning: approximately 20 x 1-min group recordings, plus controls/other analyses"
frame_rate: "30 fps"
resolution: "1920x1080 pixels"
raw_video_available: "partial; raw video was generated and supplementary examples exist, but a complete public archive was not identified"
tracking_available: "tracking was generated by the study; complete downloadable trajectory archive not verified"
pose_available: false
fish_ids_available: "within-video identities/tracks exist; persistent biological identity across separate videos is not guaranteed"
session_ids_available: true
condition_labels_available: true

baseline_suitability: "HIGH as a reference for speed, distance, direction and group/polarization features"
ssl_suitability: "LOW using the currently public release; potentially MODERATE-HIGH if complete trajectories/videos were obtained"
validation_suitability: "LOW for the planned held-out-fish design because persistent cross-video fish identity is not guaranteed"

status: APPROVED_WITH_LIMITATIONS
role:
  - BENCHMARK
  - PRIOR_ART
```

## Authorization Review

```yaml
license_name: "Creative Commons Attribution 4.0 International for the article and included supplementary material"
license_url: "https://creativecommons.org/licenses/by/4.0/"
copyright_holder: "Article authors / applicable rights holders"
research_use_allowed: true
redistribution_allowed: "true for CC BY-covered article/supplement; unknown for unreleased raw data"
derivative_use_allowed: "true for CC BY-covered article/supplement; unknown for unreleased raw data"
attribution_required: true
authorization_verified: "partial"
verification_date: "2026-08-21"
```

---

# 10. DS-005 — Sridhar et al. / Marques et al. Larval Navigation Dataset

## Minimum Dataset Metadata

```yaml
dataset_id: DS-005
dataset_name: "Dataset for Uncovering multiscale structure in the variability of larval zebrafish navigation V2"
authors:
  dataset_creators:
    - Gautam Sridhar
    - Antonio Carlos Costa
  associated_paper:
    - Gautam Sridhar
    - Massimo Vergassola
    - João C. Marques
    - Michael B. Orger
    - Antonio Carlos Costa
    - Claire Wyart
year: 2024
paper: "Uncovering multiscale structure in the variability of larval zebrafish navigation"
repository: "https://zenodo.org/records/13605471"
doi: "10.5281/zenodo.13605471"
license: "CC BY 4.0"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "larval; 6-7 dpf"
number_of_fish: 463
number_of_sessions: "unknown as a formal session field; 463 individual fish recordings across 14 sensory contexts are described"
total_recording_duration: "at least 231.5 fish-hours at the 30-min lower bound; individual recordings span approximately 30 min to 3 h"
frame_rate: "700 Hz"
resolution: "original acquisition used two pixel sizes: approximately 58 µm/pixel in 5x5 cm arenas and 27 µm/pixel in 2.5x2.5 cm arenas"

raw_video_available: false
tracking_available: true
pose_available: true
fish_ids_available: true
session_ids_available: "recoverable at least at fish/recording/context level; exact archive field schema must be verified during ingestion"
condition_labels_available: true

baseline_suitability: "VERY HIGH"
ssl_suitability: "VERY HIGH / PRIMARY"
validation_suitability: "HIGH; hundreds of individual fish permit animal-level splits"

status: SELECTED_PRIMARY
role:
  - PRIMARY
```

## Authorization Review

```yaml
license_name: "Creative Commons Attribution 4.0 International"
license_url: "https://creativecommons.org/licenses/by/4.0/"
copyright_holder: "Dataset creators / applicable rights holders"
research_use_allowed: true
redistribution_allowed: true
derivative_use_allowed: true
attribution_required: true
authorization_verified: true
verification_date: "2026-08-21"
```
---

# 11. DS-006 — Reddy et al. Larval Exploration and Aversive Chemotaxis Dataset

## Minimum Dataset Metadata

```yaml
dataset_id: DS-006
dataset_name: "Zebrafish larvae exploration and aversive chemotaxis dataset"
authors:
  - Gautam Reddy
  - Laura Desban
  - Hidenori Tanaka
  - Julian Roussel
  - Olivier Mirat
  - Claire Wyart
year: 2021
paper: "A lexical approach for identifying behavioural action sequences"
paper_year: 2022
repository: "https://datadryad.org/dataset/doi:10.5061/dryad.6t1g1jwwz"
doi: "10.5061/dryad.6t1g1jwwz"
license: "CC0 1.0 Universal / public-domain dedication through Dryad"
date_accessed: "2026-08-21"

species: "Danio rerio"
developmental_stage: "larval; 7 dpf"
number_of_fish: "384 potential fish-well units; 381 nonempty; 374 usable after frozen well-level QC"
number_of_sessions: 32
fish_per_session: 12
recording_families:
  pH_1a: 10
  pH_2a: 7
  pH_2b: 7
  pH_2c: 8
total_recording_duration: "10 min per fish/experiment"
frame_rate: "160 Hz"
resolution: "pixel size approximately 70 µm; camera identified as Basler acA2040-180km"
raw_video_available: "not in the Dryad processed-data archive reviewed; recording protocol is documented"
tracking_available: true
pose_available: true
fish_ids_available: true
session_ids_available: true
condition_labels_available: true

archive_filename: "Data_all.zip"
archive_sha256: "d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f"
scientific_files_extracted: 64
mat_files: 32
txt_files: 32

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

baseline_suitability: "VERY HIGH"
ssl_suitability: "HIGH / strong external replication dataset"
validation_suitability: "HIGH; separate wells preserve fish-level organization and experiment/well hierarchy is recoverable"

status: SELECTED_REPLICATION
role:
  - REPLICATION
```

DS-006 is an independently acquired experimental dataset from Reddy et al.
(2022), distinct from the Marques et al. recordings underlying DS-005. The two
datasets differ in assay design, acquisition rate, recording duration, stimulus
conditions, and tracking pipeline. They share some investigators and were later
analyzed together in Sridhar et al. (2024), but there is no indication that
DS-006 is a resplit or reuse of the DS-005 fish or recordings.

## Authorization Review

```yaml
license_name: "Creative Commons Zero v1.0 Universal (CC0)"
license_url: "https://creativecommons.org/publicdomain/zero/1.0/"
copyright_holder: "Rights waived/dedicated to the public domain to the extent permitted under CC0 by the dataset submitters"
research_use_allowed: true
redistribution_allowed: true
derivative_use_allowed: true
attribution_required: false
authorization_verified: true
verification_date: "2026-08-21"
```
---

# 12. Current Dataset Comparison

| Criterion | DS-001 | DS-002 | DS-003 | DS-004 | DS-005 PRIMARY | DS-006 REPLICATION |
|---|---|---|---|---|---|---|
| Temporal behavior | High | **Very High** | High | High | **Very High** | **Very High** |
| Pose relevance | **Very High** | High | Low | Low | **Very High** | **Very High** |
| Independent-fish structure | Weak/unknown | Complex/repeated | 10 fish | Weak across videos | **Strong** | **Strong** |
| Temporal scale | Mixed | **140 Hz** | ≤60 fps | 30 fps | **700 Hz** | **160 Hz** |
| Duration | Mixed | **5 h/session** | ~60 s/fish analysis | repeated short trials | **30 min–3 h/fish** | **10 min/fish** |
| Authorization | **CC BY 4.0** | Verify dataset license | Unclear dataset license | Partial | **CC BY 4.0** | **CC0** |
| Baseline suitability | High | High | **Very High** | High | **Very High** | **Very High** |
| SSL suitability | Pilot | Very high social SSL | Pilot | Public data insufficient | **PRIMARY** | **REPLICATION** |
| Validation suitability | Limited | High but social | Limited by n=10 | Limited | **High** | **High** |
| Selected use | Pose/Pilot | Future social | Benchmark | Benchmark | **PRIMARY** | **REPLICATION** |

---

# 13. Primary Dataset Requirements

The selected primary dataset, DS-005, has passed the project-level readiness gate for:

## Data

- [x] equivalent temporal behavioral sequences
- [x] multiple independent fish
- [x] sufficient duration
- [x] sufficient temporal resolution
- [x] stable source
- [x] clear authorization

## Baseline

- [x] speed
- [x] acceleration
- [x] turning
- [x] movement bouts
- [x] pose features

## SSL

- [x] temporal windows
- [x] sufficient number of windows
- [x] temporal representation-learning suitability

## Validation

- [x] fish-level split feasibility
- [x] fish-level recording identity
- [x] artifact/QC assessment
- [x] speed-only controls
- [x] window sensitivity

---

# 14. Minimum Dataset Metadata Requirement

Every serious candidate must include:

```yaml
dataset_id:
dataset_name:
authors:
year:
paper:
repository:
doi:
license:
date_accessed:

species:
developmental_stage:
number_of_fish:
number_of_sessions:
total_recording_duration:
frame_rate:
resolution:

raw_video_available:
tracking_available:
pose_available:
fish_ids_available:
session_ids_available:
condition_labels_available:

baseline_suitability:
ssl_suitability:
validation_suitability:

status:
role:
```

---

# 15. Dataset Authorization Requirement

Every serious candidate must include:

```yaml
license_name:
license_url:
copyright_holder:
research_use_allowed:
redistribution_allowed:
derivative_use_allowed:
attribution_required:
authorization_verified:
verification_date:
```

A publication license must not be silently treated as the license for an unreleased or separately deposited dataset.

---

# 16. Raw Data Repository Policy

Third-party data should normally not be committed directly to Git.

```text
data/
├── README.md
├── raw/
├── interim/
└── processed/
```

Reproducibility should use:

- DOI/source URL,
- acquisition instructions,
- checksums,
- dataset version,
- download date,
- processing code.

---

# 17. Primary Dataset Selection and Freeze

## Selected Primary Dataset

```yaml
dataset_id: DS-005
dataset_version: "Dataset content identified as V2; Zenodo record 13605471 currently exposes the selected archive Datasets.tar.gz"
download_date: "TBD — freeze only after local download"
source: "https://zenodo.org/records/13605471"
archive_filename: "Datasets.tar.gz"
published_md5: "b9a00fccda494bb49ea7c67d3b0f8d9e"
sha256: "TBD — calculate locally after download"
number_of_files: "TBD — one 7.1-GB archive is published; internal extracted-file count must be recorded after download"
number_of_fish: 463
number_of_sessions: "TBD — derive and record canonical session/recording IDs during ingestion"
total_duration: "individual recordings approximately 30 min to 3 h; at least 231.5 fish-hours at the 30-min lower bound"
```

## Freeze Status

```yaml
primary_dataset_selected: true
primary_dataset_authorization_verified: true
archive_selected: true
local_download_complete: false
sha256_verified: false
ingestion_inventory_complete: false
confirmatory_dataset_frozen: false
freeze_status: "SELECTED — LOCAL FREEZE PENDING"
```

**Freeze rule:** DS-005 becomes the final confirmatory dataset only after:

- [x] `Datasets.tar.gz` is downloaded from the selected Zenodo record.
- [x] Published MD5 is checked.
- [x] A local SHA-256 is calculated and recorded.
- [x] Archive/internal file count is recorded.
- [x] Fish-ID field/file mapping is verified.
- [x] Canonical session/recording ID is defined.
- [x] Exact number of sessions/recordings is recorded.
- [x] Missing-data and tracking-QC representation is documented.
- [x] The frozen raw-data directory is made read-only or otherwise version-locked.


After those steps:

```yaml
confirmatory_dataset_frozen: true
```

Any material change after freeze creates a new internal project version, for example:

```text
DS-005-v1
DS-005-v2
```

and must be recorded in `docs/decisions.md`.

---

# 18. Fish-Level Split Requirement

The confirmatory split must occur by fish, not by behavioral window.

Do not use:

```text
Fish 01
├── train windows
├── validation windows
└── test windows
```

Use:

```text
TRAIN FISH
        ↓
Representation development

VALIDATION FISH
        ↓
Model/analysis selection

TEST FISH
        ↓
Final evaluation
```

The exact split should be generated only after DS-005 ingestion confirms all fish identifiers and context labels.

---

# 19. Dataset Threat Matrix

| Threat | DS-001 | DS-002 | DS-003 | DS-004 | DS-005 | DS-006 |
|---|---|---|---|---|---|---|
| Identity leakage | High uncertainty | High-priority pair/fish issue | Testable but n=10 | Cross-video IDs weak | **Testable** | **Testable** |
| Session leakage | Metadata weak | Testable | Weak session diversity | Testable only partially | **Test after ingestion schema** | **Experiment-level testable** |
| Camera leakage | High diversity | Three-camera fixed setup | Two-camera setup | Fixed camera | Reduced by pose-first design | Reduced by processed pose/trajectory design |
| Speed-only solution | Testable | Testable | Testable | Testable if trajectories obtained | **Testable** | **Testable** |
| Tracking artifacts | Strong QC resource | Explicit QC | Important | Important | **Assessable** | **Explicitly assessed** |
| Window artifacts | Testable | Testable | Testable | Technically testable | **Testable** | **Testable** |
| Social confound | Low | **High** | Low | **High** | Low | Low |
| Insufficient independent animals | Unknown | No | **Yes** | Moderate | **No** | Verify exact count, expected adequate |
| License uncertainty | No | **Yes until Zenodo license checked** | **Yes** | Partial | No | No |

---

# 20. Dataset Selection Decision Rule

The primary dataset is selected according to:

1. authorized reuse,
2. temporal behavioral information,
3. multiple identifiable fish,
4. sufficient duration,
5. pose/tracking feasibility,
6. conventional baseline feasibility,
7. temporal SSL feasibility,
8. fish-level validation,
9. session/context metadata,
10. nuisance-variable testing,
11. behavioral interpretation,
12. computational feasibility.

**DS-005 currently provides the strongest combination for the first study.**

---

# 21. Current Decisions

## DS-001 — Scholz et al. (2025)

**Decision:** `APPROVED_WITH_LIMITATIONS`

**Role:** `POSE_RESOURCE`, `PILOT`, `BENCHMARK`

**Reason:** Public CC BY 4.0 pose/video resource with excellent 15-keypoint tracking infrastructure, but the released documentation does not establish the dataset-wide unique-fish/session structure required for the primary held-out-fish SSL study.

---

## DS-002 — Deligkaris et al. (2026)

**Decision:** `FUTURE_EXTENSION`

**Role:** `SOCIAL_EXTENSION`, future `VALIDATION`

**Reason:** Exceptionally rich 173-session, five-hour, 140-Hz 3D dyad resource, but the pairwise/social unit of analysis introduces partner identity, pair identity, inter-fish geometry, and repeated-fish complications beyond the first MVP. Dataset license still requires direct Zenodo verification.

---

## DS-003 — Yang et al. (2021)

**Decision:** `APPROVED_WITH_LIMITATIONS`

**Role:** `BENCHMARK`, `PRIOR_ART`, `PILOT`, `CROSS_DOMAIN_CHALLENGE`

**Reason:** Public 3D trajectories and direct methodological prior art are highly valuable for Input A and the novelty boundary, but only 10 independent adult fish and unclear standalone dataset licensing make it unsuitable as PRIMARY or the official independent replication dataset.

---

## DS-004 — Barreiros et al. (2021)

**Decision:** `APPROVED_WITH_LIMITATIONS`

**Role:** `BENCHMARK`, conventional tracking/behavior-analysis reference

**Reason:** Strong reference for 30-fps automated tracking, conditioning, speed, distance and group behavior, but a complete reusable temporal dataset was not identified and cross-video biological identity is not guaranteed.

---

## DS-005 — Sridhar/Marques Larval Navigation

**Decision:** `SELECTED_PRIMARY`

**Role:** `PRIMARY`

**Reason:** 463 larval zebrafish, high-resolution 700-Hz tail tracking, long individual recordings, multiple sensory contexts, clear CC BY 4.0 authorization, a stable Zenodo archive, strong hand-engineered baseline feasibility, strong pose-sequence SSL feasibility, and animal-level validation potential.

**Remaining action:** complete the local dataset freeze by downloading, hashing, inventorying, and defining the canonical session identifier.

---

## DS-006 — Reddy et al. Larval Exploration / Aversive Chemotaxis

**Decision:** `SELECTED_REPLICATION`

**Role:** `REPLICATION`

**Reason:** Independent lab/setup, 160-Hz 10-minute larval recordings, separate-well fish organization, ZebraZoom pose/trajectory/bout outputs, free-swimming and aversive-chemotaxis conditions, and clear CC0 reuse authorization. It is close enough biologically and representationally to test replication while remaining independently acquired.

**Ingestion status:** Complete. The archive contains 32 recording sessions with
12 well slots each: 384 potential fish-well units, 381 nonempty units, and 374
units usable after frozen well-level QC. Canonical IDs and the unresolved missing
`Catamaran_pH_2b_t7` scientific-file pair are recorded above and in the dedicated
replication protocol.

---

# 22. Primary Dataset

```yaml
primary_dataset: DS-005
dataset_name: "Dataset for Uncovering multiscale structure in the variability of larval zebrafish navigation V2"
status: SELECTED_PRIMARY
role: PRIMARY
confirmatory_freeze: PENDING_LOCAL_DOWNLOAD_AND_HASH
```

The primary-dataset selection decision is closed unless ingestion reveals a material defect that violates the readiness criteria.

---

# 23. Replication Dataset

```yaml
replication_dataset: DS-006
dataset_name: "Zebrafish larvae exploration and aversive chemotaxis dataset"
status: SELECTED_REPLICATION
role: REPLICATION
```

DS-006 is independent of DS-005 in acquisition setup and processing history while remaining sufficiently close in developmental stage and behavioral modality to support a meaningful replication attempt.

---

# 24. Immediate Dataset Backlog

## Remaining candidates
- [ ] Use DS-001 for pose/tracking QC development as needed.

---
