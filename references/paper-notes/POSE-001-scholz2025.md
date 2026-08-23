# POSE-001 — Scholz et al. (2025)

## Citation

Scholz LA, Mancienne T, Stednitz SJ, Scott EK, Lee CCY. *Plug-and-Play automated behavioral tracking of zebrafish larvae with DeepLabCut and SLEAP: pre-trained networks and datasets of annotated poses.* bioRxiv. 2025. DOI: 10.1101/2025.06.04.657938.

## Publication Status

Preprint, version 1 reviewed. Refresh status before manuscript submission.

## Why I Read It

Direct larval-zebrafish pose-estimation resource and tracking-quality precedent.

## Data

- Annotated larval zebrafish videos.
- Free-swimming and head-embedded preparations.
- Diverse imaging conditions.

## Method

- 15-keypoint pose schema.
- Four pretrained pose networks:
  - DeepLabCut variants.
  - SLEAP variants.

## Validation

- Ground-truth pose annotations.
- Model benchmarking.
- Evaluation across varying imaging conditions.

## Main Result

Provides reusable annotated datasets and pretrained networks that reduce setup burden for detailed zebrafish pose analysis.

## Limitations for Present Project

Tracking resource rather than SSL behavioral-discovery study.

## What This Means for Zebrafish SSL

Pose is viable, but tracking/model quality must be evaluated across recording conditions.

## Action for This Project

Use tracking-QC concerns to justify exclusion/sensitivity handling of unreliable coordinate-derived features.
