# SSL-003 — BEAST (Wang et al., ICLR 2026)

## Citation

Wang Y, Yu H, Blau A, Zhang Y, International Brain Laboratory, Paninski L, Hurwitz C, Whiteway MR. *Animal Behavioral Analysis and Neural Encoding with Transformer-Based Self-Supervised Pretraining.* ICLR 2026. arXiv:2507.09513.

## Why I Read It

Strong recent animal-behavior SSL precedent for temporal contrastive learning.

## Method

Experiment-specific transformer pretraining for behavioral video.

## SSL Objective

- masked autoencoding,
- temporal contrastive learning.

## Downstream Evaluation

- neural encoding,
- pose estimation,
- action segmentation,
- single- and multi-animal settings,
- multiple species/datasets.

## Validation

Cross-dataset/task/species evidence is strong. The reviewed material did not establish the exact full subject-identity/session/speed leakage battery planned here.

## Main Result

Self-supervised behavioral-video pretraining improves several neurobehavioral downstream tasks.

## What This Means for Zebrafish SSL

Temporal contrastive learning has strong contemporary animal-behavior precedent.

## Action for This Project

Supports temporal contrastive objective selection while retaining a smaller 1D CNN suited to DS-005 bout sequences.
