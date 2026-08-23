# SSL-001 — ContrastivePose (Zhou et al., 2023)

## Citation

Zhou T, Cheah CCH, Chin EWM, Chen J, Farm HJ, Goh ELK, Chiam KH. *ContrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavorial classification of interacting animals.* Computers in Biology and Medicine. 2023;165:107416. DOI: 10.1016/j.compbiomed.2023.107416.

## Why I Read It

Closest methodological precedent for comparing self-supervised learned pose features against handcrafted animal-behavior features.

## Method

Self-supervised contrastive learning on pose-estimation data.

## SSL Objective

Positive pairs are behavior-preserving augmented versions of the same pose example; other examples act as negatives.

## Augmentations

Reviewed methodology includes geometric behavior-preserving transformations such as flipping, rotation, and translation.

## Downstream Evaluation

Supervised behavioral classification.

## Validation

- Learned representation evaluated on known behavioral labels.
- Compared with handcrafted features.
- Evaluated across animal-interaction datasets.
- No complete fish-identity/session/speed leakage battery relevant to this project was identified.

## Main Result

Self-supervised learned features improved downstream behavioral classification relative to handcrafted features.

## Limitations for Present Project

The primary evaluation is supervised classification, not unsupervised state discovery.

## What This Means for Zebrafish SSL

Directly supports a controlled handcrafted-vs-learned representation comparison without requiring a new SSL architecture.

## Action for This Project

Use contrastive learning as a practical primary family while keeping the scientific claim centered on discovery and validation.
