# DATA-001 — Deligkaris et al. (2026)

## Citation

Deligkaris K, Neiman R, Hiroi M, Izawa T, O'Shaughnessy L, Carretero Rodriguez L, Masai I, Stephens GJ. *A dataset of fine-grained zebrafish interactions in health and disease.* Scientific Data. 2026;13:583. DOI: 10.1038/s41597-026-06953-6.

## Why I Read It

High-resolution adult zebrafish tracking resource with persistent identity and explicit QC limitations.

## Data

- Dyadic adult zebrafish interactions.
- 3D tracking.
- Persistent fish identity.
- Health/disease and other condition metadata.

## Representation

Three anatomical landmarks in 3D plus identity/tracking metadata.

## Tracking / Identity Method

SLEAP body-point detections combined with identity tracking.

## Validation / QC

- Explicit tracking-quality workflow.
- Persistent individual identity.
- Residual identity swaps and body-part errors are acknowledged, especially during close interactions.

## Main Result

Provides a reusable fine-grained social-interaction dataset and highlights practical identity/pose errors that survive automated tracking.

## What This Means for Zebrafish SSL

Identity and tracking artifacts are credible nuisance variables capable of masquerading as behavioral structure.

## Action for This Project

- Quantify fish-identity predictability.
- Keep tracking-QC sensitivity analyses.
- Never interpret clusters without artifact checks.
