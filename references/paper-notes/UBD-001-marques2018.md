# UBD-001 — Marques et al. (2018)

## Citation

Marques JC, Lackner S, Félix R, Orger MB. *Structure of the Zebrafish Locomotor Repertoire Revealed with Unsupervised Behavioral Clustering.* Current Biology. 2018;28(2):181–195.e5. DOI: 10.1016/j.cub.2017.12.002.

## Why I Read It

Direct zebrafish precedent for genuinely unsupervised behavioral discovery and natural bout-level analysis.

## Research Question

Can a zebrafish larval locomotor repertoire be recovered from high-resolution movements without predefined behavior labels?

## Data

- Larval zebrafish.
- Millions of naturally segmented swim bouts.
- Multiple behavioral/sensory contexts.
- High temporal resolution.

## Method

- Kinematic representation of swim bouts.
- Robust unsupervised clustering using the authors' `clusterdv` approach.
- Analysis at multiple hierarchical levels.

## Behavioral Representation

Naturally segmented swim bouts represented in kinematic movement space, including tail/movement structure.

## Discovery Method

Unsupervised clustering.

## Validation

- Recovery of known swim types.
- Six additional clusters beyond previously recognized types.
- Context/stimulus usage analysis.
- Hierarchical consistency across movement organization.
- No complete identity/session/speed-leakage battery matching the present project was identified.

## Main Result

Thirteen basic swimming patterns were identified and used flexibly across contexts.

## Limitations for Present Project

- No self-supervised learned representation.
- No matched handcrafted-vs-SSL comparison.
- Does not provide the full held-out-fish/nuisance validation battery planned here.

## What This Means for Zebrafish SSL

Unsupervised discovery itself is not novel. Natural bouts are biologically defensible analysis units.

## Action for This Project

- Do not claim first unsupervised zebrafish behavioral discovery.
- Use bout-level analysis as literature-supported precedent.
- Require stronger nuisance controls than behavior recovery alone.
