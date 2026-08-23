# UBD-002 — Ghosh & Rihel (2020)

## Citation

Ghosh M, Rihel J. *Hierarchical Compression Reveals Sub-Second to Day-Long Structure in Larval Zebrafish Behavior.* eNeuro. 2020;7(4):ENEURO.0408-19.2020. DOI: 10.1523/ENEURO.0408-19.2020.

## Why I Read It

Direct zebrafish precedent for unsupervised discovery across multiple temporal scales and for resampling/repeated-clustering style robustness.

## Research Question

How is larval zebrafish behavior organized from sub-second movements through longer motifs and circadian timescales?

## Data

- Hundreds of larval zebrafish.
- Continuous recordings over multiple day/night cycles.
- Millions of movement and pause bouts.

## Method

- Active/inactive bout features.
- Evidence-accumulation clustering to define modules.
- Hierarchical compression to identify recurrent motifs.

## Behavioral Representation

Sequences of active/inactive bouts and derived modules/motifs.

## Discovery Method

Unsupervised evidence-accumulation clustering plus hierarchical compression.

## Validation

- Repeated/resampled clustering is integral to evidence accumulation.
- Day/night association.
- Pharmacological perturbation association.
- Genetic phenotype association.
- Authors show some higher-order compressibility effects are not reducible to overall activity level.
- No complete held-out-fish identity/session leakage battery identified.

## Main Result

Behavior is structured across sub-second, minute, and day-long timescales, with modules/motifs sensitive to biological conditions.

## Limitations for Present Project

- Engineered activity representation rather than learned SSL representation.
- Different recording/data modality from DS-005.

## What This Means for Zebrafish SSL

Behavioral structure is multi-scale. Stability and biological perturbation association are important forms of validation.

## Action for This Project

- Treat temporal scale as a sensitivity/interpretation issue.
- Include stability and speed/activity controls.
