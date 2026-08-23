# ZF-002 — Barreiros et al. (2021)

## Citation

Barreiros MO, Barbosa FG, Dantas DO, dos Santos DML, Ribeiro S, Santos GCO, Barros AK. *Zebrafish automatic monitoring system for conditioning and behavioral analysis.* Scientific Reports. 2021;11:9330. DOI: 10.1038/s41598-021-87502-6.

## Why I Read It

Evidence for conventional zebrafish behavioral metrics and automated monitoring.

## Research Question

Can an automated system condition zebrafish and quantify individual/group behavioral responses online?

## Data

Adult zebrafish under visual, vibration, and food-reward conditioning paradigms.

## Method

Automated fish detection/tracking with controlled stimulus delivery.

## Behavioral Representation

Reported metrics include:

- distance traveled,
- speed,
- route/spatial behavior,
- polarization/group coordination.

## Discovery Method

Not a primary unsupervised-discovery paper.

## Validation

- Stimulated vs control comparisons.
- Expected conditioning-related responses.
- Automated tracking/detection system evaluation.
- Speed is an outcome, not explicitly controlled as a representation confound.

## Main Result

Stimulated fish showed measurable differences in movement and group coordination.

## Limitations for Present Project

Does not compare learned and handcrafted representations.

## What This Means for Zebrafish SSL

Speed and movement magnitude are deeply conventional behavioral variables and therefore must be included in Input A and explicitly tested as potential SSL confounds.

## Action for This Project

Supports timing/speed/movement/orientation feature families.
