# Provenance, License, Checksum, and Third-Party Attribution Audit

**Project:** Zebrafish Behavior SSL  
**Repository:** `zebrafish-behavior-ssl`  
**Audit date:** 2026-08-23  
**Audit status:** IN PROGRESS — core checksum and third-party software license checks completed; remaining provenance/license items should be verified before release.

---

## 1. Purpose

This audit records the integrity, provenance, licensing, and attribution status of datasets, processed artifacts, and third-party software used by the project.

The audit is intended to support:

- reproducibility;
- preregistration and publication readiness;
- traceable dataset provenance;
- verification that frozen artifacts have not changed;
- compliance with third-party software and dataset licenses;
- clear separation between project-owned code and external materials.

Checksum verification is an integrity check only. Verifying a checksum for a protected TEST artifact does **not** inspect its scientific contents or use it for method selection.

---

## 2. Audit Summary

| Item | Checksum / Commit | Provenance | License | Attribution | Status |
|---|---|---|---|---|---|
| DS-005 processed baseline artifacts | Verified | Documented in project records | Dataset license must remain documented separately | Required dataset citation/attribution must be preserved | PASS — integrity |
| DS-005 baseline clustering artifacts | Manifest present; verification still to be recorded in this audit | Documented in project records | N/A for generated project artifacts | Project-generated | PENDING CHECKSUM VERIFICATION |
| DS-006 processed artifacts | Verified | Dryad dataset provenance documented | Dataset license should be recorded from authoritative source | Dataset citation/attribution required | PASS — integrity |
| DS-006 author code (`BASS-master`) | External code | Author repository/code bundle | MIT | Copyright notice + MIT permission text must be retained | PASS |
| `Markov_Fish` | Commit `c850359dc9d57dbb850c11ce56dfa294af2c3fca` | Git submodule | MIT | Copyright notice + MIT permission text must be retained | PASS |
| Project original code/docs | Git-controlled | This repository | MIT | Project `LICENSE` applies only to original project materials unless noted | PASS |
| DS-002 | External dataset | Zenodo record 18569873 | CC BY 4.0 | Attribution required | VERIFIED IN PROJECT RECORDS |
| DS-003 | Prior-art / benchmark reference | External study | Not audited here | Preserve source citation | PENDING LICENSE REVIEW IF CODE/DATA DISTRIBUTED |
| DS-004 | Prior-art / benchmark reference | External study | Not audited here | Preserve source citation | PENDING LICENSE REVIEW IF CODE/DATA DISTRIBUTED |

---

# 3. DS-005 Integrity Audit

## 3.1 Processed baseline artifacts

Command used:

```bash
cd data/processed/DS-005/baseline
shasum -a 256 -c SHA256SUMS
```

Observed verification results:

```text
build_audit_core.json: OK
feature_schema_core.json: OK
scaler_core.json: OK
test_core_raw.npz: OK
test_core_scaled.npz: OK
train_core_raw.npz: OK
train_core_scaled.npz: OK
validation_core_raw.npz: OK
validation_core_scaled.npz: OK
```

### Result

**PASS**

All files listed in:

```text
data/processed/DS-005/baseline/SHA256SUMS
```

matched their recorded SHA-256 hashes at the time of this audit.

This includes TRAIN, VALIDATION, and protected TEST files.

No scientific TEST values were inspected during checksum verification.

---

## 3.2 Baseline clustering selection artifacts

Expected manifest:

```text
data/processed/DS-005/baseline_clustering/SELECTION_SHA256SUMS
```

Expected generated artifacts include:

```text
pca_diagnostics.json
selected_configuration.json
selection_results.json
SELECTION_SHA256SUMS
```

Frozen selected configuration:

```yaml
method: gmm
k: 2
seed: 20260822
pca_components: 6
pca_variance_target: 0.95
selection_score: 0.649252350828649
```

### Status

**PENDING CHECKSUM VERIFICATION**

Run from the repository root:

```bash
cd /Users/yasamean/zebrafish-behavior-ssl
cd data/processed/DS-005/baseline_clustering
shasum -a 256 -c SELECTION_SHA256SUMS
```

After successful verification, update this section to **PASS** and record the output.

---

# 4. DS-006 Integrity Audit

## 4.1 Processed artifact manifest

Command used:

```bash
cd /Users/yasamean/zebrafish-behavior-ssl
shasum -a 256 -c data/manifests/DS-006/processed-sha256.txt
```

Observed verification results:

```text
data/processed/DS-006/baseline/feature_manifest.json: OK
data/processed/DS-006/baseline/normalization.json: OK
data/processed/DS-006/baseline/test_core_raw.npz: OK
data/processed/DS-006/baseline/test_core_scaled.npz: OK
data/processed/DS-006/baseline/train_core_raw.npz: OK
data/processed/DS-006/baseline/train_core_scaled.npz: OK
data/processed/DS-006/baseline/validation_core_raw.npz: OK
data/processed/DS-006/baseline/validation_core_scaled.npz: OK
data/processed/DS-006/metadata/bout_metadata.csv: OK
data/processed/DS-006/metadata/qc_summary.json: OK
data/processed/DS-006/metadata/split_assignments.csv: OK
data/processed/DS-006/ssl/input_manifest.json: OK
data/processed/DS-006/ssl/normalization.json: OK
data/processed/DS-006/ssl/test.npz: OK
data/processed/DS-006/ssl/train.npz: OK
data/processed/DS-006/ssl/validation.npz: OK
```

### Result

**PASS**

All files listed in:

```text
data/manifests/DS-006/processed-sha256.txt
```

matched their recorded SHA-256 hashes at the time of this audit.

Protected TEST artifacts were hash-verified only. Their scientific contents were not inspected.

---

## 4.2 Raw archive

Recorded project provenance:

```text
Dataset: Reddy et al. larval exploration / aversive chemotaxis dataset
Repository: Dryad
DOI: 10.5061/dryad.6t1g1jwwz
Raw archive: data/raw/DS-006/Data_all.zip
Recorded SHA-256:
d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f
```

### Status

**RECORDED — RAW HASH SHOULD BE REVERIFIED BEFORE RELEASE**

Recommended command:

```bash
shasum -a 256 data/raw/DS-006/Data_all.zip
```

Expected SHA-256:

```text
d94261a2ed89356cd0dd5f9fe69219aaae567eeac31cf46d90769c9aba40094f
```

---

# 5. Third-Party Software Audit

## 5.1 DS-006 author code — `BASS-master`

Path:

```text
external/DS-006-author-code/BASS-master/
```

License file:

```text
external/DS-006-author-code/BASS-master/LICENSE
```

Observed license:

```text
MIT License
Copyright (c) 2020 greddy992
```

### License obligations

The MIT license permits use, copying, modification, distribution, sublicensing, and sale.

The project must preserve:

1. the copyright notice; and
2. the MIT permission notice

in copies or substantial portions of the third-party software.

### Status

**PASS**

The upstream license file is present in the external-code directory.

The project-level MIT license does **not** replace or reassign the copyright of this third-party code.

---

## 5.2 `Markov_Fish`

Path:

```text
external/Markov_Fish/
```

License file:

```text
external/Markov_Fish/LICENSE
```

Observed license:

```text
MIT License
Copyright (c) 2024 Gautam Sridhar
```

### Pinned commit

Command:

```bash
git rev-parse HEAD
```

Observed commit:

```text
c850359dc9d57dbb850c11ce56dfa294af2c3fca
```

### Working-tree status

Command:

```bash
git status --short
```

Observed result:

```text
<no output>
```

Therefore, the checked-out `Markov_Fish` working tree was clean at the time of the audit.

### Superproject submodule status

Command:

```bash
git submodule status
```

Observed:

```text
-c850359dc9d57dbb850c11ce56dfa294af2c3fca external/Markov_Fish
```

The commit hash matches the commit returned by `git rev-parse HEAD`.

The leading `-` in the superproject output should be investigated if exact submodule initialization state matters for release automation. It does **not** change the recorded commit identity above.

### License obligations

The MIT license requires preservation of:

1. `Copyright (c) 2024 Gautam Sridhar`
2. the accompanying MIT permission notice

in copies or substantial portions of the software.

### Status

**PASS — LICENSE / COMMIT / CLEAN TREE**

Release tooling should additionally confirm the submodule is initialized in the intended way.

---

# 6. Repository License Audit

License files found by:

```bash
find . -maxdepth 4 \( -iname "LICENSE*" -o -iname "COPYING*" -o -iname "NOTICE*" \) | sort
```

Observed:

```text
./external/DS-006-author-code/BASS-master/LICENSE
./external/Markov_Fish/LICENSE
./LICENSE
```

### Project license

The repository root contains:

```text
LICENSE
```

Project policy:

> Original software and documentation created for this repository are released under the MIT License unless otherwise noted.

This project license must not be interpreted as relicensing:

- external datasets;
- external repositories;
- author-supplied code;
- pretrained models;
- copied assets;
- or any other third-party material.

Each third-party item retains its own copyright and license.

---

# 7. Dataset License and Attribution Audit

## 7.1 DS-002

Project governance records identify DS-002 as the Deligkaris et al. social-dyad dataset.

Authoritative Zenodo metadata previously verified:

```text
Zenodo record: 18569873
License: CC BY 4.0
```

### Attribution requirement

CC BY 4.0 requires appropriate attribution, a license reference, and indication of changes when applicable.

### Project role

```text
FUTURE_SOCIAL_EXTENSION
```

### Status

**VERIFIED**

---

## 7.2 DS-003

Project role:

```text
PRIOR_ART / BENCHMARK
```

Use:

- conventional baseline reference;
- 3D tracking;
- engineered features;
- PCA;
- FuzzyART.

### Status

**REFERENCE-ONLY AUDIT CURRENTLY SUFFICIENT**

If DS-003 data or code are ever redistributed, incorporated, or executed as part of the reproducible pipeline, its exact license must be independently verified first.

---

## 7.3 DS-004

Project role:

```text
PRIOR_ART / BENCHMARK / TRACKING-CONVENTIONAL-ANALYSIS REFERENCE
```

### Status

**REFERENCE-ONLY AUDIT CURRENTLY SUFFICIENT**

If DS-004 data or code are redistributed, incorporated, or executed as part of the reproducible pipeline, its exact license must be independently verified first.

---

## 7.4 DS-005

Project role:

```text
PRIMARY
```

Integrity of the processed baseline artifacts is verified above.

### Remaining license audit requirement

Before publication or public data-distribution instructions are finalized, record the authoritative DS-005:

- source;
- DOI / repository identifier;
- dataset version;
- license;
- required citation;
- redistribution conditions;
- access conditions, if any.

### Status

**PROVENANCE DOCUMENTED IN PROJECT — LICENSE FIELD SHOULD BE RECONFIRMED BEFORE RELEASE**

---

## 7.5 DS-006

Project role:

```text
EXTERNAL_REPLICATION
```

Authoritative source recorded:

```text
Dryad
DOI: 10.5061/dryad.6t1g1jwwz
```

### Remaining license audit requirement

Before release, record the exact dataset license from the authoritative Dryad record and ensure the repository's data-availability statement preserves any citation and redistribution requirements.

Do not infer the dataset license from the MIT license of the accompanying author code. Dataset and code licenses are separate.

### Status

**PROVENANCE VERIFIED / DATASET LICENSE TO BE RECORDED FROM AUTHORITATIVE SOURCE**

---

# 8. Third-Party Attribution Requirements

The following attribution must remain visible in the repository when relevant third-party code is included:

### DS-006 author code

```text
BASS author code
Copyright (c) 2020 greddy992
Licensed under the MIT License.
See external/DS-006-author-code/BASS-master/LICENSE.
```

### Markov_Fish

```text
Markov_Fish
Copyright (c) 2024 Gautam Sridhar
Licensed under the MIT License.
See external/Markov_Fish/LICENSE.
Pinned project commit:
c850359dc9d57dbb850c11ce56dfa294af2c3fca
```

Recommended repository documentation locations:

```text
README.md
docs/dataset-register.md
docs/audits/provenance-license-audit.md
external/<project>/LICENSE
```

If a formal `THIRD_PARTY_NOTICES.md` file is later added, these notices should be included there as well.

---

# 9. Protected TEST Governance During Audit

The following protected TEST files were included in checksum manifests:

```text
DS-005:
data/processed/DS-005/baseline/test_core_raw.npz
data/processed/DS-005/baseline/test_core_scaled.npz

DS-006:
data/processed/DS-006/baseline/test_core_raw.npz
data/processed/DS-006/baseline/test_core_scaled.npz
data/processed/DS-006/ssl/test.npz
```

Audit action performed:

```text
SHA-256 comparison only
```

Scientific contents inspected:

```text
NO
```

Used for feature definition, model selection, hyperparameter tuning, clustering selection, interpretation, or claim decisions:

```text
NO
```

Therefore, checksum verification does not constitute confirmatory TEST evaluation.

---

# 10. Remaining Audit Actions

Before publication/release, complete the following:

- [ ] Verify `data/processed/DS-005/baseline_clustering/SELECTION_SHA256SUMS`.
- [ ] Reverify the raw DS-006 archive SHA-256.
- [ ] Confirm the authoritative DS-005 dataset license and required citation.
- [ ] Confirm the authoritative DS-006 **dataset** license from Dryad.
- [ ] Confirm the exact provenance/version identifier for DS-005.
- [ ] Record DS-005 and DS-006 dataset citations in `docs/dataset-register.md`.
- [ ] Decide whether to add a root `THIRD_PARTY_NOTICES.md`.
- [ ] Confirm all Git submodules are initialized/pinned reproducibly before release.
- [ ] Confirm no third-party files are accidentally covered by statements implying project ownership.
- [ ] Re-run checksum audits immediately before preregistration archive/public release.
- [ ] Record the Git commit associated with the final audit.

---

# 11. Current Audit Conclusion

At the time of this audit:

- DS-005 processed baseline artifacts matched their recorded SHA-256 hashes.
- DS-006 processed artifacts matched their recorded SHA-256 hashes.
- Protected TEST files were integrity-checked only and were not scientifically inspected.
- The DS-006 author-code license is MIT.
- `Markov_Fish` is MIT-licensed.
- `Markov_Fish` is checked out at commit `c850359dc9d57dbb850c11ce56dfa294af2c3fca`.
- The `Markov_Fish` working tree was clean.
- The repository root contains a project `LICENSE`.
- Third-party software licenses remain separate from the project license.
- Remaining release-readiness work is primarily authoritative dataset-license confirmation, dataset citation completion, DS-005 clustering checksum verification, and final pre-release re-audit.

**Overall current status: PARTIAL PASS — core integrity and third-party software licensing verified; dataset-license and final release-attribution checks remain open.**
