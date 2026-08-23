# Literature Screening Protocol

**Project:** Zebrafish Behavior SSL  
**File paired with this protocol:** `screening-ledger.csv`  
**Purpose:** Create a complete, reproducible title/abstract and full-text inclusion-exclusion audit trail.

---

## 1. Objective

Every paper or record that is actually screened should receive a ledger entry.

The screening ledger is intended to answer:

- Which records were reviewed?
- At what stage were they reviewed?
- Were they included, excluded, marked maybe, or identified as duplicates?
- Why was each record excluded?
- Why was each retained paper considered relevant?
- Which search produced the record?
- When and by whom was the decision made?

The ledger should be maintained prospectively whenever possible.

Do **not** invent retrospective screening counts or exclusion decisions that were not recorded.

---

## 2. Relationship to `search-log.csv`

`search-log.csv` summarizes each search.

`screening-ledger.csv` records the paper-level decisions behind those summaries.

Recommended relationship:

```text
results_screened
=
number of title/abstract ledger rows associated with a search_id
```

and, when full-text screening is performed:

```text
papers_retained
=
number of records ultimately marked include after full-text review
```

If an exact historical count was never recorded, retain `NR` in the search log rather than reconstructing it from memory.

---

## 3. Screening Stages

Use only the following values in `screening_stage`.

### `title_abstract`

Use when screening:

- title;
- abstract;
- keywords;
- basic bibliographic metadata;
- obvious study scope.

### `full_text`

Use only after a paper passes title/abstract screening and the full paper or sufficiently complete methods/results have been reviewed.

For papers that reach full-text screening, create a **second ledger row** rather than overwriting the title/abstract decision.

Example:

```text
SCR-014 | title_abstract | include
SCR-014 | full_text      | include
```

or:

```text
SCR-014 | title_abstract | include
SCR-014 | full_text      | exclude | insufficient_methods_detail
```

---

## 4. Allowed Decisions

Use only:

```text
include
exclude
maybe
duplicate
```

### `include`

The paper is relevant enough to retain for the current review stage.

### `exclude`

The paper is outside scope or fails an inclusion criterion.

### `maybe`

The available information is insufficient for a confident decision.

A `maybe` record should later be resolved to `include` or `exclude` whenever possible.

### `duplicate`

The same paper or record was already screened elsewhere.

When marking a duplicate, note the retained record ID in `notes`.

---

## 5. Inclusion Criteria

Retain a paper if it materially informs at least one of the following project areas.

### Zebrafish behavior

- zebrafish behavioral representation;
- zebrafish behavioral-state discovery;
- zebrafish locomotion or pose analysis;
- zebrafish temporal behavior analysis;
- zebrafish unsupervised clustering or behavioral embedding.

### Self-supervised / representation learning

- self-supervised learning for animal behavior;
- contrastive learning for behavioral sequences;
- temporal representation learning;
- pose-sequence representation learning;
- video-based behavioral representation learning.

### Unsupervised discovery

- unsupervised behavioral-state discovery;
- clustering of animal behavior;
- latent-state or embedding-based behavior discovery;
- state-number selection methods relevant to behavioral clustering.

### Validation and nuisance controls

- held-out-subject evaluation;
- cross-animal generalization;
- identity leakage;
- session/camera/context leakage;
- locomotor speed dependence;
- tracking-quality or artifact sensitivity;
- seed or representation stability.

### Temporal scale and segmentation

- bout-based behavioral units;
- fixed temporal windows;
- adaptive segmentation;
- temporal-scale sensitivity;
- movement-bout detection relevant to comparable zebrafish data.

### Datasets / tracking methods

- directly relevant zebrafish datasets;
- pose/tracking schemas used in comparable behavioral analysis;
- tracking-QC methods that materially affect representation validity.

A paper does **not** need to cover all categories.

---

## 6. Standardized Exclusion Reasons

Use one primary exclusion reason in `exclusion_reason`.

Allowed values:

```text
wrong_organism
not_behavioral
not_ssl
not_unsupervised
not_temporal
not_pose_or_tracking
not_empirical
review_only
insufficient_methods_detail
no_relevant_validation
duplicate
not_accessible
outside_scope
```

### Definitions

#### `wrong_organism`

The study organism is outside the intended comparative scope and the methods do not materially inform the project.

#### `not_behavioral`

The study does not analyze behavior or behavioral representations.

#### `not_ssl`

The record was screened for an SSL-specific search but does not actually use or materially discuss self-supervised representation learning.

Use this only when SSL relevance was required for that search stream.

#### `not_unsupervised`

The paper does not address unsupervised discovery when that was required by the specific search stream.

#### `not_temporal`

The method lacks a temporal or sequence component when temporal representation was required by the search stream.

#### `not_pose_or_tracking`

The record does not materially involve pose, tracking, movement representation, or comparable behavioral inputs where those were required.

#### `not_empirical`

The work is purely conceptual, editorial, or otherwise lacks relevant empirical methodology.

#### `review_only`

The paper is a review and does not provide primary empirical methods needed for the extraction target.

A review may still be retained if it is being used explicitly for background or citation chaining.

#### `insufficient_methods_detail`

The record appears relevant but does not provide enough methodological information for the review objective.

#### `no_relevant_validation`

The work lacks the specific validation procedure targeted by that search stream.

Use carefully; absence of one validation type is not a general reason to discard an otherwise relevant paper.

#### `duplicate`

Duplicate of a record already screened.

#### `not_accessible`

The record could not be accessed sufficiently to make the required screening decision.

#### `outside_scope`

The paper does not materially inform the preregistered project questions despite superficial keyword overlap.

---

## 7. Inclusion Reason

For included papers, `inclusion_reason` should contain one short, concrete explanation.

Good examples:

```text
Direct zebrafish unsupervised behavioral-state discovery study.
```

```text
Animal-behavior temporal contrastive learning with held-out-subject evaluation.
```

```text
Provides zebrafish pose schema and tracking QC relevant to Input B preprocessing.
```

```text
Defines conventional zebrafish locomotor features used as Input A precedent.
```

Avoid vague entries such as:

```text
relevant
good paper
important
```

---

## 8. Record IDs

Use a stable identifier such as:

```text
SCR-001
SCR-002
SCR-003
```

The same paper should retain the same `record_id` across screening stages.

Example:

```text
SCR-021 | title_abstract
SCR-021 | full_text
```

Do not assign a new ID merely because the screening stage changed.

---

## 9. Search IDs

`search_id` should correspond to the search entry in `search-log.csv`.

Examples:

```text
S01
S02
S03
```

If an older search log currently lacks explicit IDs, add stable IDs there before completing the ledger.

A paper found through more than one search may either:

1. retain the first search ID and note later rediscovery in `notes`; or
2. receive duplicate rows marked `duplicate` for later searches.

Use one approach consistently.

---

## 10. Title / Abstract Screening Rule

At title/abstract stage:

Include if the record appears plausibly relevant to at least one inclusion category.

Exclude only when the available metadata makes the exclusion clear.

When uncertain:

```text
decision = maybe
```

Do not exclude based on assumptions about methods that are not visible from the title/abstract.

---

## 11. Full-Text Screening Rule

At full-text stage, determine whether the paper actually supports the extraction goal.

Check, where relevant:

- organism and developmental stage;
- sample size;
- behavioral unit;
- video / pose / trajectory input;
- hand-engineered features;
- SSL or representation-learning method;
- clustering method;
- cluster-number selection;
- validation split;
- held-out subjects;
- leakage controls;
- speed dependence;
- tracking/QC handling;
- window/bout duration;
- code and data availability.

If included, record a concrete `inclusion_reason`.

If excluded, use one standardized primary exclusion reason.

---

## 12. Duplicate Handling

If a duplicate is found:

```text
decision = duplicate
exclusion_reason = duplicate
```

and add:

```text
Duplicate of SCR-0XX.
```

to `notes`.

Do not count duplicates as independently screened retained papers.

---

## 13. Retrospective Entries

For searches already completed before this ledger existed:

- add paper-level entries only when the actual screened records can be reconstructed from saved searches, notes, browser history, citation lists, or existing literature files;
- distinguish reconstructed decisions from contemporaneous decisions in `notes`;
- do not invent records simply to make historical counts look complete.

Suggested note:

```text
Retrospectively reconstructed from existing project literature notes on 2026-08-23.
```

---

## 14. Reviewer and Date

Fill:

```text
reviewer
date_screened
```

whenever possible.

Recommended date format:

```text
YYYY-MM-DD
```

If historical screening date is unknown, do not guess.

---

## 15. Counting Rules

For a given search:

### Results screened

Count title/abstract rows whose decision is one of:

```text
include
exclude
maybe
duplicate
```

if those rows genuinely represent reviewed search results.

### Papers retained

For searches with full-text review:

Count unique records whose final full-text decision is:

```text
include
```

For searches where only title/abstract screening was performed, clearly document that the retained count is provisional.

---

## 16. Audit Quality Rules

The ledger should satisfy all of the following before preregistration/publication archive:

- [ ] Every recorded screening decision has a `record_id`.
- [ ] Every record is linked to a `search_id`.
- [ ] Every row has a screening stage.
- [ ] Every row has a decision.
- [ ] Every excluded record has a standardized exclusion reason.
- [ ] Every included record has a concrete inclusion reason.
- [ ] Duplicate records point to the retained record.
- [ ] Full-text decisions do not overwrite title/abstract decisions.
- [ ] Historical/reconstructed entries are labeled as such.
- [ ] Search-log counts agree with the ledger where counts are known.
- [ ] Unknown historical counts remain `NR`, not guessed.

---

## 17. File Governance

Recommended paths:

```text
docs/literature/screening-protocol.md
docs/literature/screening-ledger.csv
docs/literature/search-log.csv
```

Treat the ledger as a research record.

Do not delete excluded rows after screening.

If a decision changes, preserve the prior decision in Git history and document the reason for the update.

---

## 18. Current Project Use

For the Zebrafish Behavior SSL project, the ledger should be used to systematically document screening decisions for literature streams involving:

- direct zebrafish SSL;
- zebrafish temporal contrastive learning;
- zebrafish pose-sequence representation learning;
- zebrafish unsupervised behavioral-state discovery;
- conventional zebrafish locomotion/pose baselines;
- held-out-animal validation;
- identity/session/context leakage;
- speed-controlled embeddings;
- tracking/QC;
- behavioral window duration and segmentation;
- relevant animal-behavior SSL precedents.

This ledger supports transparency and reproducibility. It does not retroactively convert previously undocumented search behavior into a fully prospective systematic review.
