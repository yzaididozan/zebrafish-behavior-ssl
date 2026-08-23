# Citation Chaining Protocol

**Project:** Zebrafish Behavior SSL  
**Ledger:** `references/citation-chain.csv`  
**Status:** ACTIVE  
**Purpose:** Record backward and forward citation checking for the core prior-art seeds used to support the frozen novelty boundary.

---

## 1. Core Seeds

Complete backward and forward citation checking for:

```text
UBD-001 — Marques et al. (2018)
UBD-002 — Ghosh & Rihel (2020)
ZF-002  — Yang et al. (2021)
SSL-001 — ContrastivePose (2023)
SSL-002 — Xu & Wang zebrafish SSL (2024)
SSL-003 — BEAST
```

`SSL-002` receives special attention because it is currently the closest direct zebrafish SSL precedent.

---

## 2. Ledger Columns

```text
check_date
seed_paper_id
direction
source_checked
candidate_title
candidate_authors
candidate_year
doi_or_url
screening_decision
exclusion_reason
retained_paper_id
notes
```

### `check_date`

Date on which the citation source was screened.

Format:

```text
YYYY-MM-DD
```

### `seed_paper_id`

Evidence-matrix ID of the paper whose references/citations were checked.

### `direction`

Allowed values:

```text
backward
forward
```

### `source_checked`

Record the actual citation source, for example:

```text
paper reference list
Google Scholar
Semantic Scholar
Scopus
Web of Science
Crossref
PubMed
```

For forward citation checks, always record the database because citation indexes change over time.

### `screening_decision`

Allowed values:

```text
include
exclude
duplicate
uncertain
```

### `exclusion_reason`

Required whenever:

```text
screening_decision = exclude
```

Recommended controlled reasons:

```text
outside_scope
wrong_organism
not_behavioral
not_temporal
not_ssl
not_unsupervised
not_pose_or_tracking
no_relevant_validation
review_only
insufficient_methods_detail
not_accessible
other
```

Use the most specific defensible primary reason.

### `retained_paper_id`

If included, assign or record the evidence-matrix ID.

Examples:

```text
UBD-003
METHOD-005
SSL-005
VALID-001
```

Leave blank for excluded or uncertain candidates.

---

# 3. Backward Citation Checking

For every core seed:

1. Open the complete reference list.
2. Screen all plausibly relevant references.
3. Record each plausible candidate in `citation-chain.csv`.
4. Screen for relevance to:
   - zebrafish behavioral clustering;
   - behavioral representation learning;
   - pose or trajectory analysis;
   - self-supervised learning;
   - held-out-subject evaluation;
   - identity/session/speed controls;
   - tracking-QC;
   - cluster stability and validation;
   - temporal segmentation or state-number selection.
5. Deep-read retained candidates.
6. Synchronize every included study across:
   - `docs/literature/literature-matrix.csv`;
   - `references/papers.bib`;
   - `references/paper-notes/`.
7. Repeat backward checking for newly retained papers when they materially extend the novelty or validation chain.
8. Stop when a new pass yields no materially new study.

Do not record every irrelevant reference from a paper merely to inflate the audit trail. Record every reference that was *plausibly relevant enough to require a screening decision*.

---

# 4. Forward Citation Checking

For every core seed:

1. Search later papers citing the seed.
2. Record:
   - database;
   - check date;
   - candidate bibliographic details;
   - screening decision.
3. Use the same inclusion/exclusion criteria as backward screening.
4. Deep-read included candidates.
5. Synchronize included papers across the matrix, bibliography, and notes.
6. Pay special attention to forward citations of `SSL-002`.
7. Repeat the forward check in a second pass before final literature freeze/archive.

---

# 5. Evidence-Matrix Synchronization

Every:

```text
screening_decision = include
```

must have:

```text
retained_paper_id != blank
```

and must be synchronized to all three research records:

```text
docs/literature/literature-matrix.csv
references/papers.bib
references/paper-notes/<paper_id>-*.md
```

A paper already represented in the matrix should be marked:

```text
screening_decision = duplicate
```

and `retained_paper_id` should point to the existing matrix ID.

---

# 6. Search-Log Synchronization

After completing one backward or forward check, add one summary row to:

```text
references/search-log.csv
```

The summary row should record the true counts derived from `citation-chain.csv`.

Recommended interpretation:

```text
results_screened
=
number of candidate rows screened for that seed/direction/source/date
```

```text
papers_retained
=
number of unique candidates marked include for that completed check
```

Example summary:

```text
2026-08-23,Google Scholar,Forward citations of Xu & Wang 2024,42,3,"Candidate-level decisions recorded in references/citation-chain.csv; no study matched the complete frozen comparison and validation framework."
```

Do not use `NR` for newly completed citation-chain checks. Once a check is intentionally performed under this protocol, record the actual counts.

---

# 7. Completion Rules

Citation-chain auditing is considered complete only when:

- [x] Every core seed has a backward citation check for the targeted first pass.
- [x] Every core seed has a forward citation check for the targeted first pass.
- [x] Each check records the date and citation source/database.
- [x] Every plausibly relevant candidate identified in the first pass has a ledger row.
- [x] Every candidate has a screening decision.
- [x] Every excluded candidate has an exclusion reason.
- [x] Every included candidate has a retained matrix ID.
- [x] Included papers are synchronized across matrix, bibliography, and paper notes.
- [x] Search-log summary counts agree with the candidate-level ledger.
- [ ] Newly retained papers that materially affect the novelty chain receive appropriate follow-up chaining.
- [ ] A second pass identifies no study that materially changes the frozen novelty boundary.

---

# 8. Novelty-Boundary Rule

Finding additional relevant literature does **not** automatically require changing the frozen research question.

The novelty boundary should be amended only if a newly identified study materially overlaps the full combination of:

```text
zebrafish
+
temporal representation learning
+
self-supervised learning
+
unsupervised behavioral discovery
+
direct handcrafted baseline comparison
+
held-out animals
+
identity/session/context controls
+
speed controls
+
tracking-artifact controls
+
stability analysis
+
independent replication
```

If a close match is found, document it immediately in:

```text
docs/decision-log.md
docs/novelty.md
docs/literature.md
```

before opening the final TEST partition.

---

# 9. File Governance

Recommended paths:

```text
references/citation-chain.csv
references/citation-chain-protocol.md
references/search-log.csv
references/papers.bib
references/paper-notes/
docs/literature/literature-matrix.csv
```

Do not delete excluded rows after screening.

If a decision changes later, update the row with an explanatory note and preserve the earlier state in Git history.

---

# 10. Current Completion State

After the first recorded citation-chain pass on 2026-08-23:

```yaml
core_seeds_defined: true
candidate_level_ledger_created: true
core_seed_backward_coverage: COMPLETE_FIRST_PASS
core_seed_forward_coverage: COMPLETE_FIRST_PASS
candidate_rows_screened: 37
included: 7
excluded: 20
duplicates: 10
uncertain: 0
included_records_synchronized: COMPLETE
search_log_summaries: COMPLETE_FIRST_PASS
second_pass_complete: false
```

Every core seed now has at least one backward and one forward check. This is a targeted first pass, not a claim that every citation in every index was exhaustively screened. Final saturation still requires synchronization of retained records and a dated second pass before manuscript submission.
