# De-identification

Not implemented in this repository. This directory documents the required interface and
its validation burden. The synthetic demo needs no de-identification, and no stage here
performs one.

De-identification is specified in two phases because phase 1 alone does not make a
document safe to release.

## Phase 1 — Direct identifier removal

Deterministic patterns plus a clinical NER model detect and replace direct identifiers.
This phase is measurable and must be measured before use: report entity-level recall per
identifier type against a held-out annotated set, and treat recall — not F1 — as the
release gate. A false negative is a disclosure; a false positive is a readability cost.

Phase 1 must preserve character offsets or emit an offset map. Grounding downstream
depends on exact spans, and a scrubber that silently reflows text breaks provenance.

## Phase 2 — Residual re-identification risk

Removing direct identifiers leaves quasi-identifiers — age, sex, dates, rare diagnoses,
geography, admission timing — that re-identify in combination. Phase 2 generalizes those
into a quasi-identifier key and assesses residual risk across the release cohort
(k-anonymity as a floor, not a proof).

This phase is a property of the corpus, not of the note. It cannot be evaluated one
document at a time, which is why it belongs to orchestration rather than to this
pipeline.

## Release authorization

Neither phase authorizes release. Both are inputs to a documented determination that is
jurisdiction-specific and human-owned. Passing phase 1 and phase 2 means a release
decision can be made, not that it has been.

Implement here only after validating recall, residual risk, and jurisdiction-specific
requirements. De-identification does not by itself authorize public release or provider
transmission.
