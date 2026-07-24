# Clinical extraction gold set

Frozen fixtures for comparing extraction models (MedGemma 27B vs Qwen3.6 and any
later candidate) on the metrics named in [`docs/architecture.md`](../docs/architecture.md). Every
fixture is **synthetic** — no MIMIC content, no PHI — so the set can live in Git
and run in CI. Scoring never calls a model; inference happens once per model,
its outputs are stored, and all scoring runs on the stored files.

Status: `review_status: draft` on every fixture. Per the project ground rules
these annotations are not clinical validation until a clinician has reviewed
them; the field flips to `clinician-reviewed` only after that happens.

## Layout

- `fixtures/*.json` — one case per file; filename equals the fixture `id`.
- `VERSION` — gold set version. Reports embed this plus a SHA-256 checksum of
  all fixture files, so a comparison run is verifiably tied to a frozen set.

## Workflow

~~~text
# once per model, requires local Ollama (skip while inference is unavailable)
python goldset_predict.py --model medgemma:27b
python goldset_predict.py --model qwen3.6:35b

# any time, no inference, CI-safe
python goldset_score.py output/goldset/medgemma-27b --report output/goldset/medgemma-27b/report.json
python goldset_score.py output/goldset/qwen3.6-35b --report output/goldset/qwen3.6-35b/report.json
~~~

Predictions are one JSON file per fixture id in the `FACT_SCHEMA` shape from
`baseline_extraction_test.py`. `test_goldset.py` lints the fixtures and
self-tests the scorer with synthetic predictions; it runs in CI with no model.

## Fixture format

Top level: `id`, `axis`, `description`, `review_status`, `note`, `patient`,
`encounter`, `facts`, optional `forbidden`. Each gold fact carries every
`FACT_SCHEMA` field plus:

- `match_terms` — lowercase substrings that must all appear in a predicted
  fact's `name` for a name-based match.
- `exclude_terms` (optional) — substrings that disqualify a name-based match
  (e.g. plain `hemoglobin` must not swallow `hemoglobin A1c`).

`forbidden` entries name hallucination traps: predicted facts that match no
gold fact and contain the trap terms are flagged with the stated reason
(section headers, reaction descriptions extracted as conditions, …).

## Matching and metrics

Predicted facts are matched one-to-one to gold facts in three passes: exact
`evidence_quote` equality, then `match_terms` on the name, then evidence-span
overlap. Verbatim quotes are the strongest signal, so a model may expand
abbreviations in `name` (T2DM → type 2 diabetes mellitus) without losing the
match.

Reported metrics: schema-valid rate, entity precision/recall/F1, enum-attribute
accuracy (category, assertion, temporality, clinical_status, experiencer),
detail accuracy (date, value, unit, dose, route, frequency), demographics
accuracy, exact and overlapping evidence-span rates, unsupported-fact rate
(quote is not a verbatim substring of the note), spurious-fact rate, and
forbidden-trap hits.

## Annotation conventions

- Gold nulls are authoritative: if the note does not state a birth date,
  gender, or dose, the gold value is `null` and a non-null prediction is wrong
  (inferring gender from a name is an error we deliberately test).
- Negated or denied items are `assertion: absent` with `clinical_status:
  unknown`.
- Return precautions and if-then therapy are `assertion: conditional`.
- Family history is `experiencer: family`.
- One fact per clinical entity: a medication mentioned in the hospital course
  and again in the discharge list is one fact, anchored to the discharge line;
  duplicates count as spurious.
- `unit` comparison treats `percent` and `%` as equal; values compare
  numerically with tolerance.

## Freeze policy

Once a model comparison report has been recorded against a version, fixtures in
that version are frozen. Fixing an annotation error or adding cases means
bumping `VERSION`; never silently edit a fixture that has scored runs behind
it. Contested annotations (rationale in each fixture's `description`) should be
settled during clinician review, as a version bump.
