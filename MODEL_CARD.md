# Model and pipeline card

## Intended use

This system creates candidate, evidence-backed structured facts and proposed FHIR R4 resources from synthetic clinical-style documents for research, evaluation, and human-review workflow demonstrations.

## Not intended for use

It must not be used for diagnosis, treatment, patient triage, autonomous clinical action, billing, population-health decisions, or unreviewed EHR writes.

## Limitations

Extraction models can hallucinate, omit context, confuse negation or temporality, match the wrong patient, and map concepts incorrectly. Exact text grounding helps detect unsupported assertions but does not establish clinical truth. Terminology coverage and FHIR validity do not establish clinical correctness.
