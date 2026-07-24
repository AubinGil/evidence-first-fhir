# Threat model

Clinical documents, patient identity, extracted facts, terminology mappings, review decisions, audit records, and FHIR endpoints are sensitive assets. Documents and model/provider output are untrusted.

| Threat | Default control | Residual risk |
|---|---|---|
| Prompt injection in documents | Data-only document handling; fixed pipeline policy | Misleading candidate facts |
| PHI leakage | Synthetic fixtures; telemetry off; body-free audit logs | Deployer misconfiguration |
| Unsupported evidence | Exact document-span provenance required; otherwise quarantine | Exact quotes can be clinically misleading |
| Incorrect patient matching | Explicit identity match; ambiguity fails closed | Bad upstream identifiers |
| Duplicate FHIR writes | Writes off; idempotency and conditional create when enabled | Misconfigured server behavior |
| Terminology error | Versioned sources; validation; reviewer visibility | Valid but inappropriate codes |
| Automation bias | Candidate labels, source display, and explicit approval | Reviewer over-trust |

## Security invariants

1. A document may not configure tools, policy, identity, or persistence.
2. No candidate fact is eligible for mapping without exact source provenance.
3. Ambiguous patient identity prevents mapping and persistence.
4. Persistence requires both an approved review decision and explicit deployment-time write enablement.
5. Audit records are append-only and omit document bodies by default.
