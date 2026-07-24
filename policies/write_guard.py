"""Deliberately conservative persistence authorization."""

from __future__ import annotations

import os

from pipelines.contracts import CandidateFact


def persistence_allowed(fact: CandidateFact) -> bool:
    """Writes require both a reviewed fact and an explicit environment opt-in."""
    return (
        fact.review_status == "approved"
        and os.getenv("CLINICAL_FHIR_ALLOW_WRITES", "false").lower() == "true"
        and os.getenv("CLINICAL_FHIR_FHIR_READ_ONLY", "true").lower() == "false"
    )
