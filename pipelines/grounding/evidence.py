"""Exact-span grounding gate. Untrusted model output never bypasses this."""

from __future__ import annotations

from pipelines.contracts import CandidateFact, Evidence, document_hash


def ground_candidate(*, document: str, subject: str, category: str, value: str, quote: str, confidence: float | None = None) -> CandidateFact | None:
    """Return a pending candidate only when the evidence is an exact document span."""
    if not quote or quote not in document:
        return None
    start = document.index(quote)
    return CandidateFact(
        subject=subject,
        category=category,
        value=value,
        evidence=Evidence(document_hash(document), start, start + len(quote), quote),
        confidence=confidence,
    )
