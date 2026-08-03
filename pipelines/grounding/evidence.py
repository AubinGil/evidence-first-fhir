"""Exact-span grounding gate. Untrusted model output never bypasses this."""

from __future__ import annotations

import re

from pipelines.contracts import CandidateFact, Evidence, document_hash

_TERM = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _TERM.findall((text or "").lower())


def unsupported_value_terms(value: str, quote: str) -> list[str]:
    """Terms asserted in a value that its own evidence quote does not contain.

    Matching is per term rather than by substring: a value claiming "type 2"
    against a quote reading "diagnosed 2018" would otherwise count the "2" as
    supported by the "2" inside "2018".
    """
    quote_terms = set(_terms(quote))
    return [term for term in _terms(value) if term not in quote_terms]


def quote_is_section_label(quote: str) -> bool:
    """A quote that terminates at a colon is a section label, not evidence.

    A label states where information would appear; it does not state that any
    was found. "Major Surgical or Invasive Procedure:" grounds perfectly -- it
    is a real span, and its terms appear in the fact naming it -- so nothing
    downstream rejects a section header promoted to a clinical fact.

    Deliberately narrow. Across the gold set no legitimate evidence quote ends
    with a colon, while bare terms that *are* their own evidence ("HTN",
    "productive cough") do not, so this does not disturb them. It does not
    address reaction-as-condition confusion, which is semantic.
    """
    return quote.strip().endswith(":")


def ground_candidate(*, document: str, subject: str, category: str, value: str, quote: str, confidence: float | None = None) -> CandidateFact | None:
    """Return a pending candidate only when the evidence is an exact document span.

    Verifying the quote alone is not sufficient. The quote is evidence; `value`
    is the assertion built into the resource downstream, and a model can cite a
    genuine span while asserting something that span does not say. Both must
    hold, or the candidate is not grounded.
    """
    if not quote or quote not in document:
        return None
    if quote_is_section_label(quote):
        return None
    if unsupported_value_terms(value, quote):
        return None
    start = document.index(quote)
    return CandidateFact(
        subject=subject,
        category=category,
        value=value,
        evidence=Evidence(document_hash(document), start, start + len(quote), quote),
        confidence=confidence,
    )
