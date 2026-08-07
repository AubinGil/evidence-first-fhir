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
    "productive cough") do not, so this does not disturb them.
    """
    return quote.strip().endswith(":")


# Connectives after which a quote names an effect rather than its cause.
# Stems rather than literal forms: a list spelling out "causes", "caused" and
# "causing" misses "which cause hives", which is one of the two cases this
# exists to catch. Same for "leading to" beside "leads to" and "led to".
# "associated with" is weaker than the rest -- it states co-occurrence, not
# causation -- but in an allergy quote it introduces the manifestation just as
# reliably, and it is the phrasing a discharge summary most often uses.
_CAUSAL = re.compile(
    r"\b(?:caus\w*|result\w*\s+in|lead\w*\s+to|led\s+to|produc\w*|trigger\w*|"
    r"manifest\w*\s+as|present\w*\s+with|develop\w*|associated\s+with)\b",
    re.IGNORECASE,
)
_ALLERGY_CATEGORIES = {"allergy", "allergyintolerance", "allergy_intolerance"}


def names_the_reaction_not_the_substance(category: str, value: str, quote: str) -> bool:
    """An allergy whose name is its manifestation rather than its allergen.

    "Penicillin allergy causes a rash" supports one AllergyIntolerance, coded
    to penicillin, whose reaction.manifestation is the rash. Models routinely
    emit a second one coded to the rash itself, and it survives every other
    check here honestly: the quote is a real span, and "rash" genuinely appears
    in it, so both the span test and the value test pass.

    What is wrong is not the words but the direction. The allergen precedes the
    causal connective and the manifestation follows it, so a name drawn only
    from after the connective is the effect being recorded as the cause. That
    is decidable from the quote alone, without a model and without a
    terminology server.

    The consequence is why it is worth a rule of its own: a reviewer scanning an
    allergy list sees "rash" where "Penicillin" should be, and the substance
    that must never be prescribed again may appear nowhere. Both surviving
    forbidden facts across every model measured were this -- qwen3.6:27b
    emitting "rash", Gemma-4-26B emitting "hives" -- and nothing downstream
    stopped either.

    Conservative by construction. With no connective, or with the name also
    appearing before one, the quote gives no evidence of inversion and the fact
    is left alone.
    """
    if category.strip().casefold() not in _ALLERGY_CATEGORIES:
        return False
    causal = _CAUSAL.search(quote or "")
    if not causal:
        return False
    terms = _terms(value)
    if not terms:
        return False
    lowered = (quote or "").lower()
    earliest = min(
        (position.start() for position in
         (re.search(rf"\b{re.escape(term)}", lowered) for term in terms)
         if position is not None),
        default=None,
    )
    return earliest is not None and earliest >= causal.end()


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
    if names_the_reaction_not_the_substance(category, value, quote):
        return None
    start = document.index(quote)
    return CandidateFact(
        subject=subject,
        category=category,
        value=value,
        evidence=Evidence(document_hash(document), start, start + len(quote), quote),
        confidence=confidence,
    )
