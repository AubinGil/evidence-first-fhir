"""A section label states where information would be, not that any was found.

"Major Surgical or Invasive Procedure:" passed every gate: it is a real span
of the document, and every term in the fact naming it appears in the quote. So
a heading was promoted to a clinical Procedure and nothing downstream objected.

The rule is deliberately narrow -- a quote that terminates at a colon -- because
the obvious wider rule is unsafe. Requiring evidence to add context beyond the
value would quarantine 14 of the gold set's 51 facts, since an abbreviated
history legitimately lists conditions as bare terms whose evidence is the term
itself ("HTN", "T2DM", "productive cough").
"""

from __future__ import annotations

from discharge_to_fhir import grounded_facts
from pipelines.grounding import ground_candidate, quote_is_section_label

NOTE = ("Major Surgical or Invasive Procedure:\n"
        "Past Medical History: HTN, T2DM\n"
        "Patient reports productive cough on admission.\n")


def fact(name: str, quote: str) -> dict[str, object]:
    return {"name": name, "category": "condition", "experiencer": "patient",
            "assertion": "present", "evidence_quote": quote}


def test_section_label_is_quarantined() -> None:
    accepted, quarantined = grounded_facts(
        NOTE, {"facts": [fact("Major Surgical or Invasive Procedure",
                              "Major Surgical or Invasive Procedure:")]})
    assert not accepted
    assert any("section label" in reason for reason in quarantined[0]["reasons"])


def test_bare_term_that_is_its_own_evidence_is_still_accepted() -> None:
    """The gold set contains 14 of these; the rule must not disturb them."""
    accepted, quarantined = grounded_facts(NOTE, {"facts": [fact("HTN", "HTN")]})
    assert len(accepted) == 1
    assert not quarantined


def test_prose_evidence_is_unaffected() -> None:
    accepted, _ = grounded_facts(
        NOTE, {"facts": [fact("productive cough",
                              "Patient reports productive cough on admission.")]})
    assert len(accepted) == 1


def test_candidate_gate_rejects_section_labels_too() -> None:
    assert ground_candidate(document=NOTE, subject="synthetic-001",
                            category="procedure",
                            value="Major Surgical or Invasive Procedure",
                            quote="Major Surgical or Invasive Procedure:") is None


def test_rule_recognises_only_a_terminating_colon() -> None:
    assert quote_is_section_label("Allergies:")
    assert quote_is_section_label("  Discharge Diagnoses:  ")
    assert not quote_is_section_label("Allergies: penicillin")
    assert not quote_is_section_label("HTN")


def test_reaction_as_condition_is_not_covered_by_this_rule() -> None:
    """Stated so the gap is visible rather than assumed closed.

    "Sulfa drugs, which cause hives." is prose, not a label, so a fact naming
    "hives" as a condition still passes. Separating a reaction from a condition
    is semantic, and model selection remains the only control for it -- which
    is why forbidden_hits matters more than F1 when choosing an extractor.
    """
    note = "Allergies: Sulfa drugs, which cause hives.\n"
    accepted, _ = grounded_facts(
        note, {"facts": [fact("hives", "Sulfa drugs, which cause hives.")]})
    assert len(accepted) == 1
