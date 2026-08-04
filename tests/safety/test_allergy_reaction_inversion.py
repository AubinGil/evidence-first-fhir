"""An allergy must be named for its substance, never for its reaction.

"Penicillin allergy causes a rash" supports one AllergyIntolerance, coded to
penicillin. Models also emit a second coded to the rash, and it passes the span
test and the value test honestly -- the quote is real and "rash" is in it. Only
the direction is wrong: the allergen precedes the causal connective and the
manifestation follows it.

The two cases below are the only forbidden facts that survived the gate across
every model measured on the gold set, quoted verbatim from their runs.
"""

from __future__ import annotations

import pytest

from discharge_to_fhir import grounded_facts
from pipelines.grounding import names_the_reaction_not_the_substance


def fact(**overrides):
    base = {"category": "allergy", "name": "rash", "assertion": "present",
            "temporality": "current", "clinical_status": "active",
            "experiencer": "patient", "date": None, "value": None, "unit": None,
            "dose": None, "route": None, "frequency": None,
            "evidence_quote": "which causes a rash"}
    return {**base, "facts": None, **overrides}


# --- the survivors, verbatim -------------------------------------------------

def test_qwen_rash_is_quarantined() -> None:
    note = "Penicillin allergy causes a rash. The patient tolerates cephalosporins."
    extraction = {"facts": [
        {"category": "allergy", "name": "rash", "assertion": "present",
         "experiencer": "patient", "evidence_quote": "causes a rash"},
    ]}
    accepted, quarantined = grounded_facts(note, extraction)
    assert accepted == []
    assert "reaction" in " ".join(quarantined[0]["reasons"])


def test_gemma_hives_is_quarantined() -> None:
    note = "Allergies: Sulfa drugs, which cause hives."
    extraction = {"facts": [
        {"category": "allergy", "name": "hives", "assertion": "present",
         "experiencer": "patient",
         "evidence_quote": "Sulfa drugs, which cause hives."},
    ]}
    accepted, quarantined = grounded_facts(note, extraction)
    assert accepted == []


# --- the substance in the same quote must still be accepted ------------------

def test_the_allergen_itself_is_untouched() -> None:
    """The rule must not reject the fact it exists to protect."""
    note = "Allergies: Sulfa drugs, which cause hives."
    extraction = {"facts": [
        {"category": "allergy", "name": "Sulfa drugs", "assertion": "present",
         "experiencer": "patient",
         "evidence_quote": "Sulfa drugs, which cause hives."},
    ]}
    accepted, _ = grounded_facts(note, extraction)
    assert [f["name"] for f in accepted] == ["Sulfa drugs"]


# --- the predicate, directly --------------------------------------------------

@pytest.mark.parametrize("quote,name", [
    ("which causes a rash", "rash"),
    ("Sulfa drugs, which cause hives.", "hives"),
    ("Penicillin, resulting in anaphylaxis", "anaphylaxis"),
    ("contrast dye, leading to urticaria", "urticaria"),
    ("latex exposure triggering bronchospasm", "bronchospasm"),
    ("Penicillin associated with rash", "rash"),
    # Bare stems. A literal list of "causes/caused/causing" misses this one,
    # which is a case measured off a real model rather than an invented one.
    ("Sulfa drugs, which cause hives.", "hives"),
    ("contrast dye, leading to urticaria", "urticaria"),
])
def test_names_drawn_from_after_the_connective_are_reactions(quote, name) -> None:
    assert names_the_reaction_not_the_substance("allergy", name, quote)


@pytest.mark.parametrize("quote,name", [
    ("Penicillin allergy causes a rash.", "Penicillin"),
    ("Sulfa drugs, which cause hives.", "Sulfa drugs"),
    ("Allergic to peanuts.", "peanuts"),           # no connective at all
    ("Shellfish causes shellfish reactions", "shellfish"),  # also appears before
])
def test_substances_are_left_alone(quote, name) -> None:
    assert not names_the_reaction_not_the_substance("allergy", name, quote)


def test_only_allergies_are_examined() -> None:
    """A condition caused by something is ordinary clinical prose, not inversion."""
    assert not names_the_reaction_not_the_substance(
        "condition", "pneumonia", "sepsis resulting in pneumonia")
    assert not names_the_reaction_not_the_substance(
        "observation", "hypokalemia", "diuresis causing hypokalemia")
