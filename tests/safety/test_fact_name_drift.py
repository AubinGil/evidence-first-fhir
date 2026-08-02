"""`grounded_facts` builds resources from `name`, so `name` must be grounded too.

The quote was checked against the note and `name` was trusted unchecked, which
let a fact assert a term the source does not contain while citing a real span.
"""

from discharge_to_fhir import grounded_facts


def fact(**overrides: object) -> dict[str, object]:
    base = {
        "name": "essential hypertension",
        "category": "condition",
        "experiencer": "patient",
        "assertion": "present",
        "evidence_quote": "Essential hypertension",
    }
    base.update(overrides)
    return base


NOTE = "ACTIVE CONDITIONS Essential hypertension Type diabetes mellitus diagnosed 2018"


def test_faithful_fact_is_accepted() -> None:
    accepted, quarantined = grounded_facts(NOTE, {"facts": [fact()]})
    assert len(accepted) == 1
    assert not quarantined
    assert accepted[0]["evidence_start"] == NOTE.index("Essential hypertension")


def test_name_asserting_a_token_the_source_lacks_is_quarantined() -> None:
    """Observed 2026-08-02 on a real OCR run.

    OCR dropped the "2" from "Type 2 diabetes mellitus". The model restored it
    in `name` from clinical knowledge while quoting the corrupted span, so the
    resource asserted something the document did not contain. The correction
    was right that time; nothing about the mechanism guarantees the next one.
    """
    drifted = fact(
        name="type 2 diabetes mellitus",
        evidence_quote="Type diabetes mellitus diagnosed 2018",
    )
    accepted, quarantined = grounded_facts(NOTE, {"facts": [drifted]})
    assert not accepted
    assert len(quarantined) == 1
    assert any("absent from its evidence quote" in reason
               for reason in quarantined[0]["reasons"])


def test_name_unrelated_to_a_real_quote_is_quarantined() -> None:
    unrelated = fact(name="diabetes", evidence_quote="Essential hypertension")
    accepted, quarantined = grounded_facts(NOTE, {"facts": [unrelated]})
    assert not accepted
    assert quarantined


def test_transcription_error_is_not_this_gate_s_job() -> None:
    """A name faithful to a *corrupted* quote still passes, by design.

    OCR produced "Hemoglobin Alc" for "Hemoglobin A1c" and the model quoted it
    faithfully. Name and evidence agree, so this gate accepts. Detecting that
    the transcript itself is wrong needs transcription confidence or a second
    reader, not this check.
    """
    note = "LABORATORY HISTORY Hemoglobin Alc 8.2 % High"
    corrupted = fact(name="Hemoglobin Alc", evidence_quote="Hemoglobin Alc 8.2 %")
    accepted, _ = grounded_facts(note, {"facts": [corrupted]})
    assert len(accepted) == 1
