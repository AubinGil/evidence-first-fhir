"""A quote that is a real document span does not make the asserted value true.

The gate verified the quote and then trusted `value` unchecked, so a model
could cite a genuine span while asserting something the span does not say.
"""

from pipelines.grounding import ground_candidate, unsupported_value_terms


DOCUMENT = "SYNTHETIC: The patient reports a penicillin allergy."


def test_value_absent_from_its_own_quote_is_rejected() -> None:
    candidate = ground_candidate(
        document=DOCUMENT,
        subject="synthetic-001",
        category="condition",
        value="diabetes",
        quote="penicillin allergy",
    )
    assert candidate is None


def test_value_supported_by_its_quote_is_accepted() -> None:
    candidate = ground_candidate(
        document=DOCUMENT,
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin allergy",
    )
    assert candidate is not None
    assert candidate.value == "penicillin"


def test_model_restoring_a_token_the_source_lacks_is_rejected() -> None:
    """Observed 2026-08-02 on a real OCR run.

    OCR dropped the "2" from "Type 2 diabetes mellitus". The model restored it
    from clinical knowledge in the value while quoting the corrupted span, so
    the fact asserted something the document did not contain and the gate
    accepted it. Here the correction was right; nothing about the mechanism
    guarantees the next one will be.
    """
    document = "ACTIVE CONDITIONS Type diabetes mellitus diagnosed 2018"
    candidate = ground_candidate(
        document=document,
        subject="synthetic-001",
        category="condition",
        value="type 2 diabetes mellitus",
        quote="Type diabetes mellitus diagnosed 2018",
    )
    assert candidate is None
    assert unsupported_value_terms(
        "type 2 diabetes mellitus", "Type diabetes mellitus diagnosed 2018"
    ) == ["2"]


def test_digits_are_matched_as_terms_not_substrings() -> None:
    """"2" must not count as supported by the "2" inside "2018"."""
    assert unsupported_value_terms("type 2 diabetes", "Type diabetes diagnosed 2018") == ["2"]


def test_gate_does_not_claim_to_detect_transcription_error() -> None:
    """A value faithful to a *corrupted* quote still passes, by design.

    Observed on the same run: OCR produced "Hemoglobin Alc" for "Hemoglobin
    A1c" and the model quoted it faithfully. Value/evidence agreement is
    intact, so this gate accepts it. Detecting that the transcript itself is
    wrong requires transcription confidence or a second reader, not this check.
    """
    document = "LABORATORY HISTORY Hemoglobin Alc 8.2 % High"
    candidate = ground_candidate(
        document=document,
        subject="synthetic-001",
        category="observation",
        value="Hemoglobin Alc",
        quote="Hemoglobin Alc 8.2 %",
    )
    assert candidate is not None
