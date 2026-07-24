from dataclasses import replace

from pipelines.grounding import ground_candidate
from policies.write_guard import persistence_allowed


def test_writes_are_disabled_by_default(monkeypatch) -> None:
    candidate = ground_candidate(
        document="SYNTHETIC: Allergy to penicillin.",
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin",
    )
    assert candidate is not None
    monkeypatch.delenv("CLINICAL_FHIR_ALLOW_WRITES", raising=False)
    assert not persistence_allowed(replace(candidate, review_status="approved"))


def test_pending_fact_cannot_be_mapped() -> None:
    from pipelines.fhir_mapping import candidate_to_observation

    candidate = ground_candidate(
        document="SYNTHETIC: Allergy to penicillin.",
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin",
    )
    assert candidate is not None
    try:
        candidate_to_observation(candidate)
    except PermissionError:
        pass
    else:
        raise AssertionError("pending candidates must not be mapped")
