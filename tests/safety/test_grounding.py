from pipelines.grounding import ground_candidate


def test_unsupported_evidence_is_quarantined() -> None:
    candidate = ground_candidate(
        document="SYNTHETIC: No known allergies.",
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin allergy",
    )
    assert candidate is None


def test_evidence_has_exact_offsets() -> None:
    document = "SYNTHETIC: Allergy to penicillin documented."
    candidate = ground_candidate(
        document=document,
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin",
    )
    assert candidate is not None
    assert document[candidate.evidence.start:candidate.evidence.end] == "penicillin"
