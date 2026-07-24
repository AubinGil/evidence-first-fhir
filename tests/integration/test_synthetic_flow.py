from dataclasses import replace

from pipelines.fhir_mapping import candidate_to_observation
from pipelines.grounding import ground_candidate


def test_approved_synthetic_candidate_becomes_a_proposal() -> None:
    candidate = ground_candidate(
        document="SYNTHETIC: The patient reports a penicillin allergy.",
        subject="synthetic-001",
        category="allergy",
        value="penicillin",
        quote="penicillin allergy",
    )
    assert candidate is not None
    proposal = candidate_to_observation(replace(candidate, review_status="approved"))
    assert proposal["resourceType"] == "Observation"
    assert proposal["status"] == "preliminary"
