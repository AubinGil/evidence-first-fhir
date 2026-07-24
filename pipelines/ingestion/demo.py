"""Run a synthetic, no-network grounding demonstration."""

from pipelines.grounding import ground_candidate


def main() -> None:
    document = "SYNTHETIC: The patient reports a penicillin allergy."
    candidate = ground_candidate(
        document=document,
        subject="synthetic-patient-001",
        category="allergy",
        value="penicillin",
        quote="penicillin allergy",
        confidence=0.98,
    )
    print(candidate.as_dict() if candidate else "quarantined")


if __name__ == "__main__":
    main()
