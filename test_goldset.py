"""Fixture lint and scorer self-tests for the gold set. No inference, no network."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from baseline_extraction_test import FACT_SCHEMA
from goldset_score import (
    FIXTURES_DIR,
    GOLD_ONLY_KEYS,
    aggregate_cases,
    goldset_checksum,
    load_fixtures,
    name_matches,
    schema_errors,
    score_case,
)

FACT_FIELDS = set(FACT_SCHEMA["properties"]["facts"]["items"]["required"])
FIXTURE_KEYS = {"id", "axis", "description", "review_status", "note", "patient", "encounter", "facts"}


def perfect_prediction(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient": dict(fixture["patient"]),
        "encounter": dict(fixture["encounter"]),
        "facts": [
            {key: value for key, value in fact.items() if key not in GOLD_ONLY_KEYS}
            for fact in fixture["facts"]
        ],
    }


class FixtureLintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = load_fixtures(FIXTURES_DIR)

    def test_fixtures_exist(self) -> None:
        self.assertGreaterEqual(len(self.fixtures), 12)

    def test_ids_match_filenames_and_are_unique(self) -> None:
        seen = set()
        for fixture in self.fixtures:
            self.assertEqual(f"{fixture['id']}.json", fixture["_file"])
            self.assertNotIn(fixture["id"], seen)
            seen.add(fixture["id"])

    def test_required_top_level_keys(self) -> None:
        for fixture in self.fixtures:
            missing = FIXTURE_KEYS - set(fixture)
            self.assertFalse(missing, f"{fixture['id']}: missing {missing}")
            self.assertTrue(fixture["note"].strip())
            self.assertEqual(set(fixture["patient"]), {"full_name", "birth_date", "gender"})
            self.assertEqual(set(fixture["encounter"]), {"admit_date", "discharge_date"})

    def test_gold_facts_carry_exactly_the_schema_fields(self) -> None:
        for fixture in self.fixtures:
            for fact in fixture["facts"]:
                extra = set(fact) - FACT_FIELDS - GOLD_ONLY_KEYS
                missing = FACT_FIELDS - set(fact)
                self.assertFalse(extra, f"{fixture['id']}/{fact.get('name')}: extra {extra}")
                self.assertFalse(missing, f"{fixture['id']}/{fact.get('name')}: missing {missing}")

    def test_perfect_prediction_is_schema_valid(self) -> None:
        for fixture in self.fixtures:
            errors = schema_errors(perfect_prediction(fixture), FACT_SCHEMA)
            self.assertFalse(errors, f"{fixture['id']}: {errors}")

    def test_evidence_quotes_are_unique_exact_substrings(self) -> None:
        for fixture in self.fixtures:
            note = fixture["note"]
            for fact in fixture["facts"]:
                quote = fact["evidence_quote"]
                count = note.count(quote)
                self.assertEqual(
                    count, 1,
                    f"{fixture['id']}/{fact['name']}: quote occurs {count} times: {quote!r}",
                )

    def test_match_terms_hit_their_own_gold_name(self) -> None:
        for fixture in self.fixtures:
            for fact in fixture["facts"]:
                self.assertTrue(
                    name_matches(fact, fact),
                    f"{fixture['id']}/{fact['name']}: match/exclude terms reject the gold name itself",
                )

    def test_forbidden_entries_are_well_formed(self) -> None:
        for fixture in self.fixtures:
            for trap in fixture.get("forbidden", []):
                self.assertEqual(set(trap), {"match_terms", "reason"})
                self.assertTrue(trap["match_terms"])


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = load_fixtures(FIXTURES_DIR)
        cls.by_id = {fixture["id"]: fixture for fixture in cls.fixtures}

    def case(self, fixture_id: str, mutate=None) -> dict[str, Any]:
        fixture = self.by_id[fixture_id]
        prediction = perfect_prediction(fixture)
        if mutate:
            mutate(prediction)
        return score_case(fixture, prediction)

    def test_perfect_predictions_score_clean_on_every_fixture(self) -> None:
        cases = [score_case(fixture, perfect_prediction(fixture)) for fixture in self.fixtures]
        for result in cases:
            self.assertTrue(result["schema_valid"], result["id"])
            self.assertEqual(result["matched"], result["gold_facts"], result)
            self.assertFalse(result["spurious"], result)
            self.assertFalse(result["missed"], result)
            self.assertEqual(result["enum_correct"], result["enum_total"], result)
            self.assertEqual(result["detail_correct"], result["detail_total"], result)
            self.assertEqual(result["demographics_correct"], result["demographics_total"], result)
            self.assertEqual(result["evidence_exact"], result["matched"], result)
            self.assertEqual(result["unsupported"], 0, result)
            self.assertFalse(result["forbidden_hits"], result)
        aggregate = aggregate_cases(cases)
        self.assertEqual(aggregate["entity_precision"], 1.0)
        self.assertEqual(aggregate["entity_recall"], 1.0)
        self.assertEqual(aggregate["schema_valid_rate"], 1.0)
        self.assertEqual(aggregate["evidence_exact_rate"], 1.0)
        self.assertEqual(aggregate["unsupported_fact_rate"], 0.0)
        self.assertEqual(aggregate["forbidden_hits"], 0)

    def test_dropped_fact_reduces_recall_only(self) -> None:
        result = self.case("negation-basic", lambda p: p["facts"].pop())
        self.assertEqual(result["matched"], result["gold_facts"] - 1)
        self.assertEqual(len(result["missed"]), 1)
        self.assertFalse(result["spurious"])

    def test_flipped_assertion_counts_one_enum_error(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            prediction["facts"][2]["assertion"] = "present"

        result = self.case("negation-basic", mutate)
        self.assertEqual(result["enum_correct"], result["enum_total"] - 1)
        self.assertIn("WRONG 'deep vein thrombosis'.assertion: expected 'absent', got 'present'", result["messages"])

    def test_fabricated_fact_is_unsupported_and_spurious(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            fact = copy.deepcopy(prediction["facts"][0])
            fact["name"] = "fabricated syndrome"
            fact["evidence_quote"] = "this text does not appear in the note"
            prediction["facts"].append(fact)

        result = self.case("negation-basic", mutate)
        self.assertEqual(result["unsupported"], 1)
        self.assertEqual(result["spurious"], ["fabricated syndrome"])

    def test_partial_quote_is_matched_but_not_exact(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            prediction["facts"][1]["evidence_quote"] = "type 2 diabetes"

        result = self.case("negation-basic", mutate)
        self.assertEqual(result["matched"], result["gold_facts"])
        self.assertEqual(result["evidence_exact"], result["matched"] - 1)
        self.assertEqual(result["evidence_overlap"], result["matched"])

    def test_duplicate_prediction_is_spurious(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            prediction["facts"].append(copy.deepcopy(prediction["facts"][0]))

        result = self.case("negation-basic", mutate)
        self.assertEqual(result["matched"], result["gold_facts"])
        self.assertEqual(len(result["spurious"]), 1)

    def test_expanded_abbreviation_still_matches_by_quote(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            prediction["facts"][1]["name"] = "type 2 diabetes mellitus"

        fixture = self.by_id["abbreviated-history"]
        prediction = perfect_prediction(fixture)
        mutate(prediction)
        result = score_case(fixture, prediction)
        self.assertEqual(result["matched"], result["gold_facts"])
        self.assertFalse(result["spurious"])

    def test_wrong_demographics_and_inferred_gender_are_penalized(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            prediction["patient"]["gender"] = "male"

        result = self.case("demographics-and-dates", mutate)
        self.assertEqual(result["demographics_correct"], result["demographics_total"] - 1)

    def test_section_header_extraction_hits_forbidden_trap(self) -> None:
        def mutate(prediction: dict[str, Any]) -> None:
            fact = copy.deepcopy(prediction["facts"][0])
            fact["name"] = "Major Surgical or Invasive Procedure"
            fact["category"] = "procedure"
            fact["evidence_quote"] = "Major Surgical or Invasive Procedure"
            prediction["facts"].append(fact)

        result = self.case("section-header-trap", mutate)
        self.assertEqual(len(result["forbidden_hits"]), 1)
        self.assertEqual(result["forbidden_hits"][0]["reason"], "section header, not a clinical fact")

    def test_schema_validator_flags_missing_key_and_bad_enum(self) -> None:
        fixture = self.by_id["negation-basic"]
        prediction = perfect_prediction(fixture)
        del prediction["facts"][0]["assertion"]
        prediction["facts"][1]["category"] = "diagnosis"
        result = score_case(fixture, prediction)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("missing required key 'assertion'" in e for e in result["schema_errors"]))
        self.assertTrue(any("'diagnosis' not in" in e for e in result["schema_errors"]))

    def test_missing_facts_list_scores_zero_matches(self) -> None:
        fixture = self.by_id["negation-basic"]
        result = score_case(fixture, {})
        self.assertFalse(result["schema_valid"])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["predicted_facts"], 0)
        self.assertEqual(len(result["missed"]), result["gold_facts"])

    def test_checksum_is_deterministic(self) -> None:
        self.assertEqual(goldset_checksum(FIXTURES_DIR), goldset_checksum(FIXTURES_DIR))


if __name__ == "__main__":
    unittest.main()
