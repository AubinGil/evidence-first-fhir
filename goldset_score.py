"""Score gold-set extraction predictions deterministically, with no inference.

Predictions are one JSON file per fixture id, in the extraction schema emitted
by ``baseline_extraction_test.call_ollama`` (FACT_SCHEMA). Matching and scoring
are pure functions of the stored files, so this runs on cached outputs and in
CI without a model, GPU, or network access.

Usage:
    python goldset_score.py output/goldset/medgemma-27b \
        --report output/goldset/medgemma-27b/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from baseline_extraction_test import FACT_SCHEMA

FIXTURES_DIR = Path(__file__).resolve().parent / "goldset" / "fixtures"
VERSION_FILE = Path(__file__).resolve().parent / "goldset" / "VERSION"

ENUM_ATTRIBUTES = ("category", "assertion", "temporality", "clinical_status", "experiencer")
DETAIL_ATTRIBUTES = ("date", "value", "unit", "dose", "route", "frequency")
GOLD_ONLY_KEYS = {"match_terms", "exclude_terms"}
UNIT_ALIASES = {"percent": "%", "pct": "%"}

TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def schema_errors(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Minimal structural validator covering the subset of JSON Schema used by FACT_SCHEMA."""
    errors: list[str] = []
    kinds = schema.get("type")
    if kinds:
        allowed = kinds if isinstance(kinds, list) else [kinds]
        matched = False
        for kind in allowed:
            if kind in ("number", "integer") and isinstance(instance, bool):
                continue
            if isinstance(instance, TYPE_CHECKS[kind]):
                matched = True
                break
        if not matched:
            errors.append(f"{path}: expected {allowed}, got {type(instance).__name__}")
            return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, str) and len(instance) < schema.get("minLength", 0):
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, subschema in properties.items():
            if key in instance:
                errors.extend(schema_errors(instance[key], subschema, f"{path}.{key}"))
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
    return errors


def find_span(note: str, quote: Any) -> tuple[int, int] | None:
    """First occurrence of an exact substring; gold quotes are lint-checked unique."""
    if not isinstance(quote, str) or not quote:
        return None
    index = note.find(quote)
    return None if index < 0 else (index, index + len(quote))


def name_matches(fact: dict[str, Any], gold: dict[str, Any]) -> bool:
    name = str(fact.get("name", "")).casefold()
    if not all(term.casefold() in name for term in gold.get("match_terms", [])):
        return False
    return not any(term.casefold() in name for term in gold.get("exclude_terms", []))


def match_facts(
    note: str, gold_facts: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> tuple[dict[int, int], list[int]]:
    """One-to-one greedy matching: exact evidence quote, then name terms, then span overlap.

    Exact quotes are the strongest signal (the schema demands verbatim quotes, so
    abbreviation expansion in ``name`` cannot break them). Span overlap is the
    last resort for partial quotes.
    """
    pairs: dict[int, int] = {}
    used: set[int] = set()
    gold_spans = [find_span(note, gold["evidence_quote"]) for gold in gold_facts]
    predicted_spans = [find_span(note, fact.get("evidence_quote")) for fact in predicted]

    def claim(gold_index: int, condition) -> None:
        if gold_index in pairs:
            return
        for predicted_index, fact in enumerate(predicted):
            if predicted_index not in used and condition(predicted_index, fact):
                pairs[gold_index] = predicted_index
                used.add(predicted_index)
                return

    for gold_index, gold in enumerate(gold_facts):
        claim(gold_index, lambda _pi, fact: fact.get("evidence_quote") == gold["evidence_quote"])
    for gold_index, gold in enumerate(gold_facts):
        claim(gold_index, lambda _pi, fact: name_matches(fact, gold))
    for gold_index, gold in enumerate(gold_facts):
        span = gold_spans[gold_index]
        if span is None:
            continue
        claim(
            gold_index,
            lambda pi, _fact: predicted_spans[pi] is not None
            and predicted_spans[pi][0] < span[1]
            and span[0] < predicted_spans[pi][1],
        )
    spurious = [index for index in range(len(predicted)) if index not in used]
    return pairs, spurious


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def normalized_unit(value: Any) -> str:
    text = str(value).strip().casefold()
    return UNIT_ALIASES.get(text, text)


def detail_equal(field: str, actual: Any, expected: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if field == "value":
        expected_number = numeric(expected)
        actual_number = numeric(actual)
        if expected_number is not None and actual_number is not None:
            return abs(actual_number - expected_number) < 1e-4
        return str(actual).strip().casefold() == str(expected).strip().casefold()
    if field == "unit":
        return normalized_unit(actual) == normalized_unit(expected)
    if field == "date":
        return str(actual).strip() == str(expected).strip()
    actual_text = str(actual).strip().casefold()
    expected_text = str(expected).strip().casefold()
    return actual_text == expected_text or actual_text in expected_text or expected_text in actual_text


def demographics_equal(actual: Any, expected: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    return str(actual).strip().casefold() == str(expected).strip().casefold()


def score_case(fixture: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    note = fixture["note"]
    gold_facts = fixture["facts"]
    validation = schema_errors(prediction, FACT_SCHEMA)
    predicted = prediction.get("facts")
    if not isinstance(predicted, list):
        predicted = []
    predicted = [fact for fact in predicted if isinstance(fact, dict)]

    pairs, spurious_indexes = match_facts(note, gold_facts, predicted)
    messages: list[str] = []

    enum_correct = enum_total = 0
    detail_correct = detail_total = 0
    for gold_index, predicted_index in pairs.items():
        gold = gold_facts[gold_index]
        fact = predicted[predicted_index]
        for field in ENUM_ATTRIBUTES:
            enum_total += 1
            actual, expected = fact.get(field), gold.get(field)
            if str(actual).casefold() == str(expected).casefold():
                enum_correct += 1
            else:
                messages.append(f"WRONG {gold.get('name')!r}.{field}: expected {expected!r}, got {actual!r}")
        for field in DETAIL_ATTRIBUTES:
            detail_total += 1
            actual, expected = fact.get(field), gold.get(field)
            if detail_equal(field, actual, expected):
                detail_correct += 1
            else:
                messages.append(f"WRONG {gold.get('name')!r}.{field}: expected {expected!r}, got {actual!r}")

    evidence_exact = evidence_overlap = 0
    for gold_index, predicted_index in pairs.items():
        gold = gold_facts[gold_index]
        quote = predicted[predicted_index].get("evidence_quote")
        if quote == gold["evidence_quote"]:
            evidence_exact += 1
            evidence_overlap += 1
            continue
        gold_span = find_span(note, gold["evidence_quote"])
        span = find_span(note, quote)
        if gold_span and span and span[0] < gold_span[1] and gold_span[0] < span[1]:
            evidence_overlap += 1
        else:
            messages.append(f"EVIDENCE {gold.get('name')!r}: no overlap with gold span, got {quote!r}")

    grounded = sum(1 for fact in predicted if find_span(note, fact.get("evidence_quote")) is not None)
    unsupported = len(predicted) - grounded

    missed = [gold_facts[index].get("name") for index in range(len(gold_facts)) if index not in pairs]
    for name in missed:
        messages.append(f"MISS gold fact {name!r}")

    spurious_names = []
    forbidden_hits = []
    for index in spurious_indexes:
        fact = predicted[index]
        name = str(fact.get("name"))
        spurious_names.append(name)
        messages.append(f"SPURIOUS predicted fact {name!r}")
        for trap in fixture.get("forbidden", []):
            if all(term.casefold() in name.casefold() for term in trap["match_terms"]):
                forbidden_hits.append({"name": name, "reason": trap["reason"]})
                messages.append(f"FORBIDDEN {name!r}: {trap['reason']}")
                break

    demographics_correct = demographics_total = 0
    for section in ("patient", "encounter"):
        expected_section = fixture.get(section, {})
        actual_section = prediction.get(section)
        if not isinstance(actual_section, dict):
            actual_section = {}
        for field, expected in expected_section.items():
            demographics_total += 1
            if demographics_equal(actual_section.get(field), expected):
                demographics_correct += 1
            else:
                messages.append(
                    f"WRONG {section}.{field}: expected {expected!r}, got {actual_section.get(field)!r}"
                )

    return {
        "id": fixture["id"],
        "schema_valid": not validation,
        "schema_errors": validation[:10],
        "gold_facts": len(gold_facts),
        "predicted_facts": len(predicted),
        "matched": len(pairs),
        "missed": missed,
        "spurious": spurious_names,
        "forbidden_hits": forbidden_hits,
        "enum_correct": enum_correct,
        "enum_total": enum_total,
        "detail_correct": detail_correct,
        "detail_total": detail_total,
        "demographics_correct": demographics_correct,
        "demographics_total": demographics_total,
        "evidence_exact": evidence_exact,
        "evidence_overlap": evidence_overlap,
        "unsupported": unsupported,
        "messages": messages,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = lambda key: sum(case[key] for case in cases)  # noqa: E731
    gold = total("gold_facts")
    predicted = total("predicted_facts")
    matched = total("matched")
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases_scored": len(cases),
        "schema_valid_rate": ratio(sum(case["schema_valid"] for case in cases), len(cases)),
        "gold_facts": gold,
        "predicted_facts": predicted,
        "matched": matched,
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
        "enum_attribute_accuracy": ratio(total("enum_correct"), total("enum_total")),
        "detail_accuracy": ratio(total("detail_correct"), total("detail_total")),
        "demographics_accuracy": ratio(total("demographics_correct"), total("demographics_total")),
        "evidence_exact_rate": ratio(total("evidence_exact"), matched),
        "evidence_overlap_rate": ratio(total("evidence_overlap"), matched),
        "unsupported_fact_rate": ratio(total("unsupported"), predicted),
        "spurious_fact_rate": ratio(sum(len(case["spurious"]) for case in cases), predicted),
        "forbidden_hits": sum(len(case["forbidden_hits"]) for case in cases),
    }


def load_fixtures(fixtures_dir: Path) -> list[dict[str, Any]]:
    fixtures = []
    for path in sorted(fixtures_dir.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        fixture["_file"] = path.name
        fixtures.append(fixture)
    return fixtures


def goldset_checksum(fixtures_dir: Path) -> str:
    """Line endings are normalized so git autocrlf cannot change the checksum."""
    digest = hashlib.sha256()
    for path in sorted(fixtures_dir.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def goldset_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_dir", type=Path)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--report", type=Path, help="Optional path for the JSON report")
    args = parser.parse_args()

    if not args.predictions_dir.is_dir():
        print(f"Predictions directory not found: {args.predictions_dir}")
        return 2
    fixtures = load_fixtures(args.fixtures)
    if not fixtures:
        print(f"No fixtures found in {args.fixtures}")
        return 2

    manifest = None
    manifest_path = args.predictions_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    for fixture in fixtures:
        prediction_path = args.predictions_dir / f"{fixture['id']}.json"
        if not prediction_path.is_file():
            missing.append(fixture["id"])
            continue
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            prediction = {}
            print(f"[{fixture['id']}] unreadable prediction: {error}")
        if not isinstance(prediction, dict):
            prediction = {}
        cases.append(score_case(fixture, prediction))

    print(f"Gold set version {goldset_version()} ({len(fixtures)} fixtures), predictions: {args.predictions_dir}")
    if manifest and manifest.get("model"):
        print(f"Model: {manifest['model']}")
    print()
    header = f"{'case':<36} {'gold':>4} {'pred':>4} {'match':>5} {'spur':>4} {'enum':>9} {'exact-ev':>8}"
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['id']:<36} {case['gold_facts']:>4} {case['predicted_facts']:>4} "
            f"{case['matched']:>5} {len(case['spurious']):>4} "
            f"{case['enum_correct']}/{case['enum_total']:<8} {case['evidence_exact']}/{case['matched']}"
        )
        for message in case["messages"]:
            print(f"  - {message}")
    aggregate = aggregate_cases(cases)
    print()
    for key, value in aggregate.items():
        print(f"{key}: {value}")
    if missing:
        print(f"\nMISSING predictions for {len(missing)} fixture(s): {', '.join(missing)}")

    if args.report:
        report = {
            "goldset_version": goldset_version(),
            "goldset_sha256": goldset_checksum(args.fixtures),
            "fixtures_dir": str(args.fixtures),
            "predictions_dir": str(args.predictions_dir),
            "manifest": manifest,
            "missing": missing,
            "cases": cases,
            "aggregate": aggregate,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport: {args.report}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
