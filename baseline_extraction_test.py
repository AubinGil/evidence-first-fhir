"""Adversarial baseline test for discharge-summary clinical extraction.

Uses Ollama structured output and scores clinically important attributes before
we invest in SFT or generate FHIR resources. No third-party Python packages are
required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_NUM_CTX = int(os.getenv("CLINICAL_OLLAMA_NUM_CTX", "8192"))

FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["patient", "encounter", "facts"],
    "properties": {
        "patient": {
            "type": "object",
            "additionalProperties": False,
            "required": ["full_name", "birth_date", "gender"],
            "properties": {
                "full_name": {"type": ["string", "null"]},
                "birth_date": {"type": ["string", "null"]},
                "gender": {"type": ["string", "null"]},
            },
        },
        "encounter": {
            "type": "object",
            "additionalProperties": False,
            "required": ["admit_date", "discharge_date"],
            "properties": {
                "admit_date": {"type": ["string", "null"]},
                "discharge_date": {"type": ["string", "null"]},
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "category",
                    "name",
                    "assertion",
                    "temporality",
                    "clinical_status",
                    "experiencer",
                    "date",
                    "value",
                    "unit",
                    "dose",
                    "route",
                    "frequency",
                    "evidence_quote",
                ],
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "condition",
                            "observation",
                            "medication",
                            "procedure",
                            "allergy",
                        ],
                    },
                    "name": {"type": "string"},
                    "assertion": {
                        "type": "string",
                        "enum": ["present", "absent", "possible", "conditional"],
                    },
                    "temporality": {
                        "type": "string",
                        "enum": ["current", "historical", "planned", "unknown"],
                    },
                    "clinical_status": {
                        "type": "string",
                        "enum": [
                            "active",
                            "resolved",
                            "completed",
                            "discontinued",
                            "planned",
                            "unknown",
                        ],
                    },
                    "experiencer": {
                        "type": "string",
                        "enum": ["patient", "family"],
                    },
                    "date": {"type": ["string", "null"]},
                    "value": {"type": ["number", "string", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "dose": {"type": ["string", "null"]},
                    "route": {"type": ["string", "null"]},
                    "frequency": {"type": ["string", "null"]},
                    "evidence_quote": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

SYSTEM_INSTRUCTIONS = """You extract clinical facts from a discharge summary.
Return only facts explicitly supported by the document, using the supplied JSON schema.
Copy evidence_quote verbatim as the shortest text span that supports the fact.
Do not convert a negated, hypothetical, family-history, or discontinued item into an active patient condition.
Use assertion=absent for explicit negation and possible for uncertainty.
Use experiencer=family when the fact concerns a relative.
Use temporality=historical for remote history, planned for future actions, and current for this encounter.
Do not invent codes or facts. Use null when a field is not stated."""


@dataclass(frozen=True)
class ExpectedFact:
    name_terms: tuple[str, ...]
    attributes: dict[str, Any]


TESTS = [
    {
        "name": "negation",
        "note": (
            "Jane Doe has type 2 diabetes mellitus. "
            "There is no evidence of pneumonia."
        ),
        "expected": [
            ExpectedFact(("diabetes",), {"category": "condition", "assertion": "present"}),
            ExpectedFact(("pneumonia",), {"category": "condition", "assertion": "absent"}),
        ],
    },
    {
        "name": "history_and_experiencer",
        "note": (
            "The patient's mother had breast cancer. The patient has no personal history of cancer. "
            "An appendectomy was performed in 1998."
        ),
        "expected": [
            ExpectedFact(("breast", "cancer"), {"experiencer": "family", "assertion": "present"}),
            ExpectedFact(("cancer",), {"experiencer": "patient", "assertion": "absent"}),
            ExpectedFact(
                ("appendectomy",),
                {"category": "procedure", "temporality": "historical", "clinical_status": "completed"},
            ),
        ],
    },
    {
        "name": "medication_status",
        "note": (
            "Lisinopril was discontinued because of cough. Start amlodipine 5 mg by mouth daily. "
            "Continue metformin 500 mg by mouth twice daily."
        ),
        "expected": [
            ExpectedFact(("lisinopril",), {"category": "medication", "clinical_status": "discontinued"}),
            ExpectedFact(("amlodipine",), {"category": "medication", "clinical_status": "active"}),
            ExpectedFact(("metformin",), {"category": "medication", "clinical_status": "active"}),
        ],
    },
    {
        "name": "uncertainty_and_plan",
        "note": (
            "Pulmonary embolism is possible but not confirmed. CT pulmonary angiography is planned tomorrow. "
            "Anticoagulation has not been started."
        ),
        "expected": [
            ExpectedFact(("pulmonary", "embol"), {"category": "condition", "assertion": "possible"}),
            ExpectedFact(("angiograph",), {"category": "procedure", "temporality": "planned"}),
            ExpectedFact(("anticoag",), {"category": "medication", "assertion": "absent"}),
        ],
    },
    {
        "name": "allergy_and_tolerance",
        "note": "Penicillin allergy causes a rash. The patient tolerates cephalosporins.",
        "expected": [
            ExpectedFact(("penicillin",), {"category": "allergy", "assertion": "present"}),
            ExpectedFact(("cephalosporin",), {"category": "allergy", "assertion": "absent"}),
        ],
    },
    {
        "name": "laboratory_value",
        "note": "Potassium was 3.1 mmol/L and hemoglobin was 9.4 g/dL on 2026-07-11.",
        "expected": [
            ExpectedFact(("potassium",), {"category": "observation", "value": 3.1, "unit": "mmol/L"}),
            ExpectedFact(("hemoglobin",), {"category": "observation", "value": 9.4, "unit": "g/dL"}),
        ],
    },
]


def call_ollama(model: str, note: str, timeout: int) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": FACT_SCHEMA,
        # A 27B model at its 32K default context leaves too little VRAM on a
        # 24 GB GPU for reliable document extraction. Keep this configurable
        # for larger installations, with a safe default for local workbenches.
        "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
        "prompt": f"{SYSTEM_INSTRUCTIONS}\n\nDISCHARGE SUMMARY:\n{note}",
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.load(response)
    elapsed = time.monotonic() - started
    return json.loads(envelope["response"]), elapsed


def fact_matches_name(fact: dict[str, Any], terms: tuple[str, ...]) -> bool:
    name = str(fact.get("name", "")).lower()
    return all(term.lower() in name for term in terms)


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 0.0001
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


def score_case(note: str, result: dict[str, Any], expected: list[ExpectedFact]) -> tuple[int, int, list[str]]:
    facts = result.get("facts", [])
    passed = 0
    total = 0
    messages: list[str] = []

    for wanted in expected:
        candidates = [fact for fact in facts if fact_matches_name(fact, wanted.name_terms)]
        total += 1
        if not candidates:
            messages.append(f"MISS fact containing {wanted.name_terms}")
            continue
        fact = candidates[0]
        passed += 1

        for key, expected_value in wanted.attributes.items():
            total += 1
            if values_equal(fact.get(key), expected_value):
                passed += 1
            else:
                messages.append(
                    f"WRONG {fact.get('name')!r}.{key}: expected {expected_value!r}, got {fact.get(key)!r}"
                )

        total += 1
        quote = fact.get("evidence_quote", "")
        if quote and quote in note:
            passed += 1
        else:
            messages.append(f"UNGROUNDED {fact.get('name')!r}: {quote!r}")

    return passed, total, messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma:27b")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json-output", help="Optional path for complete raw results")
    args = parser.parse_args()

    all_results = []
    total_passed = 0
    total_checks = 0
    print(f"Model: {args.model}")
    print(f"Cases: {len(TESTS)}\n")

    for index, test in enumerate(TESTS, start=1):
        try:
            result, elapsed = call_ollama(args.model, test["note"], args.timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            print(f"[{index}/{len(TESTS)}] {test['name']}: ERROR {error}")
            return 2

        passed, checks, messages = score_case(test["note"], result, test["expected"])
        total_passed += passed
        total_checks += checks
        all_results.append({"test": test["name"], "note": test["note"], "result": result})
        label = "PASS" if passed == checks else "FAIL"
        print(f"[{index}/{len(TESTS)}] {test['name']}: {label} {passed}/{checks} ({elapsed:.1f}s)")
        for message in messages:
            print(f"  - {message}")

    percent = 100.0 * total_passed / total_checks if total_checks else 0.0
    print(f"\nTOTAL: {total_passed}/{total_checks} ({percent:.1f}%)")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Raw results: {args.json_output}")

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
