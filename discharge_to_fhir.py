"""Discharge summary -> grounded facts -> FHIR R4 transaction Bundle -> HAPI validation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baseline_extraction_test import call_ollama


FHIR_BASE = "http://localhost:8080/fhir"
UUID_NAMESPACE = uuid.UUID("614847fc-21b9-4c9a-9ad4-fb8f31ab3b77")


def clean(value: Any) -> Any:
    if isinstance(value, str) and value.strip().casefold() in {"", "null", "none", "unknown"}:
        return None
    return value


def stable_id(document_hash: str, resource_type: str, key: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, f"{document_hash}:{resource_type}:{key}"))


def urn(resource_id: str) -> str:
    return f"urn:uuid:{resource_id}"


def concept(text: str, system: str | None = None, code: str | None = None, display: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"text": text}
    if system and code:
        result["coding"] = [{"system": system, "code": code, "display": display or text}]
    return result


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


FHIR_TEMPORAL = re.compile(
    r"^\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?$"
)


def fhir_temporal(value: Any) -> str | None:
    """Return only syntactically and calendrically valid FHIR date/dateTime values."""
    candidate = clean(value)
    if not isinstance(candidate, str) or not FHIR_TEMPORAL.fullmatch(candidate):
        return None
    if len(candidate) >= 10:
        try:
            datetime.fromisoformat(candidate[:10])
        except ValueError:
            return None
    return candidate


def grounded_facts(note: str, extraction: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for original in extraction.get("facts", []):
        fact = {key: clean(value) for key, value in original.items()}
        quote = fact.get("evidence_quote")
        reasons = []
        if not isinstance(quote, str) or quote not in note:
            reasons.append("evidence quote is not an exact source substring")
        if fact.get("experiencer") != "patient":
            reasons.append("fact does not concern the patient")
        if fact.get("assertion") != "present":
            reasons.append(f"assertion is {fact.get('assertion')!r}")
        if reasons:
            quarantined.append({"fact": fact, "reasons": reasons})
            continue
        fact["evidence_start"] = note.index(quote)
        fact["evidence_end"] = fact["evidence_start"] + len(quote)
        accepted.append(fact)
    return accepted, quarantined


def category(fact: dict[str, Any]) -> str:
    """Apply only high-precision corrections; ambiguous facts stay quarantinable later."""
    name = str(fact.get("name", "")).casefold()
    evidence = str(fact.get("evidence_quote", "")).casefold()
    if any(term in name for term in ("hba1c", "hemoglobin a1c", "potassium", "hemoglobin")):
        return "observation"
    if "appendectom" in name:
        return "procedure"
    if re.search(r"\b(?:was|is)\s+\d+(?:\.\d+)?\s*(?:percent|%|mmol/l|g/dl)\b", evidence):
        return "observation"
    return str(fact.get("category", ""))


def entry(resource: dict[str, Any]) -> dict[str, Any]:
    resource_type = resource["resourceType"]
    resource_id = resource["id"]
    return {
        "fullUrl": urn(resource_id),
        "resource": resource,
        "request": {"method": "PUT", "url": f"{resource_type}/{resource_id}"},
    }


def build_bundle(note: str, extraction: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(note.encode()).hexdigest()
    patient_data = {key: clean(value) for key, value in extraction.get("patient", {}).items()}
    encounter_data = {key: clean(value) for key, value in extraction.get("encounter", {}).items()}
    patient_id = stable_id(digest, "Patient", str(patient_data.get("full_name") or digest))
    encounter_id = stable_id(digest, "Encounter", json.dumps(encounter_data, sort_keys=True))
    patient_ref = urn(patient_id)
    encounter_ref = urn(encounter_id)

    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{"system": "urn:discharge-to-fhir:document", "value": digest}],
    }
    if patient_data.get("full_name"):
        patient["name"] = [{"text": patient_data["full_name"]}]
    if patient_data.get("gender") in {"male", "female", "other", "unknown"}:
        patient["gender"] = patient_data["gender"]
    birth_date = fhir_temporal(patient_data.get("birth_date"))
    if birth_date:
        patient["birthDate"] = birth_date

    encounter: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "IMP",
            "display": "inpatient encounter",
        },
        "subject": {"reference": patient_ref},
    }
    if encounter_data.get("admit_date") or encounter_data.get("discharge_date"):
        encounter["period"] = {}
        admit_date = fhir_temporal(encounter_data.get("admit_date"))
        discharge_date = fhir_temporal(encounter_data.get("discharge_date"))
        if admit_date:
            encounter["period"]["start"] = admit_date
        if discharge_date:
            encounter["period"]["end"] = discharge_date

    resources: list[dict[str, Any]] = [patient, encounter]
    for index, fact in enumerate(facts):
        kind = category(fact)
        name = str(fact.get("name") or "Unspecified clinical fact")
        resource_id = stable_id(digest, kind, f"{index}:{name}:{fact.get('evidence_quote')}")

        if kind == "condition":
            coded = concept(name)
            if "type 2 diabetes" in name.casefold():
                coded = concept(
                    name,
                    "http://hl7.org/fhir/sid/icd-10-cm",
                    "E11.9",
                    "Type 2 diabetes mellitus without complications",
                )
            resource = {
                "resourceType": "Condition",
                "id": resource_id,
                "clinicalStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }]},
                "verificationStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                }]},
                "code": coded,
                "subject": {"reference": patient_ref},
                "encounter": {"reference": encounter_ref},
            }
        elif kind == "observation":
            coded = concept(name)
            if name.casefold() in {"hba1c", "hemoglobin a1c"}:
                coded = concept(name, "http://loinc.org", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood")
            resource = {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": coded,
                "subject": {"reference": patient_ref},
                "encounter": {"reference": encounter_ref},
            }
            magnitude = number(fact.get("value"))
            unit = fact.get("unit")
            if magnitude is not None:
                resource["valueQuantity"] = {"value": magnitude}
                if unit:
                    unit = "%" if str(unit).casefold() == "percent" else str(unit)
                    resource["valueQuantity"].update({
                        "unit": unit,
                        "system": "http://unitsofmeasure.org",
                        "code": unit,
                    })
            fact_date = fhir_temporal(fact.get("date"))
            if fact_date:
                resource["effectiveDateTime"] = fact_date
        elif kind == "medication":
            resource = {
                "resourceType": "MedicationStatement",
                "id": resource_id,
                "status": "stopped" if fact.get("clinical_status") == "discontinued" else "active",
                "medicationCodeableConcept": concept(name),
                "subject": {"reference": patient_ref},
                "context": {"reference": encounter_ref},
            }
            dosage = " ".join(str(fact.get(key)) for key in ("dose", "route", "frequency") if fact.get(key))
            if dosage:
                resource["dosage"] = [{"text": dosage}]
        elif kind == "procedure":
            resource = {
                "resourceType": "Procedure",
                "id": resource_id,
                "status": "preparation" if fact.get("temporality") == "planned" else "completed",
                "code": concept(name),
                "subject": {"reference": patient_ref},
            }
            fact_date = fhir_temporal(fact.get("date"))
            if fact_date:
                resource["performedDateTime"] = fact_date
        elif kind == "allergy":
            clinical_status = {
                "active": "active",
                "resolved": "resolved",
                "discontinued": "inactive",
            }.get(str(fact.get("clinical_status")), "active")
            resource = {
                "resourceType": "AllergyIntolerance",
                "id": resource_id,
                "clinicalStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                    "code": clinical_status,
                }]},
                "verificationStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                    "code": "confirmed",
                }]},
                "code": concept(name),
                "patient": {"reference": patient_ref},
                "encounter": {"reference": encounter_ref},
            }
        else:
            continue
        resource["text"] = {
            "status": "generated",
            "div": (
                '<div xmlns="http://www.w3.org/1999/xhtml">'
                f"{html.escape(name)}</div>"
            ),
        }
        resources.append(resource)

    return {
        "resourceType": "Bundle",
        "id": stable_id(digest, "Bundle", "transaction"),
        "type": "transaction",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entry": [entry(resource) for resource in resources],
    }


def validate(bundle: dict[str, Any], fhir_base: str, timeout: int) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{fhir_base.rstrip('/')}/Bundle/$validate",
        data=json.dumps(bundle).encode(),
        headers={"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, {"resourceType": "OperationOutcome", "issue": [{"severity": "fatal", "diagnostics": body}]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="medgemma:27b")
    parser.add_argument("--fhir-base", default=FHIR_BASE)
    parser.add_argument("--output-dir", type=Path, default=Path("clinical_fhir/output"))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    note = args.input.read_text(encoding="utf-8").strip()
    extraction, elapsed = call_ollama(args.model, note, args.timeout)
    accepted, quarantined = grounded_facts(note, extraction)
    bundle = build_bundle(note, extraction, accepted)
    status, outcome = validate(bundle, args.fhir_base, args.timeout)
    errors = [issue for issue in outcome.get("issue", []) if issue.get("severity") in {"error", "fatal"}]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "extraction.json": extraction,
        "review.json": {"accepted": accepted, "quarantined": quarantined},
        "bundle.json": bundle,
        "validation.json": {"http_status": status, "operation_outcome": outcome},
    }
    for filename, payload in artifacts.items():
        (args.output_dir / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Model: {args.model} ({elapsed:.1f}s)")
    print(f"Facts: {len(extraction.get('facts', []))} extracted, {len(accepted)} accepted, {len(quarantined)} quarantined")
    print(f"Bundle: {len(bundle['entry'])} resources")
    print(f"HAPI $validate: HTTP {status}, {len(errors)} error(s)")
    for issue in errors:
        print(f"  - {issue.get('diagnostics') or issue.get('details', {}).get('text')}")
    print(f"Artifacts: {args.output_dir.resolve()}")
    return 0 if status < 400 and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
