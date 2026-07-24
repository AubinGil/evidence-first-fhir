"""Generate per-model predictions for the gold set. Requires local Ollama.

This is the only inference step in the gold-set workflow; run it once per
candidate model when the GPU is free. Scoring (goldset_score.py) and CI never
call a model.

Usage:
    python goldset_predict.py --model medgemma:27b
    python goldset_predict.py --model qwen3.6:35b --timeout 900
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from baseline_extraction_test import (
    FACT_SCHEMA,
    OLLAMA_URL,
    SYSTEM_INSTRUCTIONS,
    call_ollama,
)
from goldset_score import FIXTURES_DIR, goldset_checksum, goldset_version, load_fixtures


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.casefold()).strip("-")


def call_openai(model: str, note: str, timeout: int, base_url: str) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"DISCHARGE SUMMARY:\n{note}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "clinical_facts", "strict": True, "schema": FACT_SCHEMA},
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.load(response)
    elapsed = time.monotonic() - started
    return json.loads(envelope["choices"][0]["message"]["content"]), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url", default="http://localhost:8012/v1")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--out", type=Path, help="Defaults to output/goldset/<model-slug>")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true", help="Regenerate existing predictions")
    parser.add_argument("--limit", type=int, help="Only run the first N fixtures (smoke test)")
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures)
    if args.limit:
        fixtures = fixtures[: args.limit]
    out_dir = args.out or Path("output/goldset") / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "model": args.model,
        "provider": args.provider,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "goldset_version": goldset_version(),
        "goldset_sha256": goldset_checksum(args.fixtures),
        "cases": {},
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(previous.get("cases"), dict):
            manifest["cases"] = previous["cases"]

    failures = 0
    for index, fixture in enumerate(fixtures, start=1):
        case_id = fixture["id"]
        target = out_dir / f"{case_id}.json"
        if target.is_file() and not args.force:
            print(f"[{index}/{len(fixtures)}] {case_id}: exists, skipping (use --force to redo)")
            continue
        try:
            if args.provider == "ollama":
                prediction, elapsed = call_ollama(args.model, fixture["note"], args.timeout)
            else:
                prediction, elapsed = call_openai(
                    args.model, fixture["note"], args.timeout, args.base_url
                )
        except urllib.error.URLError as error:
            endpoint = OLLAMA_URL if args.provider == "ollama" else args.base_url
            print(f"{args.provider} endpoint is not reachable at {endpoint}: {error}")
            print("This step needs local inference; scoring existing predictions does not.")
            return 2
        except (TimeoutError, json.JSONDecodeError, KeyError) as error:
            failures += 1
            manifest["cases"][case_id] = {"error": f"{type(error).__name__}: {error}"}
            print(f"[{index}/{len(fixtures)}] {case_id}: ERROR {error}")
            continue
        target.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        manifest["cases"][case_id] = {"elapsed_seconds": round(elapsed, 1)}
        print(f"[{index}/{len(fixtures)}] {case_id}: {len(prediction.get('facts', []))} facts ({elapsed:.1f}s)")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPredictions: {out_dir}")
    print(f"Score with: python goldset_score.py {out_dir} --report {out_dir / 'report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
