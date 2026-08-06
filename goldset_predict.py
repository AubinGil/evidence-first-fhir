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
from typing import Any

from baseline_extraction_test import (
    FACT_SCHEMA,
    OLLAMA_URL,
    SYSTEM_INSTRUCTIONS,
    RunawayGeneration,
    ServingArtifact,
    call_ollama,
    parse_model_json,
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


SCHEMA_INSTRUCTION = (
    "\n\nReturn ONLY a JSON object validating against this JSON Schema. Every listed "
    "required field must be present on every object; use null when unknown. No markdown "
    "fences, no commentary.\n\nJSON SCHEMA:\n" + json.dumps(FACT_SCHEMA)
)


def call_openai_prompt_schema(
    model: str, note: str, timeout: int, base_url: str
) -> tuple[dict[str, Any], float]:
    """Same request as call_openai, with the schema moved into the prompt.

    For servers that cannot constrain decoding at all. A diffusion LM has no
    left-to-right token commitment for a grammar to attach to, so vLLM serving
    one answers json_schema and json_object with HTTP 500 -- which arrives here
    as an HTTPError, a subclass of URLError, and so gets reported as an
    unreachable endpoint rather than an unsupported feature.

    The dangerous path is not that one. guided_json and guided_choice are
    accepted and silently ignored: measured on diffusiongemma-26B-A4B, a
    guided_json request for a schema-conforming object returned 200 with the
    body "**Emerald Green**", and guided_choice restricted to present/absent
    returned a free-text hedge. A caller that assumes those flags bind gets
    unconstrained output that never raises. So the schema is carried in the
    prompt, where it is visibly only a request, and the reply is parsed by the
    same parse_model_json the other providers use.

    A run made this way is NOT comparable to a grammar-enforced one, because
    shape compliance is now a model property rather than a decoder guarantee.
    The manifest records schema_enforcement for exactly this reason.
    """
    payload = {
        "model": model,
        "temperature": 0,
        "seed": 2026,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + SCHEMA_INSTRUCTION},
            {"role": "user", "content": f"DISCHARGE SUMMARY:\n{note}"},
        ],
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
    choice = envelope["choices"][0]
    # Unenforced output can also just stop when it runs out of room. Ask why
    # generation ended before parsing, for the same reason call_ollama does.
    if str(choice.get("finish_reason", "")).casefold() == "length":
        raise RunawayGeneration(
            f"{model} generated {envelope['usage']['completion_tokens']} tokens without "
            f"stopping (finish_reason=length); output is truncated"
        )
    return parse_model_json(choice["message"]["content"]), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument(
        "--schema-mode",
        choices=("grammar", "prompt"),
        default="grammar",
        help="openai provider only: 'prompt' carries the schema in the prompt for "
        "servers that cannot constrain decoding (see call_openai_prompt_schema)",
    )
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
        # Ollama's `format` and the openai grammar path both bind the decoder;
        # "prompt" does not. A reader comparing reports needs to know which,
        # because an unenforced run can lose shape compliance on its own.
        "schema_enforcement": "grammar" if args.provider == "ollama" else args.schema_mode,
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
        retried = False
        try:
            if args.provider == "ollama":
                try:
                    prediction, elapsed = call_ollama(args.model, fixture["note"], args.timeout)
                except ServingArtifact as artifact:
                    # Retry exactly this class of failure, and only this class.
                    # The model is warm by the time it is raised, so a second
                    # request measures the model rather than the swap. A
                    # runaway is never retried: it reproduces by definition.
                    print(f"[{index}/{len(fixtures)}] {case_id}: {artifact}; retrying warm")
                    prediction, elapsed = call_ollama(args.model, fixture["note"], args.timeout)
                    retried = True
            elif args.schema_mode == "prompt":
                prediction, elapsed = call_openai_prompt_schema(
                    args.model, fixture["note"], args.timeout, args.base_url
                )
            else:
                prediction, elapsed = call_openai(
                    args.model, fixture["note"], args.timeout, args.base_url
                )
        except urllib.error.URLError as error:
            endpoint = OLLAMA_URL if args.provider == "ollama" else args.base_url
            print(f"{args.provider} endpoint is not reachable at {endpoint}: {error}")
            print("This step needs local inference; scoring existing predictions does not.")
            return 2
        except ServingArtifact as error:
            # Twice in a row is no longer a transient, but it still is not a
            # model property, so it must not be scored as one.
            failures += 1
            manifest["cases"][case_id] = {"serving_artifact": True, "error": str(error)}
            print(f"[{index}/{len(fixtures)}] {case_id}: SERVING ARTIFACT {error}")
            continue
        except RunawayGeneration as error:
            # Recorded separately because it is a property of the model, not of
            # the run. Retrying at temperature 0 reproduces it, so a run that
            # reports this is complete, not interrupted.
            failures += 1
            manifest["cases"][case_id] = {"runaway": True, "error": str(error)}
            print(f"[{index}/{len(fixtures)}] {case_id}: RUNAWAY {error}")
            continue
        # ValueError, not just JSONDecodeError: parse_model_json raises the bare
        # parent when the reply parses but is not an object. Unenforced output
        # reaches that branch, and a model that returns the wrong shape should be
        # recorded as a failed case, not crash the run before it is written.
        except (TimeoutError, ValueError, KeyError) as error:
            failures += 1
            manifest["cases"][case_id] = {"error": f"{type(error).__name__}: {error}"}
            print(f"[{index}/{len(fixtures)}] {case_id}: ERROR {error}")
            continue
        target.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        manifest["cases"][case_id] = {"elapsed_seconds": round(elapsed, 1)}
        if retried:
            # Kept so a reader can tell a clean first answer from a recovered
            # one, and so a run full of retries reads as a serving problem.
            manifest["cases"][case_id]["recovered_from_serving_artifact"] = True
        print(f"[{index}/{len(fixtures)}] {case_id}: {len(prediction.get('facts', []))} facts ({elapsed:.1f}s)")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPredictions: {out_dir}")
    print(f"Score with: python goldset_score.py {out_dir} --report {out_dir / 'report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
