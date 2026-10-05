#!/usr/bin/env python3
"""Smoke-check a running prediction service. Standard library only.

    python tools/smoke.py http://127.0.0.1:8080 --expect-version ci-fixture
    python tools/smoke.py "$SERVICE_URL" --expect-version 2026-10-05.1 --require-real

Checks liveness, readiness (waiting up to --wait seconds), the pinned artifact
version, one valid prediction built from the service's own /teams and /ready
answers, one invalid request, and the model-comparison endpoint. Exits non-zero
with the failed check named.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def call(base: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(  # noqa: S310 - operator-supplied URL
        base.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw.decode(errors="replace")[:200]


def fail(check: str, detail: Any) -> None:
    print(f"FAIL  {check}: {detail}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--expect-version", help="artifact version the service must report")
    parser.add_argument("--require-real", action="store_true", help="fail on a synthetic artifact")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for readiness")
    parser.add_argument(
        "--allow-missing-evaluation",
        action="store_true",
        help="accept 503 evaluation_unavailable from /models/comparison",
    )
    args = parser.parse_args()
    base = args.base_url

    deadline = time.monotonic() + args.wait
    while True:
        try:
            status, ready = call(base, "/ready")
        except (urllib.error.URLError, OSError) as exc:
            status, ready = 0, str(exc)
        if status == 200 or time.monotonic() > deadline:
            break
        time.sleep(2)
    if status != 200:
        fail("readiness", f"/ready -> {status} {ready}")
    model = ready["model"]
    print(f"ok    ready: artifact {model['artifact_version']} (synthetic={model['synthetic']})")

    status, health = call(base, "/health")
    if status != 200 or health != {"status": "ok"}:
        fail("liveness", f"/health -> {status} {health}")
    print("ok    health")

    if args.expect_version and model["artifact_version"] != args.expect_version:
        fail("artifact version", f"expected {args.expect_version}, got {model['artifact_version']}")
    if args.require_real and model["synthetic"]:
        fail("artifact kind", "service is running a SYNTHETIC test artifact")

    status, teams = call(base, "/teams")
    if status != 200 or len(teams["teams"]) < 2:
        fail("teams", f"/teams -> {status}")
    home, away = teams["teams"][0], teams["teams"][1]
    fixture = {"home_team": home, "away_team": away, "date": model["training_cutoff"]}

    status, prediction = call(base, "/predict", fixture)
    if status != 200:
        fail("prediction", f"/predict -> {status} {prediction}")
    probs = prediction["probabilities"]
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6 or not all(0.0 <= p <= 1.0 for p in probs.values()):
        fail("prediction", f"not a probability distribution: {probs}")
    if prediction["model"]["artifact_version"] != model["artifact_version"]:
        fail("prediction", "response reports a different artifact version than /ready")
    print(f"ok    predict {home} v {away}: {probs}")

    status, rejected = call(base, "/predict", {**fixture, "away_team": home})
    if status != 422 or rejected.get("error", {}).get("code") != "validation_error":
        fail("invalid input", f"expected 422 validation_error, got {status} {rejected}")
    status, rejected = call(base, "/predict", {**fixture, "away_team": "no_such_team_zz"})
    if status != 422 or rejected.get("error", {}).get("code") != "unknown_team":
        fail("unknown team", f"expected 422 unknown_team, got {status} {rejected}")
    print("ok    invalid input rejected (422 validation_error, 422 unknown_team)")

    status, comparison = call(base, "/models/comparison")
    if status == 200:
        hist = comparison["evaluation"].get("historical", {})
        print(f"ok    comparison: {len(hist.get('models', []))} models on the historical track")
    elif (
        status == 503
        and args.allow_missing_evaluation
        and comparison.get("error", {}).get("code") == "evaluation_unavailable"
    ):
        print("ok    comparison: unavailable (artifact has no stored evaluation)")
    else:
        fail("model comparison", f"/models/comparison -> {status} {comparison}")

    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
