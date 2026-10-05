#!/usr/bin/env python3
"""Small, bounded load test for the prediction service. Standard library only.

    python tools/loadtest.py http://127.0.0.1:8080 --concurrency 8 --duration 30

Closed-loop: each of ``--concurrency`` threads sends one request, waits for the
answer, then sends the next (keep-alive connections). Request mix: 90% POST
/predict over random supported team pairs, 10% GET /ready. A warm-up phase is
run and discarded. Prints a JSON summary; nothing is extrapolated.

Hard caps (duration <= 300s, concurrency <= 64) keep an accidental run against
a deployed endpoint cheap. Point it only at a service you own.
"""

import argparse
import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse
import urllib.request
from typing import Any


def get_json(base: str, path: str) -> Any:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=10) as response:  # noqa: S310
        return json.loads(response.read())


def worker(
    base: urllib.parse.SplitResult,
    bodies: list[bytes],
    stop_at: float,
    seed: int,
    out: list[tuple[str, int, float]],
) -> None:
    rng = random.Random(seed)
    make = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
    conn = make(base.netloc, timeout=10)
    while time.monotonic() < stop_at:
        is_predict = rng.random() < 0.9
        started = time.perf_counter()
        try:
            if is_predict:
                conn.request(
                    "POST",
                    "/predict",
                    body=rng.choice(bodies),
                    headers={"content-type": "application/json"},
                )
            else:
                conn.request("GET", "/ready")
            response = conn.getresponse()
            response.read()
            status = response.status
        except (OSError, http.client.HTTPException):
            status = 0
            conn.close()
            conn = make(base.netloc, timeout=10)
        out.append(("predict" if is_predict else "ready", status, time.perf_counter() - started))


def run_phase(
    base: urllib.parse.SplitResult, bodies: list[bytes], concurrency: int, seconds: float
) -> list[tuple[str, int, float]]:
    results: list[list[tuple[str, int, float]]] = [[] for _ in range(concurrency)]
    stop_at = time.monotonic() + seconds
    threads = [
        threading.Thread(target=worker, args=(base, bodies, stop_at, i, results[i]))
        for i in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [row for rows in results for row in rows]


def percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30.0, help="measured seconds")
    parser.add_argument("--warmup", type=float, default=5.0, help="discarded seconds")
    args = parser.parse_args()
    if not (1 <= args.concurrency <= 64) or not (1 <= args.duration <= 300):
        parser.error("concurrency must be 1-64 and duration 1-300 seconds")

    ready = get_json(args.base_url, "/ready")
    teams = get_json(args.base_url, "/teams")["teams"]
    rng = random.Random(0)
    bodies = []
    for _ in range(500):
        home, away = rng.sample(teams, 2)
        bodies.append(
            json.dumps(
                {
                    "home_team": home,
                    "away_team": away,
                    "date": ready["model"]["training_cutoff"],
                    "neutral": rng.random() < 0.5,
                }
            ).encode()
        )

    base = urllib.parse.urlsplit(args.base_url)
    run_phase(base, bodies, args.concurrency, args.warmup)
    started = time.monotonic()
    rows = run_phase(base, bodies, args.concurrency, args.duration)
    elapsed = time.monotonic() - started

    latencies = sorted(seconds * 1000 for _, _, seconds in rows)
    errors = [row for row in rows if not 200 <= row[1] < 300]
    summary = {
        "target": args.base_url,
        "artifact_version": ready["model"]["artifact_version"],
        "synthetic_artifact": ready["model"]["synthetic"],
        "request_mix": "90% POST /predict (random supported pairs), 10% GET /ready",
        "concurrency": args.concurrency,
        "warmup_s": args.warmup,
        "duration_s": round(elapsed, 2),
        "total_requests": len(rows),
        "requests_per_s": round(len(rows) / elapsed, 1),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(rows), 5) if rows else None,
        "error_statuses": sorted({row[1] for row in errors}),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "max": round(latencies[-1], 2),
            "mean": round(statistics.fmean(latencies), 2),
        }
        if latencies
        else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
