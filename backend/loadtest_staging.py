#!/usr/bin/env python3
"""Staging load test runner for /chart and /ai_reading."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


DEFAULT_CHART_PARAMS = {
    "year": 1992,
    "month": 8,
    "day": 15,
    "hour": 14.5,
    "lat": 37.5665,
    "lon": 126.9780,
    "house_system": "W",
    "include_nodes": 1,
    "include_d9": 1,
    "gender": "male",
    "timezone": 9.0,
}

DEFAULT_AI_READING_PARAMS = {
    "year": 1992,
    "month": 8,
    "day": 15,
    "hour": 14.5,
    "lat": 37.5665,
    "lon": 126.9780,
    "house_system": "W",
    "include_nodes": 1,
    "include_d9": 1,
    "language": "ko",
    "gender": "male",
    "use_cache": 0,
    "production_mode": 0,
    "timezone": 9.0,
}


@dataclass
class RoundResult:
    endpoint: str
    concurrency: int
    total_requests: int
    ok_count: int
    error_count: int
    throughput_rps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_s: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    arr = sorted(values)
    k = (len(arr) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1


async def run_round(
    *,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    concurrency: int,
    total_requests: int,
    timeout_s: float,
) -> RoundResult:
    url = f"{base_url.rstrip('/')}{endpoint}"
    queue: asyncio.Queue[int] = asyncio.Queue()
    latencies_ms: list[float] = []
    ok_count = 0
    error_count = 0
    lock = asyncio.Lock()

    for i in range(total_requests):
        queue.put_nowait(i)

    timeout = httpx.Timeout(timeout_s)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal ok_count, error_count
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            started = time.perf_counter()
            try:
                resp = await client.get(url, params=params)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                async with lock:
                    latencies_ms.append(elapsed_ms)
                    if 200 <= resp.status_code < 300:
                        ok_count += 1
                    else:
                        error_count += 1
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                async with lock:
                    latencies_ms.append(elapsed_ms)
                    error_count += 1
            finally:
                queue.task_done()

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        workers = [asyncio.create_task(worker(client)) for _ in range(concurrency)]
        await queue.join()
        await asyncio.gather(*workers)
    total_s = time.perf_counter() - start

    avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else math.nan
    return RoundResult(
        endpoint=endpoint,
        concurrency=concurrency,
        total_requests=total_requests,
        ok_count=ok_count,
        error_count=error_count,
        throughput_rps=(total_requests / total_s) if total_s > 0 else 0.0,
        avg_ms=avg_ms,
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
        total_s=total_s,
    )


def print_table(results: list[RoundResult]) -> None:
    print(
        "endpoint,concurrency,total,ok,error,total_s,rps,avg_ms,p50_ms,p95_ms,p99_ms"
    )
    for row in results:
        print(
            f"{row.endpoint},{row.concurrency},{row.total_requests},{row.ok_count},{row.error_count},"
            f"{row.total_s:.3f},{row.throughput_rps:.2f},{row.avg_ms:.2f},{row.p50_ms:.2f},"
            f"{row.p95_ms:.2f},{row.p99_ms:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staging load test for Vedic AI backend endpoints.")
    parser.add_argument("--base-url", required=True, help="Target base URL, e.g. https://staging.example.com")
    parser.add_argument(
        "--targets",
        default="chart,ai_reading",
        help="Comma-separated targets: chart,ai_reading",
    )
    parser.add_argument(
        "--concurrency",
        default="4,8,12,16",
        help="Comma-separated concurrency levels.",
    )
    parser.add_argument(
        "--requests-per-round",
        type=int,
        default=60,
        help="Total requests for each endpoint/concurrency round.",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0, help="Per-request timeout seconds.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional file path to write JSON summary.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    targets = {x.strip() for x in args.targets.split(",") if x.strip()}
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    endpoint_defs: list[tuple[str, dict[str, Any]]] = []
    if "chart" in targets:
        endpoint_defs.append(("/chart", DEFAULT_CHART_PARAMS))
    if "ai_reading" in targets:
        endpoint_defs.append(("/ai_reading", DEFAULT_AI_READING_PARAMS))

    if not endpoint_defs:
        raise SystemExit("No valid targets selected. Use chart and/or ai_reading.")

    print(f"base_url={args.base_url.rstrip('/')}")
    print(f"targets={','.join(t for t in ['chart', 'ai_reading'] if t in targets)}")
    print(f"concurrency_levels={levels}")
    print(f"requests_per_round={args.requests_per_round}")

    results: list[RoundResult] = []
    for endpoint, params in endpoint_defs:
        for level in levels:
            qs = urlencode(params, doseq=True)
            print(f"running endpoint={endpoint} concurrency={level} sample_query={qs[:120]}...")
            out = await run_round(
                base_url=args.base_url,
                endpoint=endpoint,
                params=params,
                concurrency=level,
                total_requests=args.requests_per_round,
                timeout_s=args.timeout_s,
            )
            results.append(out)

    print_table(results)

    if args.output_json:
        payload = [r.__dict__ for r in results]
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved_json={args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
