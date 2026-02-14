# Staging Load Test and Concurrency Tuning

This guide benchmarks `/chart` and `/ai_reading` against a staging URL and tunes runtime limits.

## 1) Run load test against staging

```bash
python backend/loadtest_staging.py \
  --base-url https://your-staging-host \
  --targets chart,ai_reading \
  --concurrency 4,8,12,16 \
  --requests-per-round 80 \
  --timeout-s 90 \
  --output-json backend/loadtest_result.json
```

Output columns:
- `rps`: throughput
- `p95_ms`, `p99_ms`: tail latency (primary KPI)
- `error`: failed or non-2xx requests

## 2) Runtime knobs for tuning

Container runtime env:
- `WEB_CONCURRENCY`: number of uvicorn worker processes
- `CHART_MAX_CONCURRENCY`: semaphore limit for heavy chart calculations in each worker
- `UVICORN_LIMIT_CONCURRENCY`: server-level concurrent connection cap (optional)
- `UVICORN_BACKLOG`: listen backlog
- `UVICORN_TIMEOUT_KEEP_ALIVE`: keep-alive timeout

## 3) Recommended tuning order

1. Fix worker count first (`WEB_CONCURRENCY`): start with CPU count and validate memory.
2. Tune chart semaphore (`CHART_MAX_CONCURRENCY`): start at `8`, then test `6/8/10/12`.
3. If p99 spikes or 5xx bursts appear, set `UVICORN_LIMIT_CONCURRENCY` to a safe cap.
4. Re-run the same load profile and compare p95/p99 and error rate.

## 4) Suggested baseline for staging

- `WEB_CONCURRENCY=2` on small instance, `3-4` on larger instances.
- `CHART_MAX_CONCURRENCY=8` as default baseline.
- Set `UVICORN_LIMIT_CONCURRENCY` to about `workers * CHART_MAX_CONCURRENCY * 2` and adjust after measurement.

## 5) Pass/Fail gates before production

- `/chart`: p95 < 500ms, error rate < 1%
- `/ai_reading`: p95 target should reflect OpenAI upstream latency budget
- No health-check failures during sustained load (10+ minutes)
