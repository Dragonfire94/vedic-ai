# Backend API Notes

## Timezone Parameter Policy

- `timezone` query/body parameter means **UTC offset hours (float)** only.
- Valid examples: `9`, `9.0`, `-5`.
- Valid range: `-12.0` to `+14.0`.
- IANA timezone names (for example `Asia/Seoul`) are **not supported** in `timezone`.
- If `timezone` is omitted, the server auto-resolves offset from coordinates; if resolution fails, the API returns `400`.

## Temporal Cache + Transit Timing Contract

- `as_of_bucket` is normalized to month only: `m_YYYY-MM`.
- New cache keys never use day buckets (`d_YYYY-MM-DD`).
- Legacy day-bucket values are read-only compatibility at payload/meta normalization level.

Transit timing rows (`transits.timing_map`) are interval-based:
- `month_1.end_utc = month_2.start_utc - 1 second`
- `month_2.end_utc = month_3.start_utc - 1 second`
- `month_3.end_utc = month_4.start_utc - 1 second` (month_4 is a calculation-only anchor)

Only `month_*` keys are mapped into `timing_map`; non-month keys (for example `trend`) are excluded.

## POST `/btr/analyze`

Request body includes:
- `year`, `month`, `day`, `lat`, `lon`
- `events`
- `tune_mode: bool` (default `false`)

Router gate order:
- `BTR_ENABLED=0` -> endpoint disabled (`503`)
- `BTR_ENABLED=1` but engine import unavailable -> (`500`)
- otherwise continue normal validation/analysis

`tune_mode` is first gated in the router by `BTR_ENABLE_TUNE_MODE` and then forwarded to the engine as an effective value.

- requested: `tune_mode` from request body
- effective: `tune_mode && (BTR_ENABLE_TUNE_MODE=1)`
- when requested is `true` but env gate is off, the request is accepted but tuning is ignored (warning log is emitted)

Only the effective `true` value stores model-tuning payloads to `data/tuning_inputs.log`.
