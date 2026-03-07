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

## GET `/ai_reading` (v1.2.25 target public contract)

- Route shape remains query-string GET.
- Existing chart/query inputs remain additive; `product_type` is an optional query parameter.
- Allowed `product_type` values: `life_cycle`, `yearly_forecast`, `compatibility`.
- v1.2.25 P0 productization target is `product_type=life_cycle`; `yearly_forecast` and `compatibility` remain bugfix-only until separately shipped.
- `life_cycle` personalization inputs:
  - `subject_name: string`
  - `onboarding_goal: career_money | relationship | condition | life_direction`
  - `focus_tokens: list[string], max 2`
  - `concern_tokens: list[string], max 3`
  - `occupation_context: string`
  - `relationship_status: string`
- For GET requests, `focus_tokens` and `concern_tokens` are serialized as comma-separated query values (CSV), not repeated keys or JSON strings.
- If `onboarding_goal` is omitted or invalid, the normalized fallback value is `life_direction` and that normalized value is written into response meta.
- `life_cycle` response meta contract includes: `as_of_utc`, `as_of_local`, `timezone_offset`, `valid_until`, `valid_until_fallback`, `onboarding_goal`, `current_mahadasha_planet`, `next_mahadasha_date`, `product_type`, `contract_version`, `render_profile`.
- `valid_until` and `next_mahadasha_date` serialize as local dates (`YYYY-MM-DD`); `next_mahadasha_date` may be `null`.
- This section documents the v1.2.25 target public contract. Until productization lands in code, older runtimes may still return the legacy generic `/ai_reading` behavior.
- Current runtime still uses the legacy query signature and generic cache isolation; treat this section as target-state documentation, not shipped runtime behavior.

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

