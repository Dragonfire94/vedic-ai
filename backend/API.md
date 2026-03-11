# Backend API Notes

## Runtime Source of Truth

- Current shipped product-specific runtime contract is `GET /ai_reading?product_type=life_cycle` baseline `contract_version=v1.4.0` / `render_profile=life_cycle_lite_v1`.
- Primary source of truth for that path is `PRD/PRODUCT_SPEC_PRD_v1_4_0.md`, `PRD/IMPLEMENTATION_CHECKLIST_v1_4_0.md`, and `PRD/release_evidence/v1_4_0/`.
- If legacy generic docs and PRD differ, prefer the v1.4.0 baseline documents for `product_type=life_cycle`.
- Omitting `product_type` still routes to the legacy generic `/ai_reading` behavior.

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

## GET `/ai_reading` (current runtime baseline contract: v1.4.0)

- Route shape remains query-string GET.
- Existing chart/query inputs remain additive.
- `product_type` is optional. Current shipped product-specific path is `product_type=life_cycle`.
- Normalizer accepts `life_cycle`, `yearly_forecast`, `compatibility`, but only `life_cycle` has a dedicated runtime contract today.
- If `product_type` is omitted, runtime stays on the legacy generic path.
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
- Current baseline response also guarantees:
  - top-level `product_type == "life_cycle"`
  - non-empty `polished_reading`
  - deterministic `reading` / `summary.structured_summary`
  - product-aware `ai_cache_key` and polished cache namespace isolation
- `life_cycle` polished markdown uses this exact H2 order:
  - `cover/meta`
  - `How to use 1p`
  - `인생 구조 한 장 요약`
  - `4단계 인생 구조`
  - `현재 위치`
  - `마하다샤 단계 목록`
  - `방법론 카드`
  - `valid_until 설명`
  - `CTA-lite`
  - `면책/윤리/데이터 보호`
- P1-only target sections (`인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`) are intentionally absent from the current baseline runtime.
- `/pdf` forwards the same `product_type`, personalization inputs, and cache alignment into the internal `get_ai_reading()` call so render/finalize/cache policy stays product-aware.
- Release review artifacts for the current baseline live under `PRD/release_evidence/v1_4_0/`.

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
