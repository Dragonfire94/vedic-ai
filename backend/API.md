# Backend API Notes

## POST `/btr/analyze`

Request body includes:
- `year`, `month`, `day`, `lat`, `lon`
- `events`
- `tune_mode: bool` (default `false`)

`tune_mode` is first gated in the router by `BTR_ENABLE_TUNE_MODE` and then forwarded to the engine as an effective value.

- requested: `tune_mode` from request body
- effective: `tune_mode && (BTR_ENABLE_TUNE_MODE=1)`
- when requested is `true` but env gate is off, the request is accepted but tuning is ignored (warning log is emitted)

Only the effective `true` value stores model-tuning payloads to `data/tuning_inputs.log`.
