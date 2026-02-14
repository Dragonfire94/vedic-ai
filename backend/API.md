# Backend API Notes

## POST `/btr/analyze`

Request body includes:
- `year`, `month`, `day`, `lat`, `lon`
- `events`
- `tune_mode: bool` (default `false`)

`tune_mode` stores model-tuning payloads to `data/tuning_inputs.log` only when `BTR_ENABLE_TUNE_MODE=1`.
