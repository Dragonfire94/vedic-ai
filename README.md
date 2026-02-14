# Vedic AI

## BTR Engine Stabilization (Signal Normalization)

### `backend/config/event_signal_mapping.json`
Maps frontend `event_type` values to internal engine profile keys.

```json
{
  "career_change": "career",
  "relationship": "relationship",
  "relocation": "relocation",
  "health": "health",
  "finance": "finance",
  "other": "other"
}
```

### `backend/config/event_signal_profile.json` schema
Single source-of-truth for event signal weights.

- `houses: number[]`
- `planets: string[]`
- `dasha_lords: string[]`
- `conflict_factors: string[]`
- `base_weight: number`

### Confidence calibration (Phase 4)
The engine now keeps a **raw confidence** value from scoring and applies a statistical post-processing layer to produce **calibrated confidence**.

- Raw confidence: direct result from event-matching score ratio logic
- Calibrated confidence: raw confidence adjusted for distribution uncertainty

Calibration features (computed from candidate score/probability distribution):

- `gap`: top-1 vs top-2 score separation
- `entropy`: `-Σ p_i log(p_i + 1e-9)` (higher means flatter, more uncertain)
- `score_variance`: normalized to `[0, 1]`
- `top_probability`: largest normalized candidate probability

Statistical safety rules prevent overconfidence:

- high entropy and small gap dampen confidence
- low top probability caps confidence
- only slight boost is allowed for very strong separation (`+0.05` max)
- final confidence is always clamped to `[0.05, 0.95]`

This helps avoid brittle, overconfident outputs when multiple time brackets are similarly plausible.

### Calibration log output (`production_mode=True`)
The engine emits structured JSON via logger `btr.calibration`:

```json
{
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "input_events": [{"event_type": "career_change", "precision_level": "exact", "year": 2010}],
  "normalized_scores": [{"raw_score": 9.2, "probability": 0.61}],
  "raw_confidence": 0.88,
  "calibrated_confidence": 0.83,
  "entropy": 1.12,
  "gap": 0.74,
  "top_probability": 0.61,
  "top_candidate_time_range": "00:00-03:00",
  "separation_gap": 0.12,
  "signal_strength_contributions": []
}
```

PII must not be logged.

### `tune_mode` usage and warning
`analyze_birth_time(..., tune_mode=True)` appends records to `data/tuning_inputs.log`, but only when environment variable `BTR_ENABLE_TUNE_MODE=1` is set.

> Do not expose tune mode in production without authorization controls.
