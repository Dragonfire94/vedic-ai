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

### Empirical tuning (Phase 5)
Empirical tuning uses accumulated `data/tuning_inputs.log` (JSONL) to derive conservative base-weight multipliers per `event_type`.

Flow (admin-triggered only):
1. `analyze_tuning_data(...)` computes per-event statistics:
   - `avg_gap` (mean top-1 vs top-2 probability separation)
   - `avg_confidence` (mean calibrated confidence)
   - `event_count`
2. `compute_weight_adjustments(...)` generates multipliers:
   - `+5%` when `avg_gap > 1.5` and `avg_confidence > 0.7`
   - `-5%` when `avg_gap < 0.5`
   - otherwise `1.0`
3. `apply_weight_adjustments(...)` writes `backend/config/event_signal_profile_adjusted.json` without overwriting the original profile.

To trigger recalculation:
- Set `BTR_ENABLE_TUNE_MODE=1`
- Call `POST /btr/admin/recalculate-weights`

Safety constraints:
- No automatic recalculation on normal runs (admin endpoint only)
- Multipliers are capped to `±10%` (`0.9` to `1.1`)
- `base_weight` is never reduced below `0.1`

> Warning: Empirical tuning requires sufficient sample size (>100 events recommended).

## Deterministic Report Engine (Multi-Block)

The report pipeline now supports deterministic **multi-block selection per chapter**.
Selector behavior remains rule-based and does not add additional GPT calls.

### Template schema
All templates in `backend/report_templates/*.json` use:

```json
{
  "id": "high_tension_axis_behavior",
  "chapter": "Psychological Architecture",
  "conditions": [
    {"field": "psychological_tension_axis.score", "operator": ">=", "value": 70},
    {"field": "behavioral_risk_profile.impulsivity_risk", "operator": ">=", "value": 60}
  ],
  "logic": "AND",
  "priority": 95,
  "content": {
    "title": "High Tension with Impulsive Tendencies",
    "summary": "...",
    "analysis": "...",
    "implication": "..."
  }
}
```

Rules:
- `conditions` is always a list.
- `logic` supports `AND` / `OR` (defaults to `AND`).
- `priority` is numeric (defaults to `0`) and used for sorting.
- `content` is the only payload returned to GPT.

### Operator support

| Operator | Meaning |
|---|---|
| `==` | equal |
| `!=` | not equal |
| `<` | less than |
| `<=` | less than or equal |
| `>` | greater than |
| `>=` | greater than or equal |

### Nested field support
`field` accepts dot-path lookup, e.g. `behavioral_risk_profile.impulsivity_risk`.

### Selection model
1. Evaluate every block against structural summary conditions.
2. Keep all passing blocks per chapter.
3. Sort by `priority` descending.
4. Cap to maximum `5` blocks per chapter.
5. If no block matches, use JSON fallback blocks from `backend/report_templates/default_patterns.json`.

### Payload shape
`build_report_payload(...)` always returns all 15 chapters with list payloads:

```json
{
  "chapter_blocks": {
    "Executive Summary": [
      {
        "title": "...",
        "summary": "...",
        "analysis": "...",
        "implication": "..."
      }
    ]
  }
}
```

### Safety constraints
- Deterministic selection only.
- No additional GPT calls.
- Maximum 5 blocks per chapter.
- All 15 chapters always present.
- No raw astrological values (e.g., longitude/aspects) in report payload.

## Report Engine Depth Extension (Intensity, Chaining, Reinforcement)

The deterministic report engine now adds narrative depth without changing core BTR mechanics and without additional GPT calls.

### Extended template schema
Template blocks can now include optional density/linking keys:

```json
{
  "id": "high_tension_risk_aggro",
  "chapter": "Psychological Architecture",
  "conditions": [{"field": "psychological_tension_axis.score", "operator": ">=", "value": 70}],
  "logic": "AND",
  "priority": 95,
  "intensity_tiers": [
    {"threshold": 0.7, "modifier": "strong"},
    {"threshold": 0.4, "modifier": "moderate"}
  ],
  "chain_followups": ["tension_stability_interaction"],
  "content": {
    "title": "...",
    "summary": "...",
    "analysis": "...",
    "implication": "...",
    "examples": "..."
  }
}
```

### Intensity & depth definition
`compute_block_intensity(...)` normalizes intensity into `[0.0, 1.0]` from structural summary only:

- psychological tension signal
- averaged behavioral risk profile signal
- inverse stability index signal (`100 - stability_index`)

Intensity is used only to modulate narrative depth (field inclusion), not to expose raw engine signals.

### Cross-chapter reinforcement guide
`REINFORCE_RULES` adds deterministic linking blocks when specific block combinations are selected.

Current rules:
- `high_tension_risk_aggro` + `high_stability_fall` ⇒ add `tension_stability_interaction` to **Psychological Architecture**
- `career_conflict_karma` + `career_purushartha` ⇒ add `career_karma_pattern_reinforcement` to **Executive Summary**

### Chain followup behavior
After initial matching, each selected block can append same-chapter `chain_followups` by block id.

Safety:
- duplicate IDs are ignored
- cyclic references are prevented by seen-id tracking
- chapter payload still caps at 5 blocks

### Priority + intensity sort order
Selected blocks are sorted descending by:

| Order | Key |
|---|---|
| 1 | `priority` |
| 2 | `_intensity` |
| 3 | internal match order tie-break |

### Dynamic field rules
Per selected block:

- intensity `> 0.8`: include `title`, `summary`, `analysis`, `implication`, `examples`
- intensity `> 0.5`: include `title`, `summary`, `analysis`, `implication`
- otherwise: include `title`, `summary`, `analysis`

This increases report density only where structural pressure is stronger.

### Example expanded output block JSON
```json
{
  "title": "Tension–Stability Interaction Pattern",
  "summary": "Psychological pressure and stability dynamics interact in ways that amplify reactivity cycles.",
  "analysis": "When internal friction rises while stability weakens, attention narrows and recovery latency increases.",
  "implication": "Deliberate pacing and reset rituals become essential to protect judgment quality.",
  "examples": "You may notice conflict spillover from one domain into unrelated decisions unless decompression boundaries are enforced."
}
```

### Safety constraints
- No additional GPT calls
- No raw astrological positions in GPT payload
- No direct numeric signal leakage to final prose
- Deterministic selection/chaining only
- Max blocks per chapter = 5

## Insight Spike Generator (Phase 7)

Phase 7 adds deterministic high-impact narrative spike fragments that are inserted at the top of a chapter when both condition matching and intensity thresholds are satisfied.

### `insight_spike` schema

Template blocks in `backend/report_templates/*.json` may include this optional object:

```json
"insight_spike": {
  "text": "High-impact declarative sentence.",
  "min_intensity": 0.8
}
```

Rules:
- `text` is required and must be a string.
- `min_intensity` is required and must be a float between `0.0` and `1.0`.
- Spike text is injected only when the block matches and its computed intensity is `>= min_intensity`.
- Blocks without `insight_spike` are processed normally (backward compatible).

### Injection behavior

Inside `build_report_payload(...)`:
1. Selected blocks are evaluated for eligible spikes.
2. Spike texts are de-duplicated while preserving order.
3. Spikes are inserted first in chapter output as top-level fragments:

```json
{"spike_text": "..."}
```

4. Normal content fragments are appended after spikes.
5. The chapter cap remains deterministic and strict: maximum `5` total fragments (spikes + content).

### Deterministic and safe by design

- No additional GPT calls.
- No changes to BTR or structural engines.
- No structural numeric data is exposed in the payload.
- Chapter order and structure remain unchanged.

### Example block with spike

```json
{
  "id": "identity_control_paradox",
  "chapter": "Psychological Architecture",
  "conditions": [
    {"field": "psychological_tension_axis.score", "operator": ">=", "value": 75},
    {"field": "stability_metrics.stability_index", "operator": "<=", "value": 50}
  ],
  "logic": "AND",
  "priority": 90,
  "insight_spike": {
    "text": "Your inner drive to control outcomes paradoxically undermines your sense of agency.",
    "min_intensity": 0.8
  },
  "content": {
    "title": "Identity-Control Paradox",
    "summary": "...",
    "analysis": "...",
    "implication": "..."
  }
}
```
