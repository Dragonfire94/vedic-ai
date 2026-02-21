# Phase11 Run4 Scorecard (Auto Snapshot)

- generated_at: 2026-02-21T23:25:06.470468
- mode: manual 3-sample regeneration (highest/lowest/balanced)
- gate_status: forbidden/style/timeout = pass (from fast gate)

## 1. highest_stability
- status: OK
- audit_score: 88
- forbidden_hits(PDF): 0
- style_errors: none
- directive_phrase_hits: 2
- soft_ban_residual: 12
- hard_ban_residual: 19
- pdf_path: logs\qa_phase11_run4\01_high_stability\highest_stability_20260221_232340.pdf
- text_path: logs\qa_phase11_run4\01_high_stability\highest_stability_20260221_232340.extracted.txt

## 2. lowest_stability
- status: OK
- audit_score: 88
- forbidden_hits(PDF): 0
- style_errors: none
- directive_phrase_hits: 8
- soft_ban_residual: 14
- hard_ban_residual: 13
- pdf_path: logs\qa_phase11_run4\02_low_stability\lowest_stability_20260221_232416.pdf
- text_path: logs\qa_phase11_run4\02_low_stability\lowest_stability_20260221_232416.extracted.txt

## 3. most_balanced
- status: OK
- audit_score: 88
- forbidden_hits(PDF): 0
- style_errors: none
- directive_phrase_hits: 5
- soft_ban_residual: 9
- hard_ban_residual: 15
- pdf_path: logs\qa_phase11_run4\03_balanced\most_balanced_20260221_232450.pdf
- text_path: logs\qa_phase11_run4\03_balanced\most_balanced_20260221_232450.extracted.txt

## Provisional QA Scores
- Note: current PDF/ai_text Korean text is mojibake-corrupted, so below is a provisional score from gate/style diagnostics.

### 1) highest_stability (provisional)
- Hook Quality: 4/5
- Narrative Rhythm: 2/5
- Humanized Language: 2/5
- Strategic Compression: 3/5
- Chapter Differentiation: 3/5
- Emotional Resonance: 2/5
- Structural Integrity Feel: 3/5
- Total: **19/35** (REWORK)

### 2) lowest_stability (provisional)
- Hook Quality: 4/5
- Narrative Rhythm: 2/5
- Humanized Language: 1/5
- Strategic Compression: 2/5
- Chapter Differentiation: 2/5
- Emotional Resonance: 2/5
- Structural Integrity Feel: 3/5
- Total: **16/35** (REWORK)

### 3) most_balanced (provisional)
- Hook Quality: 4/5
- Narrative Rhythm: 2/5
- Humanized Language: 2/5
- Strategic Compression: 3/5
- Chapter Differentiation: 3/5
- Emotional Resonance: 2/5
- Structural Integrity Feel: 3/5
- Total: **19/35** (REWORK)

## Aggregate (provisional)
- Average: **18.0 / 35**
- 판정: **REWORK**
- Priority:
  - Korean encoding/mojibake first (blocks reliable human QA)
  - hard/soft residual reduction (consulting-tone vocabulary)
  - directive phrase further suppression
