# Phase 11 QA Summary

## Baseline Gate Snapshot
- ok: 2
- timeout: 0
- failed_no_llm: 0
- forbidden_hits_total: 0
- style_failures_total: 0
- retry_count: 0
- retried_profiles: []

## Execution Commands
- baseline_structural: `python -m backend.golden_sample_runner --mode structural`
- baseline_fast_gate: `python -m backend.fast_llm_gate --samples 2 --profile-mode extremes`
- pdf_generation: `internal python call: get_chart -> get_ai_reading -> pdf_service.generate_pdf_report`

## Sample Outputs
### highest_stability (OK)
- seed: 33
- stability_index: 12.71
- risk_factor: 0.1867
- opportunity_factor: 0.6067
- influence_score: 0.6217
- pdf_path: logs\qa_phase11\01_high_stability\highest_stability_20260221_220018.pdf
- text_path: logs\qa_phase11\01_high_stability\highest_stability_20260221_220018.extracted.txt
- forbidden_hits: 0
- label_warn_hits: []

### lowest_stability (OK)
- seed: 36
- stability_index: 0.76
- risk_factor: 0.5144
- opportunity_factor: 0.2744
- influence_score: 0.275
- pdf_path: logs\qa_phase11\02_low_stability\lowest_stability_20260221_220107.pdf
- text_path: logs\qa_phase11\02_low_stability\lowest_stability_20260221_220107.extracted.txt
- forbidden_hits: 0
- label_warn_hits: []

### most_balanced (OK)
- seed: 44
- stability_index: 6.05
- risk_factor: 0.3422
- opportunity_factor: 0.47
- influence_score: 0.4704
- pdf_path: logs\qa_phase11\03_balanced\most_balanced_20260221_220153.pdf
- text_path: logs\qa_phase11\03_balanced\most_balanced_20260221_220153.extracted.txt
- forbidden_hits: 0
- label_warn_hits: []

## Manual QA Scoring (To Fill)
- Hook Quality (1-5)
- Narrative Rhythm (1-5)
- Humanized Language (1-5)
- Strategic Compression (1-5)
- Chapter Differentiation (1-5)
- Emotional Resonance (1-5)
- Structural Integrity Feel (1-5)

## Layout Capture Checklist
- Save at least 1 screenshot for chart-page spacing (especially post-D9 area).