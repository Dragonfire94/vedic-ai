# Quality Gates

## Runtime Source of Truth

- Current release source of truth for `GET /ai_reading?product_type=life_cycle` is `PRD/PRODUCT_SPEC_PRD_v1_4_0.md`, `PRD/IMPLEMENTATION_CHECKLIST_v1_4_0.md`, and `PRD/release_evidence/v1_4_0/`.
- If legacy generic gate notes and PRD v1.4.0 differ, prefer the v1.4.0 baseline documents for `life_cycle` decisions.
- Legacy generic gates still exist for maintaining the older generic commercial surface.

## Legacy Generic Gates

### PR Gate (required for generic maintenance)

Run:

```bash
python -m backend.golden_sample_runner --mode structural
python -m backend.fast_llm_gate --samples 2 --profile-mode extremes
```

Fail conditions:
- `backend.fast_llm_gate` exits non-zero
- `forbidden_hits_total > 0` in `logs/golden_samples_fast_gate/fast_gate_summary.json`

Notes:
- `--profile-mode extremes` prioritizes `highest_stability` and `lowest_stability` for representative contrast.
- This gate is intentionally lighter than full golden run.
- PDF scanner is intentionally excluded from the generic PR gate.

### Nightly / Release Gate (required for generic maintenance)

Run generic PR Gate, then add:

```bash
python -m pytest backend/test_pdf_output_scanner.py -q
python -m backend.golden_sample_runner --mode full
```

Fail conditions:
- PDF scanner test fails
- Full golden run fails
- In Nightly/Release, PDF scanner is mandatory for the generic report path.

## Life Cycle Baseline Gate (required for v1.4.0 baseline)

Run:

```bash
python -m pytest backend/test_life_cycle_gate_metrics.py backend/test_cheap_validation_gate_metrics.py backend/test_life_cycle_helpers.py backend/test_life_cycle_lite_renderer.py backend/test_life_cycle_route_contract.py backend/test_llm_token_limits.py backend/test_vedic_technical_appendix.py -q -p no:cacheprovider
```

Review required release evidence:
- `PRD/release_evidence/v1_4_0/life_cycle_lite_manual_qa.md`
- `PRD/release_evidence/v1_4_0/life_cycle_lite_sample_response.json`
- `PRD/release_evidence/v1_4_0/life_cycle_lite_gate_summary.json`
- `PRD/release_evidence/v1_4_0/life_cycle_lite_release_manifest.json`

Fail conditions:
- Any listed test fails.
- `life_cycle_lite_gate_summary.json` shows `life_cycle_release_ok != true`.
- `hard_fail_count > 0`.
- `front_contract_ok != true` or `action_steps_contract_ok != true`.
- Evidence identity mismatch across `contract_version`, `render_profile`, `request_fingerprint`, `evidence_case_id`, `commit_sha`.
- Manifest SHA-256 values do not match the three referenced evidence files.

Notes:
- `life_cycle_lite` direct input scans use product-aware release semantics in `scripts/cheap_validation_gate.py`.
- Year/quarter timing hits are filtered only for `life_cycle_lite`; other forbidden-pattern rules remain active.
- Target-report sections are not part of the baseline gate and must remain absent in this cut.

## Environment Notes

- `pytest` must be installed in the same interpreter used for backend commands.
- On the current Windows runner, `-p no:cacheprovider` is recommended to avoid `.pytest_cache` permission noise from the shared temp root.
- PDF scanning is not part of the current `life_cycle` baseline release decision.
