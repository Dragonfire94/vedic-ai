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

## Life Cycle Baseline Fast Contract Check (dev loop only)

Run:

```bash
python -m pytest backend/test_life_cycle_lite_contract.py -q
```

Notes:
- This is a micro-fixture contract check for baseline `life_cycle-lite` meta/structure/P1-hidden guarantees.
- It is meant for fast local verification during development.
- It does not replace the baseline release gate above.

## Life Cycle Target Candidate Pack (pre-cutover)

Run:

```bash
python scripts/build_life_cycle_target_editorial_pack.py
```

Review artifacts:
- `PRD/release_evidence/v1_4_0/life_cycle_target_manual_qa.md`
- `PRD/release_evidence/v1_4_0/life_cycle_target_sample_response.json`
- `PRD/release_evidence/v1_4_0/life_cycle_target_gate_summary.json`
- `PRD/release_evidence/v1_4_0/life_cycle_target_release_manifest.json`
- `PRD/release_evidence/v1_4_0/life_cycle_target_editorial_rubric.md`
- `PRD/release_evidence/v1_4_0/life_cycle_target_editorial_review_20.md`
- `PRD/release_evidence/v1_4_0/life_cycle_target_editorial_case_matrix.json`

Notes:
- This pack is for internal target-report review before route cutover.
- `life_cycle_target_v1` is not yet the shipped `/ai_reading?product_type=life_cycle` render profile.
- Final cutover should be claimed only after target evidence is regenerated on a clean commit and a human spot-check confirms the editorial pack.

## Life Cycle Final Cutover Readiness Check

Run:

```bash
python scripts/check_life_cycle_target_cutover_ready.py
```

Notes:
- This command is expected to fail until the git worktree is clean and the Human Spot Check section is fully completed.
- After editing `life_cycle_target_manual_qa.md`, run `python scripts/refresh_life_cycle_target_manifest.py` once before the final evidence commit so manifest hashes stay aligned.
- A passing result means the target evidence pack, manifest hashes, gate summary, editorial case matrix, and human spot-check are all aligned for final cutover review.

## Frontend Repo-Wide Integration Check (dev / repo-wide follow-up)

Run:

```bash
cd frontend
npm run type-check
npm run test:e2e -- tests/e2e/btr-flow.spec.ts
```

Notes:
- This verifies that the frontend entry path, `/ai_reading` client contract, chart consumer, and Playwright E2E actually consume the current `product_type=life_cycle` baseline contract.
- The checked flow currently covers: home exact-time launch, BTR query pass-through, chart auto-load for `life_cycle`, and same-session client cache reuse.
- This is a repo-wide integration check, not the backend release gate source of truth.
- A passing result here does not imply `life_cycle_target_v1` cutover readiness; target cutover is still controlled by the target evidence pack and human spot-check.

## Clean Environment / CI Backend Setup

Run:

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend -q
```

Notes:
- `backend/requirements-dev.txt` is the clean-environment / CI install source of truth for backend tests.
- Runtime-only local app launch may still use `backend/requirements.txt`.
- Release decisions still use the gate commands and evidence pack documented above; the commands here are for reproducible backend test setup.

## Environment Notes

- `pytest` is included in `backend/requirements-dev.txt`; use the same interpreter for install and test commands.
- On the current Windows runner, `-p no:cacheprovider` is recommended to avoid `.pytest_cache` permission noise from the shared temp root.
- PDF scanning is not part of the current `life_cycle` baseline release decision.
