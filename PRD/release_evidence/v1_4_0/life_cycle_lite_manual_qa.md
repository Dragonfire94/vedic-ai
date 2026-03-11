# Life Cycle Lite Manual QA

- contract_version: v1.4.0
- release_evidence_dir: PRD/release_evidence/v1_4_0
- render_profile: life_cycle_lite_v1
- request_fingerprint: 490a7a73f1cd2658
- evidence_case_id: life_cycle_lite_baseline_primary
- commit_sha: 99b73d9cec19fe0d20f1545c2fa21b5b117f30ef
- release_manifest_path: PRD/release_evidence/v1_4_0/life_cycle_lite_release_manifest.json
- dirty_worktree: false

## Case 1: life_cycle_lite_baseline_primary

- input fixture / request summary: fake chart primary (`birth_jd=2447892.5`, `dasha_reference_jd=2461110.5`) + `GET /ai_reading?product_type=life_cycle&subject_name=민서&onboarding_goal=life_direction&focus_tokens=커리어,리듬&concern_tokens=우선순위,전환`
- expected points:
  - `meta.contract_version == v1.4.0`
  - `meta.render_profile == life_cycle_lite_v1`
  - `meta.valid_until_fallback == false`
  - `polished_reading` exact H2 order 10개 유지
  - target-only H2(`인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`) 미노출
  - gate summary `hard_fail_count == 0`
- actual summary:
  - `valid_until=2029-03-11`, `next_mahadasha_date=2031-03-11`, `current_mahadasha_planet=Moon`
  - headings: cover/meta, How to use 1p, 인생 구조 한 장 요약, 4단계 인생 구조, 현재 위치, 마하다샤 단계 목록, 방법론 카드, valid_until 설명, CTA-lite, 면책/윤리/데이터 보호
  - gate: `front_contract_ok=true`, `action_steps_contract_ok=true`, `life_cycle_release_ok=true`
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): 2026-03-11T17:02:50+09:00

## Case 2: life_cycle_lite_valid_until_fallback

- input fixture / request summary: fake chart fallback (`birth_jd=2447892.5`, `dasha_reference_jd=9999999.0`) + same request params
- expected points:
  - `meta.valid_until_fallback == true`
  - `meta.next_mahadasha_date == null`
  - `valid_until 설명`에 fallback 안내 문구 포함
  - target-only H2 미노출
- actual summary:
  - `valid_until=2029-03-11`, `next_mahadasha_date=None`, `valid_until_fallback=true`
  - headings: cover/meta, How to use 1p, 인생 구조 한 장 요약, 4단계 인생 구조, 현재 위치, 마하다샤 단계 목록, 방법론 카드, valid_until 설명, CTA-lite, 면책/윤리/데이터 보호
  - fallback sentence present: True
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): 2026-03-11T17:02:50+09:00
