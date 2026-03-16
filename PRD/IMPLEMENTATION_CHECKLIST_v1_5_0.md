# IMPLEMENTATION_CHECKLIST v1.5.0 (Life Cycle Long-Form Commercial Promotion Audit)

- 기준 PRD: `PRODUCT_SPEC_PRD_v1_5_0.md`
- 점검 기준일: 2026-03-16
- 점검 범위: 현재 저장소의 `life_cycle` baseline/target 구현 상태, generic 장문 엔진 재사용 가능성, long-form adapter/render/evidence/checker 구현 필요 범위
- 목적: "`life_cycle_target_v1`를 출발점으로 해서, 실제 판매 가능한 `life_cycle_longform_v1` 장문 상업 리포트를 어떤 순서로 닫을지"를 코드 기준으로 고정하는 실행 체크리스트

---

## 0. 상태 표기

- `DONE`: 현재 코드에 재사용 가능한 구현이 이미 있음
- `PARTIAL`: 관련 구현은 있지만 `v1.5.0` long-form 상품 수준까지는 아님
- `TODO`: 구현이 필요함
- `BLOCKED`: 구현보다 먼저 계약/증빙/판정 조건을 해결해야 함
- `P1`: long-form prototype 이후 다루는 후속 범위

---

## 0.5 Current State vs Long-Form Target

- Current state: shipped runtime 기본 경로는 여전히 `life_cycle_lite_v1`이며, `product_type=life_cycle`는 deterministic baseline surface를 반환합니다.
- Current state: `life_cycle_target_v1`는 구조/게이트/evidence/manual QA 기준으로 이미 닫힌 pre-longform candidate입니다.
- Current state: `PRD/release_evidence/v1_4_0/` 아래 target sample/gate summary/manifest/manual QA는 실제로 존재하며, target candidate cutover readiness까지 닫힌 경험이 있습니다.
- Current state: generic 장문 엔진은 이미 `backend/report_engine.py`, `backend/llm_service.py`, generic `/ai_reading` 경로에 존재합니다.
- Current state: frontend/client/BTR/PDF 소비 경로는 `life_cycle` baseline contract를 실제로 소비하며 repo-wide follow-up과 E2E도 반영돼 있습니다.
- Current state: 반대로 `life_cycle_longform_v1` 전용 adapter/renderer/checker와 `PRD/release_evidence/v1_5_0/` 증빙 번들은 아직 없습니다.
- Current state: 따라서 현재 저장소는 "장문 엔진 없음" 상태가 아니라, "generic 장문 엔진을 life_cycle 상품 경로에 아직 연결하지 않은 상태"입니다.
- Target state: `life_cycle` 상품은 current Korean H2 order와 personalization 흐름을 유지한 채, 기존 generic 장문 엔진을 재사용하는 `life_cycle_longform_v1` surface를 생성합니다.
- Target state: `PRD/release_evidence/v1_5_0/` 아래 clean sample/gate summary/release manifest/manual QA가 실제로 존재하고, dedicated checker 기준 `READY`가 나와야 합니다.
- Target state: long-form이 닫히기 전까지 shipped route default는 유지되며, cutover는 separate decision으로만 다룹니다.

---

## 0.6 필수 리스크 8개 (2026-03-16 latest)

1. 현재 `life_cycle_target_v1`는 꽤 괜찮은 상업형 후보 리포트지만, 장문 상업 리포트 그 자체는 아닙니다.
2. generic 장문 엔진은 이미 있으나, `life_cycle` 상품 구조/H2 order/행동선과 바로 맞물리는 adapter 계층이 없습니다.
3. generic finalizer/front prepend 경로를 그대로 재사용하면 long-form `life_cycle` 표면이 generic front로 재오염될 수 있습니다.
4. polished cache/PDF/source identity는 `life_cycle_longform_v1` namespace가 분리되지 않으면 baseline/target과 충돌할 수 있습니다.
5. 기본 shipped route를 바꾸지 않은 채 long-form sample/evidence를 생성할 내부 review/evidence mode 진입점이 아직 없습니다.
6. `v1.4.0` evidence는 copy bank로는 유용하지만, `v1.5.0` long-form 완료 증빙을 대체하지 못합니다.
7. long-form 품질의 핵심 리스크는 엔진 부재보다 카피입니다. 특히 `현재 위치`, `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`를 bullet-heavy 톤에서 narrative paragraph로 승격해야 합니다.
8. long-form readiness는 자동 게이트만으로 닫히지 않습니다. clean evidence + human `YES` + dedicated checker `READY`가 모두 필요합니다.
9. frontend는 이미 baseline contract를 소비하지만, 실제 long-form cutover 전에도 chart/PDF/session cache/UI rhythm smoke를 확인해야 user-facing regression을 늦게 발견하지 않습니다.

---

## 0.7 핵심 결론

현재 프로젝트는 "장문 상업 리포트를 처음부터 새로 발명해야 하는 상태"가 아닙니다.

- 점성 계산 코어 있음
- generic 장문 리포트 엔진 있음
- `life_cycle` baseline/target 상품 구조 있음
- target candidate evidence/manual QA 경험 있음
- frontend consumer/PDF/E2E도 이미 baseline 기준으로 닫혀 있음

문제의 본질은 **엔진 개발 부재가 아니라 상품 연결 계층 부재**입니다.

즉, 이번 구현의 본질은 아래 7개입니다.

1. `life_cycle` payload를 generic 장문 엔진이 먹을 수 있는 블록 구조로 매핑
2. `life_cycle` 전용 long-form render profile/meta 계약 추가
3. generic finalizer/front 경로에서 product surface를 보호
4. long-form cache/PDF/hash/evidence identity 분리
5. long-form editorial/copy regression 기준 추가
6. `v1_5_0` sample/gate/manifest/manual QA/checker 생성
7. 그 이후에만 shipped route cutover 판단

---

> 주의: 아래 파일명(`life_cycle_longform_adapter.py`, `life_cycle_longform_renderer.py`, `build_life_cycle_longform_editorial_pack.py`, `check_life_cycle_longform_cutover_ready.py`)은 권장 구조입니다. 동일한 behavior / test / evidence 계약을 만족하면 다른 구조도 허용합니다.

## 1. 현재 프로젝트 스냅샷

| 영역 | 상태 | 현재 판단 |
|---|---|---|
| 점성 계산 코어 (`astro_engine`, `dasha_core`) | `DONE` | long-form에서도 그대로 재사용 |
| generic 장문 block 엔진 (`report_engine.py`) | `DONE` | 이미 상업형 장문 block 구성 가능 |
| generic LLM refinement (`llm_service.py`) | `DONE` | 장문 내러티브 리파인먼트 재사용 가능 |
| generic `/ai_reading` 장문 경로 | `DONE` | 기존 generic 상품에서는 실제 동작 |
| `life_cycle_lite_v1` baseline path | `DONE` | shipped runtime 기본 경로 |
| `life_cycle_target_v1` 구조형 candidate | `DONE` | pre-longform copy bank / 기준 샘플로 사용 가능 |
| `life_cycle_longform_v1` adapter | `TODO` | 권장 파일/동등 계층 모두 없음 |
| `life_cycle_longform_v1` renderer/surface shaping | `TODO` | 권장 파일/동등 계층 모두 없음 |
| long-form route branch/meta contract | `PARTIAL` | baseline/target contract는 있으나 long-form profile 없음 |
| long-form cache namespace | `TODO` | baseline/target/long-form 3중 분리 필요 |
| long-form PDF parity | `TODO` | narrative source selection 전용 검증 필요 |
| long-form evidence bundle (`v1_5_0`) | `TODO` | 폴더 자체 없음 |
| long-form readiness checker | `TODO` | 전용 스크립트 없음 |
| baseline/target evidence/checker 패턴 | `DONE` | `v1.4.0`에서 재사용 가능한 패턴 존재 |
| frontend baseline consumer | `DONE` | baseline life_cycle contract 실제 소비 |
| frontend long-form cutover smoke | `P1` | long-form runtime이 생긴 뒤 별도 확인 필요 |

---

## 2. 현재 코드에서 이미 있는 것

### 2.1 long-form 기반 인프라

`AUD-LF-01` `DONE` generic 장문 commercial prompt와 block builder 존재  
  - 근거: `backend/report_engine.py`
  - 의미: long-form을 위해 새 엔진을 발명할 필요는 없음

`AUD-LF-02` `DONE` generic LLM refinement / polished narrative 경로 존재  
  - 근거: `backend/llm_service.py`
  - 의미: chapter block 기반 장문 narrative 생성 경로가 이미 있음

`AUD-LF-03` `DONE` generic `/ai_reading` 런타임에서 report engine + refinement 조합이 실제 사용됨  
  - 근거: `backend/main.py`
  - 의미: life_cycle도 이 인프라를 재사용하는 방향이 현실적

### 2.2 life_cycle 상품 기반

`AUD-LF-04` `DONE` `life_cycle_lite_v1` baseline deterministic path 존재  
  - 근거: `backend/life_cycle_helpers.py`, `backend/life_cycle_lite_renderer.py`, `backend/main.py`
  - 의미: long-form에서도 deterministic source/meta를 재사용 가능

`AUD-LF-05` `DONE` `life_cycle_target_v1` 구조형 candidate와 copy bank 존재  
  - 근거: `backend/life_cycle_target_renderer.py`, `PRD/release_evidence/v1_4_0/`
  - 의미: long-form은 target 구조를 버리는 것이 아니라 narrative depth를 얹는 승격 단계

`AUD-LF-06` `DONE` target evidence/manual QA/readiness 운영 패턴 존재  
  - 근거: `scripts/build_life_cycle_target_editorial_pack.py`, `scripts/check_life_cycle_target_cutover_ready.py`
  - 의미: `v1_5_0` long-form도 같은 패턴으로 증빙 번들을 만들 수 있음

### 2.3 repo-wide 소비 경로

`AUD-LF-07` `DONE` frontend/chart/BTR/PDF baseline consumer와 E2E 존재  
  - 근거: `frontend/lib/api.ts`, `frontend/app/chart/ChartClient.tsx`, `frontend/tests/e2e/btr-flow.spec.ts`
  - 의미: long-form은 새 프론트 제품 개발보다 runtime response 교체/확인이 중심

`AUD-LF-08` `DONE` clean-env/CI 및 backend baseline pytest 계약 존재  
  - 근거: `backend/requirements-dev.txt`, `backend/QUALITY_GATES.md`, `README.md`
  - 의미: long-form 작업도 같은 clean-env 기준 위에서 검증 가능

---

## 3. 현재 코드에서 없는 것 또는 모자란 것

### 3.1 상품 연결 계층

`AUD-LF-09` `TODO` `life_cycle` -> `report_engine` adapter 부재  
  - 해야 할 일:
    - `life_cycle` payload를 chapter block 친화 구조로 매핑
    - target H2 order와 long-form paragraph strategy를 함께 보존
    - deterministic baseline/target payload를 evidence anchor로 주입

`AUD-LF-10` `TODO` `life_cycle_longform_v1` render profile/meta contract 부재  
  - 해야 할 일:
    - `render_profile=life_cycle_longform_v1`
    - `generation_mode`
    - `source_render_profile`
    - `narrative_profile`
    - `longform_cutover_candidate`
  - 주의: baseline/target contract와 혼동되면 안 됨

`AUD-LF-11` `TODO` long-form surface shaping 계층 부재  
  - 해야 할 일:
    - generic 장문 결과를 life_cycle H2 order와 문체 규칙에 맞게 최종 shape
    - bullet-heavy output을 narrative paragraph 중심으로 조정

### 3.2 런타임/표면 보호

`AUD-LF-12` `BLOCKED` generic finalizer/front 재오염 위험  
  - 근거: 현재 generic front prepend 경로는 reference일 뿐이며, long-form `life_cycle`은 product-specific 보호가 필요함
  - 해야 할 일:
    - long-form 경로에서 generic front 재부착 우회
    - product 전용 postprocess/finalize branch 추가 또는 동등 보호

`AUD-LF-13` `TODO` long-form polished cache namespace 부재  
  - 해야 할 일:
    - `product_type=life_cycle`
    - `render_profile=life_cycle_longform_v1`
    - generation mode/hash identity
    - baseline/target/long-form collision 방지

`AUD-LF-14` `TODO` long-form PDF narrative parity 검증 부재  
  - 해야 할 일:
    - `/pdf`가 long-form polished surface를 정확히 선택하는지 확인
    - JSON/sample/manual QA와 hash identity 일치 확인

`AUD-LF-14-a` `TODO` 기본 route 변경 없이 long-form을 생성할 내부 review/evidence mode 진입점 부재  
  - 해야 할 일:
    - shipped default는 `life_cycle_lite_v1`로 유지
    - 내부 호출/스크립트/명시적 render profile branch 중 하나로 long-form sample 생성 경로 추가
    - evidence 생성 스크립트가 이 진입점을 사용하도록 고정
  - 금지:
    - evidence를 만들기 위해 baseline shipped branch를 먼저 바꾸는 것

### 3.3 증빙/판정 계층

`AUD-LF-15` `TODO` `PRD/release_evidence/v1_5_0/` evidence bundle 부재  
  - 필수 산출물:
    - `life_cycle_longform_sample_response.json`
    - `life_cycle_longform_gate_summary.json`
    - `life_cycle_longform_release_manifest.json`
    - `life_cycle_longform_manual_qa.md`

`AUD-LF-16` `TODO` long-form editorial pack builder 부재  
  - 해야 할 일:
    - sample/manual QA/gate summary/manifest 생성
    - human spot-check block 포함
    - reviewer identity/hash 일치 보장

`AUD-LF-17` `TODO` long-form cutover readiness checker 부재  
  - 해야 할 일:
    - clean worktree 여부
    - evidence identity/hash
    - human `cutover_ready`
    - gate summary PASS
    - 최종 `READY/NOT READY` 판정

### 3.4 카피/품질 계층

`AUD-LF-18` `PARTIAL` target candidate 카피는 우수하지만 long-form narrative 기준은 아직 미달  
  - 현재 상태:
    - `인생 구조 한 장 요약`, `현재 위치`, `인생 고점/저점 지도`는 컷오버 직전 수준까지 옴
    - 그러나 장문 commercial narrative의 paragraph-driven 흐름은 아직 없음
  - 해야 할 일:
    - 핵심 4개 섹션을 bullet-heavy 구조에서 narrative paragraph 중심으로 승격
    - 설명문 톤/템플릿감/리스트 느낌 제거

`AUD-LF-19` `TODO` long-form editorial regression 테스트 부재  
  - 해야 할 일:
    - 첫 문장 자기인식 규칙
    - 반복 표현 억제
    - jargon 과노출 금지
    - 행동선 연결
    - 고점/저점 문단 맥락 문장 존재

---

## 4. 구현 순서 (권장)

### Phase A. 문서/계약 고정

`LF-01` `DONE` `PRODUCT_SPEC_PRD_v1_5_0.md` 작성  
`LF-02` `DONE` `IMPLEMENTATION_CHECKLIST_v1_5_0.md` 작성  
`LF-03` `TODO` `PRODUCT_SPEC_PRD_v1_5_0.md` / `IMPLEMENTATION_CHECKLIST_v1_5_0.md` / `CHANGELOG_PRD_v1_5_0.md`를 함께 커밋해 버전 맥락 보존

### Phase B. long-form prototype 연결

`LF-04` `TODO` `life_cycle_longform_adapter.py` 또는 동등 계층 추가  
`LF-05` `TODO` `report_engine` 기반 chapter block 생성 연결  
`LF-06` `TODO` `llm_service` refinement를 `life_cycle_longform_v1` profile로 호출  
`LF-07` `TODO` 기본 route를 바꾸지 않는 internal review/evidence mode 진입점 추가  
`LF-08` `TODO` sample 1건이 deterministic target보다 충분히 길고 풍부한지 확인

### Phase C. runtime/meta/cache/PDF 보호

`LF-09` `TODO` long-form render profile/meta contract 추가  
`LF-10` `TODO` long-form finalize branch 또는 동등 protection 추가  
`LF-11` `TODO` polished cache namespace 분리  
`LF-12` `TODO` `/pdf` narrative selection parity 고정

### Phase D. 카피/편집 승격

`LF-13` `TODO` `현재 위치` narrative paragraph 확장  
`LF-14` `TODO` `인생 고점/저점 지도` 맥락 문장/설득력 강화  
`LF-15` `TODO` `반복 패턴 분석`을 분석 보고서 톤에서 self-recognition 톤으로 승격  
`LF-16` `TODO` `다음 3년 구체화`를 분절 action list가 아니라 하나의 행동선으로 연결  
`LF-17` `TODO` long-form editorial regression 테스트 추가

### Phase E. evidence/checker/QA

`LF-18` `TODO` `scripts/build_life_cycle_longform_editorial_pack.py` 추가  
`LF-19` `TODO` `PRD/release_evidence/v1_5_0/` 번들 생성  
`LF-20` `TODO` `scripts/check_life_cycle_longform_cutover_ready.py` 추가  
`LF-21` `TODO` chart/PDF/session cache long-form smoke 및 최소 E2E 확인  
`LF-22` `TODO` clean evidence + human spot-check + checker `READY` 달성

### Phase F. cutover decision

`LF-23` `BLOCKED` shipped route default를 `life_cycle_longform_v1`로 바꿀지 최종 판단  
  - 선행 조건:
    - long-form evidence clean
    - human `cutover_ready: YES`
    - checker `READY`
    - chart/PDF/session cache smoke 및 최소 E2E 통과
    - baseline/target fallback 경로 보존

### Phase G. repo-wide 후속

`LF-24` `P1` long-form 기준 확장 E2E / visual QA / UI polish 보강  
`LF-25` `P1` cutover 이후 README/API/QUALITY_GATES source of truth 갱신

---

## 5. 테스트/검증 체크리스트

### 5.1 자동 테스트 (필수)

- `life_cycle_longform` adapter contract test
- long-form render profile routing test
- long-form cache namespace test
- long-form PDF narrative selection test
- long-form evidence identity/hash test
- long-form editorial regression test
- chart/PDF/session cache long-form smoke test
- backend baseline regression (`python -m pytest backend -q`) 재확인

### 5.2 샘플 검토 (필수)

- sample 1건: deterministic target보다 의미 있게 길고 풍부한가
- 핵심 3개 섹션 첫 문장이 설명문이 아닌가
- 고점/저점 문단이 리스트형 정보 전달을 넘어 맥락 문장을 가지는가
- `How to use -> 현재 위치 -> 다음 3년 -> valid_until -> CTA-lite` 행동선이 이어지는가

### 5.3 evidence 존재 체크 (필수)

- `PRD/release_evidence/v1_5_0/life_cycle_longform_sample_response.json`
- `PRD/release_evidence/v1_5_0/life_cycle_longform_gate_summary.json`
- `PRD/release_evidence/v1_5_0/life_cycle_longform_release_manifest.json`
- `PRD/release_evidence/v1_5_0/life_cycle_longform_manual_qa.md`

---

## 6. 현시점 우선순위 5개

1. `life_cycle_longform` adapter/prototype부터 만들기
2. generic finalizer/front 재오염 없이 long-form surface 보호하기
3. cache/PDF/meta identity를 long-form namespace로 분리하기
4. `v1_5_0` evidence/checker 생성 루프 닫기
5. 핵심 4개 섹션을 paragraph-driven commercial copy로 승격하기

---

## 7. 현재 미달 항목 (2026-03-16)

아래가 현재 실제로 비어 있습니다.

1. `backend/life_cycle_longform_adapter.py` 또는 동등 adapter
2. `backend/life_cycle_longform_renderer.py` 또는 동등 surface shaping 계층
3. `scripts/build_life_cycle_longform_editorial_pack.py`
4. `scripts/check_life_cycle_longform_cutover_ready.py`
5. `PRD/release_evidence/v1_5_0/` long-form evidence bundle
6. long-form cache/PDF/meta contract 테스트
7. long-form editorial regression 테스트

즉, `v1.5.0`은 현재 "현실적인 설계 문서 + 실행 체크리스트" 단계이며,
제품 구현과 증빙은 아직 시작 전입니다.
