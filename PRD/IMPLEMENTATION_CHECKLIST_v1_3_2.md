# IMPLEMENTATION_CHECKLIST v1.3.2 (Current Project Audit)

- 기준 PRD: `PRODUCT_SPEC_PRD_v1_3_2.md`
- 점검 기준일: 2026-03-11
- 점검 범위: 현재 저장소의 backend 중심 구현 상태, 테스트 환경, API 표면, 후처리/게이트, 문서-코드 불일치
- 목적: "무엇이 이미 구현되어 있고, 무엇이 아직 없으며, 무엇부터 손대야 하는가"를 코드 기준으로 잠그는 실행 체크리스트

---

## 0. 상태 표기

- `DONE`: 현재 코드에 재사용 가능한 구현이 이미 있음
- `PARTIAL`: 관련 구현은 있지만 PRD 기준 상품 수준까지는 아님
- `TODO`: 구현이 필요함
- `BLOCKED`: 구현보다 먼저 환경/계약 불일치를 해결해야 함
- `P1`: v1.3.2 범위 밖이므로 지금 구현하지 않음

---

## 0.5 Current State vs Target State

- Current state: runtime backend의 `/ai_reading` 시그니처에는 아직 `product_type` / `life_cycle` 개인화 입력이 없고, cache key에도 `product_type`가 없습니다.
- Current state: frontend 진입점과 client contract는 여전히 legacy `/chart` + BTR + `AIReadingResponse.polished_reading` 기준입니다.
- Current state: 현재 워크스페이스에서는 `python -m pytest`가 실행 가능하며, `python -m pytest backend -q -p no:cacheprovider` baseline은 `367 passed, 14 failed, 1 skipped`입니다.
- Current state: 위 baseline은 현재 로컬 dirty worktree(`backend/test_llm_token_limits.py` 정렬분 + PRD 문서 수정분)를 전제로 하므로, 구현 전에 baseline 정렬분을 먼저 커밋해 재현 기준을 고정하는 것이 안전합니다.
- Current state: `backend/API.md`, `backend/QUALITY_GATES.md`, `README.md`가 아직 정렬되기 전까지 현재 runtime / release contract의 interim source of truth는 PRD v1.3.2입니다.
- Current state: generic `_finalize_ai_reading_result()` -> `_build_polished_reading_surface()` -> `prepend_front_modules()` 경로가 여전히 generic front modules를 다시 붙일 수 있어, `life_cycle-lite` renderer를 추가해도 상품 표면이 오염될 수 있습니다.
- Current state: polished narrative cache는 아직 `chapter_blocks_hash + language` 기준이며 `/pdf`는 `get_ai_reading()`을 직접 호출하므로, route cache key만 분리해서는 product isolation이 닫히지 않습니다.
- Current state: 운영 문서의 release gate source of truth는 아직 `golden_sample_runner` / `fast_llm_gate` 기준이고, cheap gate는 아직 generic metric semantics 중심입니다.
- Target state: v1.3.2 P0는 backend-only `life_cycle-lite` productization 계약을 잠그는 문서이며, repo-wide 완료 주장은 README/frontend migration까지 반영된 뒤에만 가능합니다. 단, backend-only라도 product-aware finalizer/cache/PDF contract는 함께 닫혀 있어야 합니다.

---

## 0.6 필수 리스크 7개 (2026-03-11 latest)

1. 현재 워크스페이스에서는 `python -m pytest`가 실행 가능하지만, backend baseline에는 stale contract test cluster와 temp write permission cluster가 남아 있습니다.
2. 런타임 `/ai_reading` API와 route cache key는 아직 target contract가 아니며, `product_type` / `life_cycle` 요청 스키마와 cache isolation이 미구현입니다.
3. generic `_finalize_ai_reading_result()` / `_build_polished_reading_surface()` / `prepend_front_modules()` 경로가 여전히 generic front modules를 다시 붙일 수 있어 `life_cycle-lite` 전용 표면 격리가 보장되지 않습니다.
4. polished narrative cache는 아직 `chapter_blocks_hash + language` 기준이고, `/pdf`는 `get_ai_reading()`을 직접 호출하므로 product-aware cache namespace와 PDF contract를 같은 단계에서 닫아야 합니다.
5. frontend는 아직 `life_cycle-lite`를 노출하거나 소비할 수 없으므로, backend-only 실험이 아니라 repo-wide 적용 기준이면 blocker입니다.
6. 출고 gate source of truth가 아직 옛 기준이므로, `life_cycle-lite` 계약을 검증하지 못한 채 구 게이트만 통과하고 출고될 위험이 있습니다.
7. `release_evidence` reviewer entrypoint가 없으면 같은 버전/커밋이라도 서로 다른 `render_profile` / `request_fingerprint` 기준 산출물이 섞여 최종 sign-off가 흔들릴 수 있습니다.

---

## 0.7 핵심 결론

현재 프로젝트는 "life_cycle-lite P0를 새로 시작해야 하는 상태"가 아니라, 아래 조합이 이미 있는 상태입니다.

- 엔진 코어 있음
- generic AI reading 파이프라인 있음
- polished surface / cache / cheap gate 있음
- 다샤 timing 보조 컨텍스트 있음

문제는 상품 계층이 없다는 것입니다.

즉, 이번 구현의 본질은 엔진 개발이 아니라 아래 5개입니다.

1. 상품 오케스트레이션 추가
2. `life_cycle-lite` 전용 deterministic payload/renderer 추가
3. product-aware finalizer / front 격리
4. 메타 + cache + PDF 계약 고정
5. 테스트 baseline + micro fixture 정리

이 5개를 먼저 닫으면 P0는 충분히 현실적인 범위입니다.

---

> 주의: 아래 파일명(`product_orchestrator.py`, `render_contract.py`, `life_cycle_helpers.py`, `commercial_gate_helpers.py`)은 **권장 구조**입니다. 동일한 behavior / test / gate 계약을 만족하면 다른 구조도 허용합니다.

## 1. 현재 프로젝트 스냅샷

| 영역 | 상태 | 현재 판단 |
|---|---|---|
| 다샤 계산 코어 | `DONE` | 재사용 가능 |
| generic AI reading 파이프라인 | `DONE` | 이미 운영 가능한 수준 |
| chapter_blocks -> 상업 표면 렌더 | `DONE` | generic 12챕터 기준으로 존재 |
| front modules / front contract / cheap gate | `DONE` | generic 리포트 기준으로 존재 |
| life_cycle 관련 LLM 보조 흐름 | `PARTIAL` | Current Phase 보강 수준만 존재 |
| `life_cycle-lite` 전용 상품 경로 | `TODO` | 현재 없음 |
| `product_type` 기반 오케스트레이션 | `TODO` | 현재 없음 |
| full P0 meta contract / personalization request contract | `TODO` | 런타임 backend에는 없음 |
| `commercial_quality_constants.py` 상수층 | `PARTIAL` | 파일은 존재하지만 `life_cycle` shared constants는 아직 없음 |
| `commercial_gate_helpers.py` | `TODO` | 권장 helper 분리 기준으로는 파일 없음 (동등 구현 허용) |
| `life_cycle_helpers.py` | `TODO` | 권장 pure-function 분리 기준으로는 파일 없음 (동등 구현 허용) |
| `product_orchestrator.py` | `TODO` | 권장 분리 기준으로는 파일 없음 (동등 구현 허용) |
| `render_contract.py` | `TODO` | 권장 메타 계약 분리 기준으로는 파일 없음 (동등 구현 허용) |
| `pytest` 실행 환경 | `DONE` | 현재 워크스페이스 인터프리터에서 표준 명령 실행 가능 |
| backend 전체 pytest baseline | `PARTIAL` | `367 passed / 14 failed / 1 skipped` (`-p no:cacheprovider`) |
| stale contract test cluster | `BLOCKED` | atomic/prompt/pdf/report_engine 계열 기대값 재정렬 필요 |
| temp write permission cluster | `BLOCKED` | tuning analyzer / tuning mode file creation 계열 `PermissionError` |
| generic finalizer / front isolation | `BLOCKED` | 전용 renderer만 추가해도 generic front가 다시 붙을 수 있음 |
| polished narrative cache namespace | `TODO` | `chapter_blocks_hash + language`만으로는 상품 격리 부족 |
| frontend/client migration note | `PARTIAL` | 체크리스트 기준으로만 정렬됨. `backend/API.md` / frontend 구현 / `README.md` 반영 전까지 repo-wide 적용 blocker |
| release evidence / reviewer manifest | `TODO` | 증적 4종을 같은 `render_profile` / `request_fingerprint` 기준으로 묶는 reviewer entrypoint 없음 |
| micro fixture 체계 | `TODO` | generic fixture 1개만 있음 |
| token budget v1.3.2 정렬 | `PARTIAL` | PRD는 정렬되었지만 코드 product-layer 적용은 미구현 |
| yearly/compat bugfix-only 보호 | `TODO` | 문서상 합의만 있고 코드 분기 없음 |

---

## 2. 현재 코드에서 이미 있는 것

### 2.1 엔진 / 타이밍 코어

`AUD-01` `DONE` deterministic Vimshottari dasha 계산 코어 존재
   - 근거: `backend/dasha_core.py:77`, `backend/dasha_core.py:123`, `backend/dasha_core.py:159`
   - 활용 방식: `life_cycle-lite`는 이 코어를 직접 재사용해야 함
   - 의미: P0에서 엔진 산식을 새로 만들 필요 없음

`AUD-02` `PARTIAL` dasha narrative context helper 존재
   - 근거: `backend/report_engine.py:2176`
   - 현재 상태: `mahadasha`, `antardasha`, intensity, pressure 정도만 제공
   - 부족한 점: `next_mahadasha_date`, `valid_until_fallback`, 4단계 life stage, 현재 위치용 계약이 없음

`AUD-03` `PARTIAL` life timeline LLM 보조 함수 존재
   - 근거: `backend/llm_service.py:2493`, `backend/llm_service.py:2561`, `backend/llm_service.py:3170`
   - 현재 상태: generic `Current Phase` 챕터를 더 자연스럽게 다시 쓰는 용도
   - 부족한 점: `life_cycle-lite` 상품 전체 렌더러가 아님
   - 결론: 재사용 후보는 맞지만, 상품 구현 완료로 간주하면 안 됨

### 2.2 AI reading / 표면 / 캐시

`AUD-04` `DONE` `/ai_reading` 엔드포인트 존재
   - 근거: `backend/main.py:2872`
   - 현재 상태: generic 12챕터 AI reading 중심
   - 의미: 새 상품은 이 경로를 확장하거나 thin wrapper를 두는 방식이 현실적

`AUD-05` `DONE` `chapter_blocks_hash` / `polished_reading` 캐시 경로 존재
   - 근거: `backend/main.py:340`, `backend/main.py:345`, `backend/main.py:354`, `backend/main.py:3114`
   - 의미: PRD의 단일 LLM 호출 + 캐시 재사용 전략과 잘 맞음
   - 주의: 현재 polished cache key는 `chapter_blocks_hash + language`만 사용하므로, 상품 namespace를 추가하지 않으면 `life_cycle-lite`와 generic 경로가 충돌할 수 있음

`AUD-06` `DONE` commercial surface builder 존재
   - 근거: `backend/main.py:2732`, `backend/main.py:2764`, `backend/main.py:2800`
   - 현재 상태: chapter blocks 기반 polished surface 조립 가능
   - 부족한 점: `life_cycle-lite` 전용 구조가 아님
   - 관련 상수층: `backend/commercial_quality_constants.py:57` 존재
   - 의미: PRD lock constant를 둘 파일을 새로 만들 필요는 없음
`AUD-07` `DONE` generic commercial markdown renderer 존재
   - 근거: `backend/commercial_surface_renderer.py:557`
   - 현재 상태: 12챕터 generic 구조를 deterministic markdown으로 렌더
   - 테스트 근거: `backend/test_commercial_surface_renderer.py:6`
   - 부족한 점: `life_cycle-lite`의 4단계 구조/valid_until UX/방법론 카드/CTA-lite 전용 구조는 없음

`AUD-08` `DONE` front modules prepend 가능
   - 근거: `backend/commercial_surface_renderer.py:132`, `backend/main.py:2800`
   - 의미: generic front 개념은 이미 있음
   - 주의: `life_cycle-lite`는 front contract를 그대로 재사용할지, 별도 product front를 둘지 결정이 필요하며, 현재 generic finalizer/prepend 경로를 그대로 타면 product-specific H2 contract가 오염될 수 있음
### 2.3 품질 게이트 / cheap gate / 리포트 품질 스캔

`AUD-09` `DONE` cheap validation gate와 true-path surface metrics 존재
   - 근거: `scripts/cheap_validation_gate.py:793`
   - 현재 상태: scored surface, front contract, actionable coverage, postprocess alignment 측정 가능
   - 테스트 근거: `backend/test_cheap_validation_gate_metrics.py:177`, `backend/test_cheap_validation_gate_metrics.py:218`, `backend/test_cheap_validation_gate_metrics.py:373`

`AUD-10` `DONE` fallback front 적용 탐지 로직 존재
    - 근거: `scripts/cheap_validation_gate.py:532`
    - 의미: `front_contract_fallback_applied` 같은 front 계약을 추적하는 방식은 이미 있음
    - 부족한 점: `life_cycle-lite`의 `valid_until_fallback`과는 다른 계약임

`AUD-11` `DONE` quality gate 문서와 golden runner 존재
    - 근거: `backend/QUALITY_GATES.md`, `backend/golden_sample_runner.py:67`, `backend/golden_sample_runner.py:398`
    - 의미: 완전히 처음부터 QA 체계를 만들 필요는 없음
    - 주의: v1.2.x는 golden 중심이 아니라 micro fixture 중심으로 가야 함

---

## 3. 현재 코드에서 없는 것 또는 모자란 것

### 3.1 상품 오케스트레이션 / 계약 계층

`AUD-12` `TODO` `product_type` 기반 단일 분기 없음
    - 근거: runtime backend code 기준 `product_type` 분기 매치 없음 (`backend/API.md` 문서 제외)
    - 현재 상태: `/ai_reading` 단일 generic 상품만 사실상 존재
    - 해야 할 일:
      - 요청 입력에 `product_type` 추가
      - `life_cycle` 전용 개인화 입력(`subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status`) 추가/정규화
      - `life_cycle` / `yearly_forecast` / `compatibility` 분기 단일화
      - P0에서는 `life_cycle`만 productized, 나머지는 bugfix-only 유지

`AUD-13` `TODO` 권장 분리 기준에서 `product_orchestrator.py` 부재
    - 근거: `backend/product_orchestrator.py` 파일 없음
    - 해야 할 일:
      - thin orchestration 레이어 추가
      - main route에서 상품별 분기 규칙이 다시 흩어지지 않게 고정

`AUD-14` `TODO` 권장 분리 기준에서 `render_contract.py` 부재
    - 근거: `backend/render_contract.py` 파일 없음
    - 해야 할 일:
      - 응답 메타 계약 (`as_of_utc`, `as_of_local`, `timezone_offset`, `valid_until`, `valid_until_fallback`, `onboarding_goal`, `current_mahadasha_planet`, `next_mahadasha_date`, `product_type`, `contract_version`, `render_profile`) 정의
      - `valid_until`, `next_mahadasha_date` local date 직렬화 규칙(`YYYY-MM-DD`) 고정
      - PDF/JSON/cheap gate가 같은 계약을 보게 만들 것

`AUD-15` `TODO` 권장 분리 기준에서 `commercial_gate_helpers.py` 부재
    - 근거: `backend/commercial_gate_helpers.py` 파일 없음
    - 역할: HF 게이트 helper 전용 모듈
    - 해야 할 일:
      - `parse_sections()`
      - `check_empathy()`
      - `check_name_token_exposure()`
      - 기타 섹션 9 계열 helper를 `cheap_validation_gate.py`에서 분리할 위치로 고정
    - 금지:
      - `PLANET_LABEL_MAP`
      - `assign_life_stages()`
      - `compute_valid_until_lifecycle()`

`AUD-15-a` `BLOCKED` generic finalizer가 `life_cycle-lite` 표면을 다시 generic front로 감쌀 수 있음
    - 근거: `backend/main.py:2732`, `backend/main.py:2805`, `backend/commercial_surface_renderer.py:132`
    - 현재 상태: `_finalize_ai_reading_result()`는 generic `_build_polished_reading_surface()` 경로를 통해 front modules를 다시 붙일 수 있음
    - 해야 할 일:
      - `product_type == life_cycle` 경로에서 product-aware finalize branch 추가
      - `life_cycle-lite` renderer 출력이 generic `prepend_front_modules()`에 다시 들어가지 않도록 고정

`AUD-15-b` `TODO` polished narrative cache namespace가 product-aware가 아님
    - 근거: `backend/main.py:336`, `backend/main.py:345`, `backend/main.py:3501`
    - 현재 상태: polished narrative cache는 `chapter_blocks_hash + language` 기준이며 `product_type` / `render_profile`를 보지 않음
    - 해야 할 일:
      - route cache key와 별개로 polished cache namespace에도 `product_type`와 stable product render identity를 반영
      - `/pdf` 재사용 경로와 동일 규칙 사용

 life_cycle-lite 상품 자체

`AUD-16` `TODO` `life_cycle-lite` 전용 markdown renderer 없음
    - 현재 상태: generic renderer만 존재
    - 해야 할 일:
      - 4단계 인생 구조
      - 현재 위치 표시
      - 마하다샤 단계 목록
      - 방법론 카드
      - valid_until fallback UX
      - CTA-lite
    - 권장 구현 위치:
      - `backend/life_cycle_lite_renderer.py` 신설 또는
      - `backend/render_contract.py` + dedicated render function

`AUD-17` `TODO` 4단계 life stage grouping 로직 없음
    - 현재 상태: PRD 요구 pure function 부재
    - 재사용 입력: `backend/dasha_core.py`의 mahadasha list
    - 해야 할 일:
      - 9개 마하다샤를 소비자용 4단계로 그룹화
      - 각 단계 start/end/year span 계산
      - 현재 시점이 어느 단계인지 표시
    - 권장 구현 위치: `backend/life_cycle_helpers.py`

`AUD-18` `TODO` `next_mahadasha_date` / `valid_until` lifecycle contract 없음
    - 현재 상태: generic `build_dasha_narrative_context()`는 next transition date를 제공하지 않음
    - 해야 할 일:
      - 현재 maha end date 추출
      - `min(as_of_local + 3년, next_mahadasha_date)` 계산
      - null/past일 때 fallback + soft UX 문구
    - 권장 구현 위치: `backend/life_cycle_helpers.py`

`AUD-19` `TODO` `PLANET_LABEL_MAP` / consumer label mapping shared constants 없음
    - 현재 상태: `backend/commercial_quality_constants.py`에 PRD 수준 lock constant 없음
    - 해야 할 일:
      - 9개 행성 코드 매핑을 shared constant로 고정
      - `VALID_PLANET_CODES`, `PLANET_DOMAIN_MAP`까지 같은 파일에서 함께 정의
      - renderer는 이 맵만 사용

`AUD-20` `TODO` `LIFE_STAGE_LABELS` / `TRANSITION_INTENSITY_THRESHOLDS` shared constants 없음
    - 현재 상태: `backend/commercial_quality_constants.py`에 4단계 라벨/전환 강도 상수 계약이 없음
    - 해야 할 일:
      - stage label set과 전환 강도 기준을 같은 상수 파일에 고정
      - `life_cycle_helpers.py`는 상수를 정의하지 말고 import만 사용
      - test fixture와 gate에서 동일 상수 사용

`AUD-21` `P1` 고점/저점 지도, 반복 패턴, 다음 3년 구체화
    - 현재 상태: P0 대상 아님
    - 해야 할 일: 지금 구현하지 않음
    - 주의: 함수 이름이나 stub은 만들 수 있으나 제품에 노출하면 안 됨

### 3.3 메타 계약 / 응답 스키마

`AUD-22` `TODO` full P0 meta contract가 실제 backend 응답에 없음
    - 근거: runtime backend response path 기준 `as_of_local`, `onboarding_goal`, `next_mahadasha_date`, `contract_version`, `render_profile`, `product_type` 응답 매치 없음 (`backend/API.md` 문서 제외)
    - 해야 할 일:
      - JSON 응답 최상위 또는 `meta` 아래에 full field set 고정
      - `current_mahadasha_planet`, `next_mahadasha_date`는 값 없을 때도 키를 유지하고 `null` 허용 여부를 동일 계약으로 맞출 것
      - PDF 변환 경로에서도 유지
      - cheap gate summary에 같이 남기기

`AUD-23` `TODO` `valid_until_fallback` 전용 flag 없음
    - 현재 상태: generic `fallback`은 LLM fallback이고, `front_contract_fallback_applied`는 front module fallback 전용 metric임
    - 선행 감사:
      - 기존 `fallback`, `front_contract_fallback_applied` 사용처를 먼저 grep으로 분리 기록
      - life_cycle 전용 `valid_until_fallback`과 generic fallback semantics가 섞이지 않게 audit note 남기기
    - 해야 할 일:
      - life_cycle 전용 메타 키를 `valid_until_fallback`으로 고정
      - 응답 `meta.valid_until_fallback`과 `gate_summary.valid_until_fallback`가 같은 boolean을 보게 만들 것
      - cheap gate / 수동 QA / 로그가 모두 같은 키만 보게 만들 것

`AUD-24` `TODO` `render_profile` 버전 문자열 전략 없음
    - 해야 할 일:
      - 예: `life_cycle_lite_v1`
      - 렌더 profile 바뀔 때만 값 변경
      - QA 로그와 응답 payload에 동일하게 남길 것

### 3.4 테스트 / 환경 / fixture

`AUD-25` `PARTIAL` `python -m pytest` 실행은 가능하지만 release blocker가 남아 있음
    - 근거: `python -m pytest --version` -> `pytest 9.0.2`
    - 근거: `python -m pytest backend/test_llm_token_limits.py -q` -> `4 passed`
    - 근거: `python -m pytest backend -q -p no:cacheprovider` -> `367 passed, 14 failed, 1 skipped`
    - 의미: 표준 명령 자체는 재현되지만, suite 정리와 runner baseline 고정이 아직 남아 있음
    - release 기준: test runner 부재 이슈는 닫혔지만, stale cluster / temp write permission cluster가 남아 있으면 v1.3.2 P0 sign-off를 닫지 않음
    - 해야 할 일:
      - backend baseline failure inventory를 문서 기준으로 잠글 것
      - temp/cache 권한 의존 여부를 표준 runner 명령과 분리 기록할 것
      - clean-environment 설치/실행 문서와 CI 명령을 현재 기준으로 고정할 것

`AUD-26` `BLOCKED` stale contract test cluster 정렬 필요
    - 근거: 2026-03-11 baseline(`python -m pytest backend -q -p no:cacheprovider`)에서 14 fail 중 다수가 stale contract 계열임
    - 이미 정리된 항목: `backend/test_llm_token_limits.py`는 현재 런타임 계약 기준으로 수정되어 개별 통과함
    - 남은 대표 범주:
      - `backend/test_atomic_dominance.py`
      - `backend/test_llm_refinement_pipeline.py`
      - `backend/test_markdown_flowable_parser.py`
      - `backend/test_pdf_narrative_selection.py`
      - `backend/test_report_engine_insight_spike.py`
      - `backend/test_report_engine_korean_localization.py`
      - `backend/test_report_engine_psychological_depth.py`
    - 해야 할 일:
      - Phase A에서는 failure inventory와 deferred rewrite 범위를 먼저 잠글 것
      - product-layer touched 범위 밖 generic stale test 재작성은 orchestrator/product path 이후로 미룰 것
      - product-layer 정책은 global constant보다 product contract 기준으로 검사

`AUD-26-a` `BLOCKED` temp write permission cluster 정리 필요
    - 근거: `backend/test_tuning_analyzer.py`, `backend/test_tuning_mode_file_creation.py`는 현재 temp root에서 `PermissionError [WinError 5]`로 실패
    - 현재 재현: `tempfile.TemporaryDirectory()`가 `C:\Users\Public\Documents\ESTsoft\CreatorTemp\...` 아래를 사용하며 `write_text()` / `mkdir()`가 막힘
    - 해야 할 일:
      - 테스트에서 repo writable temp root를 명시적으로 사용하거나
      - 러너 환경의 temp path 권한을 표준화할 것


`AUD-27` `TODO` life_cycle-lite 전용 micro fixture 부재
    - 근거: `backend/tests/fixtures`에는 `chapter_blocks_pre_llm_sample.json` 1개만 존재
    - 해야 할 일:
      - 최소 2개 fixture 추가
      - 정상 경로 1개
      - valid_until_fallback=True 경로 1개
      - 가능하면 현재 maha 전환이 임박한 케이스 1개 추가

`AUD-28` `TODO` life_cycle-lite 전용 contract test 부재
    - 현재 상태: generic surface/gate 테스트는 있으나 상품별 테스트 없음
    - 해야 할 일:
      - meta keys 존재 검사
      - 4단계 구조 개수 검사
      - current stage single-mark 검사
      - `valid_until` fallback UX 검사
      - CTA-lite 존재 검사

`AUD-29` `TODO` `cheap_validation_gate`에 life_cycle-lite release mode 없음
    - 현재 상태: generic front-contract/12챕터 기준
    - P0 출고 기준: 이 항목이 닫히기 전에는 HF 16개 회귀 없음 판정을 닫을 수 없음
    - 해야 할 일:
      - 새 상품용 lightweight scan mode 추가
      - `valid_until_fallback`를 response meta와 같은 boolean으로 관찰 지표에 기록
      - HF2/HF3는 전용 product-specific helper에서 계산하고 generic gate 반환값을 source of truth로 재사용하지 말 것

### 3.5 PDF / 전달 채널

`AUD-30` `PARTIAL` `/pdf` 엔드포인트는 존재하나 상품-aware contract는 아님
    - 근거: `backend/main.py:3556`
    - 현재 상태: generic ai_reading narrative를 PDF로 내보내는 구조이며, 내부적으로 `get_ai_reading()`을 직접 호출함
    - 해야 할 일:
      - `life_cycle-lite` payload를 PDF로 넘길 때 메타/유효기간/방법론 카드/CTA-lite가 그대로 보존되는지 확인
      - 필요 시 PDF template 분기 추가

`AUD-31` `TODO` PDF 선택 경로에 `product_type` 반영 필요
    - 현재 상태: `ai_cache_key` 기반 generic reuse이며, route-level key와 별개로 polished narrative cache namespace도 generic임
    - 해야 할 일:
      - 동일 chart라도 상품이 다르면 cache/pdf 키도 달라져야 함
      - `/pdf` -> `get_ai_reading()` direct call path가 `life_cycle` request normalization / finalize / cache policy를 같은 규칙으로 타게 만들 것

---

## 4. 구현 전에 먼저 고정할 최소 아키텍처
### 4.1 권장 구조 (현재 프로젝트에 맞는 최소 분리)

1. `backend/product_orchestrator.py`
   - 역할: `product_type` 기준 thin branch
   - 원칙: `main.py`에서 상품별 if/else 확산 방지

2. `backend/render_contract.py`
   - 역할: 공통 메타 계약 생성
   - 포함: `as_of_utc`, `as_of_local`, `timezone_offset`, `valid_until`, `valid_until_fallback`, `onboarding_goal`, `current_mahadasha_planet`, `next_mahadasha_date`, `product_type`, `contract_version`, `render_profile`

3. `backend/life_cycle_helpers.py`
   - 역할: `life_cycle-lite` pure function 모음
   - 포함: stage grouping, current stage 계산, valid_until lifecycle, P1 stub
   - 주의: `PLANET_LABEL_MAP`, `VALID_PLANET_CODES`, `PLANET_DOMAIN_MAP`, `LIFE_STAGE_LABELS`, `TRANSITION_INTENSITY_THRESHOLDS`는 `commercial_quality_constants.py`에서 import만 사용
   - 금지: 운영 알림, logger 호출, side-effect

4. `backend/commercial_gate_helpers.py`
   - 역할: HF 게이트 helper 모듈
   - 포함: empathy/name/section parsing 등 cheap gate 분리 대상

5. `backend/life_cycle_lite_renderer.py`
   - 역할: 상품 전용 deterministic markdown 조립
   - 이유: 현재 `commercial_surface_renderer.py`는 generic 12챕터용이라 섞으면 파일이 더 무거워짐

6. `backend/main.py` 또는 동등 진입점의 product-aware finalize hook
   - 역할: `life_cycle-lite` 응답은 generic `_build_polished_reading_surface()` / `prepend_front_modules()` 경로와 분리
   - 이유: renderer를 추가해도 finalize 단계에서 generic front가 다시 붙으면 exact H2 contract가 깨짐
### 4.2 권장하지 않는 방식

- `main.py`에 `life_cycle-lite` 규칙을 직접 계속 추가하는 방식
- `report_engine.py`의 12챕터 generic block builder 안에 life-cycle product logic를 섞는 방식
- global token constant를 바로 `7000/9000`으로 조정해 generic `/ai_reading`, `yearly_forecast`, `compatibility`까지 동시에 영향 주는 방식 (문제는 수치 자체보다 전역 정책 변경이라는 점)
- cheap gate generic front-contract를 그대로 `life_cycle-lite`에 강제 적용하는 방식
- `life_cycle-lite` renderer 결과를 generic `_build_polished_reading_surface()` / `prepend_front_modules()` 경로에 다시 넣는 방식
- route cache key에만 `product_type`를 넣고, polished narrative cache namespace는 그대로 두는 방식

참고:
- metric key(`front_contract_ok`, `action_steps_contract_ok`)는 유지하되, 계산 의미는 `life_cycle-lite` release mode의 product-specific semantics로 바꿉니다.
### 4.3 토큰 정책 구현 권장안

현재 코드:
- `backend/main.py:153-155` -> global default/hard limit는 generic `/ai_reading` 기준
- 이 값은 v1.3.2에서 직접 변경 대상이 아님

PRD v1.3.2 요구 (`life_cycle-lite` product-layer):
- default `7000`
- soft range `6000–8000`
- hard cap `9000`

계층별 정의:
1. global default/hard limit (`18000 / 8000 / 22000`)는 generic `/ai_reading` 경로용이며 변경 금지
2. product-layer default `7000`은 `product_type == life_cycle` 경로에서만 적용
3. product-layer soft QA range `6000–8000`은 `life_cycle` 수동 QA/contract validation 기준
4. product-layer hard cap `9000`은 `life_cycle` 경로 절대 상한
5. 구현은 global 상수 overwrite가 아니라 product contract/orchestrator에서 통제

---

## 5. 상세 구현 체크리스트

## Phase A. 환경 / 테스트 복구

1. `PARTIAL` backend pytest baseline 고정
   - 현재 상태: 현재 인터프리터에서 `python -m pytest`는 실행 가능하며, `python -m pytest backend -q -p no:cacheprovider` 기준 `367 passed, 14 failed, 1 skipped`
   - Done 기준: 표준 runner 명령과 baseline failure inventory가 문서/CI 기준으로 고정됨

2. `TODO` clean-environment / CI 설치 경로 정리
   - 선택안 A: `backend/requirements-dev.txt`
   - 선택안 B: `backend/requirements.txt` + test runner 문서화
   - Done 기준: 로컬/CI에서 같은 설치 명령과 실행 명령이 현재 기준으로 재현 가능

3. `PARTIAL` failure inventory / stale contract cluster 분리 고정
   - 현재 문제:
     - atomic dominance / prompt contract / PDF narrative / report_engine depth 계열 테스트가 현재 코드 계약과 불일치
     - `backend/test_llm_token_limits.py`는 정리되었지만 나머지 stale cluster가 남아 있음
   - Phase A 목표:
     - 어떤 실패가 stale contract drift인지 문서/CI 기준으로 먼저 잠글 것
     - product path와 무관한 generic stale test 재작성은 orchestrator/product path 이후로 미룰 것
   - Done 기준: backend baseline failure inventory와 deferred rewrite 범위가 문서 기준으로 고정됨

4. `BLOCKED` temp write permission cluster 정리
   - 현재 문제:
     - tuning analyzer / tuning mode file creation 테스트가 temp root 권한으로 실패
   - Done 기준: 표준 러너에서 tempfile write가 재현 가능하거나 테스트가 writable root를 명시적으로 사용

5. `TODO` 최소 PR용 테스트 명령 확정
   - 권장:
     - `python -m pytest backend/test_commercial_surface_renderer.py -q`
     - `python -m pytest backend/test_cheap_validation_gate_metrics.py -q`
     - 새로 추가할 `life_cycle` 전용 테스트 2~4개

6. `TODO` micro fixture 폴더 구조 생성
   - 권장 경로: `backend/tests/fixtures/life_cycle_lite/`
   - 파일 최소 구성:
     - `normal_case.json`
     - `fallback_case.json`
     - `expected_contract.json`

## Phase B. 상품 계약 / API 분기

> Phase B 착수 전/종료 전 잠기는 결정:
> - 기존 `/ai_reading` 유지 + optional `product_type` 추가 방식으로 간다.
> - `life_cycle` 요청 입력 이름은 `subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status`로 고정한다.
> - 방법론 카드는 generic front 재사용이 아니라 `life_cycle` 전용 deterministic card로 간다.
> - cache key는 `product_type`를 반드시 포함한다.
> - generic `_finalize_ai_reading_result()` / `prepend_front_modules()` 경로는 `life_cycle`에 그대로 재사용하지 않는다.
> - polished narrative cache namespace에도 `product_type` + stable product render identity를 반영한다.
> - P0 CTA-lite는 미출시 상품명 대신 알림/업데이트 리마인드 액션으로만 노출한다.
> - `/pdf` contract 정렬은 backend P0의 일부로 보고, `get_ai_reading()` direct call path까지 같은 단계에서 맞춘다.
> - frontend/client migration은 backend P0 blocker는 아니지만, repo-wide 적용 완료를 주장하려면 `frontend/app/page.tsx`, `frontend/lib/api.ts`, `frontend/app/chart/ChartClient.tsx`, 관련 E2E가 새 계약을 실제로 소비해야 한다. migration note만으로는 부족하다.
6. `TODO` `product_type` enum + `life_cycle` 요청 스키마 도입
   - 권장 값: `life_cycle`, `yearly_forecast`, `compatibility`
   - 추가 요청 필드: `subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status`
   - Done 기준: request parsing + response echo 모두 가능

7. `TODO` thin orchestrator entrypoint 도입
   - 권장 위치: `product_orchestrator.py`
   - 동등 구현 허용: `main.py` 내부 전용 분기 함수 또는 기존 모듈 재구성
   - 핵심 함수 예시:
     - `build_product_response(product_type, ...)`
     - `route_life_cycle(...)`
   - Done 기준: `main.py`의 상품 분기 진입점이 1곳으로 줄어듦

8. `TODO` `/ai_reading` 확장 방식 고정 및 적용
   - 이 문서에서 잠근 결정: 기존 `/ai_reading` 유지 + optional `product_type` 추가
   - 이유: 현재 캐시/LLM/audit 경로 재사용 가능
   - Done 기준: 사용자 요청이 `life_cycle`이면 product branch로 들어감

9. `TODO` `yearly_forecast`, `compatibility`를 code-level bugfix-only로 묶기
   - Done 기준:
     - 새 productization 분기 없음
     - 기존 generic 흐름만 유지
     - 관련 TODO/주석/guard 명시

10. `TODO` full meta contract builder 고정
    - 권장 위치: `render_contract.py`
    - 동등 구현 허용: 기존 렌더 경로 내부의 dedicated meta builder
    - 공통 메타 생성 함수 예시:
      - `build_render_meta(...)`
    - 포함 필드:
      - `as_of_utc`
      - `as_of_local`
      - `timezone_offset`
      - `valid_until`
      - `valid_until_fallback`
      - `onboarding_goal`
      - `current_mahadasha_planet`
      - `next_mahadasha_date`
      - `product_type`
      - `contract_version`
      - `render_profile`
    - Done 기준: JSON 응답과 PDF 입력 payload에 동일 메타 존재

## Phase C. life_cycle-lite 데이터 조립

11. `TODO` `life_cycle` pure helper 계층 도입
    - 권장 위치: `life_cycle_helpers.py`
    - 동등 구현 허용: 기존 모듈 내부 pure helper 정리
    - shared constants 위치: `backend/commercial_quality_constants.py`
      - `PLANET_LABEL_MAP`
      - `VALID_PLANET_CODES`
      - `PLANET_DOMAIN_MAP`
      - `LIFE_STAGE_LABELS`
      - `TRANSITION_INTENSITY_THRESHOLDS`
    - P0 필수 함수:
      - `assign_life_stages()`
      - `compute_valid_until_lifecycle()`
    - P1 함수는 stub 또는 미구현 유지:
      - `compute_life_highs_lows()`
      - `compute_repeat_patterns()`
    - 주의: `life_cycle_helpers.py`는 shared constants를 import만 사용
    - 주의: HF regex/helper는 `commercial_gate_helpers.py`로 분리
    - 주의: `compute_valid_until_lifecycle()`는 pure function으로 유지하고, past-date 운영 알림은 caller/orchestrator에서 처리

12. `TODO` mahadasha 리스트 -> 4단계 구조 변환
    - 입력: `dasha_core.calculate_vimshottari_dasha()` 결과
    - 출력 요구:
      - stage id
      - label
      - start/end
      - dominant planet
      - summary label
    - Done 기준: 정상 fixture에서는 4 stages 반환, edge fixture에서는 `min(len(dashas), 4)`개만 반환하고 패딩하지 않음

13. `TODO` 현재 위치 계산
    - 입력: `as_of_local` 또는 현재 시점
    - 출력 요구:
      - current stage index 1개
      - current mahadasha info
      - remaining range 또는 next transition
    - Done 기준: 정상 fixture와 1~3 stage edge fixture에서는 current stage가 정확히 1개, 빈 stage fixture에서는 current stage가 없음

14. `TODO` `next_mahadasha_date` 계산
    - 재사용 입력: mahadasha list + current position
    - Done 기준: 정상값/None/과거값 3개 케이스 처리

15. `TODO` `valid_until` lifecycle fallback 함수 구현
    - 규칙:
      - 정상: `min(as_of_local + 3년, next_mahadasha_date)`
      - 결측/과거: `as_of_local + 3년`
      - fallback flag 저장
    - Done 기준: return `(valid_until, valid_until_fallback)` 후 직렬화 계층에서 `YYYY-MM-DD`로 노출

16. `TODO` 행성 코드 -> 소비자 라벨 변환 고정
    - Done 기준: renderer가 임의 문자열 생성하지 않고 상수맵만 사용

17. `TODO` 4단계 요약 데이터 구조 설계
    - 필드 예시:
      - `stage_label`
      - `start_date`
      - `end_date`
      - `planet_code`
      - `planet_consumer_label`
      - `is_current` (현재 위치 계산 후 주입)
    - Done 기준: renderer가 stage count 0~4를 추가 계산 없이 바로 출력 가능하고, `is_current`를 새로 계산하지 않음

18. `P1` 전환 강도/반복 패턴/다음 3년 구체화는 구현 금지
    - Done 기준: P0 코드에서 이 섹션이 노출되지 않음

## Phase D. life_cycle-lite 렌더러

19. `TODO` life_cycle-lite 전용 renderer 함수 구현
    - 권장 파일: `backend/life_cycle_lite_renderer.py`
    - 권장 함수: `render_life_cycle_lite_markdown(payload: dict) -> str`

20. `TODO` 섹션 구조 고정
    - 최소 섹션(H2 기준 고정 명칭 / exact order):
      1. cover/meta
      2. How to use 1p
      3. 인생 구조 한 장 요약
      4. 4단계 인생 구조
      5. 현재 위치
      6. 마하다샤 단계 목록
      7. 방법론 카드
      8. valid_until 설명
      9. CTA-lite
      10. 면책/윤리/데이터 보호
    - 주의: alias/변형(`한 장 요약`, `마하다샤 인생 단계 목록`, `CTA`)은 release mode 기준 실패
    - 주의: P0에서는 고점/저점 지도, 반복 패턴, 다음 3년 구체화 섹션을 넣지 않음
    - 주의: `CTA-lite`는 반드시 `valid_until 설명` 바로 다음 H2에 와야 하며, 면책/윤리/데이터 보호는 마지막 H2여야 함
    - 주의: stage가 1~3개면 존재하는 stage만 출력하고 current mark는 정확히 1개여야 하며, 0개면 fallback summary만 출력하며 current line을 생략함
    - Done 기준: 입력이 같으면 deterministic markdown이 동일하게 나옴

21. `TODO` 방법론 카드 6줄 deterministic copy 구현
    - 방향은 Phase B에서 이미 잠금: generic front 재사용이 아니라 `life_cycle` 전용 deterministic card
    - 권장: product renderer 내부 고정 문구로 먼저 닫기
    - 여기서는 구현만 진행
    - 차단 규칙: 아이템 21 구현 전에는 아이템 26~30 착수 금지 (HF15/방법론 카드 기준을 먼저 잠가야 함)

22. `TODO` `valid_until_fallback` 소비자 UX 문구 구현
    - Done 기준: fallback일 때 날짜만 튀지 않고 설명 문구가 함께 붙음

23. `TODO` CTA-lite 구현
    - 원칙:
      - 최적화 금지
      - 고정 문구 기반
      - life_cycle-lite 범위 안에서만 동작
      - 미출시 상품명(`yearly_forecast`, `compatibility`, `전환점 심화 리포트`) 노출 금지
      - `valid_until 설명` 바로 뒤 H2에 위치하고, valid_until 본문과 의미 토큰 1개 이상 공유
      - P0 backend markdown에서는 secondary consent button/control 병기 금지
    - Done 기준: CTA가 존재하되 업셀 과적 없고 HF14 source token 조건을 만족하며 버튼/행동은 1개만 존재

24. `TODO` `render_profile` 문자열 고정
    - 권장 값: `life_cycle_lite_v1`
    - Done 기준: response meta / logs / QA 문서에서 동일 값 사용

## Phase E. 검증 / 게이트 / QA

25. `PARTIAL` `python -m pytest` 실행은 가능하지만 release blocker가 남아 있음
    - source of truth: `Phase A` item 1 참조
    - 의미: execution contract 자체는 복구되었지만, renderer/contract/gate 작업을 진행하더라도 baseline 고정, source-of-truth 정렬, tempdir 분리가 닫히기 전에는 P0 출고를 닫지 않음
26. `TODO` life_cycle-lite micro fixture contract test 추가
    - 목적: 개발 루프에서 빠르게 메타/구조/fallback/P1 미노출을 확인
    - 형태: `backend/test_life_cycle_lite_contract.py` + pure checks
    - 주의: 이 테스트는 최종 출고 게이트를 대체하지 않는다
    - 주의: cache key / method card / route shape 결정은 이미 Phase B에서 잠근 상태를 전제로 한다

27. `TODO` 최소 contract test 1: 메타 필드
    - 체크:
      - `product_type == life_cycle`
      - `contract_version == v1.3.2`
      - `render_profile` non-empty
      - `as_of_utc`, `as_of_local`, `timezone_offset` 존재
      - `valid_until` 존재 + `YYYY-MM-DD` 직렬화
      - `valid_until_fallback` 존재
      - `onboarding_goal` 존재
      - `current_mahadasha_planet` key 존재 (`null` 허용)
      - `next_mahadasha_date` key 존재 (`YYYY-MM-DD` 또는 `null`)

28. `TODO` 최소 contract test 2: 4단계 구조
    - 체크:
      - 정상 fixture: stage count = 4
      - edge fixture: 0개 -> 빈 리스트 / 1~3개 -> 입력 개수만큼 반환
      - 정상 fixture: current stage = exactly 1
      - edge fixture: 1~3개면 current stage = exactly 1
      - edge fixture: stage 0개면 current stage = none
      - each stage has label/start/end

28-a. `TODO` 최소 contract test 3: exact H2 order + CTA adjacency
   - 체크:
     - H2 순서가 `cover/meta` -> `How to use 1p` -> `인생 구조 한 장 요약` -> `4단계 인생 구조` -> `현재 위치` -> `마하다샤 단계 목록` -> `방법론 카드` -> `valid_until 설명` -> `CTA-lite` -> `면책/윤리/데이터 보호`
     - `CTA-lite` 직전 H2는 항상 `valid_until 설명`
     - `면책/윤리/데이터 보호`는 마지막 H2
     - legacy alias(`한 장 요약`, `마하다샤 인생 단계 목록`, `CTA`)는 없음
   - Done 기준: 동일 payload에서 markdown H2 order가 deterministic 하게 유지되고, exact H2 계약 위반 시 테스트가 실패함

29. `TODO` `cheap_validation_gate.py` life_cycle-lite release mode 추가
    - 목적: 최종 출고 게이트에서 HF2/HF3뿐 아니라 HF11/HF12/HF14/HF16의 product-specific semantics도 함께 잠금
    - 구현 원칙: generic front-contract/12챕터 scanner를 그대로 재사용하지 말고, `life_cycle-lite` 전용 helper(`front_contract_ok`, `action_steps_contract_ok`, HF11/HF12/HF14/HF16 header scope 계산용)를 추가
    - `front_contract_ok = true` 조건:
      - required H2 `cover/meta` -> `How to use 1p` -> `인생 구조 한 장 요약` -> `4단계 인생 구조` -> `현재 위치` -> `마하다샤 단계 목록` -> `방법론 카드` -> `valid_until 설명` -> `CTA-lite` -> `면책/윤리/데이터 보호` 순서로 각 1회
      - P1 H2(`인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`)는 0회
    - `action_steps_contract_ok = true` 조건:
      - `How to use 1p` 체크리스트 >= 4 + 복구 플랜 1개 이상
      - `valid_until 설명` 업데이트/리마인드 행동 라인 1개
      - `CTA-lite` 버튼/행동 라인 1개, 버튼 텍스트 20자 이내
      - secondary consent button/control 없음
    - 추가 잠금:
      - HF11 empathy target = `cover/meta`, `How to use 1p`를 건너뛴 첫 narrative H2, 정상 경로에서는 `인생 구조 한 장 요약`
      - HF12 action-required H2 = `How to use 1p`, `valid_until 설명`, `CTA-lite`만 해당
      - HF14 source tokens = `valid_until 설명` 본문
      - HF16 summary H2 exact = `인생 구조 한 장 요약`

30. `TODO` 최소 contract test 4: fallback UX + P1 미노출
    - 체크:
      - `next_mahadasha_date` None/past -> fallback flag true
      - markdown에 부드러운 설명 문구 포함
      - 고점/저점 지도 없음
      - 반복 패턴 없음
      - 다음 3년 구체화 없음

31. `TODO` 수동 QA 템플릿 작성
    - 권장 경로: `PRD/release_evidence/v1_3_2/life_cycle_lite_manual_qa.md`
    - 형식: markdown 1파일, case 2건(정상 1 + fallback/edge 1) 고정
    - 문서 상단 필수 메타:
      - `contract_version`
      - `release_evidence_dir`
      - `render_profile`
      - `request_fingerprint`
      - `evidence_case_id`
      - `commit_sha`
      - `release_manifest_path`
    - 필수 필드:
      - case id
      - input fixture / request summary
      - expected points
      - actual summary
      - PASS/FAIL
      - reviewer
      - run date (Asia/Seoul)
    - Done 기준: 2건 모두 채워져 있고, 문서 상단 메타만 읽어도 현재 정본 `v1.3.2` 기준의 release candidate와 연결된 manifest 경로를 확인할 수 있음

31-a. `TODO` release gate summary 저장
    - 권장 경로: `PRD/release_evidence/v1_3_2/life_cycle_lite_gate_summary.json`
    - 형식: `cheap_validation_gate.py` `life_cycle-lite` release mode 결과 1건의 pretty-printed JSON
    - 필수 포함:
      - `product_type`
      - `contract_version`
      - `render_profile`
      - `request_fingerprint`
      - `evidence_case_id`
      - `commit_sha`
      - `front_contract_ok`
      - `action_steps_contract_ok`
      - HARD FAIL 결과 요약 (`hard_fail_count` 또는 동등 필드)
    - Done 기준: 자동 gate가 실제 release source of truth로 통과했는지 별도 로그 검색 없이 확인 가능하고, `contract_version == v1.3.2`이며 `render_profile` / `request_fingerprint` / `evidence_case_id`가 수동 QA 문서 및 manifest와 일치함

31-b. `TODO` 샘플 응답 1개 저장
    - 권장 경로: `PRD/release_evidence/v1_3_2/life_cycle_lite_sample_response.json`
    - 형식: 실제 `life_cycle` API 응답 1건의 pretty-printed JSON
    - 필수 포함:
      - full P0 meta contract
      - `valid_until_fallback`
      - `render_profile`
      - non-empty `polished_reading`
      - `polished_reading` 안에서 exact H2 order 확인 가능
    - Done 기준: 문서/QA/release gate 검토자가 코드 실행 없이도 계약 필드와 최종 렌더 구조를 확인할 수 있고, `meta.contract_version == v1.3.2`이며 `meta.render_profile`이 gate summary 및 manifest와 일치함

31-c. `TODO` release evidence manifest 저장
    - 권장 경로: `PRD/release_evidence/v1_3_2/life_cycle_lite_release_manifest.json`
    - 형식: release evidence reviewer entrypoint 1건의 pretty-printed JSON
    - 필수 포함:
      - `contract_version`
      - `release_evidence_dir`
      - `render_profile`
      - `request_fingerprint`
      - `evidence_case_id`
      - `commit_sha`
      - `manual_qa_path`
      - `sample_response_path`
      - `gate_summary_path`
      - `manual_qa_sha256`
      - `sample_response_sha256`
      - `gate_summary_sha256`
    - Done 기준: reviewer가 manifest 하나만 읽어도 세 증적 파일의 경로/해시/버전/render/request identity를 현재 정본 `v1.3.2` 기준으로 재검수할 수 있음

32. `PARTIAL` existing cheap gate metrics는 재사용 가능
    - 재사용 대상:
      - cache/hash/logging 패턴
      - scored surface / postprocess audit 방법
    - 비재사용 대상:
      - generic front contract 그대로 적용하는 규칙

## Phase F. 캐시 / 응답 / PDF

33. `TODO` Phase B에서 잠근 cache key 정책 구현
    - 현재 cache key는 generic ai_reading 중심
    - Done 기준:
      - 같은 chart라도 상품이 다르면 다른 route cache key
      - polished narrative cache namespace도 `product_type` / `render_profile` 또는 동등 deterministic identity 기준으로 분리

34. `TODO` response payload에 full P0 meta contract 직렬화
    - Done 기준: API 응답에서 메타 확인 가능

35. `TODO` PDF 경로가 life_cycle-lite payload를 그대로 소비하도록 연결
    - Done 기준:
      - PDF에서도 `valid_until`, 방법론 카드, CTA-lite 누락 없음
      - `/pdf` -> `get_ai_reading()` direct call path가 `life_cycle` request normalization / finalize / cache policy를 동일하게 사용

36. `TODO` `chapter_blocks_hash`와 상품 렌더 결과 관계 정리
    - 권장:
      - generic hash는 generic chapter blocks용
      - life_cycle-lite는 별도 deterministic payload hash를 둘지 검토
    - 최소 기준: 캐시 충돌 없을 것

## Phase G. 문서 / 운영 정리
37. `BLOCKED` `backend/API.md`에 `/ai_reading` `product_type` / `life_cycle` 공개 계약 반영
    - Done 기준: `backend/API.md`가 `/ai_reading`의 optional `product_type`, `life_cycle` 요청 입력, full P0 meta contract, bugfix-only 범위, backend-only P0 한계를 설명
    - 현재 상태: 목표 공개 계약은 PRD/체크리스트에만 잠겨 있고, `backend/API.md`는 아직 `v1.2.25 target public contract` 기준 표현이 남아 있음
38. `BLOCKED` release gate source of truth를 운영 문서에 전환
    - 대상 문서: `backend/QUALITY_GATES.md`, `README.md` release 섹션
    - Done 기준: 최종 출고 판정 경로가 `cheap_validation_gate.py` `life_cycle-lite` release mode로 동일하게 적힘
    - 현재 상태: 운영 문서는 아직 `golden_sample_runner` / `fast_llm_gate` / PDF scanner 기준이고, cheap gate는 아직 generic `front_contract_ok` / `action_steps_contract_ok` semantics 중심
    - 주의: 기존 `golden_sample_runner` / `fast_llm_gate`는 보조 검증으로 남길 수 있어도 release source of truth로 남기면 안 됨
39. `TODO` 구현 완료 후 `PRODUCT_SPEC_PRD_v1_3_2.md` 정본과 코드 정합 재검수
40. `TODO` `README.md` 범위/출고 기준 정리
    - BTR / PDF / Next.js는 현재 backend P0 비범위 또는 향후 개발로 명시
    - 현재 구현 범위는 `life_cycle-lite` backend productization에 맞춰 설명 정리
    - repo-wide 적용 완료가 아니라면 frontend/client 구현 및 E2E 후속 작업이 남아 있음을 명시
41. `TODO` frontend/client/API/cache 구현 + E2E 반영 (repo-wide 후속)
    - 대상 후보: `frontend/app/page.tsx`, `frontend/app/chart/ChartClient.tsx`, `frontend/lib/api.ts`, 관련 E2E
    - 의미: backend P0 blocker는 아니지만 전체 저장소 적용 완료를 주장하려면 실제 consumer 구현이 필요

---

## 6. 지금 당장 손대면 안 되는 것

1. `P1` 기능인 고점/저점 지도, 반복 패턴, 다음 3년 구체화
2. `yearly_forecast`, `compatibility`의 전면 productization
3. BTR ON 전환
4. generic `/ai_reading` 전체를 깨는 global token 정책 급변
5. `report_engine.py` 대규모 리라이트

---

## 7. 추천 구현 순서 (현실적인 P0 순서)

> 주의: 섹션 7은 섹션 5의 요약 우선순위입니다. 세부 체크 박스와 정본 순서는 섹션 5를 기준으로 봅니다.

1. PRD / 체크리스트 / `backend/test_llm_token_limits.py` 정렬분을 먼저 커밋해 baseline을 고정 (`AUD-25`)
2. runner source of truth + failure inventory 고정 + tempdir 문제 분리 (`Phase A` #1~#5)
3. `product_type` + `life_cycle` 요청 스키마 + thin orchestrator 도입 (`Phase B` #6~#9)
4. full meta contract builder 고정 (`Phase B` #10)
5. product-aware finalize branch + generic front 재부착 차단 + polished cache namespace 분리 (`Phase B` + `Phase F`)
6. `life_cycle` pure helper / payload builder / renderer 정리 (`Phase C` + `Phase D`)
7. response meta local-date 직렬화 + `/pdf` direct call contract 정렬 (`Phase F` #34~#36)
8. `cheap_validation_gate` release mode + micro fixture / manual QA / release evidence 4종 정리 (`Phase E`)
9. `backend/API.md` + `backend/QUALITY_GATES.md` + `README.md` 문서 계약 동기화 (`Phase G` #37~#40)
10. repo-wide 적용이 필요하면 frontend consumer 구현까지 반영 (`Phase G` #41)
11. 그 다음에 product path touched 범위 밖 generic stale test 재작성 및 추가 개선 작업

---

## 8. P0 완료 정의

아래가 모두 만족되면 v1.3.2 기준 P0 완료로 봅니다.

> 주의: 특정 파일의 신설 여부 자체는 완료 조건이 아닙니다. 동일한 behavior / test / gate 계약을 만족하면 동등 구현도 허용합니다.

1. `python -m pytest` 표준 명령이 실행 가능하고, backend baseline failure inventory가 문서/CI 기준으로 고정됨
2. runner baseline이 고정되고, product path touched 범위의 stale / permission 실패가 release gate 판단을 오염시키지 않음
3. `life_cycle` 요청이 thin orchestrator를 통해 전용 경로로 분기됨
4. 요청 입력이 `subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status` canonical name으로 정규화됨
5. 응답에 full P0 meta contract(`as_of_utc`, `as_of_local`, `timezone_offset`, `valid_until`, `valid_until_fallback`, `onboarding_goal`, `current_mahadasha_planet`, `next_mahadasha_date`, `product_type`, `contract_version`, `render_profile`) 존재
6. `life_cycle-lite` markdown이 `cover/meta` 포함 exact H2 order로 deterministic 하게 생성되며, generic front modules가 finalize 단계에서 다시 붙지 않음
7. route cache key와 polished narrative cache namespace가 모두 product-aware contract로 분리됨
8. `/pdf` 경로가 같은 `life_cycle` request normalization / finalize / cache policy를 사용하고 메타를 보존함
9. `cheap_validation_gate.py`의 `life_cycle-lite` release mode 기준 HF 16개 회귀 없음
10. `backend/QUALITY_GATES.md`와 `README.md`의 release gate 문구가 동일 source of truth로 전환됨
11. 정상 fixture에서는 4단계 + current stage 1개, edge fixture에서는 0~3단계 허용 및 0단계면 current stage 없음
12. `next_mahadasha_date` 결측/과거 시 fallback UX가 자연스럽게 동작함
13. 단일 LLM 호출 정책 유지
14. `life_cycle` 상품 경로에서 product-layer 기본값 `7000`, soft QA range `6000–8000`, hard cap `9000`이 고정됨
15. P1 기능이 출력에 섞이지 않음
16. `backend/API.md`가 `/ai_reading`의 `product_type`, `life_cycle` 요청 입력, 응답 meta contract, backend-only P0 한계를 현재 기준으로 설명함
17. README/운영 문서가 backend P0와 repo-wide rollout을 구분함
18. micro fixture 2건 + 수동 QA 2건 통과
19. `PRD/release_evidence/v1_3_2/` 아래에 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json`, `life_cycle_lite_release_manifest.json`가 모두 존재하고, manifest를 포함한 4종이 같은 `contract_version`, `render_profile`, `request_fingerprint`, `evidence_case_id`, `commit_sha` 기준으로 현재 정본 `v1.3.2`과 일치함

> 주의: 위 완료 정의는 backend-only P0 기준입니다. 프로젝트 전체 적용 완료를 주장하려면 frontend 진입 경로, `/ai_reading` client contract, chart consumer, E2E가 새 계약을 실제로 소비해야 하며, migration note/계획만으로는 충분하지 않습니다.

---

## 9. 핵심 결론

- 핵심 결론은 섹션 `0.7 핵심 결론`로 이동했습니다.







