# IMPLEMENTATION_CHECKLIST v1.2.7 (Current Project Audit)

- 기준 PRD: `PRODUCT_SPEC_PRD_v1_2_7.md`
- 점검 기준일: 2026-03-06
- 점검 범위: 현재 저장소의 backend 중심 구현 상태, 테스트 환경, API 표면, 후처리/게이트, 문서-코드 불일치
- 목적: "무엇이 이미 구현되어 있고, 무엇이 아직 없으며, 무엇부터 손대야 하는가"를 코드 기준으로 잠그는 실행 체크리스트

---

## 0. 상태 표기

- `DONE`: 현재 코드에 재사용 가능한 구현이 이미 있음
- `PARTIAL`: 관련 구현은 있지만 PRD 기준 상품 수준까지는 아님
- `TODO`: 구현이 필요함
- `BLOCKED`: 구현보다 먼저 환경/계약 불일치를 해결해야 함
- `P1`: v1.2.7 범위 밖이므로 지금 구현하지 않음

---

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
| `contract_version` / `render_profile` / `product_type` 메타 계약 | `TODO` | 런타임 backend에는 없음 |
| `commercial_quality_constants.py` 상수층 | `DONE` | 파일은 존재, product-layer 재사용 규칙만 정리하면 됨 |
| `commercial_gate_helpers.py` | `TODO` | HF gate helper 전용 파일 없음 |
| `life_cycle_helpers.py` | `TODO` | 현재 없음 |
| `product_orchestrator.py` | `TODO` | 파일 자체 없음 |
| `render_contract.py` | `TODO` | 파일 자체 없음 |
| `pytest` 실행 환경 | `TODO` | requirements 정리로 해결 가능한 상태 |
| micro fixture 체계 | `TODO` | generic fixture 1개만 있음 |
| token budget v1.2.7 정렬 | `PARTIAL` | PRD는 정렬되었지만 코드 product-layer 적용은 미구현 |
| yearly/compat bugfix-only 보호 | `TODO` | 문서상 합의만 있고 코드 분기 없음 |

---

## 2. 현재 코드에서 이미 있는 것

### 2.1 엔진 / 타이밍 코어

1. `DONE` deterministic Vimshottari dasha 계산 코어 존재
   - 근거: `backend/dasha_core.py:77`, `backend/dasha_core.py:123`, `backend/dasha_core.py:159`
   - 활용 방식: `life_cycle-lite`는 이 코어를 직접 재사용해야 함
   - 의미: P0에서 엔진 산식을 새로 만들 필요 없음

2. `PARTIAL` dasha narrative context helper 존재
   - 근거: `backend/report_engine.py:2176`
   - 현재 상태: `mahadasha`, `antardasha`, intensity, pressure 정도만 제공
   - 부족한 점: `next_mahadasha_date`, `valid_until_fallback`, 4단계 life stage, 현재 위치용 계약이 없음

3. `PARTIAL` life timeline LLM 보조 함수 존재
   - 근거: `backend/llm_service.py:2493`, `backend/llm_service.py:2561`, `backend/llm_service.py:3170`
   - 현재 상태: generic `Current Phase` 챕터를 더 자연스럽게 다시 쓰는 용도
   - 부족한 점: `life_cycle-lite` 상품 전체 렌더러가 아님
   - 결론: 재사용 후보는 맞지만, 상품 구현 완료로 간주하면 안 됨

### 2.2 AI reading / 표면 / 캐시

4. `DONE` `/ai_reading` 엔드포인트 존재
   - 근거: `backend/main.py:2872`
   - 현재 상태: generic 12챕터 AI reading 중심
   - 의미: 새 상품은 이 경로를 확장하거나 thin wrapper를 두는 방식이 현실적

5. `DONE` `chapter_blocks_hash` / `polished_reading` 캐시 경로 존재
   - 근거: `backend/main.py:340`, `backend/main.py:345`, `backend/main.py:354`, `backend/main.py:3114`
   - 의미: PRD의 단일 LLM 호출 + 캐시 재사용 전략과 잘 맞음

6. `DONE` commercial surface builder 존재
   - 근거: `backend/main.py:2732`, `backend/main.py:2764`, `backend/main.py:2800`
   - 현재 상태: chapter blocks 기반 polished surface 조립 가능
   - 부족한 점: `life_cycle-lite` 전용 구조가 아님
   - 관련 상수층: `backend/commercial_quality_constants.py:57` 존재
   - 의미: PRD lock constant를 둘 파일을 새로 만들 필요는 없음

7. `DONE` generic commercial markdown renderer 존재
   - 근거: `backend/commercial_surface_renderer.py:557`
   - 현재 상태: 12챕터 generic 구조를 deterministic markdown으로 렌더
   - 테스트 근거: `backend/test_commercial_surface_renderer.py:6`
   - 부족한 점: `life_cycle-lite`의 4단계 구조/valid_until UX/방법론 카드/CTA-lite 전용 구조는 없음

8. `DONE` front modules prepend 가능
   - 근거: `backend/commercial_surface_renderer.py:132`, `backend/main.py:2800`
   - 의미: generic front 개념은 이미 있음
   - 주의: `life_cycle-lite`는 front contract를 그대로 재사용할지, 별도 product front를 둘지 결정 필요

### 2.3 품질 게이트 / cheap gate / 리포트 품질 스캔

9. `DONE` cheap validation gate와 true-path surface metrics 존재
   - 근거: `scripts/cheap_validation_gate.py:793`
   - 현재 상태: scored surface, front contract, actionable coverage, postprocess alignment 측정 가능
   - 테스트 근거: `backend/test_cheap_validation_gate_metrics.py:177`, `backend/test_cheap_validation_gate_metrics.py:218`, `backend/test_cheap_validation_gate_metrics.py:373`

10. `DONE` fallback front 적용 탐지 로직 존재
    - 근거: `scripts/cheap_validation_gate.py:532`
    - 의미: `front_contract_fallback_applied` 같은 front 계약을 추적하는 방식은 이미 있음
    - 부족한 점: `life_cycle-lite`의 `valid_until_fallback`과는 다른 계약임

11. `DONE` quality gate 문서와 golden runner 존재
    - 근거: `backend/QUALITY_GATES.md`, `backend/golden_sample_runner.py:67`, `backend/golden_sample_runner.py:398`
    - 의미: 완전히 처음부터 QA 체계를 만들 필요는 없음
    - 주의: v1.2.x는 golden 중심이 아니라 micro fixture 중심으로 가야 함

---

## 3. 현재 코드에서 없는 것 또는 모자란 것

### 3.1 상품 오케스트레이션 / 계약 계층

12. `TODO` `product_type` 기반 단일 분기 없음
    - 근거: backend 전체 검색 기준 `product_type` 매치 없음
    - 현재 상태: `/ai_reading` 단일 generic 상품만 사실상 존재
    - 해야 할 일:
      - 요청 입력에 `product_type` 추가
      - `life_cycle` / `yearly_forecast` / `compatibility` 분기 단일화
      - P0에서는 `life_cycle`만 productized, 나머지는 bugfix-only 유지

13. `TODO` `product_orchestrator.py` 부재
    - 근거: `backend/product_orchestrator.py` 파일 없음
    - 해야 할 일:
      - thin orchestration 레이어 추가
      - main route에서 상품별 분기 규칙이 다시 흩어지지 않게 고정

14. `TODO` `render_contract.py` 부재
    - 근거: `backend/render_contract.py` 파일 없음
    - 해야 할 일:
      - 응답 메타 계약 (`product_type`, `contract_version`, `render_profile`, `valid_until`, `valid_until_fallback`) 정의
      - PDF/JSON/cheap gate가 같은 계약을 보게 만들 것

15. `TODO` `commercial_gate_helpers.py` 부재
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

### 3.2 life_cycle-lite 상품 자체

16. `TODO` `life_cycle-lite` 전용 markdown renderer 없음
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

17. `TODO` 4단계 life stage grouping 로직 없음
    - 현재 상태: PRD 요구 pure function 부재
    - 재사용 입력: `backend/dasha_core.py`의 mahadasha list
    - 해야 할 일:
      - 9개 마하다샤를 소비자용 4단계로 그룹화
      - 각 단계 start/end/year span 계산
      - 현재 시점이 어느 단계인지 표시
    - 권장 구현 위치: `backend/life_cycle_helpers.py`

18. `TODO` `next_mahadasha_date` / `valid_until` lifecycle contract 없음
    - 현재 상태: generic `build_dasha_narrative_context()`는 next transition date를 제공하지 않음
    - 해야 할 일:
      - 현재 maha end date 추출
      - `min(as_of_local + 3년, next_mahadasha_date)` 계산
      - null/past일 때 fallback + soft UX 문구
    - 권장 구현 위치: `backend/life_cycle_helpers.py`

19. `TODO` `PLANET_LABEL_MAP` / consumer label mapping 없음
    - 현재 상태: PRD 수준 lock constant 없음
    - 해야 할 일:
      - 9개 행성 코드 매핑을 pure constant로 고정
      - renderer는 이 맵만 사용

20. `TODO` `LIFE_STAGE_LABELS` 없음
    - 현재 상태: 4단계 라벨 contract 없음
    - 해야 할 일:
      - stage label set 고정
      - test fixture와 gate에서 동일 상수 사용

21. `P1` 고점/저점 지도, 반복 패턴, 다음 3년 구체화
    - 현재 상태: P0 대상 아님
    - 해야 할 일: 지금 구현하지 않음
    - 주의: 함수 이름이나 stub은 만들 수 있으나 제품에 노출하면 안 됨

### 3.3 메타 계약 / 응답 스키마

22. `TODO` `contract_version`, `render_profile`, `product_type`가 실제 backend 응답에 없음
    - 근거: backend 전체 검색 기준 없음
    - 해야 할 일:
      - JSON 응답 최상위 또는 `meta` 아래에 고정
      - PDF 변환 경로에서도 유지
      - cheap gate summary에 같이 남기기

23. `TODO` `valid_until_fallback` 전용 flag 없음
    - 현재 상태: generic `fallback`은 LLM fallback이고, `front_contract_fallback_applied`는 front module fallback 전용 metric임
    - 선행 감사:
      - 기존 `fallback`, `front_contract_fallback_applied` 사용처를 먼저 grep으로 분리 기록
      - life_cycle 전용 `valid_until_fallback`과 generic fallback semantics가 섞이지 않게 audit note 남기기
    - 해야 할 일:
      - life_cycle 전용 메타 키를 `valid_until_fallback`으로 고정
      - 응답 `meta.valid_until_fallback`과 `gate_summary.valid_until_fallback`가 같은 boolean을 보게 만들 것
      - cheap gate / 수동 QA / 로그가 모두 같은 키만 보게 만들 것

24. `TODO` `render_profile` 버전 문자열 전략 없음
    - 해야 할 일:
      - 예: `life_cycle_lite_v1`
      - 렌더 profile 바뀔 때만 값 변경
      - QA 로그와 응답 payload에 동일하게 남길 것

### 3.4 테스트 / 환경 / fixture

25. `TODO` 현재 `pytest` 설치 안 됨
    - 근거: `python -m pytest --version` 실패 (`No module named pytest`)
    - 추가 근거: `backend/requirements.txt`에 `pytest` 없음
    - 해야 할 일:
      - dev requirements 또는 requirements 분리
      - 최소한 로컬/CI에서 `python -m pytest` 가능하게 만들 것

26. `TODO` 현재 테스트 스위트 일부가 코드와 불일치할 가능성 높음
    - 근거: `backend/test_llm_token_limits.py:66-68`은 `1500/2000/3000`을 기대
    - 실제 코드: `backend/main.py:153-155`는 `18000/8000/22000`
    - 추가 불일치: test는 `max_tokens` payload를 기대하지만 현재 구현은 `max_completion_tokens` 사용 (`backend/main.py:320`)
    - 해야 할 일:
      - 이 테스트를 현재 아키텍처에 맞게 정리
      - v1.2.6용 상품 토큰 정책은 global constant보다 product-layer에서 검사하는 방식으로 고정

27. `TODO` life_cycle-lite 전용 micro fixture 부재
    - 근거: `backend/tests/fixtures`에는 `chapter_blocks_pre_llm_sample.json` 1개만 존재
    - 해야 할 일:
      - 최소 2개 fixture 추가
      - 정상 경로 1개
      - valid_until_fallback=True 경로 1개
      - 가능하면 현재 maha 전환이 임박한 케이스 1개 추가

28. `TODO` life_cycle-lite 전용 contract test 부재
    - 현재 상태: generic surface/gate 테스트는 있으나 상품별 테스트 없음
    - 해야 할 일:
      - meta keys 존재 검사
      - 4단계 구조 개수 검사
      - current stage single-mark 검사
      - `valid_until` fallback UX 검사
      - CTA-lite 존재 검사

29. `TODO` `cheap_validation_gate`에 life_cycle-lite release mode 없음
    - 현재 상태: generic front-contract/12챕터 기준
    - P0 출고 기준: 이 항목이 닫히기 전에는 HF 16개 회귀 없음 판정을 닫을 수 없음
    - 해야 할 일:
      - 새 상품용 lightweight scan mode 추가
      - `valid_until_fallback`를 response meta와 같은 boolean으로 관찰 지표에 기록
      - generic gate에 억지로 끼워 넣지 말 것

### 3.5 PDF / 전달 채널

30. `PARTIAL` `/pdf` 엔드포인트는 존재하나 상품-aware contract는 아님
    - 근거: `backend/main.py:3556`
    - 현재 상태: generic ai_reading narrative를 PDF로 내보내는 구조
    - 해야 할 일:
      - `life_cycle-lite` payload를 PDF로 넘길 때 메타/유효기간/방법론 카드/CTA-lite가 그대로 보존되는지 확인
      - 필요 시 PDF template 분기 추가

31. `TODO` PDF 선택 경로에 `product_type` 반영 필요
    - 현재 상태: `ai_cache_key` 기반 generic reuse
    - 해야 할 일:
      - 동일 chart라도 상품이 다르면 cache/pdf 키도 달라져야 함

---

## 4. 구현 전에 먼저 고정할 최소 아키텍처

### 4.1 권장 구조 (현재 프로젝트에 맞는 최소 분리)

1. `backend/product_orchestrator.py`
   - 역할: `product_type` 기준 thin branch
   - 원칙: `main.py`에서 상품별 if/else 확산 방지

2. `backend/render_contract.py`
   - 역할: 공통 메타 계약 생성
   - 포함: `product_type`, `contract_version`, `render_profile`, `valid_until`, `valid_until_fallback`

3. `backend/life_cycle_helpers.py`
   - 역할: `life_cycle-lite` pure function 모음
   - 포함: stage grouping, current stage 계산, valid_until lifecycle, P1 stub
   - 금지: 운영 알림, logger 호출, side-effect

4. `backend/commercial_gate_helpers.py`
   - 역할: HF 게이트 helper 모듈
   - 포함: empathy/name/section parsing 등 cheap gate 분리 대상

5. `backend/life_cycle_lite_renderer.py`
   - 역할: 상품 전용 deterministic markdown 조립
   - 이유: 현재 `commercial_surface_renderer.py`는 generic 12챕터용이라 섞으면 파일이 더 무거워짐

### 4.2 권장하지 않는 방식

- `main.py`에 `life_cycle-lite` 규칙을 직접 계속 추가하는 방식
- `report_engine.py`의 12챕터 generic block builder 안에 life-cycle product logic를 섞는 방식
- global token constant를 바로 `7000/9000`으로 조정해 generic `/ai_reading`, `yearly_forecast`, `compatibility`까지 동시에 영향 주는 방식 (문제는 수치 자체보다 전역 정책 변경이라는 점)
- cheap gate generic front-contract를 그대로 `life_cycle-lite`에 강제 적용하는 방식

### 4.3 토큰 정책 구현 권장안

현재 코드:
- `backend/main.py:153-155` -> global default/hard limit는 generic `/ai_reading` 기준
- 이 값은 v1.2.6에서 직접 변경 대상이 아님

PRD v1.2.7 요구 (`life_cycle-lite` product-layer):
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

1. `TODO` `pytest` 설치 경로 확정
   - Done 기준: `python -m pytest --version` 성공

2. `TODO` dev dependency 파일 정리
   - 선택안 A: `backend/requirements-dev.txt`
   - 선택안 B: `backend/requirements.txt`에 직접 추가
   - Done 기준: 문서화된 설치 명령 1개로 테스트 가능

3. `TODO` stale test 정리: `backend/test_llm_token_limits.py`
   - 현재 문제:
     - 상수 기대값이 현재 코드와 불일치
     - payload key 기대값이 현재 함수 시그니처와 불일치
   - Done 기준: 현재 구현 또는 product-layer 정책에 맞게 테스트 재작성

4. `TODO` 최소 PR용 테스트 명령 확정
   - 권장:
     - `python -m pytest backend/test_commercial_surface_renderer.py -q`
     - `python -m pytest backend/test_cheap_validation_gate_metrics.py -q`
     - 새로 추가할 `life_cycle` 전용 테스트 2~4개

5. `TODO` micro fixture 폴더 구조 생성
   - 권장 경로: `backend/tests/fixtures/life_cycle_lite/`
   - 파일 최소 구성:
     - `normal_case.json`
     - `fallback_case.json`
     - `expected_contract.json`

## Phase B. 상품 계약 / API 분기

> Phase B 착수 전/종료 전 잠기는 결정:
> - 기존 `/ai_reading` 유지 + optional `product_type` 추가 방식으로 간다.
> - 방법론 카드는 generic front 재사용이 아니라 `life_cycle` 전용 deterministic card로 간다.
> - cache key는 `product_type`를 반드시 포함한다.
> - P0 CTA-lite는 미출시 상품명 대신 알림/업데이트 리마인드 액션으로만 노출한다.

6. `TODO` `product_type` enum 도입
   - 권장 값: `life_cycle`, `yearly_forecast`, `compatibility`
   - Done 기준: request parsing + response echo 모두 가능

7. `TODO` `product_orchestrator.py` 신설
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

10. `TODO` `render_contract.py` 신설
    - 공통 메타 생성 함수 예시:
      - `build_render_meta(...)`
    - 포함 필드:
      - `product_type`
      - `contract_version`
      - `render_profile`
      - `valid_until`
      - `valid_until_fallback`
    - Done 기준: JSON 응답과 PDF 입력 payload에 동일 메타 존재

## Phase C. life_cycle-lite 데이터 조립

11. `TODO` `life_cycle_helpers.py` 신설
    - P0 필수 상수:
      - `PLANET_LABEL_MAP`
      - `LIFE_STAGE_LABELS`
    - P0 필수 함수:
      - `assign_life_stages()`
      - `compute_valid_until_lifecycle()`
    - P1 함수는 stub 또는 미구현 유지:
      - `compute_life_highs_lows()`
      - `compute_repeat_patterns()`
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
    - Done 기준: always 4 stages 반환

13. `TODO` 현재 위치 계산
    - 입력: `as_of_local` 또는 현재 시점
    - 출력 요구:
      - current stage index 1개
      - current mahadasha info
      - remaining range 또는 next transition
    - Done 기준: 중복 current marking 없음

14. `TODO` `next_mahadasha_date` 계산
    - 재사용 입력: mahadasha list + current position
    - Done 기준: 정상값/None/과거값 3개 케이스 처리

15. `TODO` `valid_until` lifecycle fallback 함수 구현
    - 규칙:
      - 정상: `min(as_of_local + 3년, next_mahadasha_date)`
      - 결측/과거: `as_of_local + 3년`
      - fallback flag 저장
    - Done 기준: return `(valid_until, valid_until_fallback)`

16. `TODO` 행성 코드 -> 소비자 라벨 변환 고정
    - Done 기준: renderer가 임의 문자열 생성하지 않고 상수맵만 사용

17. `TODO` 4단계 요약 데이터 구조 설계
    - 필드 예시:
      - `stage_label`
      - `start_date`
      - `end_date`
      - `planet_code`
      - `planet_consumer_label`
      - `is_current`
    - Done 기준: renderer가 별도 계산 없이 바로 출력 가능

18. `P1` 전환 강도/반복 패턴/다음 3년 구체화는 구현 금지
    - Done 기준: P0 코드에서 이 섹션이 노출되지 않음

## Phase D. life_cycle-lite 렌더러

19. `TODO` life_cycle-lite 전용 renderer 함수 구현
    - 권장 파일: `backend/life_cycle_lite_renderer.py`
    - 권장 함수: `render_life_cycle_lite_markdown(payload: dict) -> str`

20. `TODO` 섹션 구조 고정
    - 최소 섹션:
      1. cover/meta
      2. How to use 1p
      3. 한 장 요약
      4. 4단계 인생 구조
      5. 현재 위치
      6. 마하다샤 단계 목록
      7. 방법론 카드
      8. valid_until 설명
      9. 면책/윤리/데이터 보호
      10. CTA-lite
    - 주의: P0에서는 고점/저점 지도, 반복 패턴, 다음 3년 구체화 섹션을 넣지 않음
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
    - Done 기준: CTA가 존재하되 업셀 과적 없음

24. `TODO` `render_profile` 문자열 고정
    - 권장 값: `life_cycle_lite_v1`
    - Done 기준: response meta / logs / QA 문서에서 동일 값 사용

25. `TODO` generic 12챕터 surface와 분리
    - Done 기준: `life_cycle-lite`는 `Current Phase`, `Executive Diagnosis` 같은 generic chapter name에 의존하지 않음

## Phase E. cheap gate / contract test / QA

26. `TODO` life_cycle-lite micro fixture contract test 추가
    - 목적: 개발 루프에서 빠르게 메타/구조/fallback/P1 미노출을 확인
    - 형태: `backend/test_life_cycle_lite_contract.py` + pure checks
    - 주의: 이 테스트는 최종 출고 게이트를 대체하지 않는다
    - 주의: cache key / method card / route shape 결정은 이미 Phase B에서 잠근 상태를 전제로 한다

27. `TODO` 최소 contract test 1: 메타 필드
    - 체크:
      - `product_type == life_cycle`
      - `contract_version == v1.2.7` 또는 구현 당시 최신값
      - `render_profile` non-empty
      - `valid_until` 존재
      - `valid_until_fallback` 존재

28. `TODO` 최소 contract test 2: 4단계 구조
    - 체크:
      - 정상 fixture: stage count = 4
      - edge fixture: 0개 -> 빈 리스트 / 1~3개 -> 입력 개수만큼 반환
      - current stage = exactly 1
      - each stage has label/start/end

29. `TODO` 최소 contract test 3: fallback UX
    - 체크:
      - `next_mahadasha_date` None/past -> fallback flag true
      - markdown에 부드러운 설명 문구 포함

30. `TODO` 최소 contract test 4: P1 미노출
    - 체크:
      - 고점/저점 지도 없음
      - 반복 패턴 없음
      - 다음 3년 구체화 없음

31. `TODO` 수동 QA 템플릿 작성
    - 정상 경로 1개
    - fallback/edge 1개
    - 관찰 항목:
      - 이해 가능성
      - current stage 명확성
      - valid_until 문구 자연스러움
      - CTA-lite 과도하지 않음

32. `PARTIAL` existing cheap gate metrics는 재사용 가능
    - 재사용 대상:
      - cache/hash/logging 패턴
      - scored surface / postprocess audit 방법
    - 비재사용 대상:
      - generic front contract 그대로 적용하는 규칙

## Phase F. 캐시 / 응답 / PDF

33. `TODO` Phase B에서 잠근 cache key 정책 구현
    - 현재 cache key는 generic ai_reading 중심
    - Done 기준: 같은 chart라도 상품이 다르면 다른 key

34. `TODO` response payload에 product meta 직렬화
    - Done 기준: API 응답에서 메타 확인 가능

35. `TODO` PDF 경로가 life_cycle-lite payload를 그대로 소비하도록 연결
    - Done 기준: PDF에서도 `valid_until`, 방법론 카드, CTA-lite 누락 없음

36. `TODO` `chapter_blocks_hash`와 상품 렌더 결과 관계 정리
    - 권장:
      - generic hash는 generic chapter blocks용
      - life_cycle-lite는 별도 deterministic payload hash를 둘지 검토
    - 최소 기준: 캐시 충돌 없을 것

## Phase G. 문서 / 운영 정리

37. `TODO` `API.md`에 `product_type` 정책 추가
38. `TODO` QA 실행 명령을 `QUALITY_GATES.md` 또는 별도 운영 문서에 추가
39. `TODO` 샘플 응답 1개 저장
40. `TODO` 구현 완료 후 `PRODUCT_SPEC_PRD_v1_2_7.md` 정본과 코드 정합 재검수
41. `TODO` `README.md` 범위 정리
    - BTR / PDF / Next.js는 현재 backend P0 비범위 또는 향후 개발로 명시
    - 현재 구현 범위는 `life_cycle-lite` backend productization에 맞춰 설명 정리

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

1. 테스트 환경 복구 (`pytest` 설치)
2. stale test 정리 (`test_llm_token_limits.py` 우선)
3. `product_type` + orchestrator 얇게 추가
4. `render_contract.py` 추가
5. `life_cycle_helpers.py` 추가
6. `life_cycle-lite` payload builder 구현
7. `life_cycle_lite_renderer.py` 구현
8. response meta 직렬화
9. contract test 4종 + `cheap_validation_gate` release mode 추가
10. fallback/edge 수동 QA 2건 수행
11. PDF 경로 점검
12. 그 다음에만 개선 작업

---

## 8. P0 완료 정의

아래가 모두 만족되면 v1.2.7 기준 P0 완료로 봅니다.

1. `python -m pytest` 실행 가능
2. `life_cycle` 요청이 thin orchestrator를 통해 전용 경로로 분기됨
3. 응답에 `product_type`, `contract_version`, `render_profile`, `valid_until`, `valid_until_fallback` 존재
4. `life_cycle-lite` markdown이 deterministic 하게 생성됨
5. `cheap_validation_gate.py`의 `life_cycle-lite` release mode 기준 HF 16개 회귀 없음
6. 4단계 구조와 current stage가 명확히 출력됨
7. `next_mahadasha_date` 결측/과거 시 fallback UX가 자연스럽게 동작함
8. 단일 LLM 호출 정책 유지
9. `life_cycle` 상품 경로에서 product-layer 기본값 `7000`, soft QA range `6000–8000`, hard cap `9000`이 고정됨
10. P1 기능이 출력에 섞이지 않음
11. micro fixture 2건 + 수동 QA 2건 통과

---

## 9. 핵심 결론

현재 프로젝트는 "life_cycle-lite P0를 새로 시작해야 하는 상태"가 아니라, 아래 조합이 이미 있는 상태입니다.

- 엔진 코어 있음
- generic AI reading 파이프라인 있음
- polished surface / cache / cheap gate 있음
- 다샤 timing 보조 컨텍스트 있음

문제는 상품 계층이 없다는 것입니다.

즉, 이번 구현의 본질은 엔진 개발이 아니라 아래 4개입니다.

1. 상품 오케스트레이션 추가
2. life_cycle-lite 전용 deterministic payload/renderer 추가
3. 메타 계약 고정
4. 테스트 환경 + micro fixture 복구

이 4개를 먼저 닫으면 P0는 충분히 현실적인 범위입니다.
