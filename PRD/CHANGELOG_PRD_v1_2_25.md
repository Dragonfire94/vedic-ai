# CHANGELOG_PRD v1.2.25

- Source PRD: `PRODUCT_SPEC_PRD_v1_2_25.md`
- Last updated (Asia/Seoul): 2026-03-07

## 변경 로그

### v1.2.25 *(현재)*
- 표준 출고 검증 명령 `python -m pytest -q`가 최신 로컬 실행 기준 `161`개 `PermissionError`로 수집 단계에서 중단되는 점을 선행 블로커로 고정
- `backend/requirements.txt`에 `pytest`가 없고 `backend/test_llm_token_limits.py`가 현재 런타임 상수/시그니처와 충돌하는 상태를 표준 테스트 계약 오염으로 명시
- 런타임 `/ai_reading` 시그니처와 cache key에는 아직 `product_type` / `life_cycle` target contract가 없으므로, API/캐시 계층 정렬 전에는 상품 전환을 닫지 않도록 명시
- frontend 홈 진입, client 타입, chart consumer가 아직 legacy `/chart` + BTR + `AIReadingResponse.polished_reading` 계약을 가정하므로 repo-wide 적용 판단과 backend-only P0를 분리
- `backend/QUALITY_GATES.md` / `README.md`의 release gate source of truth가 아직 옛 기준이며, cheap gate도 아직 generic semantics 중심이라는 점을 선행 블로커로 승격
- `backend/API.md` target public contract는 유지하되, 구현 전까지는 목표 상태 문서라는 경고를 더 분명히 함

## Current State vs Target State (2026-03-06)

- Current state: 런타임 `/ai_reading` 경로는 아직 `product_type` / `life_cycle` 입력과 product-specific cache isolation을 지원하지 않습니다.
- Current state: frontend 진입점과 E2E는 아직 `/chart` + BTR + legacy `AIReadingResponse` 계약을 가정합니다.
- Current state: `python -m pytest -q`는 temp/log/cache 디렉터리 수집 단계에서 `161`개 `PermissionError`로 중단됩니다.
- Current state: 운영 문서의 release gate source of truth는 아직 `golden_sample_runner` / `fast_llm_gate` 기준이고, cheap gate는 아직 generic metric semantics 중심입니다.
- Target state: 이 PRD는 backend-only `life_cycle-lite` P0 전환 계약을 잠그며, repo-wide 완료 주장은 frontend/README migration 이후에만 허용합니다.

### v1.2.11
- `CTA-lite`는 P0 backend markdown에서 버튼/행동 1개만 허용하고, 추가 동의 버튼/secondary control은 UI/P1 전용으로 분리
- `CTA-lite` release gate의 버튼 수 계약과 7.1.9 CTA 문구를 동일 기준으로 정렬
- 수동 QA의 stale `모듈 순서` 문구를 현재 `life_cycle-lite` P0 범위 기준으로 정리

### v1.2.10
- `life_cycle-lite` release mode에서 HF11/HF12 적용 섹션을 product-specific 기준으로 재정의하고, generic 첫 본문/행동 종결 규칙의 직접 재사용을 금지
- 요약 H2 명칭을 `인생 구조 한 장 요약`으로 단일화하고 HF16 regex 및 exact H2 계약을 같은 이름으로 정렬
- P0 exact H2 순서를 고정하고 `CTA-lite`를 `valid_until 설명` 바로 뒤에 배치해 HF14 source token 기준을 잠금
- `SECTION_EXCLUDE_ALIAS` 기반 공통 파싱 위에 `life_cycle-lite` 전용 HF11/HF12 helper 집합을 추가

### v1.2.9
- `cheap_validation_gate.py`의 `life_cycle-lite` release mode에서 HF2/HF3를 **전용 product-specific scan helper**로 계산하고, generic 12챕터/front contract path 재사용을 금지
- `assign_life_stages()` edge case와 renderer/current-stage 책임을 stage 수 0~4 범위에서 다시 잠금
- P0 수용 기준의 current-stage/edge-stage 표현을 정상 fixture와 edge fixture 기준으로 명시
- 잔여 현재 버전 문구를 `v1.2.9` 기준으로 정리

### v1.2.8
- `cheap_validation_gate.py`의 `life_cycle-lite` release mode에서 HF2/HF3를 generic front contract가 아니라 product-specific 계약으로 계산하도록 고정
- `assign_life_stages()` edge case와 renderer 출력 규칙을 stage 수 0~4 범위에서 명시적으로 잠금
- P0 수동 QA에서 `yearly_forecast` 확인 항목을 제거하고, bugfix-only touched release에서만 별도 수행하도록 수정
- 잔여 `v1.2.6` 문구를 현재 버전 기준으로 정리

### v1.2.7
- `life_cycle-lite` P0 필수 섹션 집합을 PRD와 체크리스트에서 동일하게 잠금 (`How to use 1p`, 별도 면책/윤리 포함)
- HF 16개 회귀 없음의 최종 판정 경로를 `cheap_validation_gate.py`의 `life_cycle-lite` release mode로 고정
- `valid_until_fallback`를 응답 meta 필드 + gate_summary mirror metric으로 명시
- `compute_valid_until_lifecycle()`를 pure function으로 유지하고, 운영 알림 책임을 호출 파이프라인으로 분리
- `CTA-lite`는 미출시 상품명 대신 알림/업데이트 리마인드 액션만 허용하도록 수정

### v1.2.6
- `PATCH v1.2.5`의 balanced token budget 수치를 본문 정본에 실제 반영
- token budget를 계층별로 고정: generic `/ai_reading` global 상수는 유지, `life_cycle-lite` product-layer에서만 `7000 / 6000–8000 / 9000` 적용
- `commercial_gate_helpers.py`는 HF 게이트 전용으로 고정하고, `life_cycle_helpers.py`를 인생 주기 pure function 모듈로 분리

### v1.2.5
- balanced token budget 상향: 기본 `llm_max_tokens=7000`, 권장 범위 `6000–8000`, 예외 상한 `9000`

### v1.2.4
- 섹션 10.2 `fallback_applied` 오염 정리 지시를 원문 지정 방식으로 고정
- `life_cycle-full` 착수 트리거의 "연속" 단위를 `연속 2릴리즈`로 고정

### v1.2.3
- 3.1 메타 필드의 개행/CR/제어문자 오염 제거
- 3.4 제품 목록에 `life_cycle-lite` P0 / P1 범위 구분을 직접 병기
- 7.1 blockquote 뒤 빈 줄 추가, 10.2 QA 근거 명시, 14.5-a P1 착수 트리거 추가

### v1.2.2
- `life_cycle` P0를 `life_cycle-lite`로 축소
- P0 수용 기준을 "읽히는 구조 + fallback 안전성 + HF 통과" 중심으로 단순화
- 전환 강도/반복 패턴/다음 3년 구체화는 P1로 이동

### v1.2.1
- `life_cycle` 수용 기준이 닫히기 전까지 `yearly_forecast` / `compatibility`는 bugfix-only
- balanced token budget 도입: 기본 `llm_max_tokens=6000`, 권장 범위 `5000–7000`, 예외 상한 `8000`

### v1.2.0
- golden fixture 중심 검증 제거
- 전체 snapshot 대신 micro fixture + 계약 테스트 + 제한된 수동 QA로 전환
- 현재 프로젝트 기준 구현 가능 범위 재정의
### v1.1.9

#### [수정] 마크다운 코드블록 중첩 포맷 수정
- `~~~markdown` 블록 안에 ` ```python ` 을 넣는 중첩 구조를 `~~~~` 펜스 또는
  블록 제거 방식으로 정리. 렌더러 파싱 오류 방지.

#### [수정] delta=11 테스트 케이스 주석 강화
- "실제 데이터에서는 발생 불가" 표현을 "순수 comparator 경계 검증용 — 데이터 불변식
  위반 케이스, 로직 회귀 방지 목적"으로 강화.

#### [수정] N=3에서 "상" 등급 미발생 UX 안내 문구 추가 (섹션 7.1.5)
- 전환점이 3개일 때 "하/중/중" 만 출력되면 소비자가 "왜 상이 없어?"를 느낄 수 있음.
- 렌더 계층에서 전환점 섹션 상단에 1줄 안내 문구를 조건부로 삽입하는 규칙 추가.
- PRD 게이트 계약 변경 없음.

#### [수정] valid_until_fallback 소비자 노출 UX 규칙 추가 (섹션 7.1.8-a)
- `fallback_applied=True`일 때 소비자 본문에 유효기간이 갑자기 +3년으로 노출되면 어색.
- 특히 `next_mahadasha_date`가 과거 날짜 오류일 때 설명 없이 노출되는 문제.
- 소비자 노출 문구를 부드럽게 처리하는 규칙을 LOCK으로 추가.

#### [수정] 분위수 floor 절사 "정확한 33% 분할 아님" 명시 (섹션 7.1.5)
- `int(n * 0.33)` floor 절사 방식은 N=4일 때 인덱스 1, 2로 약간 기울어진 분포가 나올 수 있음.
- 버그가 아니라 의도된 방식이지만 "정확한 33% 분할이 아님"을 한 줄로 명시.

---

### v1.1.8
- 오타 2개 (`이름` → `이르면`, `분위 tie-break` → `분위수 tie-break`)
- 분위수 인덱스 floor 방식 명시 + N=3 의도된 동작 고정

### v1.1.7
- 전환 강도 문서·코드 통일 (strict `>`)
- 파싱 계층 계약 고정 (`date | None` 시그니처)

### v1.1.6
- `PLANET_LABEL_MAP` 상수 명시
- `next_mahadasha_date` 결측/과거 fallback 규칙
- 렌더/후처리 계층 vs 엔진 산식 비범위 경계 명시
- 전환 강도 분위수 tie-break + 엣지케이스 문서화
- 자동 테스트 / 수동 QA 항목 분리

### v1.1.5
- 제품 1: 인생 주기 리포트(Life Cycle Map) 재설계
- 행성 주제 라벨 변환 테이블 / 인생 단계 그룹핑 / 고점저점 / 반복 패턴 / 예언 리스크 방어

### v1.1.4
- HF16 name 정규화 함수 책임 분리
- 짧은 이름 경계 regex — lookahead/lookbehind + 조사 허용
- H2=0 배포 임계치 분모 공식 명시
- `_name_boundary_re()` lru_cache 캐싱

### v1.1.3
- HF16 짧은 이름(≤ 2글자) 경계 매칭
- H2=0 배포 임계치 (5% warn / 20% block)
- `## [면책]` 혼합 헤더 정규화 테스트

### v1.1.2
- HF16 name 매칭 정규화 (`_normalize_name()`)
- H2 0개 문서 파서 fallback
- `_normalize_header_for_match()` 헤더 정규화 함수
- HF15 품질 조건 수치·패턴 확정

### v1.1.1
- HF16 명칭 `name_token_min_exposure` 확정
- 섹션 경계 H2 기준 고정 / SECTION_EXCLUDE_ALIAS exact match
- HF15 본문 최소 품질 조건 / 모듈 분리

### v1.1.0
- 개인화 슬롯 5종 / 온보딩 목표 분기 / 월별 12슬롯 / HF15·16 추가

### v1.0.3–v1.0.9
- 상업용 설계 철학, CTA/업셀 규칙, HARD FAIL 14개 도입

---

