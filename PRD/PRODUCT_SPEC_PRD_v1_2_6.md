# PRODUCT_SPEC_PRD v1.2.6 — 상업용 베딕 리포트 엔진 (Contract-Complete)

- **Status:** LOCKED (상세 계약 보존 + v1.2.6 실행 범위 분리)
- **Last updated (Asia/Seoul):** 2026-03-06
- **Scope focus:** "그럴듯한 품질의 리포트를 안정적으로 출력" + 개인화 분기 / 12슬롯 / HF 게이트 완전 잠금 / 인생 주기 리포트(Life Cycle Map) / balanced token budget / micro fixture 검증 / `life_cycle-lite` P0
- **Explicitly out of scope:**
  - Frontend/UI (Next.js / 결제 / 로그인 / 대시보드 / 다운로드 UX 등)
  - **BTR 생시보정:** 출시 후 개발 예정 — 현재 기능 **OFF**
  - 엔진(점성 계산) 알고리즘 변경 (다샤/트랜짓/점수 산식) — 본 PRD 범위 밖
  - LLM 추가 호출 증가 금지 (후처리로만 품질을 끌어올린다)

---

## 변경 로그

### v1.2.6 *(현재)*
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

## 목차

- 0. 문서 목적 및 상업용 설계 철학
- 1. LOCK 항목 요약
- 2. 상업용 필수 조건 A–F
- 3. 제품 라인업 (3종) + 메타 필드 계약
- 4. 공통 출력 구조
- 5. 공통 규칙 (문체/개인화/분기)
- 6. 기간/타임라인 표준 (A/B/C)
- 7. 제품별 상세 요구사항 + CTA/업셀 규칙
- 8. Scored Surface/표면/해시 정합 계약
- 9. 품질 게이트 (HARD FAIL 16개 + 패턴 정의)
- 10. 테스트/검증
- 11. 보안/개인정보/윤리
- 12. 로드맵
- 13. 변경 관리
- 14. v1.2.6 실행 우선순위 및 남은 리스크

---

## 0. 문서 목적 및 상업용 설계 철학

### 0.1 문서 목적

이 문서는 "사람들이 돈을 내고 살 만한 상업용 베딕 리포트"를 만들기 위한
**제품 설계서 + 출고 기준(게이트) 계약서**입니다.

- **LOCK 항목**은 v1.2.6 내에서 변경 금지입니다 (필요 시 후속 버전으로 버전업).

### 0.5 상업용 베딕 리포트 설계 철학 (LOCK)

#### 0.5.1 전통 베딕 vs 상업용 베딕 — 차별점 (LOCK)

| 항목 | 전통 베딕 리포트 | **상업용 베딕 리포트 (본 제품)** |
|---|---|---|
| 독자 | 점성술 관심자 / 전문가 | 점성술 비전문자 일반 소비자 |
| 목적 | 차트 해석의 정확성 / 완결성 | **읽힘 → 납득 → 행동 → 재구매** |
| 문체 | 전문 용어, 산스크리트어 중심 | 일상 언어 + 전문 용어 병기 (괄호 설명 필수) |
| 구조 | 차트 요소 순서대로 나열 | **소비자 궁금증 순서로** 배치 |
| 길이 | 완결성 우선 | **한 화면에 핵심 1개, 행동 1개**로 집약 |
| 성과 지표 | 해석의 깊이 | **구매 전환율 / 완독률 / 업셀 클릭률 / 재구매율** |

#### 0.5.2 상업용 리포트 설계 3대 원칙 (LOCK)

**원칙 1 — 공감 먼저, 정보는 그 다음**
- 리포트 본문 첫 섹션은 반드시 소비자의 "현재 감정/상황"에 공감하는 문구로 시작.
- 좋은 예: "요즘 결정을 내릴 때마다 묘하게 망설임이 생긴다면, 그건 우연이 아닐 수 있습니다."
- 나쁜 예: "현재 금성 다샤(Shukra Dasha) 7년 주기가 진행 중입니다."

**원칙 2 — 모든 섹션은 행동으로 끝난다**
- 각 섹션의 마지막 요소는 반드시 **행동 1개 (오늘 할 수 있는 것)**.
- 행동은 "동사 + 시간/횟수 + 조건" 포맷.

**원칙 3 — 리포트는 판매 채널이다**
- CTA는 리포트 내용의 "자연스러운 다음 단계"처럼 느껴져야 함.

#### 0.5.3 소비자 여정 설계

```
신규 구매
  └─ 온보딩(목표 1개 선택) → 공감 첫 문장 → 현재 시즌 납득
       └─ 행동 템플릿 실행 → "나한테 맞는 리포트다" 체감
            └─ CTA(업셀/다음 리포트) → 재구매 / 구독 전환
```

#### 0.5.4 금지 설계 패턴 (LOCK)

- 산스크리트어 단독 사용 금지 → "금성 다샤(Shukra Dasha)"처럼 반드시 한국어 병기
- 3줄 이상 연속 숫자/점수 나열 금지
- "점수가 낮으니 조심하세요" 식의 부정 결론 단독 제시 금지 → 반드시 대안 행동 병기
- 1개 섹션 내 200자 이상의 순수 설명 단독 노출 금지

### 0.6 v1.2.x 실행 제약 (LOCK)

- v1.2.6의 유일한 P0 productization 대상은 `life_cycle-lite`입니다.
- `yearly_forecast`, `compatibility`는 스펙 유지 대상이지만 v1.2.6 출고 블로커는 아닙니다.
- 비용 절감은 "최소 토큰"이 아니라 **적정 토큰 예산**을 목표로 합니다.
- 전체 문서 golden snapshot은 운영하지 않고, micro fixture + 계약 테스트 + 수동 QA로 검증합니다.
- `life_cycle` full 기능(전환 강도/반복 패턴/다음 3년 구체화)은 P1로 분리합니다.
---

## 1. LOCK 항목 요약

버전업 없이 변경 금지:

1. **상업용 설계 철학** (섹션 0.5)
2. **CTA/업셀 규칙** (섹션 7)
3. **상업용 문체 규칙** (섹션 5.1)
4. **HARD FAIL 16개 + 검사 정의** (섹션 9)
5. **날짜 구간 A/B/C** (섹션 6)
6. **수치 계약** (섹션 3): Ashtakoota 임계치, valid_until, 중요구간 선정 로직
7. **중요구간 window 정의** (섹션 7.2.2): ±window 크기, 겹침 계산식
8. **메타 필드** (섹션 3.1)
9. **Scored Surface/해시 정합** (섹션 8)
10. **온보딩 목표 분기 로직** (섹션 5.6)
11. **월별 12슬롯 밀도 판정 기준** (섹션 7.2.1)
12. **섹션 경계 파싱 규칙** (섹션 9.3): H2 기준, `_normalize_header_for_match()` 순서, `SECTION_EXCLUDE_ALIAS`
13. **HF16 name 정규화 함수 책임 분리** (섹션 9.3): `_normalize_name()` vs `_normalize_text_for_name_match()`
14. **HF16 짧은 이름 경계 분기 기준** (섹션 9.3): ≤ 2글자, lookahead/lookbehind + 조사 허용
15. **H2=0 배포 임계치 및 분모 공식** (섹션 9.3)
16. **인생 주기 리포트 렌더 알고리즘** (섹션 7.1.3–7.1.6): 행성 라벨 테이블, 단계 그룹핑, 고점저점, 반복 패턴
17. **전환 강도 분위수 판정 규칙** (섹션 7.1.5): strict `>`, floor 절사, 엣지케이스
18. **valid_until_fallback 소비자 UX 문구** (섹션 7.1.8-a)
19. **life_cycle-lite P0 범위 동결 정책** (섹션 0.6, 3.4, 14)
20. **balanced token budget** (섹션 5.7): `life_cycle-lite` product-layer 기준 기본 `llm_max_tokens=7000`, 권장 범위 `6000-8000`, 예외 상한 `9000` (generic `/ai_reading` global 상수 직접 변경 금지 포함)
21. **micro fixture 중심 검증 전략** (섹션 8.4, 10.0)
22. **yearly_forecast / compatibility bugfix-only 정책** (섹션 3.4, 14)

---

## 2. 상업용 필수 조건 A–F

### A. "Why Vedic?" 납득이 약함
- **Acceptance**: 방법론 카드 존재 — **HF15로 자동 차단** / 상대 기간 표기 0건

### B. 문장/편집 오류 존재
- **Acceptance**: HARD FAIL 16개 전부 통과

### C. 기간 예측이 "달/다음 달"로만 됨
- **Acceptance**: 상대 월/주 표현 0건 (`RELATIVE_MONTH_RE`)

### D. 개인화 레이어 약함
- **Acceptance**: 이름/호칭(①) 입력값 있을 때 요약 섹션에 1회 이상 — **HF16으로 자동 차단**

### E. 내부 점수 노출 역효과
- **Acceptance**: 내부 score/grade 원값이 소비자 본문에 직접 노출 0건

### F. 사용설명서 없음
- **Acceptance**: How to use 1페이지 존재 + 체크리스트 포함

---

## 3. 제품 라인업 (3종) + 메타 필드 계약

### 3.1 리포트 메타 필드 계약 (LOCK)

| 필드 | 형식 | 설명 |
|---|---|---|
| `as_of_utc` | ISO-8601 | 서버 UTC 기준 계산 시점 |
| `as_of_local` | ISO-8601 | 사용자 타임존 변환값 |
| `timezone_offset` | 예: +09:00 | 입력 타임존 오프셋 |
| `valid_until` | YYYY-MM-DD HH:MM:SS (local) | 제품별 유효기간 |
| `onboarding_goal` | 문자열 (4종 중 1) | 온보딩 목표 선택 결과 |
| `current_mahadasha_planet` | 행성 코드 (2자) | 현재 마하다샤 주인 행성 (인생 주기 리포트 전용) |
| `next_mahadasha_date` | ISO-8601 date | 다음 마하다샤 전환일 (valid_until 계산용) |
| `product_type` | enum: `life_cycle` / `yearly_forecast` / `compatibility` | 상품 분기 키 |
| `contract_version` | 문자열 | 적용 PRD 계약 버전 (`v1.2.6`) |
| `render_profile` | 문자열 | 현재 렌더 프로파일 식별자 |

**메타 필드 편집 위생 규칙 (LOCK):**
- 메타 필드명은 전부 백틱으로 감싼다.
- `contract_version` 값은 문서 버전과 동일한 ASCII 가시 문자열만 허용한다.
- U+000B, U+000C, literal `` `r ``, literal `\n`, literal `\r` 같은 제어문자/escape 잔류는 허용하지 않는다.
모든 날짜 범위 섹션 하단 고정 푸터:
> "표기된 기간은 계획/주의 창이며, 개인의 체감은 상황에 따라 달라질 수 있습니다."

### 3.2 제품별 valid_until (LOCK)

| 제품 | valid_until 규칙 | 소비자 표기 |
|---|---|---|
| **인생 주기 리포트** | `min(as_of_local + 3년, next_mahadasha_date)` (섹션 7.1.8-a 참조) | "유효기간: 발행 기준 3년 (또는 다음 인생 주기 전환 시점)" |
| 신년운세 리포트 | `대상 연도 12/31 23:59:59 (local)` | "유효기간: 해당 연도 종료까지" |
| 궁합 리포트 | `as_of_local + 180일` | "유효기간: 발행 기준 180일" |

### 3.3 궁합(Ashtakoota) 라벨 임계치 (LOCK)

| 점수 범위 | 라벨 |
|---|---|
| 0–17 | 낮음 |
| 18–26 | 중간 |
| 27–36 | 높음 |

### 3.4 제품 목록

1. **인생 주기 리포트 (Life Cycle Map)**: 마하다샤 기반 인생 전체 주기 지도 + 현재 위치 + 방법론/valid_until UX + 단계별 확장 구조 *(v1.2.6 구현 단계: P0는 `life_cycle-lite`; 고점/저점 지도, 반복 패턴, 다음 3년 구체화는 P1에서 복원)*
2. **신년운세 리포트**: 연간 구간 + 월별 12슬롯 + 중요구간 2개 + 분야별 모듈
3. **궁합 리포트**: Ashtakoota 36점 기반 궁합 평가 + 조율 행동

> **v1.2.6 구현 단계 계약 (LOCK)**:
> - `life_cycle`는 유일한 P0 productization 대상입니다.
> - `yearly_forecast`, `compatibility`는 bugfix-only / regression-only 정책을 적용합니다.
> - 다중 상품 동시 productization PR은 금지합니다.

---

## 4. 공통 출력 구조

### 4.1 권장 섹션 구성

| # | 섹션 | 내용 |
|---|---|---|
| 0 | 커버 | 이름 / `as_of_local` / `valid_until` / 출생정보(옵션) |
| 1 | How to use 1p | "이 리포트를 10분에 쓰는 법" + 체크리스트 + 복구 플랜 |
| 2 | 한 장 요약 | 고정 포맷 (섹션 4.3) |
| 3 | 타임라인 | 날짜 구간 + 테마 + 주의/기회/행동 |
| 4 | 분야별 모듈 4장 | **온보딩 목표 기준 순서 분기** |
| 5 | 실전 템플릿 1장 | 합의/거절/보류/협상 문장 모음 |
| 6 | 7일 시스템 | 체크박스 + 측정지표 + 복구 플랜 |
| 7 | 면책/윤리/데이터 보호 | 필수 5줄 (섹션 11.2) |
| 8 | CTA | 다음 단계 (섹션 7 규칙) |
| (옵션) | Technical Appendix | 내부 점수는 여기에만 |

### 4.2 방법론 카드 6줄 (LOCK)

| 항목 | 헤더 키워드 | 대체 키워드 | 공감 문구 예시 |
|---|---|---|---|
| 다샤(Dasha) | `다샤` | — | "지금 이 시기의 '무게'가 왜 느껴지는지 설명해줍니다." |
| 트랜짓(Transit) | `트랜짓` | — | "특정 기간에 유난히 에너지가 달리는 이유를 보여줍니다." |
| 기본 차트(D1) | `기본 차트` | `D1` | "당신 고유의 패턴이 어디서 비롯되는지 알려줍니다." |
| 해석 원칙 | `해석 원칙` | — | "예언이 아닌, 준비 방향을 잡는 지도로 씁니다." |
| 기간 표준 | `기간 표준` | `날짜 구간` | "막연한 '이번 달' 대신 실제 날짜 범위를 드립니다." |
| 면책/윤리 | `면책` | `윤리` | "선택과 결정의 책임은 언제나 본인에게 있습니다." |

### 4.3 한 장 요약 고정 포맷

- 이번 시즌 한 문장 (개인화 토큰 포함)
- 3가지 주의 — 온보딩 목표 도메인 1개 우선 배치
- 3가지 기회 — 온보딩 목표 도메인 1개 우선 배치
- 이번 주 행동 2개 (Action Steps — "동사 + 시간/횟수 + 조건")

---

## 5. 공통 규칙

### 5.1 상업용 문체 규칙 (LOCK)

1. **공감 우선 언어**: 섹션 시작 첫 문장은 정보가 아닌 독자의 현재 감정/상황 반영
2. **결핍 해소 언어**: "~하면 이 구간의 에너지를 가장 잘 활용할 수 있다" 식으로 구체적 결과 연결
3. **숫자/기간의 체감화**: 날짜 범위나 숫자 단독 제시 금지. 소비자 행동과 연결
4. **1섹션 = 1메시지**
5. **예측은 단정 금지**: "반드시/확실히/무조건" 금지
6. **불안 유발/진단 톤 금지**

포맷 표준:
```
패턴:      "이 구간에는 [패턴]이 강화될 가능성이 큽니다."
트리거:    "특히 [트리거]에서 커집니다."
대안 행동: "대신 [행동]을 먼저 하세요."
```

### 5.2 템플릿 표기 규칙 (LOCK)
- 빈칸 템플릿 금지 / 플레이스홀더는 `[대괄호]`로만 / `[ ]` (내부 공백만) 금지

### 5.3 개인화 토큰 정책 (5종, LOCK)

| # | 토큰 | 사용 위치 | 역할 |
|---|---|---|---|
| ① | 이름/호칭 | 커버, 한 장 요약, 각 모듈 첫 문장 | 개인화 체감 기반 / **HF16 검사 대상** |
| ② | 현재 목표 1~2개 | 한 장 요약, 7일 시스템 | 핵심 개인화 레이어 |
| ③ | 현재 고민/현상 2~3개 | 분야별 모듈 공감 첫 문장 | 공감 문구 재료 |
| ④ | 직업군/활동 영역 | 커리어/돈 모듈 | 문구 분기 트리거 |
| ⑤ | 관계 상태 | 관계 모듈, CTA | 궁합 CTA 자동 트리거 |

**폴백 (입력값 없을 때, LOCK):**

| 토큰 | 폴백 |
|---|---|
| ① 이름/호칭 | "당신" |
| ② 현재 목표 | "현재 중요한 목표" |
| ③ 현재 고민 | "최근 반복되는 상황" |
| ④ 직업군 | 분기 없음 (중립 문구) |
| ⑤ 관계 상태 | 궁합 CTA 트리거 없음 |

### 5.4 내부 점수/리스크 번역 규칙 (LOCK)
- 내부 score/grade는 소비자 본문에서 제거
- 소비자 본문: **라벨 + 의미 1~2문장 + 대응 행동 1~2개**로만

### 5.5 산스크리트/전문 용어 병기 규칙 (LOCK)
- 반드시 "한국어 (원어)" 형태로 병기. 예: "금성 다샤(Shukra Dasha)"
- 동일 용어는 최초 1회 병기 후 이후 섹션에서 한국어만 사용 가능

### 5.6 온보딩 목표 선택 → 문구 분기 로직 (LOCK)

#### 5.6.1 목표 선택지 4종 (LOCK)

| 목표 키 | 소비자 표시 레이블 | 연결 도메인 |
|---|---|---|
| `career_money` | 커리어 / 돈 | Career & Money 모듈 |
| `relationship` | 관계 / 인간관계 | Love & Relationship Patterns 모듈 |
| `condition` | 컨디션 / 에너지 | Health & Energy Rhythm 모듈 |
| `life_direction` | 인생방향 / 큰 그림 | Mid-Term Direction 모듈 |

#### 5.6.2 분기 적용 규칙 (결정론, LOCK)

**레이어 1 — 분야별 모듈 배치 순서:**

| `onboarding_goal` | 모듈 순서 |
|---|---|
| `career_money` | Career & Money → Health → Relationship → Mid-Term |
| `relationship` | Relationship → Career & Money → Health → Mid-Term |
| `condition` | Health & Energy Rhythm → Mid-Term → Career & Money → Relationship |
| `life_direction` | Mid-Term Direction → Career & Money → Relationship → Health |

**레이어 3 — Action Steps:**

```python
GOAL_TO_TOOLKIT_KEY = {
    "career_money":   "Career & Money",
    "relationship":   "Love & Relationship Patterns",
    "condition":      "Health & Energy Rhythm",
    "life_direction": "Mid-Term Direction",
}

def get_action_steps(onboarding_goal: str, toolkit: dict) -> list[str]:
    key = GOAL_TO_TOOLKIT_KEY.get(onboarding_goal, "Mid-Term Direction")
    return toolkit.get(key, [])[:2]
```

#### 5.6.3 분기 fallback
- `onboarding_goal` 비어있거나 4종 이외: `life_direction` fallback
- fallback 시 `goal_branch_applied = false` 관찰 지표 기록

---

### 5.7 v1.2.x Balanced Token Budget 운영 계약 (LOCK)

| 항목 | 규칙 |
|---|---|
| 리포트당 LLM 호출 수 | 최대 1회 |
| 적용 범위 | `life_cycle-lite` product-layer에만 적용 |
| generic global runtime | `backend/main.py`의 generic `/ai_reading` 상수는 이 계약만으로 직접 변경하지 않음 |
| 기본 `llm_max_tokens` 운영값 | `7000` |
| 권장 운영 범위 | `6000–8000` |
| 예외 상한 | `9000` (명시적 사유 있을 때만) |
| 운영 금지 | "최소치 경쟁" 식 하향 조정 |

**토큰 절감 허용 영역**:
- 고정 문구의 프롬프트 중복 설명 제거
- 메타/계약 설명의 반복 제거
- deterministic block을 프롬프트에서 재서술하던 부분 제거
- 디버그성 설명, 내부 용어 설명 제거

**토큰 절감 금지 영역**:
- 챕터 간 연결 문장
- 공감 시작 문장
- 행동 마감 문장
- `life_cycle-lite` 핵심 구조 설명 1줄
- fallback UX의 부드러운 설명 문구

**운영 원칙**:
- 비용 절감은 "최소 토큰"이 아니라 적정 예산을 목표로 함.
- generic `/ai_reading`의 global default/hard limit는 이 섹션만으로 직접 바꾸지 않습니다.
- `6000` 미만으로 기본 운영값을 낮추는 것은 금지.
- `8000` 초과는 기본값이 아니라 예외값.
- `9000` 초과는 별도 계약 변경 없이는 운영하지 않음.
## 6. 기간/타임라인 표준 (A/B/C) (LOCK)

| 레벨 | 이름 | 조건 | 출력 형식 |
|---|---|---|---|
| A | 정밀/이벤트 기반 | 다샤/트랜짓 이벤트 앵커 있음 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 + 행동 1개 |
| B | 슬롯 기반 | 월/분기 고정 슬롯 경계 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 + 행동 1개 |
| C | fallback | 앵커 부족 시 | `as_of_local` 기반 30일 window + 유효기간 병기 |

**금지 표현**: "이번 달/다음 달" 등 (`RELATIVE_MONTH_RE`), 날짜 단정 표현

---

## 7. 제품별 상세 요구사항 + CTA/업셀 규칙 (LOCK)

> CTA 공통: 버튼 텍스트 **20자 이내, 명령형 동사로 시작**. 직전 섹션 키워드와 의미 연결 없으면 HF14.

---

### 7.1 인생 주기 리포트 (Life Cycle Map)

> **v1.2.6 구현 범위 표기 (LOCK)**:
> - P0(`life_cycle-lite`): 7.1.1, 7.1.2의 최소 출력 구조, 7.1.3, 7.1.4, 7.1.8-a, 7.1.8-b, 7.1.9의 CTA-lite.
> - P1(`life_cycle-full`): 7.1.5, 7.1.6, 7.1.7의 full productization.
> - 아래 상세 계약은 보존되며, v1.2.6에서는 P1 기능 미구현만으로 FAIL 처리하지 않습니다.

#### 7.1.1 목적/핵심 가치

- **핵심 질문**: "내 인생의 큰 파도는 언제 오고, 언제 내려가는가?"
- **엔진 기반**: Vimshottari Dasha 120년 주기 전체 계산값 활용
- **구조 전략 (LOCK)**: 인생 전체 70% + 다음 3년 구체화 30%

#### 7.1.2 출력 구조 (LOCK)

| # | 섹션 | 엔진 입력 | 분량 캡 |
|---|---|---|---|
| 0 | 커버 | 이름, 출생정보, 적용 기간(출생~80세) | 1페이지 |
| 1 | How to use 1p | — | 1페이지 |
| 2 | 인생 구조 한 장 요약 | 4단계 구조 + 현재 위치 | 1페이지 |
| 3 | 마하다샤 인생 단계 목록 | 전체 다샤 날짜 + 주제 라벨 | 행당 4~6줄, 총 9행 이내 |
| 4 | 인생 고점/저점 지도 | pressure_score 기반 | 상승 3 + 전환점 5 + 경계 3 |
| 5 | 반복 패턴 분석 | 행성 도메인별 반복 구간 | 3개 영역 × 최대 4줄 |
| 6 | 다음 3년 구체화 | 현재 부크티 전환 일정 | 최대 5슬롯 |
| 7 | 방법론 카드 6줄 + 면책 | — | HF15 자동 검증 |
| 8 | CTA | — | 7.1.9 규칙 |

> **렌더 계층 정의 (LOCK)**: 섹션 7.1.3–7.1.6의 알고리즘(`assign_life_stages`,
> `compute_life_highs_lows`, `compute_repeat_patterns`)은 **엔진이 이미 계산한**
> 다샤 날짜·pressure_score를 받아 소비자 텍스트 구조로 변환하는
> **상업 렌더/후처리 계층**입니다. 엔진 내부 산식과 무관하며
> "엔진 산식 변경 금지" 비범위 조항은 이 함수들에 적용되지 않습니다.

#### 7.1.3 행성 다샤 → 소비자 주제 라벨 변환 테이블 (LOCK)

> 엔진이 출력하는 행성 코드를 소비자 본문으로 변환할 때 **반드시 이 테이블만 사용**.
> 구현자 임의 해석 금지.

| 행성 코드 | 한국어명 | 소비자 주제 라벨 | 핵심 키워드 | 기본 톤 |
|---|---|---|---|---|
| `SU` | 태양(Surya) | 자아 확립·리더십 | 책임, 인정, 중심 | 상승 |
| `MO` | 달(Chandra) | 감정·관계·직관 | 돌봄, 유동, 가정 | 중립 |
| `MA` | 화성(Mangal) | 에너지·도전·행동 | 추진, 충돌, 개척 | 혼합 |
| `RA` | 라후(Rahu) | 확장·혼돈·야망 | 새 환경, 불확실, 성장 | 혼합 |
| `JU` | 목성(Guru) | 성장·지혜·풍요 | 확장, 배움, 기회 | 상승 |
| `SA` | 토성(Shani) | 압축·카르마·책임 | 느림, 인내, 결실 | 경계 |
| `ME` | 수성(Budha) | 소통·분석·학습 | 커뮤니케이션, 지식, 변화 | 중립 |
| `KE` | 케투(Ketu) | 내면·해방·영성 | 과거 청산, 직관, 고독 | 경계 |
| `VE` | 금성(Shukra) | 관계·창의·물질 | 인간관계, 예술, 안락 | 상승 |

**기본 톤 사용 규칙 (LOCK)**:
- `상승`: 기회/성장 프레임 → 행동은 "확장" 계열
- `경계`: 주의/인내 프레임 → 행동은 "보호/점검" 계열 + **부정 결론 단독 제시 금지**
- `중립`: 앞뒤 다샤 맥락에 따라 결정

#### 7.1.4 인생 단계 그룹핑 알고리즘 (결정론, LOCK)

```python
# life_cycle_helpers.py

LIFE_STAGE_LABELS = ["기반 형성", "방향 탐색", "사회적 확장", "영향력 축적"]

def assign_life_stages(dashas: list[dict], birth_year: int) -> list[dict]:
    """
    마하다샤 sequence를 4단계로 그룹핑.
    규칙:
    1. 총 마하다샤 수 N을 4로 나눠 그룹 경계 인덱스 산출 (정수 나눗셈).
       예: N=9 → 그룹 크기 [2, 2, 2, 3]
    2. 각 그룹 라벨 = LIFE_STAGE_LABELS[i] (0-based).
    3. 각 그룹 기간 = 첫 번째 다샤 시작일 ~ 마지막 다샤 종료일.
    4. 그룹 내 지배적 행성(기간 합산 최장)을 그룹 대표 행성으로 선정.
    5. 현재 날짜가 속한 그룹에 "현재 위치" 마킹.
    """
    n = len(dashas)
    base, rem = divmod(n, 4)
    sizes = [base + (1 if i < rem else 0) for i in range(4)]

    stages, idx = [], 0
    for stage_i, size in enumerate(sizes):
        group = dashas[idx: idx + size]
        if not group:
            break
        dominant = max(group, key=lambda d: d["duration_days"])["planet"]
        stages.append({
            "label": LIFE_STAGE_LABELS[stage_i],
            "start": group[0]["start"],
            "end":   group[-1]["end"],
            "dominant_planet": dominant,
            "dashas": group,
        })
        idx += size
    return stages
```

**소비자 한 장 요약 출력 포맷 (고정)**:
```
당신의 인생은 "[0라벨] → [1라벨] → [2라벨] → [3라벨]" 구조로 설계되어 있습니다.

• [0시작년]~[0종료년]: [0라벨] — [0대표 행성 소비자 주제 라벨]
• [1시작년]~[1종료년]: [1라벨] — [1대표 행성 소비자 주제 라벨]
• [2시작년]~[2종료년]: [2라벨] — [2대표 행성 소비자 주제 라벨]
• [3시작년]~[3종료년]: [3라벨] — [3대표 행성 소비자 주제 라벨]

👉 현재 당신은 [현재 그룹 라벨] 구간에 있습니다.
```

#### 7.1.5 고점/저점 지도 산출 기준 (결정론, LOCK, v1.2.6 구현 단계: P1)

**전환 강도 분위수 판정 (LOCK)**

```python
# commercial_quality_constants.py
TRANSITION_INTENSITY_THRESHOLDS: tuple[float, float] = (0.33, 0.67)
```

**분위수 인덱스 계산 방식 (LOCK)**:

```python
low_thresh  = sorted_deltas[int(n * 0.33)]  # floor 절사
high_thresh = sorted_deltas[int(n * 0.67)]  # floor 절사
```

> **⚠ 정확한 33% 분할이 아님 (LOCK)**: `int()` floor 절사를 사용하므로
> N=4일 때 `low_thresh = sorted_deltas[1]`, `high_thresh = sorted_deltas[2]`로
> 33%/67% 경계가 정확히 일치하지 않을 수 있습니다. 이는 의도된 방식이며,
> 보간(interpolation) 없이 실제 존재하는 값을 경계로 사용하는 단순성을 우선합니다.

| N 값 | low_thresh 인덱스 | high_thresh 인덱스 | 비고 |
|---|---|---|---|
| ≤ 2 | — | — | "중" 고정, 이 계산 skip |
| 3 | 0 | 2 (최대값) | **"상" 나올 수 없음 — 의도된 동작** |
| 4 | 1 | 2 | 약간 기울어진 분포 — 의도된 동작 |
| 6 | 1 | 4 | 정상 분포 |
| 9 | 2 | 6 | 정상 분포 |

> **N=3에서 "상" 미발생 UX 안내 (LOCK)**: N=3일 때 "하/중/중" 만 나오면
> 소비자가 "왜 상이 없어?"를 느낄 수 있습니다.
> 렌더 계층은 전환점 섹션 상단에 아래 문구를 **조건부로 삽입**합니다.
>
> **삽입 조건**: `len(transitions) <= 3`이고 "상" 등급이 0개일 때
>
> **삽입 문구 (고정)**:
> `"전환점이 적을 때는 급격한 상승보다 안정적인 흐름이 이어지는 경우가 많습니다."`

**등급 판정 기준 (strict greater-than, LOCK)**:

| delta 값 | 등급 |
|---|---|
| `delta > high_thresh` | **상** |
| `low_thresh < delta ≤ high_thresh` | **중** |
| `delta ≤ low_thresh` | **하** |

**엣지케이스 규칙 (LOCK)**:

| 상황 | 처리 |
|---|---|
| N = 0 | 전환점 섹션 생략 |
| 1 ≤ N ≤ 2 | 전부 "중" 고정, 분위 계산 skip |
| 경계값 = `low_thresh` | **하** (`≤` 기준) |
| 경계값 = `high_thresh` | **중** (`≤` 기준) |
| 동률 delta | 별도 재승격 없음 — strict `>` 기준 그대로 적용 |

```python
# life_cycle_helpers.py

def _get_transition_intensity(delta: float, sorted_deltas: list[float]) -> str:
    """
    등급 기준 (LOCK):
    - delta > high_thresh → "상"
    - low_thresh < delta ≤ high_thresh → "중"
    - delta ≤ low_thresh → "하"
    N ≤ 2이면 호출부에서 "중" 고정, 이 함수 미호출.
    """
    n = len(sorted_deltas)
    low_thresh  = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[0])]
    high_thresh = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[1])]

    if delta > high_thresh:
        return "상"
    if delta > low_thresh:
        return "중"
    return "하"


def compute_life_highs_lows(dashas_with_score: list[dict]) -> dict:
    sorted_by_score = sorted(dashas_with_score, key=lambda d: d["pressure_score"])
    lows  = sorted_by_score[:3]
    highs = sorted_by_score[-3:]

    transitions = []
    for i in range(1, len(dashas_with_score)):
        delta = abs(dashas_with_score[i]["pressure_score"]
                    - dashas_with_score[i-1]["pressure_score"])
        transitions.append({
            "date":  dashas_with_score[i]["start"],
            "delta": delta,
            "from":  dashas_with_score[i-1]["planet"],
            "to":    dashas_with_score[i]["planet"],
        })
    top_transitions = sorted(transitions, key=lambda t: t["delta"], reverse=True)[:5]

    # 전환 강도 라벨 부여
    all_deltas = sorted([t["delta"] for t in transitions])
    for t in top_transitions:
        if len(all_deltas) <= 2:
            t["intensity"] = "중"
        else:
            t["intensity"] = _get_transition_intensity(t["delta"], all_deltas)

    return {"highs": highs, "lows": lows, "transitions": top_transitions}
```

**소비자 출력 포맷 (고정)**:
```
🔺 가장 상승 가능성 높은 3구간
  1. YYYY-MM-DD ~ YYYY-MM-DD — [행성 주제 라벨] / 기회 창

⚡ 인생 전환점 5개
  [N=3이고 "상" 0개일 때 안내 문구 조건부 삽입]
  1. YYYY-MM-DD — [이전 행성 주제] → [이후 행성 주제] / 전환 강도: [상/중/하]

⚠ 경계해야 할 구간 3개
  1. YYYY-MM-DD ~ YYYY-MM-DD — [행성 주제 라벨] / 주의 창 + 대응 행동 1개
```

#### 7.1.6 반복 패턴 분석 기준 (결정론, LOCK, v1.2.6 구현 단계: P1)

```python
# life_cycle_helpers.py

def compute_repeat_patterns(dashas: list[dict]) -> dict:
    """
    도메인별 반복 구간 리스트 산출.
    같은 도메인 행성이 2회 이상 등장하면 "반복 패턴"으로 정의.
    """
    patterns = {}
    for domain, planets in PLANET_DOMAIN_MAP.items():
        occurrences = [d for d in dashas if d["planet"] in planets]
        if len(occurrences) >= 2:
            patterns[domain] = occurrences
    return patterns
```

**소비자 출력 포맷 (고정, 도메인당 최대 4줄)**:
```
[도메인명] 반복 시기
  • YYYY ~ YYYY ([행성 주제 라벨]) — 왜 이 시기에 반복되는가 1줄
  👉 이 패턴에서 벗어나기 위한 행동: [행동 1개]
```

#### 7.1.7 다음 3년 구체화 섹션 (재구매 훅, LOCK, v1.2.6 구현 단계: P1)

**목적**: 인생 전체(70%) 이후 "지금 당장 무엇을 해야 하는가"(30%) 연결.

**출력 규칙 (LOCK)**:
- 향후 3년 이내 부크티 전환점을 날짜 구간 A 기준으로 최대 5슬롯 출력.
- 슬롯이 5개 미만이면 있는 만큼만 (패딩 금지).
- 슬롯이 0개이면 고정 문구만 출력.
- 마지막 슬롯 이후 고정 문구 삽입 (LOCK):

```
이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.
3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요.
```

#### 7.1.8-a valid_until 결측 fallback 규칙 (LOCK)

**파싱 계층 계약 (LOCK)**: `next_mahadasha_date`의 파싱(문자열 → date 변환)은
`compute_valid_until_lifecycle()` 호출 **이전** 계층(메타 파이프라인)에서 책임진다.
파싱 실패 시 파이프라인에서 `None`으로 변환한 뒤 호출한다.
`compute_valid_until_lifecycle()`은 `date | None`만 수신한다고 가정한다.

| 상황 | valid_until 값 | 소비자 노출 |
|---|---|---|
| 정상값 존재 | `min(as_of_local + 3년, next_mahadasha_date)` | 계산된 날짜 그대로 |
| null (입력 없음) | `as_of_local + 3년` + `valid_until_fallback=True` | **소비자 UX 규칙 적용** |
| 파싱 실패 | 파이프라인에서 `None` 변환 → null 케이스 동일 | **소비자 UX 규칙 적용** |
| `next_mahadasha_date < as_of_local` | `as_of_local + 3년` + 운영 알림 + `valid_until_fallback=True` | **소비자 UX 규칙 적용** |

**소비자 UX 규칙 — valid_until_fallback=True 시 (LOCK)**:
- 소비자 본문에 `valid_until` 원값을 그대로 노출하되, 아래 **부드러운 설명 문구를 함께 표기**합니다.
- 과거 날짜 오류 등 오류 원인은 소비자 본문에 노출하지 않습니다. 운영 알림은 내부 채널로만.

**fallback 시 valid_until 표기 포맷 (고정)**:
```
유효기간: [as_of_local + 3년] 까지
(현재 가용한 주기 데이터를 기준으로 산정한 기간입니다. 이후 업데이트 리포트에서 다음 전환점을 확인할 수 있습니다.)
```

```python
# life_cycle_helpers.py

from datetime import date
from dateutil.relativedelta import relativedelta

def compute_valid_until_lifecycle(
    as_of_local: date,
    next_mahadasha_date: date | None,  # 파싱 실패는 호출 전 None으로 변환 완료
) -> tuple[date, bool]:
    """
    반환: (valid_until, valid_until_fallback)
    valid_until_fallback=True → gate_summary["valid_until_fallback"] = True 기록.
    """
    default = as_of_local + relativedelta(years=3)

    if next_mahadasha_date is None:
        return default, True

    if next_mahadasha_date <= as_of_local:
        trigger_ops_alert("next_mahadasha_date_past", str(next_mahadasha_date))
        return default, True

    return min(default, next_mahadasha_date), False
```

#### 7.1.8-b 예언 리스크 방어 문체 규칙 (LOCK)

**금지 표현 (이 제품 전용 추가 목록)**:

| 금지 패턴 | 이유 |
|---|---|
| "이 시기에 반드시 ~이 일어납니다" | 단정 예언 |
| "~살이 되면 ~를 경험합니다" | 확정적 미래 기술 |
| "이 구간에서 큰돈을 벌 수 있습니다" | 투자 조언 유사 |
| "이 시기에 결혼/이별이 옵니다" | 관계 확정 예언 |
| "~다샤가 끝나면 모든 게 해결됩니다" | 근거 없는 보장 |

**권장 대체 표현**:

| 금지 | 권장 |
|---|---|
| "반드시 일어납니다" | "가능성이 높아집니다" |
| "큰돈을 법니다" | "물질적 확장의 기회가 열리는 구간입니다" |
| "결혼이 옵니다" | "관계가 깊어지거나 새로운 연결이 생기기 쉬운 구간입니다" |
| "끝나면 해결됩니다" | "이 구간 이후 새로운 방향성이 열릴 가능성이 큽니다" |

#### 7.1.9 CTA/업셀 규칙 (LOCK)

- **업셀 트리거 (결정론)**: *(v1.2.6 P0에서는 CTA-lite만 필수, 최적화는 P1)*
  - `onboarding_goal == "relationship"` 또는 ⑤ 관계 상태 입력 있음 → 궁합 리포트 CTA
  - `valid_until`이 `next_mahadasha_date` 기준인 경우 → "전환점 심화 리포트" CTA 우선
  - 그 외 → 신년운세 리포트 CTA
- **문구 포맷**: "[현재 인생 단계 라벨]에서 [다음 단계 라벨]로 넘어가는 당신에게,
  [다음 상품명]이 더 구체적인 방향을 드릴 수 있습니다."
- **버튼 텍스트**: 20자 이내. 예: "신년운세 리포트 보기", "궁합 리포트 확인하기"
- **유효기간 만료 알림**: CTA 하단에 "다음 인생 주기 전환점 알림을 받으시겠습니까?" 동의 버튼 병기

---

### 7.2 신년운세 리포트

#### 7.2.1 연간 구조 — 월별 12슬롯 (LOCK)

| `pressure_score_monthly` | 표시 방식 | 출력 항목 |
|---|---|---|
| ≥ 60 | **전체 표시** | 날짜 범위 + 테마 + 주의 + 기회 + 행동 |
| < 60 | **압축 표시** | 날짜 범위 + 테마 + 행동 |
| 결측/null | **기본 압축** | 날짜 범위 + "이 달 데이터 보완 예정" + `CORE_ACTION_TOOLKIT["Mid-Term Direction"][0]` |

#### 7.2.2 중요구간 2슬롯 선정 로직 (LOCK)

- **다샤 전환 window**: 마하다샤 ±21일 / 부크티 ±7일 (연도 경계 clip)
- **겹침 판정**: `overlap_ratio >= 0.5` → 중복

```python
overlap_days  = max(0, min(a_end, b_end) - max(a_start, b_start)).days + 1
overlap_ratio = overlap_days / min(
    (a_end - a_start).days + 1,
    (b_end - b_start).days + 1
)
```

#### 7.2.3 CTA/업셀 규칙 (LOCK)
- **인라인 CTA**: 중요구간 "주의 창"일 때만 허용

---

### 7.3 궁합 리포트 (Kuta Milan)

#### 7.3.1 출력 구조
- 총점 + 라벨 + 해석 + 조율 행동
- Ashtakoota 8요소 각각: 점수 + 해석 + 리스크 + 행동

#### 7.3.2 CTA/업셀 규칙 (LOCK)

- **CTA A (개인 업셀)**: 두 사람 각각의 이름(①토큰)으로 인생 주기 리포트 CTA.
  예: "[A이름]의 인생 주기 리포트 보기" + "[B이름]의 인생 주기 리포트 보기"
- **CTA B (심화 업셀)**: 연애/결혼 → "결혼 적기 & 관계 전환점 심화 리포트" / 사업 파트너 → "파트너십 다샤 분석 심화 리포트"

---

## 8. Scored Surface/표면/해시 정합 계약 (LOCK)

### 8.1 표면 텍스트 정의

| 파일 | 역할 |
|---|---|
| `reading.md` | raw reading (디버그용) |
| `reading_scan_surface.md` | 채점 직전 스캔 표면 |
| `reading_post_remediation.md` | 후처리 적용 최종 표면 |
| `scored_surface.md` | 게이트 채점 대상 (단일 기준) |

### 8.2 Scored Surface 선택 우선순위 (LOCK)

1. `polished_reading` non-empty → `polished 후처리 결과`
2. 아니면 → `reading 후처리 결과`
3. 둘 다 없으면 → `scan surface`

### 8.3 LF 저장/해시 정합 (LOCK)

- surface 파일 3종: **LF(\n)로만 저장**
- summary에 `scan_surface_sha256`, `post_sha256`, `scored_surface_sha256` **3개 동시 기록**

---

### 8.4 v1.2.x 경량 검증 아티팩트 계약 (LOCK)

- 전체 문서 golden snapshot은 운영하지 않습니다.
- 검증은 `micro fixture + 계약 테스트 + 수동 QA` 조합으로 수행합니다.
- micro fixture는 10~40줄 내외의 짧은 텍스트/짧은 payload를 원칙으로 합니다.
- 문서 전체 동일성보다 presence/absence, flag, section count, deterministic copy insertion을 우선 검증합니다.
- P0에서는 `life_cycle-lite` 최소 구조만 검증하고, P1 기능(반복 패턴/다음 3년/전환 강도 랭킹)은 회귀 테스트 대상에서 제외할 수 있습니다.
## 9. 품질 게이트 (HARD FAIL 16개 + 패턴 정의) (LOCK)

### 9.1 HARD FAIL vs 관찰 지표 분리

- **HARD FAIL**: 출고 불가 (자동 차단)
- **관찰 지표**: 기록만 (HARD FAIL 승격 금지 — 버전업 필수)

### 9.2 HARD FAIL 목록 (LOCK)

| # | 조건 | 분류 |
|---|---|---|
| 1 | `forbidden_hits > 0` | 형식 |
| 2 | `front_contract_ok == false` | 형식 |
| 3 | `action_steps_contract_ok == false` | 형식 |
| 4 | `definition_dasha_occurrences_after != 1` | 형식 |
| 5 | `dasha_definition_redundancy_violations > 0` | 형식 |
| 6 | `action_steps_inline_heading_violations > 0` | 형식 |
| 7 | `inline_action_chain_violations > 0` | 형식 |
| 8 | 빈칸 템플릿 (`BLANK_ANY_RE` 4종 OR) > 0 | 형식 |
| 9 | 미완성 문장 (`UNFINISHED_SENTENCE_RE`) > 0 | 형식 |
| 10 | 즉시 반복 위반 > 0 (200자 이내 8자 이상 2회 이상) | 형식 |
| 11 | 공감 문구 누락 — 첫 본문 섹션 첫 문장 trigger+emotion 미만족 | 상업용 |
| 12 | 행동 없는 섹션 마감 — 마지막 판정 가능 라인에 `오늘의 행동:` / `Action:` 없음 | 상업용 |
| 13 | 산스크리트/영문 단독 노출 (`SANSKRIT_STANDALONE_RE`) | 상업용 |
| 14 | CTA 광고성 문구 — 직전 섹션 토큰과 교집합 없음 | 상업용 |
| 15 | 방법론 카드 품질 미달 — 키워드 없음 OR 본문 25자 미만 OR 동사 패턴 없음 | 상업용 |
| 16 | name_token_min_exposure — 이름(①) 입력 있을 때 요약 섹션 0회 | 상업용 |

---

### 9.3 HARD FAIL 판정 알고리즘 (LOCK)

#### 모듈 분리 원칙 (LOCK)

| 모듈 | 역할 | 포함 금지 |
|---|---|---|
| `commercial_quality_constants.py` | 패턴 상수, 정규식, 고정값 테이블 | 알고리즘 함수 |
| `commercial_gate_helpers.py` | HF 게이트 판정 알고리즘 함수 | 패턴 상수 재정의, `life_cycle` 상품 pure function |
| `life_cycle_helpers.py` | `life_cycle` 상품용 pure function (`assign_life_stages`, `compute_valid_until_lifecycle` 등) | HF 정규식/패턴 상수 재정의 |
| `cheap_validation_gate.py` | import만 사용 | 패턴/로직 재정의 |

> 7.1의 인생 주기 결정론 함수는 `life_cycle_helpers.py`에 둡니다. `commercial_gate_helpers.py`는 섹션 9의 HARD FAIL helper에만 사용합니다.

---

#### 공통 — 섹션 경계 파싱 (LOCK)

**H2(`##`) 기준으로만 섹션 분리. H3는 내부 블록. H1은 문서 제목.**

##### 헤더 정규화 `_normalize_header_for_match()` (LOCK)

| 순서 | 처리 | 예시 입력 → 출력 |
|---|---|---|
| 1 | 마크다운 강조 제거 (`**`, `*`, `__`, `_`) | `**면책**` → `면책` |
| 2 | 헤더 마커 제거 (`#` + 공백) | `## 면책` → `면책` |
| 3 | 구분 기호 정규화 (`/`, `\|`, `:`, `-`) → 단일 공백 | `면책/윤리:` → `면책 윤리 ` |
| 4 | 괄호류 제거 (`()`, `[]`, `{}`, `（）`) | `[면책]` → `면책 ` |
| 5 | 다중 공백 → 단일 공백 | `면책  윤리` → `면책 윤리` |
| 6 | 앞뒤 공백 제거 + 소문자 | `면책 윤리 ` → `면책 윤리` |

```python
# commercial_gate_helpers.py
import re, functools

_HDR_BOLD_RE   = re.compile(r"\*{1,2}|_{1,2}")
_HDR_MARKER_RE = re.compile(r"^#+\s*")
_HDR_DELIM_RE  = re.compile(r"[/|:\-]")
_HDR_BRACE_RE  = re.compile(r"[(){}\[\]（）]")
_HDR_SPACE_RE  = re.compile(r"\s{2,}")

def _normalize_header_for_match(header: str) -> str:
    h = _HDR_BOLD_RE.sub("", header)
    h = _HDR_MARKER_RE.sub("", h)
    h = _HDR_DELIM_RE.sub(" ", h)
    h = _HDR_BRACE_RE.sub("", h)
    h = _HDR_SPACE_RE.sub(" ", h)
    return h.strip().lower()
```

##### 제외 섹션 exact match (LOCK)

```python
# commercial_quality_constants.py
SECTION_EXCLUDE_ALIAS: frozenset[str] = frozenset([
    "how to use", "이 리포트를 10분에 쓰는 법", "사용법", "how to read",
    "커버", "cover",
    "면책", "면책 및 윤리", "면책 윤리 데이터 보호", "윤리", "데이터 보호",
    "cta", "다음 단계",
    "technical appendix", "기술 부록",
    "변경 관리", "로드맵", "목차",
])
```

##### H2 0개 문서 fallback (LOCK)

```python
# commercial_gate_helpers.py
def parse_sections(text: str) -> list[dict]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        header_raw = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        excluded = _is_excluded_header(header_raw)
        sections.append({"header": header_raw, "body": text[start:end].strip(),
                          "excluded": excluded})
    return sections

def _is_excluded_header(header: str) -> bool:
    return _normalize_header_for_match(header) in SECTION_EXCLUDE_ALIAS

def get_first_body_section(sections: list[dict]) -> dict | None:
    for s in sections:
        if not s["excluded"] and s["body"]:
            return s
    return None
```

```python
# cheap_validation_gate.py 호출부
sections = parse_sections(scored_surface_text)
if not sections:
    gate_summary.update({
        "h2_section_count": 0,
        "metrics_valid": False,
        "gate_skip_reason": "no_h2_sections",
    })
    trigger_ops_alert("h2_section_count_zero", report_id)
    # HF11/12/15/16 skip, HF1–10/13/14는 계속 실행
else:
    gate_summary["h2_section_count"] = len(sections)
    gate_summary["metrics_valid"] = True
```

**H2=0 배포 임계치 (LOCK)**:

```python
# commercial_quality_constants.py
H2_ZERO_DEPLOY_WARN_THRESHOLD  = 0.05
H2_ZERO_DEPLOY_BLOCK_THRESHOLD = 0.20
```

```python
def compute_h2_zero_ratio(h2_zero_count: int, total_generated_reports: int) -> float | None:
    if total_generated_reports <= 0:
        return None
    return h2_zero_count / total_generated_reports

def get_deploy_alert_level(ratio: float | None) -> str:
    if ratio is None:
        return "none"
    if ratio > H2_ZERO_DEPLOY_BLOCK_THRESHOLD:
        return "block"
    if ratio > H2_ZERO_DEPLOY_WARN_THRESHOLD:
        return "warn"
    return "none"
```

| 조건 | 조치 |
|---|---|
| 개별 리포트 `h2_section_count = 0` | HF11/12/15/16 skip, 운영 알림, 출고 허용 |
| 배포 단위 비율 **> 5%** | deploy_warn |
| 배포 단위 비율 **> 20%** | deploy_block |
| `total_generated_reports = 0` | 판정 skip |

---

#### HARD FAIL 11 — 공감 문구 누락

```python
# commercial_quality_constants.py
EMPATHY_TRIGGER_RE = re.compile(
    r"(요즘|최근|혹시|만약|자꾸|계속|어쩌면|가끔"
    r"|어느\s*순간|문득|갑자기|이유\s*없이|왠지|그동안|한동안)"
)
EMPATHY_EMOTION_RE = re.compile(
    r"(망설|답답|불안|지치|피곤|혼란|부담|막막|버겁|힘들"
    r"|무기력|지루|외롭|괴롭|두렵|헷갈|어렵)"
)
```

```python
# commercial_gate_helpers.py
def extract_first_sentence(paragraph: str) -> str:
    m = re.search(r"[.!?\n]", paragraph)
    return paragraph[:m.start()].strip() if m else paragraph.strip()

def check_empathy(first_sentence: str) -> bool:
    return (bool(re.search(EMPATHY_TRIGGER_RE, first_sentence)) and
            bool(re.search(EMPATHY_EMOTION_RE, first_sentence)))
```

---

#### HARD FAIL 12 — 행동 없는 섹션 마감

```python
# commercial_quality_constants.py
LIST_LINE_RE    = re.compile(r"^(\s*[-*]\s|\s*-\s\[[ xX]\])")
ACTION_LABEL_RE = re.compile(r"^(오늘의\s*행동\s*:|Action\s*:)")
```

```python
# commercial_gate_helpers.py
def get_last_judgeable_line(section_lines: list[str]) -> str | None:
    for line in reversed(section_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(LIST_LINE_RE, line):
            continue
        return stripped
    return None

def check_section_action(last_judgeable_line: str | None) -> bool:
    if last_judgeable_line is None:
        return False
    return bool(re.match(ACTION_LABEL_RE, last_judgeable_line))
```

---

#### HARD FAIL 14 — CTA 광고성 문구

```python
# commercial_quality_constants.py
CTA_STOPWORDS = frozenset(
    "그리고|하지만|때문에|합니다|있습니다|됩니다|위해|통해|경우|또한|대한|이런|이와|이를".split("|")
)
CTA_DOMAIN_FALLBACK_RE = re.compile(r"(관계|돈|커리어|컨디션|인생|흐름|궁합|주기)")
```

```python
# commercial_gate_helpers.py
def normalize_tokens(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^가-힣a-z\s]", " ", text)
    return {t for t in text.split() if t not in CTA_STOPWORDS and len(t) >= 2}

def check_cta_relevance(source_tokens: set[str], cta_text: str) -> bool:
    return bool(source_tokens & normalize_tokens(cta_text))
```

---

#### HARD FAIL 15 — 방법론 카드 품질 미달

3조건 **동시 충족** 필요 (조건 1: 키워드 존재, 조건 2: 25자 이상, 조건 3: 동사 패턴):

```python
# commercial_quality_constants.py
METHODOLOGY_CARD_ITEMS: list[dict] = [
    {"key": "다샤",     "alternates": ["다샤"]},
    {"key": "트랜짓",   "alternates": ["트랜짓"]},
    {"key": "기본 차트","alternates": ["기본 차트", "d1"]},
    {"key": "해석 원칙","alternates": ["해석 원칙"]},
    {"key": "기간 표준","alternates": ["기간 표준", "날짜 구간"]},
    {"key": "면책",     "alternates": ["면책", "윤리"]},
]
METHODOLOGY_CARD_SNIPPET_LEN  = 200
METHODOLOGY_CARD_MIN_BODY_LEN = 25   # 공백·줄바꿈 제거 후 글자 수

METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|보여줍니다|알려줍니다"
    r"|이해|도움|이해합니다|도움이\s*됩니다"
    r"|활용|준비|활용합니다|준비합니다"
    r"|씁니다|있습니다|됩니다|해줍니다|드립니다|입니다)"
)
```

---

#### HARD FAIL 16 — name_token_min_exposure

##### 정규화 함수 책임 분리 (LOCK)

| 함수 | 적용 대상 | 호칭 제거 |
|---|---|---|
| `_normalize_name(name_input)` | 입력 이름값 | **YES** |
| `_normalize_text_for_name_match(text)` | 검색 대상 텍스트 | **NO — 경계 컨텍스트 보존** |

```python
# commercial_gate_helpers.py
_NAME_BRACKET_RE = re.compile(r"[（）()\[\]{}]")
_NAME_MSPACE_RE  = re.compile(r"\s{2,}")

def _normalize_name(name: str) -> str:
    n = name.strip()
    n = _NAME_MSPACE_RE.sub(" ", n)
    n = re.sub(NAME_HONORIFIC_RE, "", n).strip()
    n = _NAME_BRACKET_RE.sub("", n)
    return n.lower()

def _normalize_text_for_name_match(text: str) -> str:
    n = _NAME_MSPACE_RE.sub(" ", text)
    return n.lower()
```

##### 짧은 이름 경계 매칭 (≤ 2글자, LOCK)

```python
# commercial_gate_helpers.py
@functools.lru_cache(maxsize=256)
def _name_boundary_re(name_norm: str) -> re.Pattern:
    escaped = re.escape(name_norm)
    pattern = (
        NAME_BOUNDARY_BEFORE_RE_STR
        + escaped
        + NAME_BOUNDARY_SUFFIX_RE_STR
        + NAME_BOUNDARY_PARTICLE_RE_STR
        + NAME_BOUNDARY_AFTER_RE_STR
    )
    return re.compile(pattern, re.MULTILINE)

def check_name_token_exposure(text: str, name_input: str) -> bool:
    if not name_input.strip():
        return True
    name_norm = _normalize_name(name_input)
    if not name_norm:
        return True

    m = SUMMARY_SECTION_RE.search(text)
    if not m:
        return False

    next_h2    = SECTION_HEADER_RE.search(text, m.end())
    summary_end = next_h2.start() if next_h2 else len(text)
    text_norm   = _normalize_text_for_name_match(text[m.start():summary_end])

    if len(name_norm) <= 2:
        return bool(_name_boundary_re(name_norm).search(text_norm))
    return name_norm in text_norm
```

---

### 9.4 패턴/정규식 정의 — `commercial_quality_constants.py` 기준 (LOCK)

```python
# ── 빈칸 템플릿 (LOCK) ────────────────────────────────────────────────────────
BLANK_TEMPLATE_RE   = r"(?i)\b(우리는|나는|당신은)\s+(를|을)\s+\b"
BLANK_BRACKET_RE    = r"\[\s*\]"
BLANK_UNDERSCORE_RE = r"_{2,}"
BLANK_SPACE_RE      = r"(?m)^[^\S\n]{3,}(을|를|이|가|은|는|의|에|와|과)?\s*$"
BLANK_ANY_RE = (
    f"({BLANK_TEMPLATE_RE}|{BLANK_BRACKET_RE}"
    f"|{BLANK_UNDERSCORE_RE}|{BLANK_SPACE_RE})"
)

# ── 미완성 문장 / 즉시 반복 / 상대 기간 / 산스크리트 (LOCK) ──────────────────
UNFINISHED_SENTENCE_RE      = r"(?m)(보여주는|하는|되어|있어|으로|해서)\s*$"
IMMEDIATE_REPEAT_DISTANCE_CHARS = 200
RELATIVE_MONTH_RE = (
    r"(이번\s*달|다음\s*달|그\s*다음\s*달|이번\s*주|다음\s*주"
    r"|이달|다음달|저번\s*달|지난\s*달|전\s*달"
    r"|금월|차월|전월|익월|익주|전주)"
)
SANSKRIT_STANDALONE_RE = r"(?m)^\s*[A-Za-z][A-Za-z\s]{3,}\s*$"

# ── 섹션 경계 (LOCK) ──────────────────────────────────────────────────────────
SECTION_HEADER_RE = re.compile(r"(?m)^##\s+(.+)$")

# ── name_token_min_exposure (LOCK, HF16) ──────────────────────────────────────
SUMMARY_SECTION_RE            = re.compile(r"(?m)^##\s*한\s*장\s*요약")
NAME_HONORIFIC_RE             = re.compile(r"(님|씨|군|양|선생님?|고객님|사용자|씨\s*귀하)\s*$")
NAME_BOUNDARY_BEFORE_RE_STR   = r"(?<![가-힣A-Za-z0-9])"
NAME_BOUNDARY_SUFFIX_RE_STR   = r"(?:님|씨|군|양|선생님?|고객님|사용자)?"
NAME_BOUNDARY_PARTICLE_RE_STR = r"(?:의|이|가|은|는|을|를|에|와|과|도|만|께|한테|에게)?"
NAME_BOUNDARY_AFTER_RE_STR    = r"(?![가-힣A-Za-z0-9])"

# ── H2=0 배포 임계치 (LOCK) ───────────────────────────────────────────────────
H2_ZERO_DEPLOY_WARN_THRESHOLD  = 0.05
H2_ZERO_DEPLOY_BLOCK_THRESHOLD = 0.20

# ── 온보딩 목표 키 (LOCK) ─────────────────────────────────────────────────────
ONBOARDING_GOAL_KEYS     = frozenset(["career_money", "relationship", "condition", "life_direction"])
ONBOARDING_GOAL_FALLBACK = "life_direction"

# ── 인생 주기 리포트 (LOCK) ───────────────────────────────────────────────────
PLANET_LABEL_MAP: dict[str, dict] = {
    "SU": {"label": "자아 확립·리더십",   "tone": "상승"},
    "MO": {"label": "감정·관계·직관",     "tone": "중립"},
    "MA": {"label": "에너지·도전·행동",   "tone": "혼합"},
    "RA": {"label": "확장·혼돈·야망",     "tone": "혼합"},
    "JU": {"label": "성장·지혜·풍요",     "tone": "상승"},
    "SA": {"label": "압축·카르마·책임",   "tone": "경계"},
    "ME": {"label": "소통·분석·학습",     "tone": "중립"},
    "KE": {"label": "내면·해방·영성",     "tone": "경계"},
    "VE": {"label": "관계·창의·물질",     "tone": "상승"},
}
VALID_PLANET_CODES: frozenset[str] = frozenset(PLANET_LABEL_MAP.keys())

PLANET_DOMAIN_MAP: dict[str, list[str]] = {
    "관계":       ["VE", "MO"],
    "돈·커리어":  ["JU", "SU", "ME"],
    "건강·에너지": ["MA", "RA", "KE"],
}

LIFE_STAGE_LABELS: list[str] = ["기반 형성", "방향 탐색", "사회적 확장", "영향력 축적"]

# floor 절사 방식 — 정확한 33% 분할 아님 (LOCK, 섹션 7.1.5 참조)
TRANSITION_INTENSITY_THRESHOLDS: tuple[float, float] = (0.33, 0.67)
```

### 9.5 게이트 검사 제외 블록

- 코드블록, 헤더 라인, 테이블 구분 라인, 순수 체크박스/목록 라인
- 고정 푸터/면책 문구
- `SECTION_EXCLUDE_ALIAS` 매칭 섹션 내 본문

### 9.6 관찰 지표 (HARD FAIL 승격 금지)

- `goal_branch_applied`, `monthly_slot_density_distribution`, `personalization_token_coverage_extended`
- `h2_section_count`, `gate_skip_reason`, `h2_zero_ratio_per_deploy`, `deploy_alert`
- `name_token_exposure_norm_applied`, `name_token_short_boundary_applied`
- `valid_until_fallback`: `valid_until_fallback=True` 시 기록
- `h2_zero_ratio_per_deploy`: 배포 단위 H2=0 비율

---

## 10. 테스트/검증

### 10.0 v1.2.6 구현 단계 계약 (LOCK)

- 표준 테스트 실행 명령은 `python -m pytest`입니다.
- `pytest` CLI PATH 의존은 허용하지 않습니다.
- v1.2.6 P0 수용 기준은 `life_cycle-lite` 최소 구조 중심으로 판정합니다.
- 7.1.5 / 7.1.6 / 7.1.7의 상세 테스트는 계약 보존 대상이지만, v1.2.6 P0 미구현만으로 FAIL 처리하지 않습니다.
- `yearly_forecast` / `compatibility`는 bugfix-only 정책 위반 여부만 확인합니다.
### 10.1 자동 테스트 (필수)

**HF1–10**: 기존 단위 테스트 유지

**HF11**:
- `extract_first_sentence` 구분자별 경계 (./?/!/\n)
- trigger만 또는 emotion만 → FAIL

**HF15**:
- 키워드 + 25자 이상 + 동사 → PASS
- 25자 미만 (공백·줄바꿈 제거 기준) → FAIL
- 대체어(`"D1"`, `"날짜 구간"`) 허용

**HF16**:
- 입력 없음 → skip → PASS
- 이름 있음 + 요약에 없음 → FAIL
- 짧은 이름 `"민"` + `"민님의 이번"` → PASS
- 짧은 이름 `"민"` + `"민감한 시기"` → FAIL
- 짧은 이름 `"이"` + `"이번 시즌은"` → FAIL

**섹션 파싱 — 헤더 정규화**:
- `"## [면책]"` → `"면책"` → alias PASS
- `"## [면책 / 윤리]: 확인사항"` → `"면책 윤리 확인사항"` → alias `"면책 윤리"` PASS
- `"## [공지] 중요안내"` → `"공지 중요안내"` → alias 없음 → 제외 안 됨

**H2=0 fallback**:
- H1만 있는 문서 → `h2_section_count=0`, `metrics_valid=False`
- 배포 4건/100 → `"none"`, 6건 → `"warn"`, 21건 → `"block"`, total=0 → `"none"`

**인생 주기 리포트**:
- `assign_life_stages()`: 다샤 9개 → [2,2,2,3] / 4개 → [1,1,1,1] / 0개 → 빈 리스트  **(v1.2.6 P0 필수)**
- `compute_life_highs_lows()`: 9개 → 상위3/하위3/전환점5 / 5개 → 전환점 4개  **(v1.2.6 P1)**
- `_get_transition_intensity()`:
  - N≤2 → 호출부 "중" 고정, 함수 미호출 확인
  - delta = `low_thresh` 정확히 → **"하"** 반환 (경계값 하위 등급)
  - delta = `high_thresh` 정확히 → **"중"** 반환
  - delta > `high_thresh` → "상" 반환
  - N=3, sorted_deltas=[1,5,10]: delta=10 → "중" (N=3에서 "상" 미발생 의도된 동작 확인)
  - **[comparator 경계 검증 전용 — 데이터 불변식 위반 케이스, 로직 회귀 방지 목적]**
    N=3, sorted_deltas=[1,5,10]: delta=11 → "상" 반환 확인
    (실제 sorted_deltas에서 발생 불가. strict `>` 비교 로직 자체의 정확성 검증용)
- `compute_repeat_patterns()`: 동일 도메인 2회+ → 패턴 인식 / 1회 → 제외  **(v1.2.6 P1)**
- `PLANET_LABEL_MAP`: 9개 행성 코드 각각 → 라벨+tone 정확 변환
- `VALID_PLANET_CODES`: 유효하지 않은 코드 → 오류 처리 확인
- `compute_valid_until_lifecycle()`:
  - `next` 정상값, +3년보다 **이르면** → `next` 반환, `fallback=False`
  - `next` 정상값, +3년보다 늦음 → `+3년` 반환, `fallback=False`
  - `next = None` → `+3년`, `fallback=True`
  - `next < as_of_local` → `+3년`, 운영 알림, `fallback=True`
  - 파싱 실패 케이스: 파이프라인에서 `None` 변환 후 호출 → null 케이스와 동일 동작
- N=3 UX 안내 문구: `len(transitions) <= 3`이고 "상" 0개일 때 안내 문구 삽입 확인
- valid_until_fallback UX: `valid_until_fallback=True` 시 부드러운 설명 문구 표기 확인
- 다음 3년 구체화: 3개 슬롯 → 3개 출력 (패딩 없음) / 0개 → 고정 문구만 출력  **(v1.2.6 P1)**

**모듈 분리 정적 검사**:
- `commercial_quality_constants.py`에 함수 정의 없음
- `commercial_gate_helpers.py`에 `assign_life_stages`, `compute_valid_until_lifecycle`, `compute_repeat_patterns` 같은 `life_cycle` 상품 함수 정의 없음
- `life_cycle_helpers.py`에 HF 정규식/패턴 상수 재정의 없음
- `cheap_validation_gate.py`에서 패턴/정규식 재정의 없음
- `_name_boundary_re()` lru_cache: 동일 이름 2회 호출 시 컴파일 1회

### 10.2 수동 QA (릴리즈당 최소 2건: 정상 경로 1 + fallback/edge 1)

> **운영 근거 (v1.2.6)**:
> - golden snapshot은 운영하지 않지만, v1.2.6은 P0 범위를 `life_cycle-lite`로 축소한 상태다.
> - 자동 방어는 micro fixture + 계약 테스트 + HF 게이트로 유지한다.
> - 따라서 수동 QA는 2건으로 유지하되, 유형은 반드시 `정상 경로 1건 + valid_until_fallback=True 또는 경계 케이스 1건`으로 고정한다.

- 방법론 카드 (6줄 + 공감 문구) 가독성
- 상대 기간 표현 0건 / 빈칸 템플릿 0건
- CTA 앞 섹션과 의미 연결 확인
- 온보딩 목표가 한 장 요약/모듈 순서에 반영됐는지 육안 확인
- 신년운세 12슬롯 압축/전체 표시 및 결측 달 안내 문구 확인
- **인생 주기 리포트 전용**:
  - 각 마하다샤 섹션 분량이 4~6줄 캡 이내인지 확인
  - 경계 구간에 대안 행동이 있는지 확인
  - 예언 리스크 금지 표현 8종 육안 점검 (섹션 7.1.8-b)
  - "다음 3년 구체화" 섹션 끝 업데이트 유도 고정 문구 존재 확인
  - `valid_until_fallback=True` 리포트: 부드러운 설명 문구 자연스러운 노출 확인
  - v1.2.6 P0: 4단계 구조가 과하게 복잡하지 않고 한 번에 이해되는지 확인
  - v1.2.6 P0: 전환점/반복 패턴 부재가 오히려 문서를 더 명료하게 만드는지 확인

---

## 11. 보안/개인정보/윤리

### 11.1 소비자 본문에서 제거할 것

- `request_id`, 해시, 내부 점수 원값, debug payload, 시스템 경로
- `onboarding_goal` 메타값

### 11.2 면책/윤리 가이드 (필수 5줄)

1. 의료/법률/투자 조언 아님
2. 결과 단정 금지 (가능성/패턴 중심)
3. 선택/결정의 책임은 본인
4. 출생정보는 민감정보로 최소 수집/최소 보관
5. 데이터 보호 안내 + `valid_until`/`as_of_local` 표기

### 11.3 BTR 생시보정
현재 **OFF**. 소비자 리포트에 "BTR 미적용"을 간결하게 표기, 불안 유발 금지.

---

## 12. 로드맵

| 버전 | 범위 |
|---|---|
| v1.0.x | 출력 품질/게이트 안정화 |
| v1.1.0 | 개인화 슬롯 5종 / 온보딩 분기 / 월별 12슬롯 / HF15·16 추가 |
| v1.1.1–v1.1.4 | HF15·16 품질 잠금 / 모듈 분리 / name 정규화 / H2=0 fallback |
| v1.1.5–v1.1.9 | 인생 주기 리포트 설계 / 행성 라벨 / 단계 그룹핑 / 전환 강도 잠금 |
| v1.2.0 | 현재 프로젝트 기준 실행형 PRD / micro fixture / snapshot 제거 |
| v1.2.1 | P0 범위 동결 / balanced token budget |
| v1.2.2 | `life_cycle-lite` P0 / 난도 하향 / 최소 상업 구조 우선 출고 |
| v1.2.3 | 문서 정합성 복구 / P0·P1 범위 명확화 |
| v1.2.4 | 오염 원문 지정 교체 / 연속 2릴리즈 트리거 고정 |
| v1.2.5 | balanced token budget 상향 (`7000 / 6000–8000 / 9000`) |
| **v1.2.6 (현재)** | **token budget 본문 통합 / helper 책임 분리 / 구현 체크리스트 정렬** |
| v1.2.x | `v1.2.6` 수용 기준 전부 충족 후 `life_cycle-full` 착수 / yearly_forecast·compatibility 정리 |
| v1.3.x | 상품 번들화 / 운영 효율 개선 |
| v2.x | BTR 생시보정 ON |

---

## 13. 변경 관리

- 본 문서가 정본. 변경은 **PRD 버전업으로만 허용**.
- 아래 섹션은 **버전업 없이 단독 변경 금지**:
  - 0.5 / 1 / 3.2 / 3.3 / 5.1 / 5.6 / 6 / 7 / 8 / 9 전체
  - `_normalize_header_for_match()` 정규화 순서
  - `_normalize_name()` 정규화 순서 및 호칭 접미사 목록
  - `_normalize_text_for_name_match()` 함수 책임 범위 (호칭 제거 금지 규칙)
  - `_name_boundary_re()` 앞/뒤 경계 패턴 및 조사 허용 목록
  - H2=0 배포 임계치 및 분모 공식
  - `PLANET_LABEL_MAP` 9개 행성 코드 매핑
  - `LIFE_STAGE_LABELS` 4개 단계 라벨
  - 전환 강도 판정 기준 (strict `>`, floor 절사, N=3 의도된 동작)
  - valid_until_fallback 소비자 UX 문구 (섹션 7.1.8-a)
  - N=3 "상" 미발생 UX 안내 문구 (섹션 7.1.5)
  - `life_cycle-lite` P0 범위 동결 정책
  - balanced token budget (`7000` 기본 / `6000–8000` 권장 / `9000` 예외 상한)
  - `yearly_forecast` / `compatibility` bugfix-only 정책
  - micro fixture 중심 검증 전략 (섹션 8.4, 10.0)
  - 메타 필드명/버전 문자열/LOCK 요약 라인에는 제어문자(U+000B/U+000C) 및 escape 잔류 문자(`` `r ``, `\n`, `\r`) 삽입 금지

---

## 14. v1.2.6 실행 우선순위 및 남은 리스크

### 14.1 P0 — `life_cycle-lite` 우선 출고

1. `python -m pytest` 가능한 개발 환경 확보
2. `life_cycle-lite` 분기 단일화
3. 4단계 구조 렌더 로직 정리
4. valid_until fallback UX 유지
5. 방법론 카드 및 CTA-lite 고정 문구 정리
6. micro fixture 기반 계약 테스트 추가
7. `life_cycle-lite` product-layer 기본 `llm_max_tokens=7000` 고정
8. `yearly_forecast` / `compatibility` bugfix-only 상태 고정

### 14.2 남은 리스크 — 구조

- `main.py`가 큰 상태라 규칙이 다시 흩어질 수 있음.
- 대응: 상수/순수 함수 중복부터 제거하고, helper 추출은 필요 시에만 수행.

### 14.3 남은 리스크 — 테스트 환경

- 현재 환경에서 `python -m pytest`가 바로 안 될 수 있음.
- 대응: 기능 개발보다 dev dependency 정리를 선행.

### 14.4 남은 리스크 — snapshot 미도입에 따른 문체 회귀

- golden snapshot을 쓰지 않으므로 미세한 문체 회귀를 일부 놓칠 수 있음.
- 대응: deterministic copy 테스트 + 수동 QA 2건으로 방어.

### 14.5 남은 리스크 — `life_cycle-full` 구현 난도

- 전환점 강도/반복 패턴/다음 3년 구체화는 여전히 높은 난도.
- 대응: v1.2.6에서는 `life_cycle-lite`까지만 P0로 닫고, full 기능은 P1로 분리.

### 14.5-a `life_cycle-full` 착수 트리거

- `life_cycle-full`은 자동 착수 대상이 아닙니다.
- 아래 조건이 모두 충족된 이후에만 P1로 승격합니다.
  1. 14.6의 수용 기준 1-10 전부 충족
  2. `life_cycle-lite` 수동 QA 2건(정상 경로 1건 + fallback/edge 1건)이 연속 2릴리즈에서 모두 통과
  3. `yearly_forecast` / `compatibility` bugfix-only 정책 위반 0건 유지
- 여기서 "연속 2릴리즈"는 중간에 실패 릴리즈나 QA 미실행 릴리즈가 끼지 않은, 배포 가능한 두 번의 연속 릴리즈를 뜻합니다.

### 14.6 검증/수용 기준

1. `python -m pytest` 실행 가능
2. `life_cycle-lite` 경로 동작
3. 4단계 구조 + 현재 위치 표시 정상
4. HF 16개 회귀 없음
5. valid_until fallback UX 유지
6. 리포트당 LLM 호출 1회 초과 없음
7. `life_cycle` 경로의 product-layer 기본값 `7000`, soft QA 범위 `6000–8000`, hard cap `9000`이 유지됨
8. micro fixture 계약 테스트 통과
9. 수동 QA 2건 통과
10. `yearly_forecast` / `compatibility`는 bugfix-only 정책 위반 없음

### 14.7 비범위 (재확인)

- 엔진 산식 변경 금지.
- BTR ON 전환 금지.
- 프론트엔드/결제/로그인/대시보드 비범위.
- LLM 추가 호출 금지.
