# PRODUCT_SPEC_PRD v1.1.4 — 상업용 베딕 리포트 엔진 (Contract-Complete)

- **Status:** LOCKED (계약 변경 시 반드시 버전업 필요)
- **Last updated (Asia/Seoul):** 2026-03-02
- **Scope focus:** "그럴듯한 품질의 리포트를 안정적으로 출력" + 개인화 분기 / 12슬롯 / HF 게이트 완전 잠금
- **Explicitly out of scope:**
  - Frontend/UI (Next.js / 결제 / 로그인 / 대시보드 / 다운로드 UX 등)
  - **BTR 생시보정:** 출시 후 개발 예정 — 현재 기능 **OFF**
  - 엔진(점성 계산) 알고리즘 변경 (다샤/트랜짓/점수 산식) — 본 PRD 범위 밖
  - LLM 추가 호출 증가 금지 (후처리로만 품질을 끌어올린다)

---

## 변경 로그

### v1.1.4 *(현재)*

#### [수정] HF16 name 정규화 함수 책임 분리 (섹션 9.3)
- **v1.1.3 문제**: `_normalize_name(summary_text)` 방식으로 요약 전체 텍스트에
  호칭 제거 regex를 적용하면 "민님의" → "민의"로 바뀌어 경계 매칭 컨텍스트가 훼손됨.
- **v1.1.4 조치**: 정규화 함수를 역할에 따라 둘로 분리.
  - `_normalize_name(name_input)`: name 입력값 전용. 공백·호칭·괄호 제거 + 소문자.
  - `_normalize_text_for_name_match(text)`: 검색 대상 텍스트 전용. 다중 공백 압축 + 소문자만. 호칭 제거 금지.

#### [수정] 짧은 이름 경계 regex — lookahead/lookbehind + 조사 허용 (섹션 9.3)
- **v1.1.3 문제**: `NAME_BOUNDARY_SUFFIX_RE_STR`가 optional(`?`)이라
  lookahead/lookbehind 없이는 "민감한"에서 "민" 오탐 가능성 잔존.
  또한 "민님의 시간"처럼 호칭 뒤 조사가 붙는 경우 after-lookahead가
  조사("의")를 한글로 인식해 FAIL하는 문제.
- **v1.1.4 조치**:
  - 앞 경계: `(?<![가-힣A-Za-z0-9])` — 한글/영숫자가 선행하면 매칭 거부.
  - 뒤 경계: suffix 소비 → 조사 소비(optional) → `(?![가-힣A-Za-z0-9])` 순서로 고정.
  - 조사 허용 목록을 LOCK으로 고정.
  - `_name_boundary_re()` 결과를 `functools.lru_cache`로 캐싱.

#### [수정] H2=0 배포 임계치 분모 공식 명시 (섹션 9.3, 9.4)
- **v1.1.3 문제**: 분모를 산문("해당 배포의 전체 리포트 생성 건수")으로만 서술.
  `total_generated_reports = 0`인 경우 처리 미정의.
- **v1.1.4 조치**: `h2_zero_ratio = h2_zero_count / total_generated_reports`
  (total > 0일 때만 계산, total = 0이면 `None` → 임계치 판정 skip)을 코드 계약으로 고정.

#### [추가 보완] `_name_boundary_re()` 컴파일 캐싱 명시
- v1.1.3에서 `_name_boundary_re()`가 호출마다 `re.compile()`을 실행하는 구조.
  동일 이름 반복 검사 시 불필요한 컴파일 낭비 발생.
  `functools.lru_cache`로 캐싱하도록 PRD에 명시.

---

### v1.1.3
- HF16 짧은 이름(≤ 2글자) 경계 매칭 추가
- H2=0 배포 임계치 명시 (5% warn / 20% block)
- `## [면책]` 혼합 헤더 정규화 테스트 추가

### v1.1.2
- HF16 name 매칭 정규화 규칙 (`_normalize_name()`)
- H2 0개 문서 파서 fallback (`metrics_valid=false`, HF11/12/15/16 skip)
- `_normalize_header_for_match()` 헤더 정규화 함수 계약
- HF15 품질 조건 수치·패턴 확정 (25자, `METHODOLOGY_CARD_QUALITY_RE`)

### v1.1.1
- HF16 명칭 `name_token_min_exposure`로 확정
- 섹션 경계 파싱 H2 기준 고정, H3 내부 블록 처리
- `SECTION_EXCLUDE_ALIAS` exact match + alias 테이블 전환
- HF15 본문 최소 품질 조건 (키워드 + 25자 + 동사)
- 상수/알고리즘 모듈 분리 (`commercial_quality_constants.py` vs `commercial_gate_helpers.py`)
- 월별 12슬롯 `pressure_score_monthly` 결측 fallback 규칙

### v1.1.0
- 개인화 슬롯 5종 (④ 직업군, ⑤ 관계 상태)
- 온보딩 목표 선택 → 문구 분기 로직
- 신년운세 월별 12슬롯 구조
- HF15 (방법론 카드 6줄), HF16 (개인화 토큰) 추가
- HF11 공감 키워드 확장 (트리거 15개, 감정 17개)

### v1.0.3–v1.0.9
- 상업용 설계 철학, CTA/업셀 규칙, HARD FAIL 14개 도입, 게이트 단일소스화 등

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
- 14. v1.1.4 실행 우선순위

---

## 0. 문서 목적 및 상업용 설계 철학

### 0.1 문서 목적

이 문서는 "사람들이 돈을 내고 살 만한 상업용 베딕 리포트"를 만들기 위한
**제품 설계서 + 출고 기준(게이트) 계약서**입니다.

- **LOCK 항목**은 v1.1.4 내에서 변경 금지입니다 (필요 시 v1.2.0으로 버전업).

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

---

## 1. LOCK 항목 요약

버전업 없이 변경 금지:

1. **상업용 설계 철학** (섹션 0.5)
2. **CTA/업셀 규칙** (섹션 7)
3. **상업용 문체 규칙** (섹션 5.1)
4. **HARD FAIL 16개 + 검사 정의** (섹션 9)
5. **날짜 구간 A/B/C** (섹션 6)
6. **수치 계약** (섹션 3): Ashtakoota 임계치, valid_until, 중요구간 선정 로직
7. **중요구간 window 정의** (섹션 7.2.3): ±window 크기, 겹침 계산식
8. **메타 필드** (섹션 3.1): `as_of_utc` / `as_of_local` / `timezone_offset`
9. **Scored Surface/해시 정합** (섹션 8)
10. **온보딩 목표 분기 로직** (섹션 5.6)
11. **월별 12슬롯 밀도 판정 기준** (섹션 7.2.2): 임계치, 결측 fallback
12. **섹션 경계 파싱 규칙** (섹션 9.3): H2 기준, `_normalize_header_for_match()` 순서, `SECTION_EXCLUDE_ALIAS`
13. **HF16 name 정규화 함수 책임 분리** (섹션 9.3): `_normalize_name()` vs `_normalize_text_for_name_match()`
14. **HF16 짧은 이름 경계 분기 기준** (섹션 9.3): ≤ 2글자, lookahead/lookbehind + 조사 허용
15. **H2=0 배포 임계치 및 분모 공식** (섹션 9.3): 5% warn / 20% block, 분모 = total_generated_reports

---

## 2. 상업용 필수 조건 A–F

### A. "Why Vedic?" 납득이 약함
- **요구사항**: 방법론 카드 6줄 삽입 + 날짜 구간 구조화
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
| `as_of_local` | ISO-8601 | 사용자 타임존 변환값 (소비자 표시 기준) |
| `timezone_offset` | 예: +09:00 | 입력 타임존 오프셋 |
| `valid_until` | YYYY-MM-DD HH:MM:SS (local) | 제품별 유효기간 |
| `onboarding_goal` | 문자열 (4종 중 1) | 온보딩 목표 선택 결과 |

모든 날짜 범위 섹션 하단 고정 푸터:
> "표기된 기간은 계획/주의 창이며, 개인의 체감은 상황에 따라 달라질 수 있습니다."

### 3.2 제품별 valid_until (LOCK)

| 제품 | valid_until 규칙 | 소비자 표기 |
|---|---|---|
| 인생 흐름 리포트 | `as_of_local + 90일` | "유효기간: 발행 기준 90일" |
| 신년운세 리포트 | `대상 연도 12/31 23:59:59 (local)` | "유효기간: 해당 연도 종료까지" |
| 궁합 리포트 | `as_of_local + 180일` | "유효기간: 발행 기준 180일" |

### 3.3 궁합(Ashtakoota) 라벨 임계치 (LOCK)

| 점수 범위 | 라벨 |
|---|---|
| 0–17 | 낮음 |
| 18–26 | 중간 |
| 27–36 | 높음 |

- 라벨은 **총점 기준으로만** 산출
- "낮음" 라벨일 때도 긍정적 대안 행동 반드시 제시

### 3.4 제품 목록

1. **인생 흐름 리포트**: 인생 주기/큰 시즌 설명 + 다음 90일 실행 전략
2. **신년운세 리포트**: 연간 구간 + 월별 12슬롯 + 중요구간 2개 + 분야별 모듈
3. **궁합 리포트**: Ashtakoota 36점 기반 궁합 평가 + 조율 행동

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

- **위치**: How to use 1p 또는 한 장 요약 직후
- **자동 검증**: HF15 (키워드 + 본문 25자 + 동사 패턴 동시 충족)

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
- 3가지 주의 — **온보딩 목표 도메인 1개 우선 배치**
- 3가지 기회 — **온보딩 목표 도메인 1개 우선 배치**
- 이번 주 행동 2개 (Action Steps — "동사 + 시간/횟수 + 조건") — **온보딩 목표 기반 선택**

---

## 5. 공통 규칙

### 5.1 상업용 문체 규칙 (LOCK)

1. **공감 우선 언어**: 섹션 시작 첫 문장은 정보가 아닌 독자의 현재 감정/상황 반영
2. **결핍 해소 언어**: "~하면 이 구간의 에너지를 가장 잘 활용할 수 있다" 식으로 구체적 결과 연결
3. **숫자/기간의 체감화**: 날짜 범위나 숫자 단독 제시 금지. 소비자 행동과 연결
4. **1섹션 = 1메시지**: 독자가 읽고 난 뒤 "기억에 남는 것 1개"
5. **예측은 단정 금지**: "반드시/확실히/무조건" 금지. "가능성이 높은 패턴 + 트리거 + 대안 행동"
6. **불안 유발/진단 톤 금지**: 의료/법률/투자 유사 진단 표현 금지

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

- ②~⑤ 커버리지는 관찰 지표 `personalization_token_coverage_extended`로 기록 (v1.2.x에서 HF 승격 여부 결정)

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

**레이어 1 — 분야별 모듈 배치 순서 고정표:**

| `onboarding_goal` | 모듈 순서 |
|---|---|
| `career_money` | Career & Money → Health → Relationship → Mid-Term |
| `relationship` | Relationship → Career & Money → Health → Mid-Term |
| `condition` | Health & Energy Rhythm → Mid-Term → Career & Money → Relationship |
| `life_direction` | Mid-Term Direction → Career & Money → Relationship → Health |

**레이어 2 — 한 장 요약**: 3주의/3기회의 첫 번째 항목은 선택된 목표 도메인 관련이어야 함.

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

- `onboarding_goal`이 비어있거나 4종 이외의 값: `life_direction` fallback
- fallback 시 `goal_branch_applied = false` 관찰 지표 기록

---

## 6. 기간/타임라인 표준 (A/B/C) (LOCK)

### 6.1 날짜 구간 레벨 정의

| 레벨 | 이름 | 조건 | 출력 형식 |
|---|---|---|---|
| A | 정밀/이벤트 기반 | 다샤/트랜짓 이벤트 앵커 있음 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 + 행동 1개 |
| B | 슬롯 기반 | 월/분기 고정 슬롯 경계 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 + 행동 1개 |
| C | fallback | 앵커 부족 시 | `as_of_local` 기반 30일 window + 유효기간 병기 |

**우선순위: A > B > C**

### 6.2 금지 표현

- "이번 달/다음 달/그다음 달/이번 주/다음 주" 등 (`RELATIVE_MONTH_RE`)
- "이 날부터 운이 바뀜 / 확정" 등 날짜 단정 표현
- 날짜 범위는 항상 패턴/트리거/대안 행동과 함께 제시

---

## 7. 제품별 상세 요구사항 + CTA/업셀 규칙 (LOCK)

> CTA 공통 원칙: 버튼 텍스트 **20자 이내, 명령형 동사로 시작**. 직전 섹션 키워드와 의미 연결 없으면 HF14.

### 7.1 인생 흐름 리포트

#### 7.1.1 필수 출력 항목
- 방법론 카드 6줄 (HF15 자동 검증)
- 한 장 요약 (온보딩 목표 분기 적용)
- 90일 타임라인
- 분야별 모듈 4장 (온보딩 목표 기준 순서)
- 실전 템플릿 1장
- 7일 시스템

#### 7.1.2 CTA/업셀 규칙 (LOCK)

- **업셀 트리거**: `onboarding_goal == "relationship"` 또는 ⑤ 입력 있음 → 궁합 리포트 / 그 외 → 연간 전망 리포트
- **문구 포맷**: "[현재 시즌 키워드]에 맞는 [다음 상품명]으로 더 구체적인 계획을 세워보세요."
- **유효기간 만료 알림**: CTA 하단에 "유효기간 만료 30일 전 알림을 받으시겠습니까?" 동의 버튼 병기

### 7.2 신년운세 리포트

#### 7.2.1 연간 구조 — 월별 12슬롯 (LOCK)

**밀도 판정 기준:**

| `pressure_score_monthly` | 표시 방식 | 출력 항목 |
|---|---|---|
| ≥ 60 | **전체 표시** | 날짜 범위 + 테마 + 주의 + 기회 + 행동 |
| < 60 | **압축 표시** | 날짜 범위 + 테마 + 행동 |
| 결측/null | **기본 압축** | 날짜 범위 + "이 달 데이터 보완 예정" + `CORE_ACTION_TOOLKIT["Mid-Term Direction"][0]` |

- 중요구간 2슬롯은 월 슬롯과 **별도 callout 블록**으로 출력
- 전체 표시 월이 0개이면 → `pressure_score_monthly` 상위 2개 달 강제 전체 표시
- 결측 달 2개 이상이면 `monthly_slot_density_distribution`에 결측 카운트 기록

#### 7.2.2 중요구간 2슬롯 선정 로직 (LOCK)

- **다샤 전환 window**: 마하다샤 ±21일 / 부크티 ±7일 (연도 경계 clip)
- **선정**: 다샤 후보 `pressure_score` 상위 2개 (규칙 B) → 부족 시 월 후보 `pressure_score` 상위 보충 (규칙 A)
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
- **구독 전환**: "매 분기 업데이트 알림 받기" (v1.2 선점용, 현재 이메일 수집)

### 7.3 궁합 리포트 (Kuta Milan)

#### 7.3.1 출력 구조
- 총점 + 라벨 + 해석 + 조율 행동
- Ashtakoota 8요소 각각: 점수 + 해석 + 리스크 + 행동
- 갈등 완화 프로토콜 / 실전 템플릿 / 7일 시스템

#### 7.3.2 CTA/업셀 규칙 (LOCK)

- **CTA A (개인 업셀)**: 두 사람 각각의 이름(①토큰)으로 인생 흐름 리포트 CTA
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

1. `polished_reading` non-empty → `scored_surface = polished 후처리 결과`
2. 아니면 → `scored_surface = reading 후처리 결과`
3. 둘 다 없으면 → `scored_surface = scan surface`

### 8.3 LF 저장/해시 정합 (LOCK)

- surface 파일 3종: **LF(\n)로만 저장**
- summary에 `scan_surface_sha256`, `post_sha256`, `scored_surface_sha256` **3개 동시 기록**
- `postprocess_applied = (scan_surface_sha256 != post_sha256)`

---

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
| `commercial_gate_helpers.py` | 알고리즘 함수 (`parse_sections()` 등) | 패턴 상수 재정의 |
| `cheap_validation_gate.py` | import만 사용 | 패턴/로직 재정의 |

---

#### 공통 — 섹션 경계 파싱 (LOCK)

**H2(`##`) 기준으로만 섹션 분리. H3는 내부 블록. H1은 문서 제목.**

##### 헤더 정규화 함수 `_normalize_header_for_match()` (LOCK)

| 순서 | 처리 | 예시 입력 → 출력 |
|---|---|---|
| 1 | 마크다운 강조 제거 (`**`, `*`, `__`, `_`) | `**면책**` → `면책` |
| 2 | 헤더 마커 제거 (앞쪽 `#` + 공백) | `## 면책` → `면책` |
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
        return []   # 호출부: h2_section_count=0, metrics_valid=False

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

**H2 0개 호출부 처리 패턴 (LOCK):**

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
    # HF11/12/15/16 정상 실행
```

**H2=0 배포 임계치 및 분모 공식 (LOCK):**

```python
# commercial_quality_constants.py
H2_ZERO_DEPLOY_WARN_THRESHOLD  = 0.05   # 5% 초과 → deploy_warn
H2_ZERO_DEPLOY_BLOCK_THRESHOLD = 0.20   # 20% 초과 → deploy_block
```

```python
# 배포 파이프라인 판정 로직
def compute_h2_zero_ratio(h2_zero_count: int, total_generated_reports: int) -> float | None:
    """
    분모: total_generated_reports (해당 배포의 전체 리포트 생성 건수)
    total = 0이면 None 반환 → 임계치 판정 skip
    """
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
| 개별 리포트 `h2_section_count = 0` | HF11/12/15/16 skip, 운영 알림 트리거, 출고 허용 |
| 배포 단위 비율 **> 5%** | deploy_warn 트리거 — 담당자 확인 필수 |
| 배포 단위 비율 **> 20%** | deploy_block 트리거 — 수동 해제 없이 배포 불가 |
| `total_generated_reports = 0` | 비율 계산 skip, `deploy_alert: "none"` |

---

#### HARD FAIL 11 — 공감 문구 누락

검사 대상: `get_first_body_section()` 반환 섹션의 **첫 문장**

```python
# commercial_gate_helpers.py
def extract_first_sentence(paragraph: str) -> str:
    m = re.search(r"[.!?\n]", paragraph)
    return paragraph[:m.start()].strip() if m else paragraph.strip()
```

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

#### HARD FAIL 13 — 산스크리트/영문 단독 노출

`SANSKRIT_STANDALONE_RE` 히트 > 0 → HARD FAIL (섹션 9.4 참조)

---

#### HARD FAIL 14 — CTA 광고성 문구

```python
# commercial_quality_constants.py
CTA_STOPWORDS = frozenset(
    "그리고|하지만|때문에|합니다|있습니다|됩니다|위해|통해|경우|또한|대한|이런|이와|이를".split("|")
)
CTA_DOMAIN_FALLBACK_RE = re.compile(r"(관계|돈|커리어|컨디션|인생|흐름|궁합)")
```

```python
# commercial_gate_helpers.py
def normalize_tokens(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^가-힣a-z\s]", " ", text)
    return {t for t in text.split() if t not in CTA_STOPWORDS and len(t) >= 2}

def extract_source_tokens(section_header: str, first_two_sentences: str) -> set[str]:
    tokens = normalize_tokens(section_header + " " + first_two_sentences)
    if not tokens:
        tokens = set(re.findall(CTA_DOMAIN_FALLBACK_RE, section_header))
    return tokens

def check_cta_relevance(source_tokens: set[str], cta_text: str) -> bool:
    return bool(source_tokens & normalize_tokens(cta_text))
```

---

#### HARD FAIL 15 — 방법론 카드 품질 미달

3조건 **동시 충족** 필요:

| # | 조건 | 기준 |
|---|---|---|
| 1 | 항목 키워드(또는 대체어) 존재 | `text.lower()`에서 발견 |
| 2 | 키워드 위치부터 200자 스니펫 최소 길이 | `re.sub(r"[\s\n]", "", snippet)` 기준 **25자 이상** |
| 3 | 스니펫 내 품질 패턴 1개 이상 | `METHODOLOGY_CARD_QUALITY_RE` 매칭 |

**`METHODOLOGY_CARD_QUALITY_RE` 확정 패턴 목록 (LOCK):**

| 카테고리 | 패턴 단어 |
|---|---|
| 설명/보여주기 | 설명, 보여, 알려, 보여줍니다, 알려줍니다 |
| 이해/도움 | 이해, 도움, 이해합니다, 도움이 됩니다 |
| 활용/준비 | 활용, 준비, 활용합니다, 준비합니다 |
| 서술형 종결 | 씁니다, 있습니다, 됩니다, 해줍니다, 드립니다, 입니다 |

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

```python
# commercial_gate_helpers.py
def check_methodology_card(text: str) -> bool:
    text_lower = text.lower()
    for item in METHODOLOGY_CARD_ITEMS:
        found_pos = next(
            (text_lower.find(alt) for alt in item["alternates"]
             if text_lower.find(alt) != -1), None
        )
        if found_pos is None:
            return False  # 조건 1 실패

        snippet  = text[found_pos: found_pos + METHODOLOGY_CARD_SNIPPET_LEN]
        body_len = len(re.sub(r"[\s\n]", "", snippet))
        if body_len < METHODOLOGY_CARD_MIN_BODY_LEN:
            return False  # 조건 2 실패

        if not re.search(METHODOLOGY_CARD_QUALITY_RE, snippet):
            return False  # 조건 3 실패
    return True
```

---

#### HARD FAIL 16 — name_token_min_exposure (v1.1.4 수정)

##### 정규화 함수 책임 분리 (LOCK)

두 함수는 역할이 다릅니다. **혼용 금지.**

| 함수 | 적용 대상 | 호칭 제거 | 이유 |
|---|---|---|---|
| `_normalize_name(name_input)` | 입력 이름값 | **YES** | 이름 기준값 정제 |
| `_normalize_text_for_name_match(text)` | 검색 대상 텍스트 | **NO** | 경계 컨텍스트 보존 |

```python
# commercial_quality_constants.py
NAME_HONORIFIC_RE = re.compile(
    r"(님|씨|군|양|선생님?|고객님|사용자|씨\s*귀하)\s*$"
)
```

```python
# commercial_gate_helpers.py
_NAME_BRACKET_RE  = re.compile(r"[（）()\[\]{}]")
_NAME_MSPACE_RE   = re.compile(r"\s{2,}")

def _normalize_name(name: str) -> str:
    """입력 이름 전용: 호칭·괄호·공백 제거 + 소문자."""
    n = name.strip()
    n = _NAME_MSPACE_RE.sub(" ", n)
    n = re.sub(NAME_HONORIFIC_RE, "", n).strip()
    n = _NAME_BRACKET_RE.sub("", n)
    return n.lower()

def _normalize_text_for_name_match(text: str) -> str:
    """검색 대상 텍스트 전용: 다중 공백 압축 + 소문자만. 호칭 제거 금지."""
    n = _NAME_MSPACE_RE.sub(" ", text)
    return n.lower()
```

##### 짧은 이름 경계 매칭 규칙 (≤ 2글자, LOCK)

| 이름 길이 (정규화 후) | 매칭 방식 |
|---|---|
| ≥ 3글자 | substring (`name_norm in text_norm`) |
| ≤ 2글자 | lookahead/lookbehind + 조사 허용 경계 정규식 |
| 정규화 후 빈 문자열 | 검사 스킵 → PASS |

**경계 정규식 규칙 (LOCK):**

- **앞 경계**: `(?<![가-힣A-Za-z0-9])` — 한글/영숫자가 선행하면 매칭 거부
- **뒤 경계**: 호칭 소비(optional) → 조사 소비(optional) → `(?![가-힣A-Za-z0-9])`
- **조사 허용 목록 (LOCK)**: `의|이|가|은|는|을|를|에|와|과|도|만|께|한테|에게`

```python
# commercial_quality_constants.py
NAME_BOUNDARY_SUFFIX_RE_STR  = r"(?:님|씨|군|양|선생님?|고객님|사용자)?"
NAME_BOUNDARY_PARTICLE_RE_STR = r"(?:의|이|가|은|는|을|를|에|와|과|도|만|께|한테|에게)?"
NAME_BOUNDARY_AFTER_RE_STR   = r"(?![가-힣A-Za-z0-9])"
NAME_BOUNDARY_BEFORE_RE_STR  = r"(?<![가-힣A-Za-z0-9])"
```

```python
# commercial_gate_helpers.py
@functools.lru_cache(maxsize=256)
def _name_boundary_re(name_norm: str) -> re.Pattern:
    """
    ≤ 2글자 이름용 경계 정규식 생성 + lru_cache로 컴파일 결과 캐싱.
    동일 이름 반복 검사 시 re.compile() 중복 실행 방지.
    """
    escaped = re.escape(name_norm)
    pattern = (
        NAME_BOUNDARY_BEFORE_RE_STR
        + escaped
        + NAME_BOUNDARY_SUFFIX_RE_STR
        + NAME_BOUNDARY_PARTICLE_RE_STR
        + NAME_BOUNDARY_AFTER_RE_STR
    )
    return re.compile(pattern, re.MULTILINE)
```

**동작 검증 예시:**

| 입력 이름 | 요약 텍스트 | 기대 결과 | 이유 |
|---|---|---|---|
| "민" | "민님의 이번 시즌" | PASS | 호칭+조사 경계 |
| "민" | "민감한 시기" | FAIL | "감"이 한글 → before-lookahead 차단 |
| "이" | "이 씨의 흐름" | PASS | 공백 + 호칭 경계 |
| "이" | "이번 시즌은" | FAIL | "번"이 한글 → after-lookahead 차단 |
| "홍길동" | "홍길동의 시즌" | PASS | 3글자 substring |

```python
# commercial_gate_helpers.py
def check_name_token_exposure(text: str, name_input: str) -> bool:
    if not name_input.strip():
        return True

    name_norm = _normalize_name(name_input)
    if not name_norm:
        return True

    m = SUMMARY_SECTION_RE.search(text)
    if not m:
        return False  # HARD FAIL 16

    next_h2  = SECTION_HEADER_RE.search(text, m.end())
    summary_end  = next_h2.start() if next_h2 else len(text)
    summary_text = text[m.start():summary_end]
    text_norm    = _normalize_text_for_name_match(summary_text)  # 호칭 제거 금지

    if len(name_norm) <= 2:
        pattern = _name_boundary_re(name_norm)
        matched = bool(pattern.search(text_norm))
        gate_summary_local = {"name_token_short_boundary_applied": True}
    else:
        matched = name_norm in text_norm
        gate_summary_local = {"name_token_short_boundary_applied": False}

    gate_summary_local["name_token_exposure_norm_applied"] = True
    return matched
```

---

### 9.4 패턴/정규식 정의 — `commercial_quality_constants.py` 기준 (LOCK)

> 모든 패턴은 아래 목록을 `commercial_quality_constants.py`에서 단일 정의.
> `cheap_validation_gate.py` 또는 `commercial_gate_helpers.py`에서 재정의 금지.

```python
# ── 빈칸 템플릿 (LOCK) ───────────────────────────────────────────────────────
BLANK_TEMPLATE_RE   = r"(?i)\b(우리는|나는|당신은)\s+(를|을)\s+\b"
BLANK_BRACKET_RE    = r"\[\s*\]"
BLANK_UNDERSCORE_RE = r"_{2,}"
BLANK_SPACE_RE      = r"(?m)^[^\S\n]{3,}(을|를|이|가|은|는|의|에|와|과)?\s*$"
BLANK_ANY_RE = (
    f"({BLANK_TEMPLATE_RE}|{BLANK_BRACKET_RE}"
    f"|{BLANK_UNDERSCORE_RE}|{BLANK_SPACE_RE})"
)

# ── 미완성 문장 (LOCK) ───────────────────────────────────────────────────────
UNFINISHED_SENTENCE_RE = r"(?m)(보여주는|하는|되어|있어|으로|해서)\s*$"

# ── 즉시 반복 기준 (LOCK) ────────────────────────────────────────────────────
IMMEDIATE_REPEAT_DISTANCE_CHARS = 200

# ── 상대 기간 표현 (LOCK) ────────────────────────────────────────────────────
RELATIVE_MONTH_RE = (
    r"(이번\s*달|다음\s*달|그\s*다음\s*달|이번\s*주|다음\s*주"
    r"|이달|다음달|저번\s*달|지난\s*달|전\s*달"
    r"|금월|차월|전월|익월|익주|전주)"
)

# ── 산스크리트/영문 단독 노출 (LOCK) ─────────────────────────────────────────
SANSKRIT_STANDALONE_RE = r"(?m)^\s*[A-Za-z][A-Za-z\s]{3,}\s*$"

# ── 공감 판정 (LOCK, HF11) ───────────────────────────────────────────────────
EMPATHY_TRIGGER_RE = re.compile(
    r"(요즘|최근|혹시|만약|자꾸|계속|어쩌면|가끔"
    r"|어느\s*순간|문득|갑자기|이유\s*없이|왠지|그동안|한동안)"
)
EMPATHY_EMOTION_RE = re.compile(
    r"(망설|답답|불안|지치|피곤|혼란|부담|막막|버겁|힘들"
    r"|무기력|지루|외롭|괴롭|두렵|헷갈|어렵)"
)

# ── 행동 라벨 (LOCK, HF12) ───────────────────────────────────────────────────
ACTION_LABEL_RE = re.compile(r"^(오늘의\s*행동\s*:|Action\s*:)")
LIST_LINE_RE    = re.compile(r"^(\s*[-*]\s|\s*-\s\[[ xX]\])")

# ── 섹션 경계 — H2 전용 (LOCK) ───────────────────────────────────────────────
SECTION_HEADER_RE = re.compile(r"(?m)^##\s+(.+)$")

# ── 섹션 제외 — exact match alias 테이블 (LOCK) ──────────────────────────────
SECTION_EXCLUDE_ALIAS: frozenset[str] = frozenset([
    "how to use", "이 리포트를 10분에 쓰는 법", "사용법", "how to read",
    "커버", "cover",
    "면책", "면책 및 윤리", "면책 윤리 데이터 보호", "윤리", "데이터 보호",
    "cta", "다음 단계",
    "technical appendix", "기술 부록",
    "변경 관리", "로드맵", "목차",
])

# ── CTA 불용어 (LOCK, HF14) ──────────────────────────────────────────────────
CTA_STOPWORDS = frozenset(
    "그리고|하지만|때문에|합니다|있습니다|됩니다|위해|통해|경우|또한|대한|이런|이와|이를".split("|")
)
CTA_DOMAIN_FALLBACK_RE = re.compile(r"(관계|돈|커리어|컨디션|인생|흐름|궁합)")

# ── 방법론 카드 (LOCK, HF15) ─────────────────────────────────────────────────
METHODOLOGY_CARD_ITEMS: list[dict] = [
    {"key": "다샤",     "alternates": ["다샤"]},
    {"key": "트랜짓",   "alternates": ["트랜짓"]},
    {"key": "기본 차트","alternates": ["기본 차트", "d1"]},
    {"key": "해석 원칙","alternates": ["해석 원칙"]},
    {"key": "기간 표준","alternates": ["기간 표준", "날짜 구간"]},
    {"key": "면책",     "alternates": ["면책", "윤리"]},
]
METHODOLOGY_CARD_SNIPPET_LEN  = 200
METHODOLOGY_CARD_MIN_BODY_LEN = 25
METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|보여줍니다|알려줍니다"
    r"|이해|도움|이해합니다|도움이\s*됩니다"
    r"|활용|준비|활용합니다|준비합니다"
    r"|씁니다|있습니다|됩니다|해줍니다|드립니다|입니다)"
)

# ── name_token_min_exposure (LOCK, HF16) ─────────────────────────────────────
SUMMARY_SECTION_RE = re.compile(r"(?m)^##\s*한\s*장\s*요약")
NAME_HONORIFIC_RE  = re.compile(
    r"(님|씨|군|양|선생님?|고객님|사용자|씨\s*귀하)\s*$"
)
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
GOAL_TO_TOOLKIT_KEY = {
    "career_money":   "Career & Money",
    "relationship":   "Love & Relationship Patterns",
    "condition":      "Health & Energy Rhythm",
    "life_direction": "Mid-Term Direction",
}
```

> **INLINE_CTA_RE 사용 금지**: CTA 광고성 판단은 HF14 알고리즘으로만 판정.

### 9.5 게이트 검사 제외 블록

- 코드블록 (` ``` ... ``` `)
- 헤더 라인 (`#`으로 시작)
- 테이블 구분 라인
- 순수 체크박스/목록 라인 (빈칸 템플릿은 검사)
- 고정 푸터/면책 문구
- `SECTION_EXCLUDE_ALIAS` 매칭 섹션 내 본문

### 9.6 관찰 지표 (HARD FAIL 승격 금지)

- `actionable_ratio`: 행동형 문장 비율
- `banned_fps_hits`: generic 문구 히트
- `inline_execution_residual_violations`: 인라인 체인 잔존
- `one_page_summary_dedup_repairs`: 한 장 요약 중복 제거 수
- `action_steps_non_actionable_rewrites`
- `action_steps_duplicate_replacements`
- `goal_branch_applied`: 온보딩 목표 분기 적용 여부
- `monthly_slot_density_distribution`: 월별 슬롯 전체/압축/결측 비율
- `personalization_token_coverage_extended`: ②~⑤ 토큰 커버리지
- `h2_section_count`: 0이면 `metrics_valid=false`
- `gate_skip_reason`: 비정상 skip 사유
- `h2_zero_ratio_per_deploy`: 배포 단위 H2=0 비율
- `deploy_alert`: `"none"` / `"warn"` / `"block"`
- `name_token_exposure_norm_applied`: HF16 정규화 비교 사용 여부
- `name_token_short_boundary_applied`: HF16 ≤2글자 경계 매칭 사용 여부

---

## 10. 테스트/검증

### 10.1 자동 테스트 (필수)

**HF1–10**: 기존 단위 테스트 유지

**HF11**:
- `extract_first_sentence` 구분자별 경계 (./?/!/\n)
- trigger만 또는 emotion만 → FAIL
- v1.1.0 신규 트리거/감정 단어 각각 히트
- `get_first_body_section()` 제외 섹션(`SECTION_EXCLUDE_ALIAS`) 스킵
- H3 헤더 포함 섹션에서 H2 경계 파싱 정확도

**HF12**:
- 목록 라인 스킵 후 판정
- 전체 목록 라인 섹션(None 반환) → FAIL

**HF14**: 토큰 교집합 있음/없음 + fallback + 오탐 방지

**HF15**:
- 키워드 + 본문 25자 이상 + 동사 → PASS
- 본문 24자 → FAIL (공백·줄바꿈 제거 기준)
- 동사 패턴 없음 → FAIL
- 대체어 (`"D1"`, `"날짜 구간"`) 허용
- 1개 항목 완전 누락 → FAIL

**HF16**:
- 이름 입력 있음 + 요약에 이름 → PASS
- 이름 입력 있음 + 요약에 이름 없음 → FAIL
- 이름 입력 없음 → 스킵 → PASS
- 요약 섹션(`## 한 장 요약`) 없음 → FAIL
- 짧은 이름(≤ 2글자) 경계 매칭:
  - `"민"` + `"민님의 이번"` → PASS (호칭+조사 경계)
  - `"민"` + `"민감한 시기"` → FAIL (오탐 차단)
  - `"이"` + `"이 씨의 흐름"` → PASS (공백+호칭)
  - `"이"` + `"이번 시즌은"` → FAIL (오탐 차단)
- `_normalize_name` vs `_normalize_text_for_name_match` 분리 검증:
  - 요약 텍스트에 `_normalize_name` 적용 시 "민님의" → "민의"로 변형되어 오탐 → FAIL임을 확인

**섹션 파싱**:
- H2만 섹션 경계 분리 (H3는 내부 포함)
- `SECTION_EXCLUDE_ALIAS` exact match (부분 문자열 포함 헤더 오탐 없음)
- 헤더 정규화:
  - `"**면책**"` → `"면책"` → alias PASS
  - `"## 면책 / 윤리:"` → `"면책 윤리"` → alias PASS
  - `"## [면책]"` → `"면책"` → alias PASS
  - `"## [면책 / 윤리]: 확인사항"` → `"면책 윤리 확인사항"` → alias `"면책 윤리"` PASS
  - `"## [공지] 중요안내"` → `"공지 중요안내"` → alias 없음 → 제외 안 됨 (오탐 방지)

**H2=0 fallback**:
- H1만 있는 문서 → `parse_sections()` 빈 리스트 → `h2_section_count=0`, `metrics_valid=False`
- HF1–10/13/14 정상 실행, HF11/12/15/16 skip 확인
- 출고 차단 없음 확인
- 배포 임계치:
  - 100건 중 4건 (4%) → `deploy_alert: "none"`
  - 100건 중 6건 (6%) → `deploy_alert: "warn"`
  - 100건 중 21건 (21%) → `deploy_alert: "block"`
  - `total_generated_reports = 0` → `deploy_alert: "none"` (분모 0 처리)

**온보딩 목표 분기**:
- 4종 각각 모듈 순서 검증
- 미입력 → `life_direction` fallback, `goal_branch_applied=false`
- Action Steps 2개가 올바른 toolkit 키에서 선택

**월별 12슬롯**:
- `pressure_score_monthly >= 60` → 전체 표시
- `< 60` → 압축 표시
- `null/결측` → 기본 압축 + "이 달 데이터 보완 예정" + Mid-Term fallback 행동
- 전체 표시 월 0개 → 상위 2개 강제 전체 표시

**모듈 분리 정적 검사**:
- `commercial_quality_constants.py`에 함수 정의 없음
- `cheap_validation_gate.py`에서 패턴/정규식 재정의 없음
- `_name_boundary_re()` `lru_cache` 캐싱 동작 확인 (동일 이름 2회 호출 시 컴파일 1회)

**Scored Surface 해시**:
- `scan_surface_sha256`, `post_sha256`, `scored_surface_sha256` 3개 동시 기록 확인
- LF-only 저장 확인

**PYTHONPATH 없는 gate 실행**: ROOT 자동 추가 동작 검증

### 10.2 수동 QA (릴리즈당 최소 3건)

- 방법론 카드 (6줄 + 공감 문구) 가독성
- 상대 기간 표현 0건
- 빈칸 템플릿/미완성/중복 0건
- CTA 앞 섹션과 의미 연결
- 공감 문구 섹션 첫 문장 존재
- 온보딩 목표가 한 장 요약/모듈 순서에 반영됐는지 육안 확인
- 신년운세 12슬롯 압축/전체 표시 및 결측 달 안내 문구 육안 확인

---

## 11. 보안/개인정보/윤리

### 11.1 소비자 본문에서 제거할 것

- `request_id`, 해시, 내부 점수 원값, debug payload, 시스템 경로
- `onboarding_goal` 메타값 (분기 적용에만 사용, 소비자 본문 노출 금지)

### 11.2 면책/윤리 가이드 (필수 5줄)

1. 의료/법률/투자 조언 아님
2. 결과 단정 금지 (가능성/패턴 중심)
3. 선택/결정의 책임은 본인
4. 출생정보는 민감정보로 최소 수집/최소 보관
5. 데이터 보호 안내 + `valid_until`/`as_of_local` 표기

### 11.3 BTR 생시보정

현재 **OFF**. 소비자 리포트에는 "BTR 미적용"을 간결하게 표기, 불안 유발 금지.

---

## 12. 로드맵

| 버전 | 범위 |
|---|---|
| v1.0.x | 출력 품질/게이트 안정화 |
| v1.1.0 | 개인화 슬롯 5종 / 온보딩 분기 / 월별 12슬롯 / HF15·16 추가 |
| v1.1.1 | HF16 명칭 확정 / H2 파싱 / exact match / HF15 품질 조건 / 모듈 분리 |
| v1.1.2 | 헤더 정규화 함수 / H2=0 fallback / HF16 name 정규화 / HF15 수치 확정 |
| v1.1.3 | HF16 짧은 이름 경계 / H2=0 배포 임계치 / 혼합 헤더 테스트 |
| **v1.1.4 (현재)** | **정규화 함수 책임 분리 / 조사 허용 경계 regex / 분모 공식 명시 / lru_cache** |
| v1.1.x | HF15·16 튜닝 / ②~⑤ 토큰 HF 승격 여부 결정 |
| v1.2.x | 구독형 (월간/분기 업데이트) |
| v1.3.x | 번들 상품화 |
| v2.x | BTR 생시보정 ON |

---

## 13. 변경 관리

- 본 문서가 정본. 변경은 **PRD 버전업으로만 허용**.
- PRD 변경 시: 변경 로그 + 테스트/게이트 업데이트 **동시 수행**.
- 아래 섹션은 **버전업 없이 단독 변경 금지**:
  - 0.5 / 1 / 3.2 / 3.3 / 5.1 / 5.6 / 6 / 7 / 8 / 9 (HARD FAIL 전체 + 패턴 + 알고리즘)
  - `_normalize_header_for_match()` 정규화 순서
  - `_normalize_name()` 정규화 순서 및 호칭 접미사 목록
  - `_normalize_text_for_name_match()` 함수 책임 범위 (호칭 제거 금지 규칙)
  - `_name_boundary_re()` 앞/뒤 경계 패턴 및 조사 허용 목록
  - H2=0 fallback 출고 정책 및 배포 임계치(5%/20%)
  - 배포 임계치 분모 공식 (`h2_zero_count / total_generated_reports`)
  - HF16 짧은 이름 경계 분기 기준 (2글자)

---

## 14. v1.1.4 실행 우선순위 (Gap-to-Implementation)

> 목적: v1.1.3의 짧은 이름 경계 regex 및 배포 임계치를 완전히 잠그고,
> 정규화 함수 책임 분리로 HF16 오탐 위험을 제거한다.

### 14.1 P0 — 정규화 함수 책임 분리

1. `commercial_gate_helpers.py`에 `_normalize_text_for_name_match()` 신규 추가.
   - 적용: 다중 공백 압축 + 소문자만. 호칭 제거 금지.
2. `check_name_token_exposure()` 내부에서 `summary_text`에
   `_normalize_name()` 대신 `_normalize_text_for_name_match()` 사용으로 교체.
3. 단위 테스트: `_normalize_name(summary)` 적용 시 "민님의" → "민의" 오탐 발생함을 확인 후, 분리 함수 적용으로 오탐 제거됨을 검증.

### 14.2 P0 — 짧은 이름 경계 regex 조사 허용 추가

1. `commercial_quality_constants.py`에 `NAME_BOUNDARY_PARTICLE_RE_STR` 추가.
2. `_name_boundary_re()` 패턴에 `NAME_BOUNDARY_PARTICLE_RE_STR` 삽입 (suffix 뒤, after-lookahead 앞).
3. `@functools.lru_cache(maxsize=256)` 데코레이터 추가 (컴파일 캐싱).
4. 단위 테스트: "민님의 이번" → PASS / "민감한" → FAIL / "이번 시즌" → FAIL 확인.

### 14.3 P0 — 배포 임계치 분모 공식 구현

1. `compute_h2_zero_ratio(h2_zero_count, total_generated_reports)` 함수 구현:
   - `total <= 0`이면 `None` 반환.
   - 그 외 `h2_zero_count / total_generated_reports` 반환.
2. `get_deploy_alert_level(ratio)` 함수 구현:
   - `None` → `"none"` (판정 skip).
   - `> 0.20` → `"block"`.
   - `> 0.05` → `"warn"`.
   - 그 외 → `"none"`.
3. 단위 테스트:
   - `total=0` → `deploy_alert: "none"`.
   - `h2_zero=6, total=100` → `deploy_alert: "warn"`.
   - `h2_zero=21, total=100` → `deploy_alert: "block"`.
   - `h2_zero=4, total=100` → `deploy_alert: "none"`.

### 14.4 검증/수용 기준

1. 기존 HF 16개 계약 회귀 없음.
2. `_normalize_text_for_name_match` 분리 후 "민님의" → text_norm에서 "민님의" 유지 확인.
3. 짧은 이름 경계 케이스 8개 단위 테스트 전부 통과.
4. 배포 임계치 경계값 4케이스 통과.
5. `_name_boundary_re()` lru_cache 동작 확인 (동일 이름 2회 호출 시 컴파일 1회).
6. `## [면책]` 혼합 헤더 정규화 테스트 통과.
7. 모듈 분리 정적 검사 통과.

### 14.5 비범위 (재확인)

- 엔진 산식 변경 금지.
- BTR ON 전환 금지.
- 프론트엔드/결제/로그인/대시보드 비범위.
- LLM 추가 호출 금지.
