# PRODUCT_SPEC_PRD v1.1.1 — 상업용 베딕 리포트 엔진 (Contract-Complete)

- **Status:** LOCKED (계약 변경 시 반드시 버전업 필요)
- **Last updated (Asia/Seoul):** 2026-03-02
- **Scope focus:** "그럴듯한 품질의 리포트를 안정적으로 출력" + 개인화 슬롯 확장 / 온보딩 분기 / 12슬롯 / HF 게이트 오탐 수정
- **Explicitly out of scope:**
  - Frontend/UI (Next.js / 결제 / 로그인 / 대시보드 / 다운로드 UX 등)
  - **BTR 생시보정 (Birth Time Rectification):** 출시 후 개발 예정 — 현재 기능 **OFF**
  - 엔진(점성 계산) 알고리즘 변경 (다샤/트랜짓/점수 산식) — 본 PRD 범위 밖
  - LLM 추가 호출 증가 금지 (후처리로만 품질을 끌어올린다)

---

## 변경 로그

### v1.1.1 *(현재)*

#### [수정] HARD FAIL 16 명칭 및 검사 범위 좁힘 (섹션 9.2, 9.3)
- **v1.1.0 문제**: HF16의 목적은 "개인화 5종 확장"이었으나 실제 검사 로직은 이름(①)의 요약 섹션 노출 여부만 확인. 목적과 구현이 불일치.
- **v1.1.1 조치**: HF16 명칭을 `name_token_min_exposure`로 좁혀 "이름/호칭(①) 토큰이 요약 섹션에 1회 이상 등장" 검사임을 명확히 고정. ②~⑤ 토큰 검증은 별도 관찰 지표(`personalization_token_coverage_extended`)로 유지하고 추후 버전업에서 HARD FAIL 승격 여부를 결정.

#### [수정] 섹션 경계 파싱 — H2 기준으로 고정 (섹션 9.3)
- **v1.1.0 문제**: `SECTION_HEADER_RE = ^(#{1,3})`는 `### Action Steps`, `### 7일 행동 루틴` 같은 H3 소제목도 독립 섹션으로 분리해 HF11/12 판정이 흔들리는 오탐 위험.
- **v1.1.1 조치**:
  - HF11/12/15/16 파서는 **H2(`##`) 기준으로만 섹션 경계를 분리**.
  - H3(`###`)은 섹션 내부 블록으로 처리. 섹션 본문에 포함됨.
  - 패턴: `SECTION_HEADER_RE = r"(?m)^##\s+(.+)$"` (H2 전용)
  - H1(`#`)은 문서 최상위 제목으로만 처리. 섹션 경계 판정 제외.

#### [수정] SECTION_EXCLUDE — contains 부분 매칭에서 exact match + alias 테이블로 변경 (섹션 9.3)
- **v1.1.0 문제**: `any(kw in header.lower())` 부분 문자열 매칭은 `"CTA 위치 안내"`, `"면책 문구 적용 방법"` 같은 정상 섹션도 의도치 않게 제외할 수 있음.
- **v1.1.1 조치**: 정규화된 헤더 exact match + alias 테이블 방식으로 변경. 매칭 방식이 바뀌므로 LOCK 대상으로 추가.

#### [수정] HARD FAIL 15 — 키워드 존재 외 본문 최소 품질 조건 추가 (섹션 9.3)
- **v1.1.0 문제**: 방법론 카드 6개 항목 헤더 키워드 존재 여부만 검사하므로, 빈약한 1줄 템플릿 텍스트나 "다샤"라는 단어만 포함한 섹션도 PASS 가능.
- **v1.1.1 조치**: 각 카드 항목에 대해 헤더 키워드 존재 + **본문 최소 25자** + **공감 문구 또는 행동 관련 동사 1개 이상** 동시 충족 시 PASS. 조건 미달 시 HARD FAIL.

#### [수정] 상수/알고리즘 모듈 분리 명시 (섹션 14.1)
- **v1.1.0 문제**: `commercial_quality_constants.py`에 패턴 상수와 파싱 로직 함수(`parse_sections` 등)를 모두 넣도록 명시해 책임이 섞임.
- **v1.1.1 조치**: 상수는 `commercial_quality_constants.py`, 알고리즘/헬퍼 함수는 `commercial_gate_helpers.py`(신규)로 분리. PRD에 모듈 경계 명시.

#### [수정] 월별 12슬롯 pressure_score_monthly 결측 fallback 규칙 추가 (섹션 7.2.2)
- **v1.1.0 누락**: `pressure_score_monthly` 값이 없거나 null인 달에 대한 처리 규칙 미정의.
- **v1.1.1 조치**: 결측 시 **기본 압축 표시** 적용. 전월 값 복사 금지. 결측 달의 행동 항목은 `CORE_ACTION_TOOLKIT["Mid-Term Direction"]` 첫 번째 항목으로 fallback.

#### [보강] ②~⑤ 토큰 검증 관찰 지표 명시 (섹션 9.6)
- HF16 범위에서 제외된 ②~⑤ 토큰 커버리지를 `personalization_token_coverage_extended`로 관찰 지표에 추가.
- v1.2.x 버전업 시 HARD FAIL 승격 여부를 결정하는 기준 데이터로 활용.

---

### v1.1.0
- 개인화 슬롯 5종 확장 (④ 직업군, ⑤ 관계 상태)
- 온보딩 목표 선택 → 문구 분기 로직 정의 (섹션 5.6)
- 신년운세 리포트 월별 12슬롯 구조 (섹션 7.2)
- HARD FAIL 15 (방법론 카드 6줄), HARD FAIL 16 (개인화 토큰) 추가
- HARD FAIL 11 공감 키워드 확장 (트리거 15개, 감정 17개)
- 섹션 경계 파싱 규칙 명확화 (v1.1.1에서 오탐 수정)
- BLANK_SPACE_RE 오탐 방지 수정

### v1.0.9
- 게이트 핵심 패턴/알고리즘 단일 소스화 (`commercial_quality_constants.py`)
- HARD FAIL 11/12/14 알고리즘 코드 계약 고정
- Scored Surface 정합 운영 규칙 강화
- 실행 환경 재현성 강화 (PYTHONPATH 없는 gate 실행)

### v1.0.8
- HARD FAIL 8 빈칸 템플릿 BLANK_* 4종 OR 명시
- HARD FAIL 14 substring → 토큰 교집합 변경
- HARD FAIL 11 "첫 문장" 추출 규칙 고정
- HARD FAIL 12 마지막 줄 목록 라인 스킵 규칙 추가
- SANSKRIT_STANDALONE_RE 다단어 영문 커버

### v1.0.7
- HARD FAIL 14 키워드 추출 알고리즘 정의
- HARD FAIL 11/12 판정 기준 기계화
- 중요구간 window 종류별 분리
- 겹침 50% 계산식 고정

### v1.0.3–v1.0.6
- 상업용 설계 철학, CTA/업셀 규칙, HARD FAIL 14개 도입, 메타 필드 분리 등

---

## 목차

- 0. 문서 목적 및 상업용 설계 철학
- 1. LOCK 항목 요약
- 2. 상업용 필수 조건 A–F (구매 전환 관점)
- 3. 제품 라인업 (3종) + 메타 필드 계약
- 4. 공통 출력 구조 (판매용 PDF 구조)
- 5. 공통 규칙 (문체/표현/개인화/점수 번역/분기)
- 6. 기간/타임라인 표준 (A/B/C)
- 7. 제품별 상세 요구사항 + CTA/업셀 규칙
- 8. Scored Surface/표면/해시 정합 계약
- 9. 품질 게이트 (HARD FAIL 16개 + 패턴 정의)
- 10. 테스트/검증
- 11. 보안/개인정보/윤리
- 12. 로드맵
- 13. 변경 관리
- 14. v1.1.1 실행 우선순위 (Gap-to-Implementation)

---

## 0. 문서 목적 및 상업용 설계 철학

### 0.1 문서 목적

이 문서는 "사람들이 돈을 내고 살 만한 상업용 베딕 리포트"를 만들기 위한 **제품 설계서 + 출고 기준(게이트) 계약서**입니다.

- 제품 요구사항 (무엇을 출력할지), 편집 규칙 (문체/구성/개인화/분기), 운영 기준 (표면/채점 정합), 출고 기준 (게이트)을 한 문서에서 고정합니다.
- **LOCK 항목**은 v1.1.1 내에서 변경 금지입니다 (필요 시 v1.2.0으로 버전업).

### 0.2 배경

리포트 품질을 반복 개선하는 과정에서 흔들림이 생기는 주된 원인은 "무엇을 만들고자 하는지(제품 정의)"가 문서로 고정되어 있지 않기 때문입니다. 이 PRD는 아래 세 가지를 고정합니다.

- 리포트가 **팔리는 상품**이 되기 위해 반드시 들어가야 하는 구성요소
- 돈을 받는 문서로서 **출고 가능한 교정/편집 품질 기준**
- 엔진/후처리/게이트가 따라야 하는 **정합 계약** (품질 측정 기준, 표면 파일 역할)

---

### 0.5 상업용 베딕 리포트 설계 철학 (LOCK — 버전업 없이 변경 금지)

> 이 제품은 **전통 베딕(점성술 전문가용) 리포트가 아니라**, 일반 소비자가 구매하고 읽고 행동하도록 설계된 **상업용 리포트**입니다.  
> 아래 철학은 콘텐츠 설계·문체·구조·CTA 모든 레이어에 최우선으로 적용됩니다.

#### 0.5.1 전통 베딕 vs 상업용 베딕 — 차별점 (LOCK)

| 항목 | 전통 베딕 리포트 | **상업용 베딕 리포트 (본 제품)** |
|---|---|---|
| 독자 | 점성술 관심자 / 전문가 | 점성술 비전문자 일반 소비자 |
| 목적 | 차트 해석의 정확성 / 완결성 | **읽힘 → 납득 → 행동 → 재구매** |
| 문체 | 전문 용어, 산스크리트어 중심 | 일상 언어 + 전문 용어 병기 (괄호 설명 필수) |
| 구조 | 차트 요소 순서대로 나열 | **소비자 궁금증 순서로** 배치 (지금 나는? → 다음은? → 어떻게?) |
| 길이 | 완결성 우선 (길수록 좋음) | **한 화면에 핵심 1개, 행동 1개**로 집약 |
| 성과 지표 | 해석의 깊이 | **구매 전환율 / 완독률 / 업셀 클릭률 / 재구매율** |

#### 0.5.2 상업용 리포트 설계 3대 원칙 (LOCK)

**원칙 1 — 공감 먼저, 정보는 그 다음**
- 리포트 본문의 첫 섹션(섹션 4.1 기준 섹션 #3 타임라인 이후)은 반드시 소비자의 "현재 감정/상황"에 공감하는 문구로 시작합니다.
- 차트 데이터를 먼저 나열하지 않습니다. "당신이 지금 왜 그런 느낌인지"를 먼저 설명합니다.
  - 좋은 예: "요즘 결정을 내릴 때마다 묘하게 망설임이 생긴다면, 그건 우연이 아닐 수 있습니다."
  - 나쁜 예: "현재 금성 다샤(Shukra Dasha) 7년 주기가 진행 중입니다."

**원칙 2 — 모든 섹션은 행동으로 끝난다**
- 해석/정보로 끝나는 섹션 금지. 각 섹션의 마지막 요소는 반드시 **행동 1개 (오늘 할 수 있는 것)** 이어야 합니다.
- 행동은 "동사 + 시간/횟수 + 조건" 포맷을 기본으로 합니다.

**원칙 3 — 리포트는 판매 채널이다**
- 본문 내 CTA는 독자의 **현재 페인포인트와 연결된 다음 상품**을 자연스럽게 제안해야 합니다.
- CTA는 광고성 문구가 아니라, 리포트 내용의 "자연스러운 다음 단계"처럼 느껴져야 합니다.
- 각 제품의 CTA 위치와 문구 규칙은 섹션 7에서 별도로 고정합니다.

#### 0.5.3 소비자 여정 (구매 → 완독 → 재구매) 설계

```
신규 구매
  └─ 온보딩(목표 1개 선택) → 공감 첫 문장 → 현재 시즌 납득
       └─ 행동 템플릿 실행 → "나한테 맞는 리포트다" 체감
            └─ CTA(업셀/다음 리포트) → 재구매 / 구독 전환
```

- **완독률 설계**: 섹션 3개 이하마다 "지금까지 요약 1줄"을 삽입해 이탈 방지.
- **재구매 설계**: 리포트 말미에 "유효기간 만료 30일 전 알림" 동의 요청을 CTA에 포함 (프론트 연동).
- **업셀 설계**: 현재 리포트에서 다루지 않은 영역을 CTA로 자연스럽게 연결.

#### 0.5.4 금지 설계 패턴 (LOCK)

아래 패턴은 상업용 가독성/전환율을 저해하므로 콘텐츠 생성 단계에서 금지합니다.

- 산스크리트어 단독 사용 금지 → "금성 다샤(Shukra Dasha)"처럼 반드시 한국어 병기
- 3줄 이상 연속 숫자/점수 나열 금지 → 표로 정리하거나 라벨 전환 필수
- "점수가 낮으니 조심하세요" 식의 부정 결론 단독 제시 금지 → 반드시 "대안 행동"과 함께 제시
- 1개 섹션 내 200자 이상의 순수 설명 (행동 없는 해설) 단독 노출 금지

---

## 1. LOCK 항목 요약

아래 항목은 "해석 흔들림/운영 충돌"을 유발하는 핵심 계약이므로 **버전업 없이 변경 금지**입니다.

1. **상업용 설계 철학** (섹션 0.5): 전통 vs 상업용 차별점, 3대 원칙, 소비자 여정, 금지 설계 패턴
2. **CTA/업셀 규칙 (제품별)** (섹션 7): 트리거 조건, 문구 포맷, 버튼 텍스트 20자 제한, 인라인 CTA 허용 조건
3. **상업용 문체 규칙** (섹션 5.1): 공감 우선 / 결핍 해소 / 1섹션=1메시지 / 예측 표현 규칙
4. **게이트 HARD FAIL 16개 + 검사 정의 (패턴/정규식/알고리즘)** (섹션 9)
5. **날짜 구간 A/B/C 레벨** (섹션 6): 정밀/슬롯/fallback + 우선순위
6. **수치 계약** (섹션 3): Ashtakoota 라벨 임계치, 제품별 valid_until, 연간 중요구간 2슬롯 선정 로직 (다샤 우선)
7. **중요구간 window 정의** (섹션 7.2.3): 전환 종류별 ±window 크기, 겹침 계산식
8. **메타 필드** (섹션 3.1): `as_of_utc` / `as_of_local` / `timezone_offset` 분리
9. **Scored Surface/표면/해시 정합** (섹션 8): 채점 표면 선택 규칙, LF-only + 바이트 해시 일치 규칙
10. **온보딩 목표 분기 로직** (섹션 5.6): 목표 선택지 4종, 분기 적용 규칙
11. **월별 12슬롯 밀도 판정 기준** (섹션 7.2.2): 압력 점수 임계치, 압축/전체 표시 규칙, 결측 fallback 규칙
12. **섹션 경계 파싱 규칙** (섹션 9.3): H2 기준 분리, H3 내부 블록 처리, 제외 섹션 exact match + alias 테이블

---

## 2. 상업용 필수 조건 A–F (구매 전환 관점)

아래 6개는 "있으면 좋음"이 아니라 **필수 요건**입니다.

### A. "Why Vedic?" 납득(방법론/근거)이 약함
- **문제**: 코칭/자기관리 톤이 강하면 "점성술 기반 근거"가 불명확해 보임.
- **요구사항**:
  1. 초반에 **방법론 카드 6줄** 삽입 (다샤/트랜짓/기본 차트/해석 원칙 + 공감 문구 병기 — 섹션 4.2 참조)
  2. "시기 (다샤/트랜짓)"를 **타임라인 (날짜 구간)**으로 구조화
- **Acceptance**:
  - 방법론 카드 6줄이 문서 초반에 존재 (각 항목에 공감 문구 포함) — **HARD FAIL 15로 자동 차단**
  - 상대 기간 표기 (이번 달/다음 달 등) 0건

### B. 문장/편집 오류 (중복/빈칸/끊김/삽입문)가 존재
- **문제**: 유료 상품 신뢰를 직접 훼손.
- **요구사항**: 출고 전 **상업용 교정 게이트 (섹션 9)**로 기계적 차단
- **Acceptance**: HARD FAIL 16개 전부 통과 / 빈칸 템플릿/미완성 문장/즉시 반복 0건

### C. 기간 예측이 "달/다음 달"로만 되어 체감이 약함
- **문제**: 유료 리포트는 명확한 날짜 범위가 있어야 체감/신뢰가 증가.
- **요구사항**: 모든 상대 기간 표현을 **날짜 구간 A/B/C** (섹션 6)로 변환
- **Acceptance**: 상대 월/주 표현 0건 (정규식 RELATIVE_MONTH_RE)

### D. 개인화 레이어가 약해 "나만을 위한 문서" 느낌이 약함
- **문제**: 범용 문장 비중이 높으면 상업적 가치가 낮음.
- **요구사항**: 개인화 토큰 5종 (섹션 5.3) 삽입 + 반복 노출. **온보딩 목표 선택에 따른 문구 분기** (섹션 5.6) 적용.
- **Acceptance**: 이름/호칭(①) 입력값이 있을 때 요약 섹션에 1회 이상 노출 — **HARD FAIL 16(name_token_min_exposure)으로 자동 차단**

### E. 내부 지표/리스크 점수류를 소비자에게 그대로 노출하면 역효과
- **문제**: 심리 진단처럼 보이거나 불안 유발.
- **요구사항**: 내부 점수는 내부용으로만 유지 / 소비자 본문에는 **라벨 + 대응 행동**으로만 번역
- **Acceptance**: 내부 score/grade 원값 (숫자/내부 등급명)이 소비자 본문에 직접 노출 0건

### F. "구매 후 무엇을 하면 되는지" 사용설명서가 없다
- **문제**: 길수록 "읽고 끝"이 됨.
- **요구사항**: 맨 앞에 **How to use 1페이지** (10분 루틴/체크리스트/알림 설정/복구 플랜 3줄)
- **Acceptance**: 사용법 1페이지 존재 + 체크리스트 포함

---

## 3. 제품 라인업 (3종) + 메타 필드 계약

### 3.1 리포트 메타 필드 계약 (LOCK)

| 필드 | 형식 | 설명 |
|---|---|---|
| `as_of_utc` | ISO-8601 | 서버 UTC 기준 계산/해석 기준 시점 |
| `as_of_local` | ISO-8601 | 사용자 타임존 변환값 (소비자 본문 표시 기준) |
| `timezone_offset` | 예: +09:00 | 입력 타임존 오프셋 |
| `valid_until` | YYYY-MM-DD HH:MM:SS (local) | 제품별 유효기간 (아래 3.2 참조) |
| `onboarding_goal` | 문자열 (4종 중 1) | 온보딩 목표 선택 결과 — 분기 로직에 사용 (섹션 5.6) |

- 소비자 본문에는 **`as_of_local` 기준 시점**을 표기합니다 (선택: UTC 병기).
- 모든 날짜 범위 섹션 하단에 다음 푸터 문구를 고정합니다:
  > "표기된 기간은 계획/주의 창이며, 개인의 체감은 상황에 따라 달라질 수 있습니다."

### 3.2 제품별 valid_until 계약 (LOCK)

| 제품 | valid_until 규칙 | 소비자 표기 |
|---|---|---|
| 인생 흐름 리포트 | `as_of_local + 90일` (동일 시각) | "유효기간: 발행 기준 90일" |
| 신년운세 리포트 | `대상 연도 12/31 23:59:59 (local)` | "유효기간: 해당 연도 종료까지" |
| 궁합 리포트 | `as_of_local + 180일` (동일 시각) | "유효기간: 발행 기준 180일 (상황 변화 시 재발행 권장)" |

### 3.3 궁합(Ashtakoota) 라벨 임계치 (LOCK)

| 점수 범위 | 라벨 |
|---|---|
| 0–17 | 낮음 |
| 18–26 | 중간 |
| 27–36 | 높음 |

추가 규칙 (LOCK):
- 라벨은 **총점 기준으로만** 산출합니다.
- 특정 Koota 리스크 (Bhakoot, Nadi 등)는 총점 라벨과 무관하게 "리스크 3 / 트리거 3"에서 반드시 행동 가이드로 노출합니다.
- **"낮음" 라벨일 때도 긍정적 대안 행동을 반드시 제시합니다** (부정 결론 단독 노출 금지, 0.5.4 참조).

### 3.4 제품 목록

1. **인생 흐름 리포트**: 인생 주기/큰 시즌(다샤 등) 설명 + 다음 90일 실행 전략
2. **신년운세 리포트**: 연간 구간(날짜) + 월별 12슬롯 + 중요구간 2개 + 분야별 모듈
3. **궁합 리포트**: Kuta Milan(Ashtakoota 36점) 기반 궁합 평가 + 조율 행동

---

## 4. 공통 출력 구조 (판매용 PDF 구조)

> 출력은 1차적으로 Markdown(리포트 본문 텍스트)이며, PDF 변환은 별도 레이어 (본 PRD 범위 밖)입니다.

### 4.1 권장 섹션 구성 (공통)

| # | 섹션 | 내용 |
|---|---|---|
| 0 | 커버 | 이름 / `as_of_local` / `valid_until` / 출생정보(옵션) |
| 1 | How to use 1p | "이 리포트를 10분에 쓰는 법" + 체크리스트 + 알림 설정 + 복구 플랜 |
| 2 | 한 장 요약 | 고정 포맷 (섹션 4.3) |
| 3 | 타임라인 | 날짜 구간 + 테마 + 주의/기회/행동 |
| 4 | 분야별 모듈 4장 | 커리어/돈/관계/컨디션 — **온보딩 목표 기준 순서 분기** |
| 5 | 실전 템플릿 1장 | 합의/거절/보류/협상 문장 모음 |
| 6 | 7일 시스템 | 체크박스 + 측정지표 + 복구 플랜 |
| 7 | 면책/윤리/데이터 보호 | 필수 5줄 (섹션 11.2) |
| 8 | CTA | 다음 단계 (제품별 규칙은 섹션 7) |
| (옵션) | Technical Appendix | 계산 설정 / 용어 / 차트 — 내부 점수는 여기에만 |

> **분야별 모듈 배치 순서**: 온보딩 목표(섹션 5.6)에 따라 결정됩니다. 선택된 목표 도메인이 첫 번째 모듈이 됩니다.

### 4.2 방법론 카드 6줄 (LOCK)

- **위치**: How to use 1p 또는 한 장 요약 직후
- **형식**: 6줄 고정. 각 항목에 **"왜 이게 당신에게 유용한지" 공감 문구 1줄 병기 의무** (과설명 금지)
- **자동 검증**: HARD FAIL 15 (섹션 9.3)로 키워드 존재 + 본문 최소 품질 동시 검사

| 항목 | 헤더 키워드 (HF15 검출용) | 대체 키워드 | 공감 문구 예시 |
|---|---|---|---|
| 다샤(Dasha) | `다샤` | — | "지금 이 시기의 '무게'가 왜 느껴지는지 설명해줍니다." |
| 트랜짓(Transit) | `트랜짓` | — | "특정 기간에 유난히 에너지가 달리는 이유를 보여줍니다." |
| 기본 차트(D1) | `기본 차트` | `D1` | "당신 고유의 패턴이 어디서 비롯되는지 알려줍니다." |
| 해석 원칙 | `해석 원칙` | — | "예언이 아닌, 준비 방향을 잡는 지도로 씁니다." |
| 기간 표준 | `기간 표준` | `날짜 구간` | "막연한 '이번 달' 대신 실제 날짜 범위를 드립니다." |
| 면책/윤리 | `면책` | `윤리` | "선택과 결정의 책임은 언제나 본인에게 있습니다." |

### 4.3 한 장 요약 고정 포맷

- 이번 시즌 한 문장 (1~2문장, 개인화 토큰 포함)
- 3가지 주의 (금기 행동과 연결) — **온보딩 목표 도메인 1개 우선 배치**
- 3가지 기회 (권장 행동과 연결) — **온보딩 목표 도메인 1개 우선 배치**
- 이번 주 행동 2개 (Action Steps 2개 고정 — "동사 + 시간/횟수 + 조건") — **온보딩 목표 기반 선택**

---

## 5. 공통 규칙 (문체/표현/개인화/점수 번역/분기)

### 5.1 상업용 문체 규칙 (LOCK)

#### 5.1.1 필수 문체 조건

1. **공감 우선 언어**: 섹션 시작 첫 문장은 정보가 아닌 독자의 현재 감정/상황을 반영한 문장으로 시작합니다.
2. **결핍 해소 언어**: 행동 제안 문장은 "~하면 좋다"가 아니라 "~하면 이 구간의 에너지를 가장 잘 활용할 수 있다" 식으로 구체적 결과와 연결합니다.
3. **숫자/기간의 체감화**: 날짜 범위나 숫자 단독 제시 금지. 반드시 소비자 행동과 연결합니다.
4. **1섹션 = 1메시지**: 각 섹션은 독자가 읽고 난 뒤 "기억에 남는 것 1개"를 명확히 전달합니다.
5. **예측은 단정 금지**: "반드시/확실히/무조건 된다" 금지. "가능성이 높은 패턴 + 트리거 + 대안 행동"으로 표현합니다.
6. **불안 유발/진단 톤 금지**: 의료/법률/투자 유사 진단 표현, "당신은 ~한 사람입니다" 식의 확정 진단 표현을 금지합니다.

#### 5.1.2 문장 포맷 표준

```
패턴:      "이 구간에는 [패턴]이 강화될 가능성이 큽니다."
트리거:    "특히 [트리거]에서 커집니다."
대안 행동: "대신 [행동]을 먼저 하세요."
```

### 5.2 템플릿 표기 규칙 (LOCK)

- 빈칸 템플릿 금지 ("우리는 를 …" 형태)
- 플레이스홀더는 `[대괄호]`로만 표현
- `[ ]` (대괄호 내부 공백만 있는 형태) 금지

### 5.3 개인화 토큰 정책 (5종, LOCK)

개인화 토큰은 아래 5종이며, 입력값이 있을 때 모두 리포트 전반에 반복 노출합니다.

| # | 토큰 | 사용 위치 | 역할 |
|---|---|---|---|
| ① | 이름/호칭 | 커버, 한 장 요약, 각 모듈 첫 문장 | 개인화 체감 기반 / **HF16 검사 대상** |
| ② | 현재 목표 1~2개 | 한 장 요약, 7일 시스템 | 핵심 개인화 레이어 |
| ③ | 현재 고민/현상 2~3개 | 분야별 모듈 공감 첫 문장 | 공감 문구 재료 |
| ④ | 직업군/활동 영역 | 커리어/돈 모듈 | 문구 분기 트리거 (섹션 5.6) |
| ⑤ | 관계 상태 | 관계 모듈, CTA | 궁합 CTA 자동 트리거 (섹션 7) |

**폴백 정의 (입력값 없을 때, LOCK):**

| 토큰 | 폴백 |
|---|---|
| ① 이름/호칭 | "당신" |
| ② 현재 목표 | "현재 중요한 목표" |
| ③ 현재 고민 | "최근 반복되는 상황" |
| ④ 직업군 | 분기 없음 (중립 문구 사용) |
| ⑤ 관계 상태 | 궁합 CTA 트리거 없음 |

- 폴백을 쓰더라도 **빈칸 템플릿이 남으면 안 됩니다**.
- ②~⑤ 커버리지는 관찰 지표 `personalization_token_coverage_extended`로 기록 (v1.2.x에서 HARD FAIL 승격 여부 결정).

### 5.4 내부 점수/리스크 번역 규칙 (LOCK)

- 내부 score/grade는 소비자 본문에서 제거합니다.
- 소비자 본문에는 **라벨 (낮음/중간/높음) + 의미 1~2문장 + 대응 행동 1~2개**로만 표현합니다.
- 내부 점수/등급은 debug/Technical Appendix에서만 유지합니다.

### 5.5 산스크리트/전문 용어 병기 규칙 (LOCK)

- 산스크리트어 또는 영문 전문 용어를 단독으로 노출하지 않습니다.
- 반드시 "한국어 (원어)" 형태로 병기합니다. 예: "금성 다샤(Shukra Dasha)"
- 동일 용어는 최초 1회 병기 후 이후 섹션에서 한국어만 사용 가능합니다.

### 5.6 온보딩 목표 선택 → 문구 분기 로직 (LOCK)

> 온보딩 단계에서 소비자가 목표 1개를 선택합니다. 이 선택값은 `onboarding_goal` 메타 필드에 저장되며, 리포트 생성 시 아래 규칙대로 콘텐츠에 반영됩니다.

#### 5.6.1 목표 선택지 4종 (LOCK)

| 목표 키 | 소비자 표시 레이블 | 연결 도메인 |
|---|---|---|
| `career_money` | 커리어 / 돈 | Career & Money 모듈 |
| `relationship` | 관계 / 인간관계 | Love & Relationship Patterns 모듈 |
| `condition` | 컨디션 / 에너지 | Health & Energy Rhythm 모듈 |
| `life_direction` | 인생방향 / 큰 그림 | Mid-Term Direction 모듈 |

#### 5.6.2 분기 적용 규칙 (결정론, LOCK)

**레이어 1 — 분야별 모듈 배치 순서 고정표 (LOCK)**

| `onboarding_goal` | 모듈 순서 |
|---|---|
| `career_money` | Career & Money → Health → Relationship → Mid-Term |
| `relationship` | Relationship → Career & Money → Health → Mid-Term |
| `condition` | Health & Energy Rhythm → Mid-Term → Career & Money → Relationship |
| `life_direction` | Mid-Term Direction → Career & Money → Relationship → Health |

**레이어 2 — 한 장 요약 "3주의 / 3기회" 첫 번째 항목**

한 장 요약의 3주의/3기회 목록에서 **첫 번째 항목은 반드시 선택된 목표 도메인 관련**이어야 합니다.

**레이어 3 — 이번 주 행동 2개 (Action Steps)**

```python
GOAL_TO_TOOLKIT_KEY = {
    "career_money":   "Career & Money",
    "relationship":   "Love & Relationship Patterns",
    "condition":      "Health & Energy Rhythm",
    "life_direction": "Mid-Term Direction",
}

def get_action_steps(onboarding_goal: str, toolkit: dict) -> list[str]:
    key = GOAL_TO_TOOLKIT_KEY.get(onboarding_goal, "Mid-Term Direction")
    pool = toolkit.get(key, [])
    return pool[:2]
```

#### 5.6.3 분기 미적용 조건 (fallback)

- `onboarding_goal`이 비어있거나 4종 이외의 값이면: `life_direction` fallback 적용.
- fallback 적용 시 `goal_branch_applied = false` 관찰 지표에 기록.

---

## 6. 기간/타임라인 표준 (A/B/C) (LOCK)

### 6.1 날짜 구간 레벨 정의

| 레벨 | 이름 | 조건 | 출력 형식 |
|---|---|---|---|
| A | 정밀/이벤트 기반 | 다샤/트랜짓 이벤트 앵커가 있는 경우 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 (기회 창/주의 창/집중 창) + 행동 1개 |
| B | 슬롯 기반 | 월/분기 고정 슬롯 경계로 생성하는 경우 | `YYYY-MM-DD~YYYY-MM-DD` + 라벨 (계획 창) + 행동 1개 |
| C | fallback | 앵커 부족 시 | `as_of_local` 기반 30일 window 생성. 반드시 `as_of_local` 소비자 노출 + 유효기간 병기. |

### 6.2 우선순위 (결정론, LOCK)

**A > B > C**

### 6.3 상대 기간 표현 금지 (게이트)

- "이번 달/다음 달/그다음 달/이번 주/다음 주" 등은 최종 표면에서 금지합니다.
- 정규식: `RELATIVE_MONTH_RE` (섹션 9.4 참조)

### 6.4 금지 표현

- "이 날부터 운이 바뀜 / 정확히 끝남 / 확정" 등 날짜를 단정하는 표현 금지
- 날짜 범위는 항상 패턴/트리거/대안 행동과 함께 제시합니다 (날짜만 단독 노출 금지).

---

## 7. 제품별 상세 요구사항 + CTA/업셀 규칙 (LOCK)

> CTA 원칙 (공통): CTA는 리포트 내용의 "자연스러운 다음 단계"처럼 느껴져야 합니다. 버튼 텍스트는 **20자 이내, 명령형 동사로 시작**합니다. CTA 광고성 판단 기준: 직전 섹션 키워드와 CTA 문구 간 의미 연결이 없으면 HARD FAIL(게이트 14).

---

### 7.1 인생 흐름 리포트

#### 7.1.1 목적/핵심 가치
- "인생 주기 (Why)" + "다음 90일 (What to do)"를 동시에 제공합니다.
- **상업적 목표**: "내 인생을 처음으로 큰 그림으로 본 경험"이 되어, 후속 연간 전망 리포트 구매로 이어집니다.

#### 7.1.2 필수 출력 항목
- 방법론 카드 6줄 (공감 문구 병기 포함) — HARD FAIL 15로 자동 검증
- 한 장 요약 (이번 시즌 한 문장 + 3주의 + 3기회 + 이번 주 행동 2개) — 온보딩 목표 분기 적용
- 90일 타임라인 (날짜 구간)
- 분야별 모듈 4장 — 온보딩 목표 기준 순서 배치
- 실전 템플릿 1장
- 7일 시스템 (체크박스 + 측정지표 1개)

#### 7.1.3 CTA/업셀 규칙 (LOCK)

- **위치**: 섹션 8(CTA) 1개 + 유효기간 만료 알림 CTA 하단 병기
- **업셀 트리거 (결정론)**:
  - `onboarding_goal == "relationship"` 또는 `관계 상태(⑤)` 입력값 있음 → 궁합 리포트 CTA
  - 그 외 → 연간 전망 리포트 CTA
- **문구 포맷 (고정)**: "[현재 시즌 키워드]에 맞는 [다음 상품명]으로 더 구체적인 계획을 세워보세요."
- **버튼 텍스트**: 20자 이내. 예: "신년운세 리포트 보기", "궁합 리포트 확인하기"
- **유효기간 만료 알림 CTA**: CTA 하단에 "유효기간 만료 30일 전 알림을 받으시겠습니까?" 동의 버튼 병기 (프론트 연동).

---

### 7.2 신년운세 리포트

#### 7.2.1 목적/핵심 가치
- "연간을 날짜 구간으로 보여준다" + "중요구간 2개를 뽑아준다"
- **상업적 목표**: 연초/생일 시즌 대표 구매 상품, 분기 업데이트 상품으로의 전환 유도.

#### 7.2.2 연간 구조 — 월별 12슬롯 (LOCK)

> 1~12월 개별 슬롯 + 중요구간 2슬롯 (총 14슬롯)

**월 슬롯 밀도 판정 기준 (LOCK)**:

| `pressure_score_monthly` | 표시 방식 | 출력 항목 |
|---|---|---|
| ≥ 60 | **전체 표시** | 날짜 범위 + 테마 1줄 + 주의 1 + 기회 1 + 행동 1 |
| < 60 | **압축 표시** | 날짜 범위 + 테마 1줄 + 행동 1 (주의/기회 생략) |
| 결측/null | **기본 압축 표시** | 전월 값 복사 금지. 날짜 범위 + "이 달 데이터 보완 예정" 안내 문구 + `CORE_ACTION_TOOLKIT["Mid-Term Direction"]` 첫 번째 항목 행동 1개 |

- 중요구간 2슬롯은 월 슬롯과 **별도 callout 블록**으로 출력 (월 슬롯과 중복 표시 금지).
- 12슬롯 전체 중 전체 표시 월이 0개이면 → 가장 높은 `pressure_score_monthly` 2개 달을 강제로 전체 표시.
- 결측 달이 2개 이상이면 `monthly_slot_density_distribution` 관찰 지표에 결측 카운트를 기록하고, 최종 리포트에 "일부 구간 데이터 보완 중" 안내를 삽입.

#### 7.2.3 중요구간 2슬롯 선정 로직 (LOCK)

- **후보 풀 생성 (결정론)**:
  1. 월 후보: 해당 연도 12개 달
  2. 다샤 전환 후보: 연도 내 마하다샤/부크티 전환이 있으면 아래 window 추가 (연도 경계 clip, inclusive)
     - **마하다샤 전환**: 전환일 ±21일 (총 43일 window)
     - **부크티 전환**: 전환일 ±7일 (총 15일 window)

- **선정 규칙 (LOCK)**:
  - **규칙 B (다샤 우선)**: 다샤 전환 후보가 있으면, `pressure_score` 상위부터 최대 2개 선정 (동률이면 더 이른 구간 우선).
  - 부족한 개수는 **규칙 A (트랜짓/압력 기반)**으로 채움: 월 후보 중 `pressure_score` 상위부터, 이미 선정된 중요구간과 중복이면 다음 후보로.
  - 최종 중요구간 **최대 2개**, **비중복**.

- **겹침(중복) 판정 계산식 (LOCK)**:
  ```python
  overlap_days = max(0, min(a_end, b_end) - max(a_start, b_start)).days + 1
  overlap_ratio = overlap_days / min(
      (a_end - a_start).days + 1,
      (b_end - b_start).days + 1
  )
  # overlap_ratio >= 0.5 이면 중복 → 해당 후보 제외
  ```

- **출력 규칙 (LOCK)**:
  - 중요구간 callout 슬롯 라벨: `중요(주의 창)` 또는 `중요(기회 창)`.

#### 7.2.4 CTA/업셀 규칙 (LOCK)

- **위치**: 섹션 8(CTA) 1개 + 분야별 모듈 "관계" 섹션 직후 인라인 1개 (조건부)
- **인라인 CTA 조건**: 중요구간이 "주의 창"일 때만 허용.
- **업셀 트리거**: 연간 전망 → 인생 흐름 리포트 / 연간 전망 → 궁합 리포트 (관계 섹션 직후, `⑤` 입력 있을 때 우선)
- **버튼 텍스트**: 20자 이내. 예: "인생 흐름 리포트 보기", "궁합 리포트 확인하기"
- **구독 전환 CTA**: "매 분기 업데이트 알림 받기" (v1.2 선점용, 현재는 이메일 수집).

---

### 7.3 궁합 리포트 (Kuta Milan)

#### 7.3.1 목적/핵심 가치
- 점수 자체가 아니라 "조율 행동"을 제공합니다 (불안 유발 최소화).
- **상업적 목표**: 연애/결혼 관계 전환점 "선물형/커플형 상품" + 두 사람 각자 인생 흐름 리포트 교차 업셀.

#### 7.3.2 출력 구조 (권장)
- 총점 + 라벨 (낮/중/높) + 해석 (근거 2~3줄) + 조율 행동
- Ashtakoota 8요소 각각: 점수(표시 가능) + 해석 + 리스크/주의 + 행동
- 갈등 완화 프로토콜 (트리거 → 대안 행동)
- 실전 템플릿 (사과/요청/경계/합의)
- 7일 시스템 (관계 루틴)

#### 7.3.3 CTA/업셀 규칙 (LOCK)

- **위치**: 섹션 8(CTA) 2개 — CTA A (개인 업셀) + CTA B (심화 업셀)
- **CTA A**: 두 사람 각각의 이름(①토큰)으로 인생 흐름 리포트 CTA.
- **CTA B**: 연애/결혼 → "결혼 적기 & 관계 전환점 심화 리포트" / 사업 파트너 → "파트너십 다샤 분석 심화 리포트"
- **선물 구매 프레임 (권장)**: "소중한 사람에게 이 리포트를 선물하기" 옵션 CTA 하단 병기.
- **버튼 텍스트**: 20자 이내.

---

## 8. Scored Surface/표면/해시 정합 계약 (LOCK)

### 8.1 표면 텍스트 정의 (런 아티팩트)

| 파일 | 역할 |
|---|---|
| `reading.md` | raw reading (디버그용, 품질 계약 대상 아님) |
| `reading_scan_surface.md` | 채점 직전 스캔 표면 |
| `reading_post_remediation.md` | 후처리 적용 사용자 최종 표면 |
| `scored_surface.md` | 실제 게이트 채점 대상 텍스트 (단일 기준) |

### 8.2 Scored Surface 선택 우선순위 (LOCK)

1. `polished_reading`이 non-empty (strip 기준)면 → `scored_surface = polished 후처리 결과`
2. 아니면 → `scored_surface = reading 후처리 결과`
3. 둘 다 없으면 → `scored_surface = scan surface`

### 8.3 LF 저장/해시 정합 (LOCK)

- surface 파일 3종은 **LF(\n)로만 저장** (\r\n/\r 금지)
- summary에 `scan_surface_sha256`, `post_sha256`, `scored_surface_sha256` **3개를 동시에** 기록하고, 각각 파일 바이트 SHA256과 대조합니다.
- `postprocess_applied = (scan_surface_sha256 != post_sha256)`

---

## 9. 품질 게이트 (HARD FAIL 16개 + 패턴 정의) (LOCK)

### 9.1 HARD FAIL vs 관찰 지표 분리

- **HARD FAIL**: 출고 불가 (자동 차단)
- **관찰 지표**: 기록만 (HARD FAIL로 승격 금지 — 정책 변경 시 버전업 필수)

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
| 8 | 빈칸 템플릿 (`BLANK_ANY_RE` 4종 OR 히트) > 0 | 형식 |
| 9 | 미완성 문장 (`UNFINISHED_SENTENCE_RE`) 히트 > 0 | 형식 |
| 10 | 즉시 반복 위반 > 0 (200자 이내 8자 이상 구절 2회 이상) | 형식 |
| 11 | 공감 문구 누락 — 첫 본문 섹션 첫 문장에 trigger+emotion 동시 미만족 | 상업용 |
| 12 | 행동 없는 섹션 마감 — 마지막 판정 가능 라인에 `오늘의 행동:` / `Action:` 없음 | 상업용 |
| 13 | 산스크리트/영문 단독 노출 (`SANSKRIT_STANDALONE_RE` 히트) | 상업용 |
| 14 | CTA 광고성 문구 — 직전 섹션 토큰과 교집합 없음 | 상업용 |
| 15 | 방법론 카드 품질 미달 — 6개 항목 중 1개 이상: 키워드 없음 OR 본문 25자 미만 OR 공감/행동 동사 없음 | 상업용 |
| 16 | name_token_min_exposure — 이름(①) 입력 있을 때 요약 섹션에 0회 노출 | 상업용 |

> **HF16 명칭 근거**: v1.1.0에서 "개인화 토큰 최소 노출 위반"으로 넓게 정의했으나, 검사 로직이 이름(①)의 요약 섹션 노출만 검사하므로 `name_token_min_exposure`로 명칭을 좁혀 목적과 구현 일치를 보장합니다. ②~⑤ 확장 검증은 관찰 지표로 유지하고 추후 버전에서 승격 여부를 결정합니다.

---

### 9.3 HARD FAIL 11–16 판정 알고리즘 (LOCK)

#### 공통 — 섹션 경계 파싱 규칙 (v1.1.1 수정, LOCK)

> HF11/12/15/16은 **H2(`##`) 기준으로만 섹션을 분리**합니다. H3(`###`)은 섹션 내부 블록으로 처리합니다. H1은 문서 제목으로만 처리합니다.

**모듈 분리 원칙 (v1.1.1 신규, LOCK)**:
- **`commercial_quality_constants.py`**: 패턴 상수, 정규식, 고정값 테이블만 포함. 알고리즘 함수 포함 금지.
- **`commercial_gate_helpers.py`** (신규): `parse_sections()`, `get_first_body_section()`, `check_methodology_card()`, `check_name_token_exposure()` 등 알고리즘 함수 포함. `commercial_quality_constants.py`를 import해서 사용.
- `cheap_validation_gate.py`는 `commercial_gate_helpers.py`와 `commercial_quality_constants.py`를 import만 하며, 핵심 패턴/로직 재정의 금지.

```python
# commercial_gate_helpers.py

import re
from commercial_quality_constants import (
    SECTION_HEADER_RE,
    SECTION_EXCLUDE_ALIAS,
)

def parse_sections(text: str) -> list[dict]:
    """
    H2(##) 기준으로만 섹션을 분리.
    H3(###)은 섹션 내부 블록으로 처리되어 본문에 포함됨.
    반환: [{"header": str, "body": str, "excluded": bool}, ...]
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        header_raw = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # exact match: 정규화된 헤더를 SECTION_EXCLUDE_ALIAS에서 조회
        excluded = _is_excluded_header(header_raw)
        sections.append({"header": header_raw, "body": body, "excluded": excluded})
    return sections

def _is_excluded_header(header: str) -> bool:
    """
    SECTION_EXCLUDE_ALIAS의 exact match 방식으로 제외 여부 판정.
    contains(부분 문자열) 매칭 금지.
    """
    normalized = header.strip().lower()
    return normalized in SECTION_EXCLUDE_ALIAS

def get_first_body_section(sections: list[dict]) -> dict | None:
    """제외 섹션을 건너뛰고 첫 번째 본문 섹션 반환."""
    for s in sections:
        if not s["excluded"] and s["body"]:
            return s
    return None
```

**제외 섹션 exact match + alias 테이블 (LOCK)**:

```python
# commercial_quality_constants.py

# v1.1.1: contains(부분 문자열) 방식 제거. exact match + alias 테이블로 고정.
SECTION_EXCLUDE_ALIAS: frozenset[str] = frozenset([
    # How to use 변형
    "how to use",
    "이 리포트를 10분에 쓰는 법",
    "사용법",
    "how to read",
    # 커버
    "커버",
    "cover",
    # 면책/윤리
    "면책",
    "면책 및 윤리",
    "면책/윤리/데이터 보호",
    "윤리",
    "데이터 보호",
    # CTA
    "cta",
    "다음 단계",
    # 부록
    "technical appendix",
    "기술 부록",
    # 문서 관리
    "변경 관리",
    "로드맵",
    "목차",
])
```

> **운영 참고**: 제외할 헤더가 새로 생길 경우 `SECTION_EXCLUDE_ALIAS`에 추가하고 PRD 버전업. contains 방식으로 되돌리는 것은 LOCK 위반.

---

#### HARD FAIL 11 — 공감 문구 누락

- **검사 대상**: `get_first_body_section()` 반환 섹션의 **첫 문장** (H3 내부 블록 포함)
- **"첫 문장" 추출 규칙 (LOCK)**:

```python
def extract_first_sentence(paragraph: str) -> str:
    m = re.search(r"[.!?\n]", paragraph)
    return paragraph[:m.start()].strip() if m else paragraph.strip()
```

- **판정 알고리즘 (LOCK)**:

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
    has_trigger = bool(re.search(EMPATHY_TRIGGER_RE, first_sentence))
    has_emotion = bool(re.search(EMPATHY_EMOTION_RE, first_sentence))
    return has_trigger and has_emotion  # False → HARD FAIL 11
```

---

#### HARD FAIL 12 — 행동 없는 섹션 마감

- **섹션 경계**: H2 기준 파싱 결과 각 섹션 (HF11과 동일 파서 사용)

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
    return None  # 전부 목록 라인 → None → HARD FAIL 12

def check_section_action(last_judgeable_line: str | None) -> bool:
    if last_judgeable_line is None:
        return False
    return bool(re.match(ACTION_LABEL_RE, last_judgeable_line))
```

---

#### HARD FAIL 13 — 산스크리트/영문 단독 노출

- **판정**: `SANSKRIT_STANDALONE_RE` 히트 > 0 → HARD FAIL (섹션 9.4 참조)

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
    cta_tokens = normalize_tokens(cta_text)
    return bool(source_tokens & cta_tokens)  # 교집합 없음 → False → HARD FAIL 14
```

---

#### HARD FAIL 15 — 방법론 카드 품질 미달 (v1.1.1 수정)

> v1.1.0: 키워드 존재 여부만 검사 → 빈약 텍스트 PASS 가능한 문제 수정.
> v1.1.1: 키워드 존재 + 본문 최소 25자 + 공감/행동 동사 1개 이상 **동시 충족** 필요.

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

METHODOLOGY_CARD_MIN_BODY_LEN = 25  # 본문 최소 글자 수 (공백 제외)

# 공감 또는 행동 동사 — 방법론 카드 본문에 최소 1개 필요
METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|이해|도움|활용|준비|씁니다|있습니다|됩니다|해줍니다|드립니다)"
)
```

```python
# commercial_gate_helpers.py

def check_methodology_card(text: str) -> bool:
    """
    방법론 카드 6개 항목 각각에 대해 아래 3조건을 모두 만족해야 PASS.
    1. 항목 키워드 (또는 대체어) 가 텍스트에 존재
    2. 해당 키워드 주변 본문이 25자 이상 (공백 제외)
    3. METHODOLOGY_CARD_QUALITY_RE 패턴 1개 이상 존재
    """
    for item in METHODOLOGY_CARD_ITEMS:
        # 1. 키워드 존재 확인
        found_pos = None
        for alt in item["alternates"]:
            idx = text.lower().find(alt)
            if idx != -1:
                found_pos = idx
                break
        if found_pos is None:
            return False  # HARD FAIL 15

        # 2. 키워드 주변 본문 추출 (최대 200자)
        snippet = text[found_pos:found_pos + 200]
        body_len = len(re.sub(r"\s", "", snippet))
        if body_len < METHODOLOGY_CARD_MIN_BODY_LEN:
            return False  # HARD FAIL 15

        # 3. 공감/행동 동사 존재
        if not re.search(METHODOLOGY_CARD_QUALITY_RE, snippet):
            return False  # HARD FAIL 15

    return True
```

---

#### HARD FAIL 16 — name_token_min_exposure (v1.1.1 명칭 확정)

- **검사 조건**: `personalization_input.name` (①토큰 입력값) 비어있지 않을 때만 검사
- **검사 대상**: 한 장 요약 섹션 내 이름 토큰 1회 이상 존재 여부

```python
# commercial_quality_constants.py

SUMMARY_SECTION_RE = re.compile(r"(?m)^##\s*한\s*장\s*요약")
```

```python
# commercial_gate_helpers.py

def check_name_token_exposure(text: str, name_input: str) -> bool:
    """
    이름 입력값이 있을 때 한 장 요약 섹션에 이름이 1회 이상 등장하면 PASS.
    요약 섹션 경계는 H2 기준 파싱 결과와 동일하게 적용.
    """
    if not name_input.strip():
        return True  # 입력값 없으면 검사 스킵
    m = SUMMARY_SECTION_RE.search(text)
    if not m:
        return False  # 요약 섹션 없음 → HARD FAIL 16
    next_h2 = re.search(r"(?m)^##\s+", text[m.end():])
    summary_end = m.end() + next_h2.start() if next_h2 else len(text)
    summary_text = text[m.start():summary_end]
    return name_input.strip() in summary_text  # False → HARD FAIL 16
```

---

### 9.4 패턴/정규식 정의 — `commercial_quality_constants.py` 기준 (LOCK)

> 모든 패턴/상수는 아래 목록을 `commercial_quality_constants.py`에서 단일 정의합니다. `cheap_validation_gate.py` 또는 `commercial_gate_helpers.py`에서 재정의 금지.

```python
# ── 빈칸 템플릿 (LOCK) — BLANK_ANY_RE: 4종 OR ──────────────────────────────
BLANK_TEMPLATE_RE   = r"(?i)\b(우리는|나는|당신은)\s+(를|을)\s+\b"
BLANK_BRACKET_RE    = r"\[\s*\]"
BLANK_UNDERSCORE_RE = r"_{2,}"
# v1.1.0 수정: 행 단위 기준으로 한정 (표/코드블록 오탐 방지)
BLANK_SPACE_RE      = r"(?m)^[^\S\n]{3,}(을|를|이|가|은|는|의|에|와|과)?\s*$"

BLANK_ANY_RE = (
    f"({BLANK_TEMPLATE_RE}"
    f"|{BLANK_BRACKET_RE}"
    f"|{BLANK_UNDERSCORE_RE}"
    f"|{BLANK_SPACE_RE})"
)

# ── 미완성 문장 (LOCK) ────────────────────────────────────────────────────────
UNFINISHED_SENTENCE_RE = r"(?m)(보여주는|하는|되어|있어|으로|해서)\s*$"

# ── 즉시 반복 기준 (LOCK) ─────────────────────────────────────────────────────
IMMEDIATE_REPEAT_DISTANCE_CHARS = 200

# ── 상대 기간 표현 (LOCK) ─────────────────────────────────────────────────────
RELATIVE_MONTH_RE = (
    r"(이번\s*달|다음\s*달|그\s*다음\s*달|이번\s*주|다음\s*주"
    r"|이달|다음달|저번\s*달|지난\s*달|전\s*달"
    r"|금월|차월|전월|익월|익주|전주)"
)

# ── 산스크리트/영문 단독 노출 (LOCK) ─────────────────────────────────────────
SANSKRIT_STANDALONE_RE = r"(?m)^\s*[A-Za-z][A-Za-z\s]{3,}\s*$"

# ── 공감 판정 (LOCK, HF11용) — v1.1.0 확장 ───────────────────────────────────
EMPATHY_TRIGGER_RE = re.compile(
    r"(요즘|최근|혹시|만약|자꾸|계속|어쩌면|가끔"
    r"|어느\s*순간|문득|갑자기|이유\s*없이|왠지|그동안|한동안)"
)
EMPATHY_EMOTION_RE = re.compile(
    r"(망설|답답|불안|지치|피곤|혼란|부담|막막|버겁|힘들"
    r"|무기력|지루|외롭|괴롭|두렵|헷갈|어렵)"
)

# ── 행동 라벨 (LOCK, HF12용) ──────────────────────────────────────────────────
ACTION_LABEL_RE = re.compile(r"^(오늘의\s*행동\s*:|Action\s*:)")

# ── 목록 라인 (LOCK, HF12 마지막 줄 스킵용) ─────────────────────────────────
LIST_LINE_RE = re.compile(r"^(\s*[-*]\s|\s*-\s\[[ xX]\])")

# ── 섹션 경계 파싱 — H2 전용 (LOCK, v1.1.1 수정) ────────────────────────────
# H2만 섹션 경계. H3는 섹션 내부 블록. H1은 문서 제목.
SECTION_HEADER_RE = re.compile(r"(?m)^##\s+(.+)$")

# ── 섹션 제외 — exact match (LOCK, v1.1.1 수정) ─────────────────────────────
SECTION_EXCLUDE_ALIAS: frozenset[str] = frozenset([
    "how to use", "이 리포트를 10분에 쓰는 법", "사용법", "how to read",
    "커버", "cover",
    "면책", "면책 및 윤리", "면책/윤리/데이터 보호", "윤리", "데이터 보호",
    "cta", "다음 단계",
    "technical appendix", "기술 부록",
    "변경 관리", "로드맵", "목차",
])

# ── CTA 불용어 + fallback (LOCK, HF14용) ─────────────────────────────────────
CTA_STOPWORDS = frozenset(
    "그리고|하지만|때문에|합니다|있습니다|됩니다|위해|통해|경우|또한|대한|이런|이와|이를".split("|")
)
CTA_DOMAIN_FALLBACK_RE = re.compile(r"(관계|돈|커리어|컨디션|인생|흐름|궁합)")

# ── 방법론 카드 (LOCK, HF15용) — v1.1.1: 품질 조건 포함 ─────────────────────
METHODOLOGY_CARD_ITEMS: list[dict] = [
    {"key": "다샤",     "alternates": ["다샤"]},
    {"key": "트랜짓",   "alternates": ["트랜짓"]},
    {"key": "기본 차트","alternates": ["기본 차트", "d1"]},
    {"key": "해석 원칙","alternates": ["해석 원칙"]},
    {"key": "기간 표준","alternates": ["기간 표준", "날짜 구간"]},
    {"key": "면책",     "alternates": ["면책", "윤리"]},
]
METHODOLOGY_CARD_MIN_BODY_LEN = 25
METHODOLOGY_CARD_QUALITY_RE = re.compile(
    r"(설명|보여|알려|이해|도움|활용|준비|씁니다|있습니다|됩니다|해줍니다|드립니다)"
)

# ── name_token_min_exposure (LOCK, HF16용) ────────────────────────────────────
SUMMARY_SECTION_RE = re.compile(r"(?m)^##\s*한\s*장\s*요약")

# ── 온보딩 목표 키 (LOCK, 분기 로직 5.6용) ───────────────────────────────────
ONBOARDING_GOAL_KEYS = frozenset(["career_money", "relationship", "condition", "life_direction"])
ONBOARDING_GOAL_FALLBACK = "life_direction"
GOAL_TO_TOOLKIT_KEY = {
    "career_money":   "Career & Money",
    "relationship":   "Love & Relationship Patterns",
    "condition":      "Health & Energy Rhythm",
    "life_direction": "Mid-Term Direction",
}
```

> **INLINE_CTA_RE 사용 금지**: v1.0.5에서 정의된 `INLINE_CTA_RE`는 오탐이 발생하므로 삭제합니다. CTA 광고성 판단은 HARD FAIL 14 알고리즘으로만 판정합니다.

### 9.5 게이트 검사 제외 블록

- 코드블록 (` ``` ... ``` `)
- 제목/헤더 라인 (`#`으로 시작)
- 테이블 구분 라인 (`|`, `---`만으로 구성된 라인)
- 순수 체크박스/목록 라인 (`- [ ] ...`)은 "종결" 검사에서 제외 (빈칸 템플릿/자리표시자는 검사)
- 고정 푸터/면책 문구 (화이트리스트)
- `SECTION_EXCLUDE_ALIAS` 매칭 섹션 내 본문

### 9.6 관찰 지표 (예시 — HARD FAIL 승격 금지)

- `actionable_ratio`: 행동형 문장 비율
- `banned_fps_hits`: 금지된 generic 문구 히트
- `inline_execution_residual_violations`: 본문 인라인 체인 잔존
- `one_page_summary_dedup_repairs`: 한 장 요약 중복 제거 수
- `action_steps_non_actionable_rewrites`: Action Steps 비행동형 재작성 수
- `action_steps_duplicate_replacements`: Action Steps 중복 교체 수
- `inline_execution_lines_moved`: 인라인 실행 체인 이동 줄 수
- `goal_branch_applied`: 온보딩 목표 분기 적용 여부 (true/false)
- `monthly_slot_density_distribution`: 월별 슬롯 전체/압축/결측 비율 분포 (신년운세 전용)
- `personalization_token_coverage_extended`: ②~⑤ 토큰 커버리지 (v1.2.x HF 승격 후보)

---

## 10. 테스트/검증

### 10.1 자동 테스트 (필수)

- HARD FAIL 16개 조건 각각에 대한 단위 테스트
  - **HF8**: `BLANK_ANY_RE` 4종 각각 히트/미히트 + `BLANK_SPACE_RE` 표/코드블록 내 오탐 안 나는 케이스
  - **HF11**:
    - `extract_first_sentence` 구분자별 경계 케이스 (./?/!/\n)
    - 트리거만 있거나 감정만 있는 케이스 → FAIL
    - v1.1.0 신규 트리거/감정 단어 각각 히트 케이스
    - `parse_sections` + `get_first_body_section` 제외 섹션(SECTION_EXCLUDE_ALIAS exact match) 스킵 케이스
    - H3 헤더가 있는 섹션에서 H2 경계 파싱 정확도 케이스 (H3가 별도 섹션으로 분리되지 않음)
  - **HF12**:
    - 목록 라인 스킵 후 판정 케이스
    - 전체 목록 라인 섹션(None 반환) → FAIL
    - H2 경계 기준 섹션 본문 범위 정확도 케이스
  - **HF14**: 토큰 교집합 있음/없음 + fallback 동작 + 오탐 방지
  - **HF15** (v1.1.1 수정):
    - 키워드 존재 + 본문 25자 이상 + 동사 → PASS
    - 키워드 존재 + 본문 24자 → FAIL
    - 키워드 존재 + 본문 30자 + 동사 없음 → FAIL
    - 대체어 허용 케이스 ("D1", "날짜 구간")
    - 1개 항목 완전 누락 → FAIL
  - **HF16 (name_token_min_exposure)**:
    - 이름 입력 있음 + 요약 섹션에 이름 → PASS
    - 이름 입력 있음 + 요약 섹션에 이름 없음 → FAIL
    - 이름 입력 없음 → 검사 스킵 → PASS
    - 요약 섹션(`## 한 장 요약`) 자체 없음 → FAIL
- **섹션 파싱 단위 테스트 (v1.1.1 신규)**:
  - H2 헤더만 섹션 경계로 분리되는 케이스
  - H3 헤더가 포함된 텍스트에서 H3가 섹션 내부 본문에 포함되는 케이스
  - `SECTION_EXCLUDE_ALIAS` exact match 케이스 (부분 문자열 포함된 다른 헤더가 오탐 안 나는 케이스)
- 날짜 표준 (A/B/C) 변환 + 상대 기간 0건 검증 (RELATIVE_MONTH_RE)
- Ashtakoota 라벨 임계치 (경계값: 17/18, 26/27)
- 중요구간 window: 마하다샤 ±21일 / 부크티 ±7일 / 연도 경계 clip
- 겹침 계산식: 경계값 0.5
- `SANSKRIT_STANDALONE_RE` 다단어 영문 라인 감지
- Scored surface 우선순위/해시 정합 (LF-only + 3종 sha256 동시 기록)
- "PYTHONPATH 없이 gate 실행 가능" 테스트
- **온보딩 목표 분기 테스트**:
  - 4종 각각 모듈 순서 검증
  - 미입력 시 `life_direction` fallback
  - Action Steps 2개가 올바른 toolkit 키에서 선택되는지
- **월별 12슬롯 테스트**:
  - `pressure_score_monthly >= 60` → 전체 표시 (4항목)
  - `< 60` → 압축 표시 (2항목)
  - `null/결측` → 기본 압축 + "이 달 데이터 보완 예정" 문구
  - 전체 표시 월 0개 → 상위 2개 달 강제 전체 표시
- **모듈 분리 테스트 (v1.1.1 신규)**:
  - `commercial_quality_constants.py`에 함수 정의 없음 (상수/패턴만 포함)
  - `commercial_gate_helpers.py`가 `commercial_quality_constants.py`를 import해서 사용하는지
  - `cheap_validation_gate.py`에서 패턴/로직 재정의 없는지 정적 검사

### 10.2 수동 QA (샘플링, 릴리즈당 최소 3건)

- 방법론 카드 (6줄 + 공감 문구) 가독성 — 존재/품질은 HF15로 자동 검증
- 상대 기간 표현 0건
- 빈칸 템플릿/미완성/중복 0건
- CTA가 앞 섹션과 의미 연결
- 공감 문구가 섹션 첫 문장에 존재
- 온보딩 목표가 한 장 요약/모듈 순서에 반영됐는지 육안 확인
- 신년운세 12슬롯 압축/전체 표시 및 결측 달 안내 문구 육안 확인

---

## 11. 보안/개인정보/윤리

### 11.1 소비자 본문에서 제거할 것

- `request_id`, 해시, 내부 점수 원값, debug payload, 시스템 경로
- 출생정보(출생시각/장소)는 옵션이며, 노출 여부 정책을 명시합니다.
- `onboarding_goal` 메타값은 소비자 본문에 노출하지 않습니다 (분기 적용에만 사용).

### 11.2 면책/윤리 가이드 (필수 5줄)

1. 의료/법률/투자 조언 아님
2. 결과 단정 금지 (가능성/패턴 중심)
3. 선택/결정의 책임은 본인
4. 출생정보는 민감정보로 최소 수집/최소 보관
5. 데이터 보호 안내 + `valid_until`/`as_of_local` 표기

### 11.3 BTR 생시보정

현재 **OFF**이며 출시 후 옵션 기능으로 제공합니다 (본 PRD 범위 밖). 소비자 리포트에는 "BTR 미적용"을 **간결하게** 표기하되 불안 유발 문구는 금지합니다.

---

## 12. 로드맵

| 버전 | 범위 |
|---|---|
| v1.0.x | 출력 품질/게이트 안정화 (결정론/교정/기간/개인화 기초) |
| v1.1.0 | 개인화 슬롯 5종 / 온보딩 분기 / 신년운세 월별 12슬롯 / HF15·16 추가 |
| **v1.1.1 (현재)** | **HF16 명칭 확정 / 섹션 파싱 H2 고정 / 제외 exact match / HF15 품질 조건 추가 / 모듈 분리 / 결측 fallback** |
| v1.1.x | HF15·16 튜닝 / 방법론 카드 문구 A/B 변형 / ②~⑤ 토큰 HF 승격 여부 결정 |
| v1.2.x | **구독형 (월간/분기 업데이트)** — 이메일 수집 → 구독 전환 퍼널 |
| v1.3.x | **번들 상품화** — 인생 흐름+연간 전망 패키지 / 커플 궁합+인생 흐름 2인 패키지 |
| v2.x | BTR 생시보정 ON (출시 이후) |
| 프리미엄 | Technical Appendix 분리 |

---

## 13. 변경 관리

- 본 문서가 정본. 변경은 **PRD 버전업으로만 허용**.
- PRD 변경 시: 계약 변경 로그 + 테스트/게이트 업데이트를 **동시에** 수행.
- 아래 섹션은 **버전업 없이 단독 변경 금지**:
  - 0.5 / 1 / 3.2 / 3.3 / 5.1 / 5.6 / 6 / 7 (CTA 규칙 + 중요구간 window + 겹침 계산식 + 월별 밀도 임계치 + 결측 fallback) / 8 / 9 (HARD FAIL 전체 + 패턴 + 판정 알고리즘 + SECTION_EXCLUDE_ALIAS)

---

## 14. v1.1.1 실행 우선순위 (Gap-to-Implementation)

> 목적: v1.1.0 신규 기능을 유지하면서 게이트 오탐 위험 4가지와 운영 안정성 2가지를 수정·보강한다.

### 14.1 P0 — 모듈 분리 + 패턴 단일소스화

1. **`commercial_gate_helpers.py` 신규 생성**: `parse_sections()`, `get_first_body_section()`, `_is_excluded_header()`, `extract_first_sentence()`, `check_empathy()`, `get_last_judgeable_line()`, `check_section_action()`, `normalize_tokens()`, `extract_source_tokens()`, `check_cta_relevance()`, `check_methodology_card()`, `check_name_token_exposure()` 포함.
2. **`commercial_quality_constants.py` 보완**: 위 섹션 9.4 전체 패턴/상수 단일 정의. 알고리즘 함수 포함 금지. 특히 `SECTION_HEADER_RE`(H2 전용), `SECTION_EXCLUDE_ALIAS`(exact match 테이블), `METHODOLOGY_CARD_ITEMS`(품질 조건 포함), `SUMMARY_SECTION_RE` 추가.
3. **`cheap_validation_gate.py`**: `commercial_gate_helpers`와 `commercial_quality_constants`를 import만 사용. 재정의 금지.

### 14.2 P0 — HF11/12 파서 H2 고정 적용

1. `parse_sections()`의 `SECTION_HEADER_RE`를 H2 전용 (`^##\s+`)으로 교체.
2. `_is_excluded_header()`를 `SECTION_EXCLUDE_ALIAS` exact match 방식으로 구현.
3. H3가 포함된 샘플 텍스트로 파서 동작 검증.

### 14.3 P0 — HF15 품질 조건 적용

1. `check_methodology_card()`에 키워드 존재 + 본문 25자 + `METHODOLOGY_CARD_QUALITY_RE` 동시 조건 추가.
2. 빈약 텍스트 케이스 단위 테스트 추가.

### 14.4 P0 — HF16 명칭 확정 + 코드 반영

1. 게이트 코드에서 HF16 변수명을 `name_token_min_exposure`로 통일.
2. ②~⑤ 관찰 지표 `personalization_token_coverage_extended` 기록 로직 추가.

### 14.5 P1 — 월별 12슬롯 결측 fallback 구현

1. `pressure_score_monthly` null/결측 달에 기본 압축 표시 적용.
2. 결측 달 행동 항목 `CORE_ACTION_TOOLKIT["Mid-Term Direction"][0]` fallback.
3. "이 달 데이터 보완 예정" 안내 문구 삽입.
4. `monthly_slot_density_distribution`에 결측 카운트 기록.

### 14.6 P1 — 모듈 분리 정적 검사 테스트

1. `commercial_quality_constants.py`에 함수 정의가 없는지 AST 또는 grep 기반 정적 검사.
2. `cheap_validation_gate.py`에서 패턴/정규식 재정의가 없는지 정적 검사.

### 14.7 P2 — PYTHONPATH 없는 gate 실행 테스트

1. `scripts/cheap_validation_gate.py` ROOT 자동 추가 동작 검증 CI 테스트 추가.

### 14.8 검증/수용 기준

1. 기존 HARD FAIL 14개 계약 유지 (회귀 없음).
2. HF15 신규 품질 조건 테스트 전부 통과.
3. HF16 명칭 변경 및 ②~⑤ 관찰 지표 기록 동작 확인.
4. H3 포함 섹션에서 HF11/12 오탐 없음.
5. `SECTION_EXCLUDE_ALIAS` exact match로 의도치 않은 제외 없음.
6. 월별 결측 달 fallback 출력 및 관찰 지표 기록 확인.
7. 모듈 분리 정적 검사 통과.

### 14.9 비범위(재확인)

- 엔진 산식(다샤/트랜짓/점수) 변경 금지.
- BTR 기능 ON 전환 금지 (현재 OFF 유지).
- 프론트엔드/결제/로그인/대시보드 작업 비범위.
- LLM 추가 호출 금지 (모든 보정은 deterministic rewrite).
