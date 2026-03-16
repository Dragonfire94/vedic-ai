# PRODUCT_SPEC_PRD v1.5.0 — 상업용 베딕 리포트 엔진 (Life Cycle Long-Form Commercial Promotion)

- **Status:** DRAFT (repo reality aligned redraft; review required before LOCK)
- **Last updated (Asia/Seoul):** 2026-03-16
- **Base version:** `PRODUCT_SPEC_PRD_v1_4_0.md`
- **Scope focus:**
  - 현재 `life_cycle_target_v1`가 이미 달성한 수준과, `life_cycle_longform_v1`에서 새로 닫아야 할 범위를 분리
  - 하나의 `Vedic Life Cycle Report`를 `target candidate render`에서 **진짜 장문 상업용 narrative report**로 승격
  - 기존 `report_engine.py` + `llm_service.py` 장문 엔진을 `life_cycle` 상품 계층에 연결
  - 사람이 읽었을 때 "내 얘기 같다 -> 끝까지 읽힌다 -> 바로 행동이 떠오른다"를 목표로 하는 commercial copy layer 확정
  - long-form sample / evidence / human spot-check / cutover contract 정의
- **Explicitly out of scope:**
  - 다샤/트랜짓/점수 산식 등 점성 계산 엔진 변경
  - 외부 상품 추가 출시 (`yearly_forecast`, `compatibility`) 또는 SKU 분리
  - 결제/로그인/대시보드/CRM 등 비리포트 제품 영역
  - 장문 품질 확보 이전의 route default cutover 강행
  - `v1.4.0 target evidence`를 그대로 `v1.5.0 long-form 완료`로 간주하는 해석
  - LLM 호출 수 무한 확장. 장문 승격은 **기존 generic 장문 인프라 재사용**을 우선한다

---

## 변경 로그

- 상세 변경 이력: `CHANGELOG_PRD_v1_5_0.md`
- 현재 버전 핵심:
  - `v1.4.0`에서 닫은 `life_cycle_target_v1` cutover readiness를 시작점으로 삼되, 이번 버전에서는 그것을 최종 상품으로 보지 않는다
  - `v1.5.0`의 목표는 "짧고 안정적인 target candidate"가 아니라 **길고 자연스러우며 실제로 돈 받고 보여줄 만한 장문 리포트**다
  - generic 장문 엔진은 이미 `report_engine.py` / `llm_service.py` / `main.py` generic `/ai_reading` 경로에 존재하므로, 본 버전은 새 엔진 발명보다 **상품 전용 adapter와 copy contract**에 집중한다
  - 외부 상품은 여전히 하나의 `Vedic Life Cycle Report`이며, `life_cycle_lite_v1`, `life_cycle_target_v1`, `life_cycle_longform_v1`은 전부 내부 render stage 이름이다
  - `v1.4.0`의 target sample/evidence는 long-form 상업 리포트의 출발점이자 copy bank이며, 그대로 최종본으로 간주하지 않는다

> **Current Transition Rule (v1.5.0)**
> `life_cycle_target_v1`는 "구조/게이트/QA 기준을 닫은 pre-longform candidate"이며,
> `life_cycle_longform_v1`가 clean evidence + manual QA + readiness를 통과하기 전까지 shipped route default는 변경하지 않는다.

## 목차

- 0. 문서 목적 및 상업용 장문 설계 철학
- 0.4 Current State vs Long-Form Target
- 1. LOCK 항목 요약
- 2. 상업용 장문 리포트 필수 조건 A-H
- 3. 제품/메타/단계 계약
- 4. 공통 출력 구조
- 5. 공통 문체/개인화/행동 규칙
- 6. 장문 narrative composition 규칙
- 7. 인생 주기 리포트 장문 상세 요구사항
- 8. 엔진/아키텍처/책임 분리 계약
- 9. 표면/캐시/PDF/해시 정합 계약
- 10. 품질 게이트 및 editorial acceptance
- 11. 테스트/검증/증적
- 12. 보안/개인정보/윤리
- 13. 로드맵
- 14. 현재 상태와 남은 리스크
- 15. 완료 정의

---

## 0. 문서 목적 및 상업용 장문 설계 철학

### 0.1 문서 목적

이 문서는 `v1.4.0`에서 안정화한 `Vedic Life Cycle Report`를
실제 판매 가능한 장문 상업 리포트로 승격하기 위한
**제품 설계서 + 편집 계약서 + cutover 기준 문서**입니다.

- 이 문서는 "구조가 맞는가"보다 한 단계 더 나아가
  **"사람이 끝까지 읽고, 자기 이야기처럼 받아들이고, 바로 행동까지 연결되는가"**를 다룹니다.
- 본 문서의 LOCK 항목은 리뷰 후 `v1.5.0` 범위에서 변경 금지입니다.

### 0.4 Current State vs Long-Form Target

- Current state: 2026-03-16 기준 shipped runtime 기본 경로는 여전히 `life_cycle_lite_v1`입니다.
- Current state: `life_cycle_target_v1`는 구조/게이트/evidence/manual QA 기준으로 **상업형 후보 리포트**까지는 도달했습니다.
- Current state: target sample은 `hard_fail_count=0`, `life_cycle_release_ok=true`, editorial 20-case PASS까지 닫힌 상태입니다.
- Current state: 하지만 `life_cycle_target_v1`는 여전히 구조형 target candidate이며, `v1.5.0`이 요구하는 장문 상품 그 자체는 아닙니다.
- Current state: generic 장문 엔진은 이미 `report_engine.py` + `llm_service.py` + generic `/ai_reading` 경로에 존재합니다.
- Current state: 반대로 `life_cycle_longform_v1` 전용 adapter/renderer/checker/evidence bundle은 아직 저장소에 없습니다.
- Target state: `life_cycle` 상품이 기존 generic 장문 엔진을 재사용하되, 현재 한국어 `life_cycle` H2 구조와 개인화/행동 흐름을 유지한 장문 상업 surface로 승격됩니다.
- Target state: `PRD/release_evidence/v1_5_0/` 아래 clean sample/manifest/manual QA/gate summary가 실제로 존재하고, 그 기준으로 `READY`가 나와야 합니다.

### 0.5 상업용 장문 리포트 설계 철학 (LOCK)

#### 0.5.1 `정확한 해석`과 `팔리는 리포트`는 다르다 (LOCK)

| 항목 | 구조형/분석형 리포트 | **장문 상업 리포트 (본 버전 목표)** |
|---|---|---|
| 읽힘 | 정보는 충분하나 설명문처럼 읽힐 수 있음 | **처음 3문장 안에 몰입감 형성** |
| 납득 | 분석은 맞지만 거리감이 남을 수 있음 | **"이거 내 얘기다" 체감이 먼저 옴** |
| 행동 | 항목별 action은 있으나 조각날 수 있음 | **리포트 전체에 하나의 행동선이 흐름** |
| 판매성 | 읽을 만함 | **읽고 저장하고 공유하고 재구매할 만함** |
| 문체 | 해설/분석 중심 | **공감 -> 인식 -> 해석 -> 선택 -> 다음 행동** 중심 |

#### 0.5.2 장문 상업 리포트 설계 4대 원칙 (LOCK)

**원칙 1 — 해석보다 먼저 독자의 현재 장면을 잡는다**
- 각 핵심 섹션의 첫 문장은 "설명"보다 "자기인식"을 우선한다.
- 독자가 읽자마자 "이 상황이 왜 반복되는지"를 느껴야 한다.

**원칙 2 — 정보는 늘리되, 설명서는 금지한다**
- 장문이란 글자 수를 늘리는 것이 아니라, 독자의 맥락과 감정선을 충분히 복원하는 것이다.
- 같은 의미를 바꾸지 않고 길게만 쓰는 것은 장문이 아니라 noise다.

**원칙 3 — 장문 전체에 하나의 행동선이 흘러야 한다**
- `How to use 1p -> 현재 위치 -> 다음 3년 구체화 -> valid_until -> CTA-lite`는 분리된 조각이 아니라 하나의 흐름이어야 한다.
- 리포트 마지막에 "그래서 나는 지금 뭘 하면 되지?"가 남지 않아야 한다.

**원칙 4 — 점성학은 근거로만 존재하고, 독자 체감은 일상 언어로 번역한다**
- 점성학 요소는 evidence/anchor로만 기능한다.
- 최종 표면은 "점성학을 설명하는 글"이 아니라 "삶을 읽어주는 글"이어야 한다.

#### 0.5.3 소비자 여정 설계 (LOCK)

```text
온보딩(목표 선택 + 이름/맥락 입력)
  -> 첫 문장 몰입
  -> 현재 시즌 이해
  -> 반복 패턴 인식
  -> 다음 3년 행동선 확보
  -> valid_until 전까지 실행할 기준 1개 선택
  -> 저장/공유/재구매/업셀
```

#### 0.5.4 금지 설계 패턴 (LOCK)

- "이 섹션에서는..." 식의 설명문 도입
- 같은 패턴의 문장 시작 3회 이상 반복
- 각 섹션이 모두 "좋습니다 / 유리합니다 / 해보세요"로 끝나는 상담 보고서 톤
- 산스크리트어/행성 용어가 독자 문면을 지배하는 현상
- 정보는 많으나 "왜 지금 이 문장이 나한테 중요한지"가 없는 상태
- 장문 분량 확보를 위해 같은 의미를 2회 이상 재진술하는 것

### 0.6 v1.5.0 실행 단계 계약 (LOCK)

- 외부 상품은 여전히 하나의 `Vedic Life Cycle Report`입니다.
- 내부 단계는 아래 세 가지로 구분합니다.
  1. `life_cycle_lite_v1`: baseline deterministic shipped path
  2. `life_cycle_target_v1`: 구조/게이트/QA가 닫힌 target candidate
  3. `life_cycle_longform_v1`: 본 문서가 목표로 하는 장문 상업 render profile
- `v1.5.0`은 아래 순서로 진행합니다.
  1. 기존 generic 장문 엔진을 `life_cycle` payload에 연결
  2. 장문 commercial copy contract를 정리
  3. long-form sample/evidence/manual QA pack을 생성
  4. long-form cutover readiness를 별도로 닫는다
- 현재 저장소는 위 1~4를 아직 완료하지 않았고, `v1.4.0 target candidate`를 시작점으로만 보유합니다.
- `life_cycle_target_v1`가 이미 `READY`여도 그것만으로 장문 출시를 의미하지 않는다.
- `life_cycle_longform_v1` evidence가 닫히기 전에는 기본 shipped route는 유지한다.

---

## 1. LOCK 항목 요약

버전업 없이 변경 금지:

1. 외부 상품은 하나의 `Vedic Life Cycle Report`로 유지
2. `life_cycle_longform_v1`는 별도 외부 SKU가 아닌 내부 render stage
3. long-form의 exact H2 order는 `v1.4.0 target` 구조를 계승
4. 장문은 generic `report_engine` 기반 + `life_cycle` adapter 방식으로 구현
5. 점성 계산 엔진은 변경 금지
6. 개인화 토큰은 `subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status`만 사용
7. 상업용 첫 문장 규칙: 설명보다 자기인식 우선
8. 한 섹션당 무제한 action 금지. 핵심 행동은 최대 3개
9. 점수/퍼센트/내부 구조 지표 직접 노출 금지
10. 상대 기간/모호한 예언형 시기 표현 금지
11. `polished_reading`이 scored surface 기준이라는 계약 유지
12. product-aware cache / PDF / manifest identity는 계속 고정
13. cutover 전제는 `clean evidence + human spot-check + readiness PASS`
14. generic front module 재부착으로 life_cycle 표면을 오염시키지 않는다
15. long-form 도입 시에도 `LLM 호출 수 폭증`은 금지. 가능한 한 기존 1회 refinement 흐름 재사용
16. `v1.4.0 target evidence`와 `v1.5.0 long-form evidence`는 별도 경로/해시/판정 체계로 공존해야 한다

---

## 2. 상업용 장문 리포트 필수 조건 A-H

### A. 첫 3문장이 잡아끄는가
- **Acceptance:** `인생 구조 한 장 요약`, `현재 위치`, `인생 고점/저점 지도` 첫 문단이 설명문보다 자기인식 문장으로 시작

### B. 장문인데 반복이 적은가
- **Acceptance:** 같은 핵심 표현이 핵심 3개 섹션에 그대로 2회 이상 반복되지 않음

### C. 장문인데 피로하지 않은가
- **Acceptance:** 각 핵심 섹션은 최소 2문단 이상이되, paragraph rhythm이 살아 있고 문단별 역할이 다름

### D. 개인화가 실제로 체감되는가
- **Acceptance:** 이름/목표/현재 맥락/관계 상태/집중 토큰 중 최소 3종이 핵심 섹션에 자연스럽게 분산 반영

### E. 행동이 리포트 전체로 이어지는가
- **Acceptance:** How to use -> 현재 위치 -> 다음 3년 -> valid_until -> CTA-lite 흐름에서 행동선이 끊기지 않음

### F. 점성 근거는 살아 있고 과잉 설명은 없는가
- **Acceptance:** evidence anchor는 충분하지만 독자 문면에서 용어 dump/메타 설명이 없음

### G. PDF/JSON/sample/manual QA가 같은 텍스트를 바라보는가
- **Acceptance:** sample response / manifest / PDF source / manual QA가 동일 hash identity를 공유

### H. 사람 검토 기준으로 "돈 받고 보여줘도 된다"가 되는가
- **Acceptance:** final human spot-check에서 `cutover_ready: YES`

---

## 3. 제품/메타/단계 계약

### 3.1 외부 상품 계약 (LOCK)

- 외부 상품명은 계속 `Vedic Life Cycle Report`
- 사용자 query는 여전히 `product_type=life_cycle`
- 장문 승격은 내부 render profile과 generation mode 교체로 구현한다

### 3.2 long-form 메타 계약 (LOCK)

`life_cycle_longform_v1` 응답은 기존 `v1.4.0` meta를 계승하되, 아래 필드를 추가 또는 고정한다.

| 필드 | 형식 | 설명 |
|---|---|---|
| `render_profile` | `life_cycle_longform_v1` | long-form render profile 식별자 |
| `generation_mode` | string | 예: `report_engine.life_cycle_adapter_v1` |
| `source_render_profile` | string | long-form 이전의 source stage (`life_cycle_target_v1`) |
| `narrative_profile` | string | 예: `commercial_longform_v1` |
| `longform_cutover_candidate` | boolean | evidence/sample 검토용 플래그 |

### 3.3 내부 단계 간 관계 (LOCK)

- `life_cycle_target_v1`는 구조형 target candidate
- `life_cycle_longform_v1`는 그것의 확장본이며, **정확히 같은 상품 계약**을 더 긴 commercial surface로 재해석한다
- long-form 승격이란 target H2 order를 버리는 것이 아니라, 같은 구조 위에 narrative depth를 얹는 것이다

### 3.4 route cutover 원칙 (LOCK)

- `life_cycle_longform_v1` readiness 전에는 default shipped route를 바꾸지 않는다
- long-form은 내부 review/evidence mode로 먼저 생성한다
- cutover 후에도 fallback은 deterministic target/lite 계층을 남긴다

---

## 4. 공통 출력 구조

### 4.1 exact H2 order (LOCK)

long-form도 아래 H2 order를 유지한다.

1. `## How to use 1p`
2. `## 인생 구조 한 장 요약`
3. `## 4단계 인생 구조`
4. `## 현재 위치`
5. `## 마하다샤 단계 목록`
6. `## 인생 고점/저점 지도`
7. `## 반복 패턴 분석`
8. `## 다음 3년 구체화`
9. `## 방법론 카드`
10. `## valid_until 설명`
11. `## CTA-lite`
12. `## 면책/윤리/데이터 보호`

`cover/meta`는 H2 밖 표면으로 유지한다.

### 4.2 장문 구조 전략 (LOCK)

- 구조는 `v1.4.0 target`을 유지하되, 분량 전략은 아래와 같이 바꾼다.
  - 현재 위치/고점저점/반복패턴/다음3년: **핵심 narrative depth 영역**
  - How to use/방법론/valid_until/CTA-lite/면책: **짧고 명확한 support 영역**

### 4.3 장문 분량 가이드 (LOCK)

| 섹션 | 목표 |
|---|---|
| `인생 구조 한 장 요약` | 2~3문단 |
| `현재 위치` | 2~3문단 + 짧은 action line |
| `인생 고점/저점 지도` | 도입 1문단 + 2~3개 구간 narrative |
| `반복 패턴 분석` | 2~3문단 |
| `다음 3년 구체화` | 슬롯별 맥락 문장 + action |
| support 섹션 | 짧고 명확하게 유지 |

---

## 5. 공통 문체/개인화/행동 규칙

### 5.1 상업용 장문 문체 규칙 (LOCK)

- 문장은 "설명"보다 "인식"을 우선한다
- 독자에게 직접 말하는 2인칭 톤은 허용하되, 상담 보고서처럼 과하게 지시하지 않는다
- 한 섹션 안에서 `관찰 -> 감정/상황 -> 패턴 -> 의미 -> 행동` 흐름이 보이게 한다
- 같은 완화형 종결을 연속 섹션에서 반복하지 않는다
- 문단마다 역할이 달라야 한다. 같은 의미의 재진술 금지

### 5.2 personalization 적용 규칙 (LOCK)

- `subject_name`은 cover 또는 요약에 1회 이상 자연스럽게 노출
- `onboarding_goal`은 요약/현재위치/다음3년 중 최소 2개 섹션에 반영
- `occupation_context`와 `relationship_status`는 현재위치 또는 반복패턴에 우선 반영
- `focus_tokens`, `concern_tokens`는 다음 3년 또는 고점/저점 문맥에서 우선 사용
- 개인화는 억지 삽입보다 문맥형 반영을 우선한다

### 5.3 행동 문장 규칙 (LOCK)

- 행동은 리포트 전체에서 하나의 흐름으로 이어져야 한다
- action bullet은 많아도 3개를 넘기지 않는다
- action은 "동사 + 범위 + 시점/조건" 형태를 우선한다
- `CTA-lite`는 광고가 아니라 "다음 행동의 자연스러운 연장"처럼 읽혀야 한다

### 5.4 jargon/내부 라벨 금지 규칙 (LOCK)

- `life_cycle-lite`, `life_cycle-target`, `target candidate`, `render profile`, `contract version` 같은 내부 표현은 소비자 본문에 노출 금지
- `cover/meta`, `How to use 1p`, `CTA-lite`, `valid_until`은 내부 파일명/게이트명이 아니라 소비자용 제목/라벨로 번역되어야 한다

---

## 6. 장문 narrative composition 규칙

### 6.1 장문은 "더 긴 설명"이 아니라 "더 깊은 맥락"이다 (LOCK)

장문 확장의 우선순위:

1. 독자의 현재 장면/심리/맥락 복원
2. 점성 근거를 일상 언어로 번역
3. 반복 패턴의 의미 설명
4. 행동 전환점 제시

금지:

- 이미 말한 핵심을 문장만 바꿔 다시 늘리는 것
- 용어 설명을 길게 붙여 분량을 채우는 것

### 6.2 문단 구성 규칙 (LOCK)

- 첫 문단: 공감/자기인식/상황 포착
- 둘째 문단: 왜 이 흐름이 반복되는지 해석
- 셋째 문단이 있다면: 지금 선택해야 할 기준 또는 행동 연결
- 한 문단이 지나치게 길어지면 2개로 분리한다

### 6.3 섹션별 opening 규칙 (LOCK)

핵심 3섹션(`인생 구조 한 장 요약`, `현재 위치`, `인생 고점/저점 지도`)은 아래 중 하나로 시작해야 한다.

- "요즘 이런 느낌이 반복된다면..."
- "겉으로는 X인데, 실제로는 Y일 수 있다"
- "이 구간에서는 속도보다 기준이 먼저 중요하다"
- "흐름이 붙는 때와 늦추는 때가 생각보다 다르게 온다"

### 6.4 evidence grounding 규칙 (LOCK)

- 장문은 새 해석을 발명하지 않는다
- `report_engine`이 만든 deterministic chapter blocks / structural summary / dasha context 안에서만 확장한다
- evidence는 anchor 역할을 하되, 독자 본문에는 생활어 번역을 우선한다

---

## 7. 인생 주기 리포트 장문 상세 요구사항

### 7.1 `How to use 1p`

- 1페이지 사용 설명은 유지하되, 문장 수를 늘리지 않는다
- long-form 전체를 어떻게 읽을지 3단계로 안내한다
- 첫 핵심 행동선이 여기서 시작되어야 한다

### 7.2 `인생 구조 한 장 요약`

- 첫 문장은 즉시 자기인식이 와야 한다
- 둘째 문단은 "왜 지금 이 흐름이 중요한지"를 풀어준다
- 셋째 문단은 있다면 "이번 리포트에서 무엇을 잡아야 하는지"를 짚는다
- 목표: 읽자마자 저장/스크린샷하고 싶은 섹션

### 7.3 `4단계 인생 구조`

- 단계 정보는 유지하되 표/나열이 아닌 짧은 맥락 문장도 포함
- 지금 단계가 전체 맵에서 어디쯤인지 느껴져야 한다

### 7.4 `현재 위치`

- 첫 bullet은 설명문이 아니라 옆에서 말해주는 문장처럼 읽혀야 한다
- `occupation_context`, `relationship_status`, `onboarding_goal`이 가장 자연스럽게 녹아드는 섹션이다
- 현재 할 일보다 "지금 무엇을 덜 해야 하는가"까지 포함할 수 있다

### 7.5 `마하다샤 단계 목록`

- 정보 과부하 금지
- 현재 마하다샤와 다음 전환일이 왜 중요한지 짧게 연결

### 7.6 `인생 고점/저점 지도`

- 단순 리스트보다 "왜 이 구간이 이 사람에게 중요한지"를 한 문장씩 붙인다
- 도입문은 기능 설명형이 아니라 "흐름을 읽는 이유"를 말해야 한다
- 각 구간은 좋은 때/늦추는 때의 의미 차이가 분명해야 한다

### 7.7 `반복 패턴 분석`

- 가장 팔리는 장문 섹션 후보이므로, 기계적인 분석 문구를 피한다
- 독자가 "이게 내가 자꾸 같은 선택을 하는 이유였구나"라고 느끼게 해야 한다
- 관계/일/돈/회복 중 1~2개 반복 축으로 묶어 설명한다

### 7.8 `다음 3년 구체화`

- 슬롯별 summary/action을 유지하되, 각 슬롯의 맥락 문장을 더 늘린다
- "무엇을 밀고 무엇을 늦추는지"가 보여야 한다
- 재구매/업셀 hook이지만 광고처럼 보이면 실패다

### 7.9 `방법론 카드`

- `Why Vedic?` 납득을 강화하는 support 섹션
- 길게 설명하지 말고, 신뢰를 올리는 짧은 카피로 유지

### 7.10 `valid_until 설명`

- 유효기간을 단순 안내가 아니라 "왜 지금 읽고 실행해야 하는지"로 연결

### 7.11 `CTA-lite`

- 판매 문구보다 자연스러운 다음 단계여야 한다
- 톤은 추천이지 압박이 아니어야 한다

### 7.12 `면책/윤리/데이터 보호`

- 짧고 명확하게 유지
- 상업 문면 전체의 신뢰를 해치지 않아야 한다

---

## 8. 엔진/아키텍처/책임 분리 계약

### 8.1 기본 원칙 (LOCK)

long-form은 새 엔진이 아니라 아래 조합으로 만든다.

- deterministic source: `life_cycle_helpers.py`
- product structure: `life_cycle_target_renderer.py` 또는 동등 adapter
- long-form block engine: `report_engine.py`
- long-form prompt / refinement: `llm_service.py`
- final surface hygiene: `output_surface_postprocess.py`

### 8.2 권장 구현 구조

권장 파일 책임:

- `backend/life_cycle_longform_adapter.py`
  - `life_cycle` payload를 `report_engine.chapter_blocks` 친화 구조로 매핑
- `backend/life_cycle_longform_renderer.py`
  - 필요 시 final markdown surface shaping
- `scripts/build_life_cycle_longform_editorial_pack.py`
  - sample/manual QA/gate summary/manifest 생성
- `scripts/check_life_cycle_longform_cutover_ready.py`
  - long-form 전용 readiness 판정

동등 구현은 허용하되, 책임 분리는 유지한다.

### 8.2-a 2026-03-16 기준 실제 저장소 상태

아래는 **권장 파일**이며, 현재 로컬에는 아직 존재하지 않습니다.

- `backend/life_cycle_longform_adapter.py`
- `backend/life_cycle_longform_renderer.py`
- `scripts/build_life_cycle_longform_editorial_pack.py`
- `scripts/check_life_cycle_longform_cutover_ready.py`
- `PRD/release_evidence/v1_5_0/` long-form evidence bundle

즉 `v1.5.0`은 현재 문서 단계이며, 구현/증빙은 아직 비어 있습니다.

### 8.3 generic front 재오염 금지 (LOCK)

- long-form `life_cycle` 결과는 generic front module 재부착 경로로 오염되면 안 된다
- `commercial_surface_renderer.py`의 generic front는 reference로만 사용하고, product 전용 surface는 별도 보호가 필요하다

### 8.4 generic report_engine 재사용 원칙 (LOCK)

- `report_engine.py`의 12챕터 장문 인프라는 재사용 가능
- 단, `life_cycle` long-form은 현재 Korean H2 order와 personal CTA 흐름을 유지해야 한다
- generic chapter naming을 그대로 소비자 표면에 노출하면 안 된다

---

## 9. 표면/캐시/PDF/해시 정합 계약

### 9.1 scored surface 정의 (LOCK)

- `polished_reading` non-empty가 최우선 scored surface
- deterministic fallback은 secondary surface

### 9.2 cache namespace (LOCK)

- `product_type=life_cycle` + `render_profile=life_cycle_longform_v1`는 별도 polished cache namespace를 사용
- baseline/target/long-form cache는 서로 충돌하면 안 된다

### 9.3 PDF parity (LOCK)

- PDF는 같은 `polished_reading`을 source로 사용해야 한다
- long-form sample / PDF / manual QA가 서로 다른 표면을 보면 안 된다

### 9.4 evidence identity (LOCK)

long-form evidence bundle은 최소 아래 4종을 가진다.

1. `life_cycle_longform_sample_response.json`
2. `life_cycle_longform_gate_summary.json`
3. `life_cycle_longform_release_manifest.json`
4. `life_cycle_longform_manual_qa.md`

release evidence 경로는 `PRD/release_evidence/v1_5_0/`로 고정한다.

---

## 10. 품질 게이트 및 editorial acceptance

### 10.1 기존 HF gate 계승

- `v1.4.0`의 HF gate는 그대로 계승한다
- 장문 승격은 HF gate를 대체하지 않고 그 위에 editorial gate를 추가한다

### 10.2 long-form 전용 editorial gate (LOCK)

아래 항목은 hard-fail 또는 final human fail 사유가 된다.

1. 핵심 3개 섹션의 첫 문장이 설명문처럼 시작
2. 핵심 표현/도입 패턴이 반복되어 템플릿감이 강함
3. 개인화 토큰이 부자연스럽거나 거의 체감되지 않음
4. 고점/저점/반복패턴/다음3년이 정보 나열만 하고 왜 중요한지 안 보임
5. 행동선이 `How to use -> 현재 위치 -> 다음 3년 -> valid_until -> CTA`로 이어지지 않음
6. jargon 또는 내부 라벨 노출
7. PDF/source/manifest/manual QA가 다른 표면을 보거나 hash가 어긋남
8. `인생 고점/저점 지도`가 여전히 정보 나열형이라 "왜 이 시기가 이 사람에게 중요한지"가 약함
9. 핵심 3섹션이 target candidate 수준의 bullet-heavy tone을 벗어나지 못함

### 10.3 human spot-check 기준 (LOCK)

최소 아래 질문에 모두 `YES`여야 `cutover_ready: YES` 가능:

1. 첫 3문장이 잡아끄는가
2. 설명서보다 사람 말처럼 읽히는가
3. "내 얘기 같다"가 실제로 드는가
4. 읽고 나면 바로 행동 기준이 떠오르는가
5. 저장/공유/재구매하고 싶은 감각이 있는가

---

## 11. 테스트/검증/증적

### 11.1 자동 테스트 (필수)

- adapter contract test
- long-form render profile routing test
- long-form cache namespace test
- long-form PDF narrative selection test
- long-form evidence identity/hash test
- long-form editorial regression test

### 11.2 샘플 검토 (필수)

- primary sample 1건
- editorial pack 최소 20건
- edge/fallback sample 최소 1건
- `v1.4.0 target` 샘플은 참고/copy bank로는 사용 가능하지만, `v1.5.0 long-form evidence`를 대체하지 못함

### 11.3 manual QA (필수)

- reviewer 1명 이상
- `cutover_ready: YES/NO` 명시
- reviewer summary와 근거 문장 필수

### 11.4 clean evidence 규칙 (LOCK)

- manifest에 `dirty_worktree=false`
- `git_worktree: clean`
- sample/gate/manual QA/manifest 해시 일치
- `PRD/release_evidence/v1_5_0/` 산출물은 실제 파일로 존재해야 하며 placeholder만으로는 불충분

---

## 12. 보안/개인정보/윤리

- 건강/임신/수술/파산 등 확정형 예언 금지
- 공포 마케팅 금지
- 개인 데이터는 리포트 생성 목적 외 장문 카피 embellishment에 사용하지 않는다
- 관계/일/돈 섹션에서 취약성 자극형 판매 문구 금지

---

## 13. 로드맵

| 단계 | 목표 |
|---|---|
| Stage 1 | `life_cycle_longform_v1` adapter/prototype 내부 생성 |
| Stage 2 | target candidate bullet-heavy tone을 paragraph-driven narrative로 승격 |
| Stage 3 | `PRD/release_evidence/v1_5_0/` sample/gate/manifest/manual QA 생성 |
| Stage 4 | clean readiness + human YES |
| Stage 5 | shipped route cutover 여부 결정 |

---

## 14. 현재 상태와 남은 리스크

### 14.1 현재 상태

- `life_cycle_target_v1`는 구조/게이트/evidence/manual QA 기준으로 이미 닫힌 상태다
- target sample은 "꽤 괜찮은 상업형 후보 리포트" 수준까지는 도달했다
- generic 장문 엔진은 repo 안에 이미 존재한다
- 현재 gap은 엔진 부재가 아니라 `life_cycle` 상품 연결 부재 + long-form evidence 부재다
- 현재 runtime 기본 경로는 여전히 `life_cycle_lite_v1`다

### 14.1-a 현재 상태 — partial yes 판단

2026-03-16 기준 판단은 "부분적으로 yes"입니다.

- 네: 지금 로컬 그대로도 `life_cycle_target_v1` 수준의 꽤 괜찮은 상업형 후보 리포트는 나옵니다.
- 아니오: 그러나 이 상태를 `v1.5.0` 장문 상업 리포트 완료로 볼 수는 없습니다.

즉 현재 상태는 `잘 만든 target candidate`이지, `장문 commercial release`는 아닙니다.

### 14.2 남은 리스크 — 구조

- generic chapter model과 `life_cycle` target H2 order가 바로 일치하지 않을 수 있다
- generic front/surface 보정 경로를 그대로 타면 product surface가 오염될 수 있다

### 14.3 남은 리스크 — 카피

- generic 장문본은 "잘 읽히는 분석형 리포트"까지는 가능하지만, "잘 팔리는 상업 카피"는 별도 editorial layer가 필요하다
- `v1.4.0 target` manual QA와 20-case review는 전반적으로 PASS지만, 고점/저점 문단의 맥락 문장 부족과 약한 템플릿감이 반복적으로 지적된다
- 따라서 `현재 위치`, `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`를 문단형 narrative로 승격하는 작업이 핵심 리스크다

### 14.4 남은 리스크 — 운영

- long-form evidence를 만들더라도 human reviewer가 `YES`를 주지 않으면 shipped cutover는 불가
- long-form PDF parity와 cache namespace가 안 닫히면 운영 중 혼선이 생길 수 있다

### 14.5 남은 리스크 — 구현 공백

- `v1.5.0` PRD가 요구하는 전용 adapter/renderer/checker/evidence path는 아직 로컬에 없다
- 즉 이 문서는 "이미 닫힌 상태"를 기록하는 문서가 아니라, **지금부터 구현해야 할 장문 promotion plan**이다

---

## 15. 완료 정의

### 15.1 long-form prototype 완료 정의

아래가 모두 만족되면 prototype 완료:

1. `life_cycle` payload에서 long-form sample 1건 생성 가능
2. `report_engine` 기반 chapter block/LLM refinement가 실제 연결됨
3. sample text가 deterministic target보다 의미 있게 길고 풍부함
4. 이 시점에도 아직 shipped route 변경은 하지 않는다

### 15.2 long-form cut 완료 정의

아래가 모두 만족되면 `v1.5.0` long-form cut 완료:

1. `life_cycle_longform_v1` sample/evidence/manual QA/manifest 존재
2. cache/PDF/hash/source identity 정합성 통과
3. 기존 HF gate 회귀 없음
4. long-form editorial gate 통과
5. final human spot-check `cutover_ready: YES`
6. `scripts/check_life_cycle_longform_cutover_ready.py` 또는 동등 checker `READY`
7. 이 상태에서만 shipped route cutover를 검토할 수 있다

### 15.2-a 현재 미달 항목 (2026-03-16)

현재 로컬은 아래 항목이 미달입니다.

1. `life_cycle_longform_v1` adapter/renderer 없음
2. `PRD/release_evidence/v1_5_0/` evidence bundle 없음
3. long-form 전용 checker 없음
4. shipped route는 여전히 baseline

즉 `v1.5.0`은 아직 완료 상태가 아니라 설계/착수 상태입니다.

### 15.3 비범위 재확인

아래는 본 문서 완료와 무관:

- `yearly_forecast`, `compatibility` 장문 productization
- BTR 정확도 productization
- 결제/로그인/리텐션 CRM
- 점성 코어 알고리즘 변경
