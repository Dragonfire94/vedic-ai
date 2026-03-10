# PATCH: v1.2.26 -> v1.2.27

> **적용 방법**: 이 패치는 `PRODUCT_SPEC_PRD_v1_2_26.md`, `IMPLEMENTATION_CHECKLIST_v1_2_26.md`, `CHANGELOG_PRD_v1_2_26.md` 위에 적용합니다.
> 적용 후 생성되는 새 정본에서는 문서 제목/파일명/current version 표기와 `contract_version` 예시 값을 `v1.2.27`로 동기화합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.2.26 기준을 유지합니다.

---

## 패치 목적

이번 패치는 아래 4가지 문서 정합성 이슈만 닫습니다.

1. `7.1.1`의 구조 전략 문구가 P0 범위 안에 P1 내용을 다시 끌어들이는 문제
2. `4.1 공통 출력 구조`와 `7.1.2 exact H2 order`가 서로 다른 순서를 갖는 문제
3. 구현 체크리스트가 `contract test 4종`이라고 적으면서 실제로는 3개만 정의한 문제
4. 수동 QA 템플릿/샘플 응답 증적의 경로, 포맷, Done 기준이 닫혀 있지 않은 문제

이번 패치는 **문서 정합성 및 종료 조건 보강**만 다루며, runtime scope/product policy/HF contract 자체는 바꾸지 않습니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.2.27 *(현재)*
- `7.1.1`의 `인생 전체 70% + 다음 3년 구체화 30%` 표현을 `life_cycle-full` 목표 구조로 한정해, `life_cycle-lite` P0 최소 구조와 충돌하지 않도록 정렬
- `4.1 공통 출력 구조`에 `life_cycle-lite` P0는 `7.1.2` exact H2 order를 우선 적용한다는 단서를 추가하고, CTA-lite -> 면책/윤리/데이터 보호 순서를 명시적으로 정렬
- 구현 체크리스트에 누락된 `contract test 3`(exact H2 order + CTA adjacency)를 추가해 `contract test 4종` 표현과 실제 정의를 일치시킴
- 수동 QA 템플릿/샘플 응답 증적의 경로, 포맷, Done 기준을 고정해 릴리즈 증적 계약을 닫음
```

---

## PRODUCT_SPEC 상단 변경 로그 — `현재 버전 핵심` 블록에 아래 4줄 추가

```markdown
  - `7.1.1`의 70/30 구조 표현은 `life_cycle-full` 목표 구조로만 유지하고, P0 최소 구조와 충돌하지 않도록 정렬
  - `4.1 공통 출력 구조`는 `life_cycle-lite` P0에서 `7.1.2` exact H2 order를 우선 적용하도록 명시
  - 구현 체크리스트의 누락된 `contract test 3`을 추가해 `contract test 4종`과 실제 정의를 일치시킴
  - 수동 QA 템플릿/샘플 응답 증적의 경로, 포맷, Done 기준을 고정
```

---

## PRODUCT_SPEC — 섹션 7.1.1 구조 전략 한 줄 교체

**기존 구조 전략 한 줄을 아래 문장으로 교체:**

```markdown
- **구조 전략 (LOCK)**: `life_cycle-full` 목표 구조는 인생 전체 70% + 다음 3년 구체화 30%를 유지한다. 단, v1.2.27의 `life_cycle-lite` P0는 `7.1.2`의 최소 출력 구조만 적용하며, 다음 3년 구체화는 P1에서만 복원한다.
```

이 교체로 `7.1.1`이 P0에 포함되어 있어도 `7.1.7` P1 미노출 정책과 충돌하지 않게 됩니다.

---

## PRODUCT_SPEC — 섹션 4.1 공통 출력 구조 정렬

### 1) 표의 7행/8행 순서를 아래로 교체

```markdown
| 7 | CTA | 다음 단계 (섹션 7 규칙) |
| 8 | 면책/윤리/데이터 보호 | 필수 5줄 (섹션 11.2) |
```

### 2) 표 바로 아래에 아래 note 블록 삽입

```markdown
> `life_cycle-lite` P0는 섹션 `7.1.2`의 exact H2 order를 우선 적용합니다.
> 즉 `CTA-lite`는 `valid_until 설명` 바로 다음 H2에 오고,
> `면책/윤리/데이터 보호`는 마지막 H2로 고정합니다.
```

이 변경으로 `4.1`만 읽고 구현해도 `7.1.2` 및 release mode gate와 반대 순서로 가지 않게 됩니다.

---

## IMPLEMENTATION_CHECKLIST — Phase E에 누락된 contract test 3 추가

**기존 `28. 최소 contract test 2` 바로 다음에 아래 `28-a` 블록을 삽입:**

```markdown
28-a. `TODO` 최소 contract test 3: exact H2 order + CTA adjacency
   - 체크:
     - H2 순서가 `cover/meta` -> `How to use 1p` -> `인생 구조 한 장 요약` -> `4단계 인생 구조` -> `현재 위치` -> `마하다샤 단계 목록` -> `방법론 카드` -> `valid_until 설명` -> `CTA-lite` -> `면책/윤리/데이터 보호`
     - `CTA-lite` 직전 H2는 항상 `valid_until 설명`
     - `면책/윤리/데이터 보호`는 마지막 H2
     - legacy alias(`한 장 요약`, `CTA`)는 없음
   - Done 기준: 동일 payload에서 markdown H2 order가 deterministic 하게 유지되고, exact H2 계약 위반 시 테스트가 실패함
```

**그리고 `추천 구현 순서`의 10번 항목은 아래 한 줄로 교체:**

```markdown
10. contract test 4종 (`#27`, `#28`, `#28-a`, `#30`) + `cheap_validation_gate` release mode 추가 (`Phase E` `#25~#32`)
```

이 변경으로 체크리스트의 `contract test 4종` 표현과 실제 상세 항목이 일치합니다.

---

## IMPLEMENTATION_CHECKLIST — 수동 QA 증적 항목 교체

**기존 `31. 수동 QA 템플릿 작성` 블록을 아래로 교체:**

```markdown
31. `TODO` 수동 QA 템플릿 작성
    - 권장 경로: `PRD/release_evidence/v1_2_27/life_cycle_lite_manual_qa.md`
    - 형식: markdown 1파일, case 2건(정상 1 + fallback/edge 1) 고정
    - 필수 필드:
      - case id
      - input fixture / request summary
      - expected points
      - actual summary
      - PASS/FAIL
      - reviewer
      - run date (Asia/Seoul)
    - Done 기준: 2건 모두 채워져 있고, 각 case의 PASS/FAIL과 근거가 문서 안에서 바로 확인 가능
```

---

## IMPLEMENTATION_CHECKLIST — 샘플 응답 증적 항목 교체

**기존 `39. 샘플 응답 1개 저장` 블록을 아래로 교체:**

```markdown
39. `TODO` 샘플 응답 1개 저장
    - 권장 경로: `PRD/release_evidence/v1_2_27/life_cycle_lite_sample_response.json`
    - 형식: 실제 `life_cycle` API 응답 1건의 pretty-printed JSON
    - 필수 포함:
      - full P0 meta contract
      - `valid_until_fallback`
      - `render_profile`
      - exact H2 order가 보이는 `polished_reading` 또는 동등 렌더 본문
    - Done 기준: 문서/QA/release gate 검토자가 코드 실행 없이도 계약 필드와 렌더 구조를 확인할 수 있음
```

---

## 적용 후 기대 결과

1. P0와 P1의 구조 문구가 같은 섹션 안에서 다시 충돌하지 않습니다.
2. `4.1 공통 출력 구조`와 `7.1.2 exact H2 order`가 같은 방향을 가리킵니다.
3. 구현 체크리스트에서 `contract test 4종`이 실제로 셀 수 있는 형태가 됩니다.
4. 릴리즈 증적이 사람별 임시 위치가 아니라 고정 경로/고정 포맷으로 남습니다.

---

## 비범위

이번 패치는 아래를 바꾸지 않습니다.

- `life_cycle-lite`를 유일한 P0로 두는 정책
- `yearly_forecast` / `compatibility` bugfix-only 정책
- `cheap_validation_gate.py`의 metric semantics
- runtime API/cache/finalizer/PDF 구현 범위
- P1 기능(고점/저점, 반복 패턴, 다음 3년 구체화)의 비범위 정책
