# PATCH: v1.2.27 -> v1.2.28

> **적용 방법**: 이 패치는 `PRODUCT_SPEC_PRD_v1_2_27.md`, `IMPLEMENTATION_CHECKLIST_v1_2_27.md`, `CHANGELOG_PRD_v1_2_27.md` 위에 적용합니다.
> 적용 후 생성되는 새 정본에서는 문서 제목/파일명/current version 표기와 `contract_version` 예시 값을 `v1.2.28`로 동기화합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.2.27 기준을 유지합니다.

---

## 패치 목적

이번 패치는 아래 4가지 문서 종료 조건/정합성 이슈만 닫습니다.

1. `contract test 3`의 legacy alias 금지 목록이 정본 exact H2 계약보다 좁아, `마하다샤 인생 단계 목록` 회귀를 놓칠 수 있는 문제
2. 수동 QA/샘플 응답 증적 경로는 고정했지만, 최종 완료 정의/수용 기준에는 artifact 존재가 아직 닫혀 있지 않은 문제
3. 샘플 응답 증적이 `polished_reading` 대신 `동등 렌더 본문`도 허용해 리뷰 기준이 다시 흔들릴 수 있는 문제
4. `4.1 공통 출력 구조`의 정렬 note가 `life_cycle-lite` P0 우선 규칙인지, 공통 구조 전체 변경인지 해석 차이를 남기는 문제

이번 패치는 **문서 정합성 및 release evidence 종료 조건 보강**만 다루며, runtime scope/product policy/HF contract 자체는 바꾸지 않습니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.2.28 *(현재)*
- `contract test 3`의 legacy alias 금지 목록에 `마하다샤 인생 단계 목록`을 추가해 exact H2 계약과 테스트 범위를 완전히 일치시킴
- `수동 QA 2건 통과`만으로는 부족하다는 점을 반영해, `release_evidence` 파일 2종 존재를 체크리스트/PRD 완료 조건에 추가
- 샘플 응답 증적의 렌더 본문 필드를 `polished_reading` non-empty로 고정해 scored surface 리뷰 기준을 다시 흔들리지 않게 잠금
- `4.1 공통 출력 구조` note에 이 정렬이 `life_cycle-lite` P0 release order 우선 규칙이며, 다른 상품은 각 제품 섹션 LOCK 구조를 따른다는 범위 문구를 추가
```

---

## PRODUCT_SPEC 상단 변경 로그 — `현재 버전 핵심` 블록에 아래 4줄 추가

```markdown
  - `contract test 3`의 alias 금지 목록에 `마하다샤 인생 단계 목록`을 추가해 exact H2 계약과 테스트 범위를 일치시킴
  - `release_evidence` 파일 2종을 체크리스트/PRD 완료 조건에 포함해, 수동 QA/샘플 응답 증적이 실제 릴리즈 종료 조건이 되도록 고정
  - 샘플 응답 증적의 렌더 본문 필드를 `polished_reading` non-empty로 고정
  - `4.1 공통 출력 구조` 정렬 note가 `life_cycle-lite` P0 release order 우선 규칙임을 명시
```

---

## PRODUCT_SPEC — 섹션 4.1 공통 출력 구조 note 블록 교체

**기존 note 블록을 아래 4줄로 교체:**

```markdown
> `life_cycle-lite` P0는 섹션 `7.1.2`의 exact H2 order를 우선 적용합니다.
> 즉 `CTA-lite`는 `valid_until 설명` 바로 다음 H2에 오고,
> `면책/윤리/데이터 보호`는 마지막 H2로 고정합니다.
> 이 정렬은 `life_cycle-lite` P0 release order에 대한 우선 규칙이며, 다른 상품은 각 제품 섹션의 LOCK 구조를 따릅니다.
```

이 교체로 `4.1`의 정렬 문구를 읽을 때 `life_cycle-lite` P0 override인지, 모든 상품 공통 순서 변경인지 혼동하지 않게 됩니다.

---

## PRODUCT_SPEC — 섹션 10.2 수동 QA에 release evidence 경로 1줄 추가

**`운영 근거 (v1.2.27)` 블록 바로 아래에 아래 1줄을 삽입:**

```markdown
> - 릴리즈 증적은 `PRD/release_evidence/v1_2_28/life_cycle_lite_manual_qa.md`와 `PRD/release_evidence/v1_2_28/life_cycle_lite_sample_response.json` 2종으로 고정합니다.
```

이 추가로 체크리스트의 artifact 경로와 정본 PRD의 수동 QA 운영 규칙이 같은 경로를 가리키게 됩니다.

---

## PRODUCT_SPEC — 섹션 14.6 검증/수용 기준에 release evidence 완료 조건 1개 추가

**기존 15번 항목 바로 아래에 아래 16번을 추가:**

```markdown
16. `PRD/release_evidence/v1_2_28/` 아래에 `life_cycle_lite_manual_qa.md`와 `life_cycle_lite_sample_response.json`가 모두 존재하고, 문서만 읽어도 QA 결과와 exact H2 렌더 구조를 확인할 수 있음
```

이 추가로 release evidence가 권장 산출물이 아니라 실제 출고 종료 조건이 됩니다.

---

## IMPLEMENTATION_CHECKLIST — Phase E `contract test 3` alias 금지 목록 보강

**기존 `28-a` 블록의 alias 금지 줄을 아래 한 줄로 교체:**

```markdown
      - legacy alias(`한 장 요약`, `마하다샤 인생 단계 목록`, `CTA`)는 없음
```

이 교체로 `7.1.2` exact H2 계약과 테스트 체크 항목이 완전히 일치합니다.

---

## IMPLEMENTATION_CHECKLIST — 수동 QA 증적 경로 버전 업데이트

**기존 `31. 수동 QA 템플릿 작성` 블록의 권장 경로 1줄을 아래로 교체:**

```markdown
    - 권장 경로: `PRD/release_evidence/v1_2_28/life_cycle_lite_manual_qa.md`
```

---

## IMPLEMENTATION_CHECKLIST — 샘플 응답 증적 항목 교체

**기존 `39. 샘플 응답 1개 저장` 블록을 아래로 교체:**

```markdown
39. `TODO` 샘플 응답 1개 저장
    - 권장 경로: `PRD/release_evidence/v1_2_28/life_cycle_lite_sample_response.json`
    - 형식: 실제 `life_cycle` API 응답 1건의 pretty-printed JSON
    - 필수 포함:
      - full P0 meta contract
      - `valid_until_fallback`
      - `render_profile`
      - non-empty `polished_reading`
      - `polished_reading` 안에서 exact H2 order 확인 가능
    - Done 기준: 문서/QA/release gate 검토자가 코드 실행 없이도 계약 필드와 최종 렌더 구조를 확인할 수 있음
```

이 교체로 scored surface 검토 기준과 샘플 응답 증적 기준이 같은 필드명을 사용하게 됩니다.

---

## IMPLEMENTATION_CHECKLIST — 섹션 8 P0 완료 정의에 release evidence 완료 조건 1개 추가

**기존 18번 항목 바로 아래에 아래 19번을 추가:**

```markdown
19. `PRD/release_evidence/v1_2_28/` 아래에 `life_cycle_lite_manual_qa.md`와 `life_cycle_lite_sample_response.json`가 모두 존재하고, 수동 QA 결과와 `polished_reading` exact H2 order를 문서만으로 재검수할 수 있음
```

이 추가로 artifact 생성이 단순 TODO가 아니라 P0 완료 정의의 일부가 됩니다.

---

## 적용 후 기대 결과

1. exact H2 회귀 테스트가 정본 계약과 같은 alias 범위를 검사합니다.
2. 수동 QA/샘플 응답 증적이 실제 릴리즈 완료 조건으로 승격됩니다.
3. 샘플 응답 리뷰 기준이 `polished_reading`으로 고정되어 scored surface 판단이 다시 흔들리지 않습니다.
4. `4.1 공통 출력 구조`의 CTA/면책 정렬 문구가 `life_cycle-lite` P0 우선 규칙이라는 점이 분명해집니다.

---

## 비범위

이번 패치는 아래를 바꾸지 않습니다.

- `life_cycle-lite`를 유일한 P0로 두는 정책
- `yearly_forecast` / `compatibility` bugfix-only 정책
- `cheap_validation_gate.py`의 metric semantics
- runtime API/cache/finalizer/PDF 구현 범위
- P1 기능(고점/저점, 반복 패턴, 다음 3년 구체화)의 비범위 정책
