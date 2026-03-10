# PATCH: v1.2.28 -> v1.2.29

> **적용 방법**: 이 패치는 `PRODUCT_SPEC_PRD_v1_2_28.md`, `IMPLEMENTATION_CHECKLIST_v1_2_28.md`, `CHANGELOG_PRD_v1_2_28.md` 위에 적용합니다.
> 적용 후 생성되는 새 정본에서는 문서 제목/파일명/current version 표기와 `contract_version` 예시 값을 `v1.2.29`로 동기화합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.2.28 기준을 유지합니다.

---

## 패치 목적

이번 패치는 아래 3가지 문서 종료 조건/운영 증적 이슈만 닫습니다.

1. 최종 출고 source of truth는 `cheap_validation_gate.py`의 `life_cycle-lite` release mode인데, 고정 `release_evidence`에는 자동 gate 결과물이 없어 sign-off 증적이 반쪽인 문제
2. 수동 QA 문서 / 샘플 응답 / 자동 gate 결과가 같은 release candidate에서 생성됐는지 추적하는 규칙이 없어, 증적 3종이 서로 다른 실행 결과로 섞일 수 있는 문제
3. `release_evidence`가 P0 완료 정의에는 들어왔지만, 구현 체크리스트의 실행 흐름상 `sample response`가 아직 Phase G에 남아 있어 검증 단계에서 누락될 수 있는 문제

이번 패치는 **release evidence 계약과 최종 수용 증적 정합성 보강**만 다루며, runtime scope/product policy/HF contract 자체는 바꾸지 않습니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.2.29 *(현재)*
- `release_evidence` 고정 산출물에 `life_cycle_lite_gate_summary.json`을 추가해, 최종 출고 source of truth인 `life_cycle-lite` release mode 자동 gate 결과까지 문서 증적으로 남기도록 고정
- 수동 QA 문서에 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`를 기록하게 해, 증적 3종이 같은 release candidate 기준인지 재검수할 수 있도록 정렬
- `sample response` 증적을 검증 단계(Phase E)로 이동시켜, 수동 QA/자동 gate/샘플 응답이 한 단계에서 함께 닫히도록 정리
```

---

## PRODUCT_SPEC 상단 변경 로그 — `현재 버전 핵심` 블록에 아래 3줄 추가

```markdown
  - `release_evidence` 고정 산출물에 `life_cycle_lite_gate_summary.json`을 추가해 최종 자동 gate 결과도 함께 보관
  - 수동 QA 문서에 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`를 기록해 증적 3종의 동일 release candidate 여부를 추적 가능하게 고정
  - `sample response` 저장을 검증 단계로 이동해, release evidence가 구현 종료 직전이 아니라 검증 단계에서 함께 닫히도록 정렬
```

---

## PRODUCT_SPEC — 섹션 8.4 경량 검증 아티팩트 계약에 2줄 추가

**기존 마지막 bullet 바로 아래에 아래 2줄을 추가:**

```markdown
- `release_evidence` 고정 산출물은 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json` 3종입니다.
- 세 아티팩트는 같은 release candidate 기준으로 생성해야 하며, 수동 QA 문서에는 최소 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`를 기록합니다.
```

이 추가로 경량 검증 전략과 실제 릴리즈 증적 계약이 같은 섹션에서 닫힙니다.

---

## PRODUCT_SPEC — 섹션 10.2 수동 QA의 release evidence 경로 1줄 교체

**기존 `운영 근거` 아래의 release evidence 고정 경로 1줄을 아래 2줄로 교체:**

```markdown
> - 릴리즈 증적은 `PRD/release_evidence/v1_2_29/life_cycle_lite_manual_qa.md`, `PRD/release_evidence/v1_2_29/life_cycle_lite_sample_response.json`, `PRD/release_evidence/v1_2_29/life_cycle_lite_gate_summary.json` 3종으로 고정합니다.
> - 수동 QA 문서는 세 아티팩트가 같은 release candidate 기준인지 확인할 수 있도록 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`를 함께 기록합니다.
```

이 교체로 수동 QA 운영 규칙이 단순 파일 존재가 아니라 실제 sign-off 추적 기준까지 포함하게 됩니다.

---

## PRODUCT_SPEC — 섹션 14.6 검증/수용 기준의 release evidence 완료 조건 1개 교체

**기존 16번 항목을 아래로 교체:**

```markdown
16. `PRD/release_evidence/v1_2_29/` 아래에 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json`가 모두 존재하고, 수동 QA 문서에 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`가 기록되어 있어 세 아티팩트가 같은 release candidate 기준인지 문서만으로 재검수할 수 있음
```

이 교체로 자동 gate 결과와 증적 traceability가 실제 수용 기준으로 승격됩니다.

---

## IMPLEMENTATION_CHECKLIST — Phase E 수동 QA 템플릿을 traceability 기준까지 포함하도록 교체

**기존 `31. 수동 QA 템플릿 작성` 블록을 아래로 교체:**

```markdown
31. `TODO` 수동 QA 템플릿 작성
    - 권장 경로: `PRD/release_evidence/v1_2_29/life_cycle_lite_manual_qa.md`
    - 형식: markdown 1파일, case 2건(정상 1 + fallback/edge 1) 고정
    - 문서 상단 필수 메타:
      - `commit_sha`
      - `sample_response_sha256`
      - `gate_summary_sha256`
    - 필수 필드:
      - case id
      - input fixture / request summary
      - expected points
      - actual summary
      - PASS/FAIL
      - reviewer
      - run date (Asia/Seoul)
    - Done 기준: 2건 모두 채워져 있고, 문서 상단의 traceability 메타만 읽어도 release evidence 3종이 같은 release candidate 기준인지 확인 가능
```

---

## IMPLEMENTATION_CHECKLIST — Phase E에 자동 gate summary 저장 항목 1개 추가

**기존 `31. 수동 QA 템플릿 작성` 바로 아래에 아래 `31-a` 블록을 삽입:**

```markdown
31-a. `TODO` release gate summary 저장
    - 권장 경로: `PRD/release_evidence/v1_2_29/life_cycle_lite_gate_summary.json`
    - 형식: `cheap_validation_gate.py` `life_cycle-lite` release mode 결과 1건의 pretty-printed JSON
    - 필수 포함:
      - `product_type`
      - `contract_version`
      - `render_profile`
      - `commit_sha`
      - `front_contract_ok`
      - `action_steps_contract_ok`
      - HARD FAIL 결과 요약 (`hard_fail_count` 또는 동등 필드)
    - Done 기준: 자동 gate가 실제 release source of truth로 통과했는지 문서/QA 검토자가 별도 로그 검색 없이 확인 가능
```

---

## IMPLEMENTATION_CHECKLIST — 샘플 응답 증적을 Phase E로 이동

### 1) 기존 Phase G의 `39. 샘플 응답 1개 저장` 블록은 삭제

### 2) `31-a` 바로 아래에 아래 `31-b` 블록을 삽입

```markdown
31-b. `TODO` 샘플 응답 1개 저장
    - 권장 경로: `PRD/release_evidence/v1_2_29/life_cycle_lite_sample_response.json`
    - 형식: 실제 `life_cycle` API 응답 1건의 pretty-printed JSON
    - 필수 포함:
      - full P0 meta contract
      - `valid_until_fallback`
      - `render_profile`
      - non-empty `polished_reading`
      - `polished_reading` 안에서 exact H2 order 확인 가능
    - Done 기준: 문서/QA/release gate 검토자가 코드 실행 없이도 계약 필드와 최종 렌더 구조를 확인할 수 있고, 파일 SHA-256이 수동 QA 문서에 기록됨
```

이 이동으로 release evidence 3종이 모두 검증 단계에서 함께 닫힙니다.

---

## IMPLEMENTATION_CHECKLIST — 추천 구현 순서 12번 항목 교체

**기존 12번 항목을 아래 한 줄로 교체:**

```markdown
12. 수동 QA 2건 + `release_evidence` 3종(`manual_qa`, `sample_response`, `gate_summary`) 저장 (`Phase E` `#30~#31-b`)
```

이 교체로 요약 순서만 보고 실행해도 필수 증적 3종을 놓치지 않게 됩니다.

---

## IMPLEMENTATION_CHECKLIST — 섹션 8 P0 완료 정의의 release evidence 완료 조건 1개 교체

**기존 19번 항목을 아래로 교체:**

```markdown
19. `PRD/release_evidence/v1_2_29/` 아래에 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json`가 모두 존재하고, 수동 QA 문서에 `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`가 기록되어 있어 세 아티팩트가 같은 release candidate 기준인지 재검수할 수 있음
```

이 교체로 체크리스트 완료 정의가 PRD 수용 기준과 같은 수준의 traceability를 요구하게 됩니다.

---

## 적용 후 기대 결과

1. 자동 gate pass 여부가 임시 로그가 아니라 고정 release evidence로 남습니다.
2. 수동 QA / 샘플 응답 / gate summary가 같은 release candidate에서 나온 것인지 문서만으로 추적할 수 있습니다.
3. 요약 구현 순서만 따라도 필수 증적 3종이 검증 단계에서 함께 닫힙니다.

---

## 비범위

이번 패치는 아래를 바꾸지 않습니다.

- `life_cycle-lite`를 유일한 P0로 두는 정책
- `yearly_forecast` / `compatibility` bugfix-only 정책
- `cheap_validation_gate.py`의 metric semantics 자체
- runtime API/cache/finalizer/PDF 구현 범위
- P1 기능(고점/저점, 반복 패턴, 다음 3년 구체화)의 비범위 정책
