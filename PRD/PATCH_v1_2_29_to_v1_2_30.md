# PATCH: v1.2.29 -> v1.2.30

> **적용 방법**: 이 패치는 `PRODUCT_SPEC_PRD_v1_2_29.md`, `IMPLEMENTATION_CHECKLIST_v1_2_29.md`, `CHANGELOG_PRD_v1_2_29.md` 위에 적용합니다.
> 적용 후 생성되는 새 정본에서는 문서 제목/파일명/current version 표기와 `contract_version` 예시 값을 `v1.2.30`로 동기화합니다.
> 이 패치가 명시하지 않은 모든 내용은 v1.2.29 기준을 유지합니다.

---

## 패치 목적

이번 패치는 아래 4가지 문서 종료 조건/운영 정합성 이슈만 닫습니다.

1. 구현 체크리스트의 `추천 구현 순서`가 `backend/API.md` 공개 계약 동기화를 빠뜨려, 요약 순서만 따라도 public contract 문서가 누락될 수 있는 문제
2. PRD의 `14.6 검증/수용 기준`이 체크리스트 `8. P0 완료 정의`보다 약해, 최종 sign-off에서 `API.md` / backend-only vs repo-wide 문구 정렬이 빠질 수 있는 문제
3. `release_evidence` 3종이 `commit_sha`/SHA-256만 추적하고 `contract_version` / evidence 폴더 버전 일치를 잠그지 않아, 이전 버전 산출물을 현재 폴더에 재사용해도 문서상 통과처럼 보일 수 있는 문제
4. `sample response`를 Phase E로 옮긴 뒤 Phase G 번호가 비어 있고 요약 참조가 덜 선명해진 문제

이번 패치는 **최종 수용 기준, 문서 source of truth, release evidence 버전 추적**만 보강하며, runtime scope/product policy/HF contract 자체는 바꾸지 않습니다.

---

## 변경 로그 — 최상단에 아래 블록 삽입

```markdown
### v1.2.30 *(현재)*
- `추천 구현 순서`에 `backend/API.md` 공개 계약 동기화를 명시적으로 포함해, 요약 순서만 따라도 public contract 문서가 누락되지 않도록 정렬
- PRD `14.6 검증/수용 기준`에 `backend/API.md`와 backend-only vs repo-wide 문구 정렬을 추가해 체크리스트 `P0 완료 정의`와 최종 sign-off 기준을 일치시킴
- `release_evidence` 3종이 같은 `contract_version` / evidence 폴더 버전을 가리켜야 한다는 규칙을 추가해, 이전 버전 산출물 재사용 위험을 차단
- `sample response` Phase 이동 뒤 남은 Phase G 번호를 정리하고, 관련 요약 참조를 새 번호 기준으로 맞춤
```

---

## PRODUCT_SPEC 상단 변경 로그 — `현재 버전 핵심` 블록에 아래 4줄 추가

```markdown
  - `추천 구현 순서`에 `backend/API.md` 공개 계약 동기화를 포함해 문서 작업 순서와 완료 정의를 정렬
  - `14.6 검증/수용 기준`에 `backend/API.md` 및 backend-only vs repo-wide 문구 정렬을 추가
  - `release_evidence` 3종은 동일한 `contract_version`과 evidence 폴더 버전을 공유해야 함을 명시
  - `sample response` 이동 후 남은 Phase G 번호/참조를 정리
```

---

## PRODUCT_SPEC — 섹션 10.2 수동 QA의 release evidence 운영 규칙 2줄 교체

**기존 release evidence 2줄을 아래 3줄로 교체:**

```markdown
> - 릴리즈 증적은 `PRD/release_evidence/v1_2_30/life_cycle_lite_manual_qa.md`, `PRD/release_evidence/v1_2_30/life_cycle_lite_sample_response.json`, `PRD/release_evidence/v1_2_30/life_cycle_lite_gate_summary.json` 3종으로 고정합니다.
> - 수동 QA 문서는 세 아티팩트가 같은 release candidate 기준인지 확인할 수 있도록 `contract_version`, `release_evidence_dir`, `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`를 함께 기록합니다.
> - `life_cycle_lite_sample_response.json`의 `meta.contract_version`, `life_cycle_lite_gate_summary.json`의 `contract_version`, `release_evidence_dir`는 모두 현재 정본 버전 `v1.2.30`과 일치해야 합니다.
```

이 교체로 수동 QA 문서가 단순 SHA 기록을 넘어, 현재 버전 기준 증적인지까지 한 번에 판단할 수 있게 됩니다.

---

## PRODUCT_SPEC — 섹션 14.6 검증/수용 기준의 release/document 항목 보강

### 1) 기존 16번 항목을 아래로 교체

```markdown
16. `PRD/release_evidence/v1_2_30/` 아래에 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json`가 모두 존재하고, 수동 QA 문서에 `contract_version`, `release_evidence_dir`, `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`가 기록되어 있으며, sample/gate/evidence dir가 모두 현재 정본 버전 `v1.2.30`을 가리킴
```

### 2) 기존 16번 바로 아래에 아래 17번, 18번을 추가

```markdown
17. `backend/API.md`가 `/ai_reading`의 optional `product_type`, `life_cycle` 요청 입력, full P0 meta contract, bugfix-only 범위, backend-only P0 한계를 현재 버전 기준으로 설명함
18. `backend/QUALITY_GATES.md`와 `README.md`가 backend-only P0와 repo-wide rollout을 명확히 구분하고, 최종 release gate source of truth 문구가 현재 계약과 동일함
```

이 추가로 PRD 최종 sign-off 기준이 체크리스트의 완료 정의와 같은 수준으로 잠깁니다.

---

## IMPLEMENTATION_CHECKLIST — Phase E 수동 QA 템플릿 메타 보강

**기존 `31. 수동 QA 템플릿 작성` 블록을 아래로 교체:**

```markdown
31. `TODO` 수동 QA 템플릿 작성
    - 권장 경로: `PRD/release_evidence/v1_2_30/life_cycle_lite_manual_qa.md`
    - 형식: markdown 1파일, case 2건(정상 1 + fallback/edge 1) 고정
    - 문서 상단 필수 메타:
      - `contract_version`
      - `release_evidence_dir`
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
    - Done 기준: 2건 모두 채워져 있고, 문서 상단 메타만 읽어도 release evidence 3종의 hash/commit/version이 현재 정본 `v1.2.30` 기준으로 묶여 있음을 확인할 수 있음
```

---

## IMPLEMENTATION_CHECKLIST — Phase E gate/sample evidence의 버전 일치 규칙 보강

### 1) 기존 `31-a. release gate summary 저장` 블록의 Done 기준 마지막 줄을 아래로 교체

```markdown
    - Done 기준: 자동 gate가 실제 release source of truth로 통과했는지 문서/QA 검토자가 별도 로그 검색 없이 확인 가능하고, `contract_version == v1.2.30`이 수동 QA 문서 및 evidence 폴더 버전과 일치함
```

### 2) 기존 `31-b. 샘플 응답 1개 저장` 블록의 Done 기준 마지막 줄을 아래로 교체

```markdown
    - Done 기준: 문서/QA/release gate 검토자가 코드 실행 없이도 계약 필드와 최종 렌더 구조를 확인할 수 있고, `meta.contract_version == v1.2.30`이며, 파일 SHA-256이 수동 QA 문서에 기록됨
```

이 교체로 샘플 응답/자동 gate 결과가 "현재 버전 산출물"인지까지 증적에 포함됩니다.

---

## IMPLEMENTATION_CHECKLIST — Phase G 번호 정리

**기존 Phase G 항목 번호를 아래처럼 정리합니다.**

```markdown
39. `TODO` 구현 완료 후 `PRODUCT_SPEC_PRD_v1_2_30.md` 정본과 코드 정합 재검수
40. `TODO` `README.md` 범위/출고 기준 정리
41. `TODO` frontend/client/API/cache 구현 + E2E 반영 (repo-wide 후속)
```

즉, 기존 `40 -> 39`, `41 -> 40`, `42 -> 41`로 당겨서 Phase G 번호 공백을 제거합니다.

---

## IMPLEMENTATION_CHECKLIST — 추천 구현 순서 11번, 13번 교체

### 1) 기존 11번 항목을 아래 한 줄로 교체

```markdown
11. `backend/API.md` + `backend/QUALITY_GATES.md` + `README.md` 문서 계약 동기화 (`Phase G` #37~#40)
```

### 2) 기존 13번 항목을 아래 한 줄로 교체

```markdown
13. repo-wide 적용이 필요하면 frontend/client note만이 아니라 `frontend/app/page.tsx`, `frontend/lib/api.ts`, `frontend/app/chart/ChartClient.tsx` 구현까지 반영 (`Phase G` #41)
```

이 교체로 요약 순서만 보고 구현해도 `API.md` 공개 계약 동기화를 빠뜨리지 않게 됩니다.

---

## IMPLEMENTATION_CHECKLIST — 섹션 8 P0 완료 정의의 release evidence 항목 교체

**기존 19번 항목을 아래로 교체:**

```markdown
19. `PRD/release_evidence/v1_2_30/` 아래에 `life_cycle_lite_manual_qa.md`, `life_cycle_lite_sample_response.json`, `life_cycle_lite_gate_summary.json`가 모두 존재하고, 수동 QA 문서에 `contract_version`, `release_evidence_dir`, `commit_sha`, `sample_response_sha256`, `gate_summary_sha256`가 기록되어 있으며, sample/gate/evidence dir가 모두 현재 정본 버전 `v1.2.30`과 일치함
```

이 교체로 체크리스트 완료 정의도 "파일 존재"를 넘어서 "현재 버전 증거"까지 요구하게 됩니다.

---

## 적용 후 기대 결과

1. 요약 구현 순서와 실제 완료 정의가 다시 같은 문서 집합을 가리킵니다.
2. PRD 최종 수용 기준이 체크리스트보다 약한 상태가 해소됩니다.
3. release evidence 3종이 "같은 커밋"일 뿐 아니라 "같은 계약 버전" 기준인지도 문서만으로 재검수할 수 있습니다.
4. Phase G 번호 공백과 요약 참조 혼선이 사라집니다.

---

## 비범위

이번 패치는 아래를 바꾸지 않습니다.

- `life_cycle-lite`를 유일한 P0로 두는 정책
- `yearly_forecast` / `compatibility` bugfix-only 정책
- `cheap_validation_gate.py`의 metric semantics 자체
- runtime API/cache/finalizer/PDF 구현 범위
- P1 기능(고점/저점, 반복 패턴, 다음 3년 구체화)의 비범위 정책
