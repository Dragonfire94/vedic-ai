# CHANGELOG_PRD v1.4.0

- 기준 버전: `PRODUCT_SPEC_PRD_v1_3_2.md` / `IMPLEMENTATION_CHECKLIST_v1_3_2.md`
- 변경일: 2026-03-11
- 목적: 짧은 roadmap 초안이 아니라, `v1.3.2` 정본 본문을 유지한 채 `v1.4.0`을 **단일 상품 canonical 문서**로 재정렬

## 핵심 변경

1. `life_cycle-lite` / `life_cycle-full`을 외부 SKU가 아니라 **내부 단계 이름**으로 재해석
2. 외부 상품은 하나의 `Vedic Life Cycle Report`로 고정
3. `v1.4.0`은 `baseline freeze`와 `target report cutover`를 같은 문서 안에서 관리하도록 정렬
4. `7.1.5` / `7.1.6` / `7.1.7`을 separate product가 아니라 같은 보고서의 target stage 계약으로 승격
5. checklist 완료 정의를 `baseline cut` / `target report cut` 2단계로 분리
6. release evidence 경로를 `PRD/release_evidence/v1_4_0/` 기준으로 갱신
7. `backend/API.md` / `backend/QUALITY_GATES.md` / `README.md`가 정렬되기 전까지 PRD v1.4.0이 interim source of truth임을 유지

## 주의

- 이번 버전은 `v1_3_2` 문서를 요약한 것이 아니라, 상세 계약/표/게이트/테스트 설명을 최대한 유지한 채 framing만 재정렬한 버전이다.
- 코드 구현 상태를 자동으로 바꾸는 문서는 아니므로, 실제 baseline/test/runtime 상태는 여전히 저장소 기준으로 재검증해야 한다.
