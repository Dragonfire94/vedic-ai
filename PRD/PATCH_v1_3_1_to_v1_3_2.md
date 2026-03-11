# PATCH v1.3.1 -> v1.3.2

- 기준 피드백: 첨부 리뷰 메모(체크리스트 pytest 상태 충돌, source of truth 미정렬, dirty baseline, stale generic test 우선순위, helper 파일의 권장 구조화)
- 수정 대상:
  - `PRODUCT_SPEC_PRD_v1_3_2.md`
  - `IMPLEMENTATION_CHECKLIST_v1_3_2.md`
  - `CHANGELOG_PRD_v1_3_2.md`

## 반영 요약

1. pytest 상태 문구 통일
   - 체크리스트 item 25와 AUD-25를 모두 `실행은 가능하지만 release blocker가 남아 있음`으로 정렬

2. interim source of truth 명시
   - `backend/API.md`, `backend/QUALITY_GATES.md`, `README.md`가 정렬되기 전까지 PRD v1.3.2를 현재 runtime / release source of truth로 간주

3. dirty baseline 경고 추가
   - `367 passed, 14 failed, 1 skipped`는 현재 로컬 dirty worktree 기준임을 문서에 명시

4. 구현 우선순위 재배치
   - Phase A는 `runner 고정 + failure inventory 고정 + tempdir 분리`까지만 우선
   - generic stale test 재작성은 orchestrator/product path 이후로 미룸

5. helper/file 구조 완화
   - `product_orchestrator.py`, `render_contract.py`, `life_cycle_helpers.py`, `commercial_gate_helpers.py`는 권장 구조로 유지
   - 완료조건은 behavior / test / gate 중심으로 유지
