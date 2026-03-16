# Life Cycle Target Manual QA

- contract_version: v1.4.0
- release_evidence_dir: PRD/release_evidence/v1_4_0
- render_profile: life_cycle_target_v1
- request_fingerprint: 05d44aaccc79aedd
- evidence_case_id: life_cycle_target_primary
- commit_sha: b59be91978b9450e01c16ecf60da3e329cc582ba
- release_manifest_path: PRD/release_evidence/v1_4_0/life_cycle_target_release_manifest.json
- editorial_rubric_path: PRD/release_evidence/v1_4_0/life_cycle_target_editorial_rubric.md
- editorial_review_path: PRD/release_evidence/v1_4_0/life_cycle_target_editorial_review_20.md
- dirty_worktree: true

## Case 1: life_cycle_target_primary

- input fixture / request summary: fake chart primary (`birth_jd=2447892.5`, `dasha_reference_jd=2461110.5`) + internal target candidate render (`subject_name=민서`, `onboarding_goal=career_money`, `focus_tokens=커리어,돈`, `concern_tokens=이직 타이밍,수입 안정`, `occupation_context=브랜드 전략 업무`, `relationship_status=싱글`)
- expected points:
  - `meta.contract_version == v1.4.0`
  - `meta.render_profile == life_cycle_target_v1`
  - target exact H2 order 13개 유지
  - `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`가 각 1회 존재
  - target gate summary `hard_fail_count == 0`
  - 개인화 흔적이 `인생 구조 한 장 요약`, `현재 위치`, `다음 3년 구체화`에 보임
- actual summary:
  - `valid_until=2029-03-11`, `next_mahadasha_date=2031-03-11`, `current_mahadasha_planet=Moon`
  - headings: cover/meta, How to use 1p, 인생 구조 한 장 요약, 4단계 인생 구조, 현재 위치, 마하다샤 단계 목록, 인생 고점/저점 지도, 반복 패턴 분석, 다음 3년 구체화, 방법론 카드, valid_until 설명, CTA-lite, 면책/윤리/데이터 보호
  - gate: `front_contract_ok=true`, `action_steps_contract_ok=true`, `life_cycle_release_ok=true`
  - personalization sections: 인생 구조 한 장 요약, 현재 위치, 다음 3년 구체화
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): 2026-03-16T13:47:36+09:00

## Human Spot Check
- reviewer: USER
- review_date_kst: 2026-03-16
- sample_path: PRD/release_evidence/v1_4_0/life_cycle_target_sample_response.json
- result: PASS

### Check 1

- item: 인생 구조 한 장 요약 첫 문장 자연스러움
- result: PASS
- note: 첫 문장은 거의 통과권입니다. 다만 이어지는 문장까지 읽으면 완전한 프리라이팅이라기보다 약한 템플릿감과 의미 반복이 남아 있어, 도입 리듬을 한 번 더 다듬는 편이 안전합니다.

### Check 2

- item: 현재 위치에 이름/관심사/맥락 자연 반영
- result: PASS
- note: 이름, 관심사, 직업 맥락, 관계 상태는 충분히 반영되어 개인화 흔적은 분명합니다. 다만 현재 위치 첫 bullet은 뜻은 맞고 맥락도 좋지만, 실제 사람이 건네는 말보다는 정리문처럼 읽혀 설명문 톤이 아직 남아 있습니다.

### Check 3

- item: target 3개 섹션(고점/저점, 반복 패턴, 다음 3년) 각 1회 존재
- result: PASS
- note: 3개 타깃 섹션은 각 1회씩 존재하고 누락이나 중복은 보이지 않습니다.

### Check 4

- item: How to use -> 다음 3년 -> valid_until -> CTA 행동선 연결
- result: PASS
- note: 읽는 법에서 제시한 흐름은 후반 행동 섹션까지 무리 없이 이어집니다. 다만 다음 3년 액션과 일부 도입 문장은 아직 정리된 설명문 톤이 남아 있어, 컷오버 직전에는 카피 polish 1회를 더 거치는 편이 좋겠습니다.

### Check 5

- item: 내부 SKU 표현/과한 영문/jargon 없음
- result: PASS
- note: 내부 SKU 노출은 없고 전반적으로 소비자용 문면입니다. 다만 cover/meta, How to use 1p, CTA-lite, valid_until 같은 기능성 라벨은 아직 소비자 문면보다 내부 문서 쪽 느낌이 조금 더 남아 있습니다.

### Check 6

- item: 전체적으로 내 얘기 같고 바로 행동이 떠오름
- result: PASS
- note: 전반적으로는 개인화와 행동 연결이 살아 있어 "내 얘기 같다"는 감각은 확보됩니다. 다만 자연스러움 기준에서는 아직 반 단계 정도 덜 다듬어진 느낌이 있고, 특히 고점/저점 섹션은 분석 결과를 설명하는 문장으로 읽히는 감각이 남아 있습니다.

### Final Note

- cutover_ready: NO
- reviewer_summary: 지금 샘플은 PASS급이고 컷오버 직전 수준까지는 왔습니다. 다만 인생 구조 한 장 요약의 이어지는 문장, 현재 위치 첫 bullet, 인생 고점/저점 지도의 도입과 본문 톤에서는 아직 설명문 성격과 약한 템플릿감이 남아 있어, cutover_ready=YES로 올려도 불안하지 않은 단계라고 보기는 어렵습니다. 현재 판단은 NO 유지가 맞고, 위 세 지점을 한 번 더 다듬은 뒤 다시 보는 편이 안전합니다.
- yes_transition_conditions: (1) 인생 구조 한 장 요약 둘째 문장 이후의 반복/템플릿감 축소, (2) 현재 위치 첫 bullet을 더 사람 말처럼 완화, (3) 인생 고점/저점 지도에 민서 맥락이 닿는 자연어 문장 보강, (4) clean commit 기준 release evidence 재생성, (5) 최종 human spot-check 완료 후 cutover_ready=YES 확정.

## Suggested First-Pass Copy

- 아래 문구는 reviewer가 그대로 복붙해 시작할 수 있는 초안입니다.
- 실제 사람 검토가 끝나기 전에는 `PASS`, `cutover_ready`, 최종 총평을 확정값으로 남기지 않습니다.

### Suggested Check 1 Note

- 첫 문장이 비교적 자연스럽고, 리포트의 진입 문장으로 읽을 만합니다. 다만 완전히 프리라이팅처럼 느껴지기보다는 약한 템플릿감은 남아 있어 최종 컷오버 전 한 번 더 문장 리듬 점검이 있으면 좋겠습니다.

### Suggested Check 2 Note

- 이름, 관심사, 직업 맥락이 `인생 구조 한 장 요약`, `현재 위치`, `다음 3년 구체화`에 반복적으로 드러나서 개인화 흔적은 분명합니다. 현재 수준에서는 "내 얘기" 감각이 기본선 이상으로 확보됩니다.

### Suggested Check 3 Note

- `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`가 각 1회씩만 등장하고 canonical H2 order도 유지됩니다. 중복 삽입이나 누락은 보이지 않습니다.

### Suggested Check 4 Note

- `How to use 1p`에서 제시한 읽는 법이 `다음 3년 구체화`, `valid_until 설명`, `CTA-lite`까지 비교적 자연스럽게 이어집니다. 읽고 난 뒤 무엇을 메모하거나 추적할지 행동선이 보이는 편입니다.

### Suggested Check 5 Note

- 외부 문면에서 `life_cycle-lite`, `life_cycle-full` 같은 내부 SKU 표현은 보이지 않고, 과한 영문 용어도 눈에 띄지 않습니다. 소비자용 문면으로는 비교적 안정적입니다.

### Suggested Check 6 Note

- 전체적으로는 "내 얘기 같다"와 "다음 행동이 떠오른다" 기준을 통과하는 쪽에 가깝습니다. 다만 고점/저점 일부 문장은 최종 컷오버 전 사람 기준으로 한 번 더 읽어 자연스러움을 확인하면 더 안전합니다.

### Suggested Final Summary

- target candidate로서는 충분히 설득력 있고, 구조/개인화/행동 연결의 균형도 안정적입니다. 다만 최종 cutover 선언은 clean commit 기준 증적 재생성과 human spot-check 완료 이후에만 하는 것이 맞습니다.

## Case 2: life_cycle_target_editorial_pack_summary

- input fixture / request summary: 20-case editorial pack (`life_cycle_target_editorial_review_20.md`) 재검토
- expected points:
  - sample 20건이 동일 rubric으로 기록됨
  - 각 case가 target gate PASS를 유지함
  - 전체 pass count와 reviewer note가 문서 상단에 요약됨
- actual summary:
  - reviewed_cases=20, pass_count=20, fail_count=0
  - rubric path: PRD/release_evidence/v1_4_0/life_cycle_target_editorial_rubric.md
  - case matrix path: PRD/release_evidence/v1_4_0/life_cycle_target_editorial_case_matrix.json
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): 2026-03-16T13:47:36+09:00
