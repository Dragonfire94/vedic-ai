# Life Cycle Target Editorial Rubric

- contract_version: v1.4.0
- release_evidence_dir: PRD/release_evidence/v1_4_0
- render_profile: life_cycle_target_v1
- reviewer_mode: Codex AI-assisted first pass
- generated_at_kst: 2026-03-12T14:43:28+09:00

이 문서는 `Vedic Life Cycle Report` target candidate를 20개 샘플로 읽을 때 같은 기준으로 점검하기 위한 루브릭입니다.
이 평가는 사람이 다시 읽을 때 기준을 맞추기 위한 첫 패스이며, 최종 컷오버 직전에는 사람 spot-check를 한 번 더 권장합니다.

## 축 정의

1. 술술 읽힘
   - 5점: 흐름이 끊기지 않고 H2 전개와 문장 리듬이 자연스럽다.
   - 4점: 전반적으로 잘 읽히며 일부 문장만 다듬으면 된다.
   - 3점 이하: 섹션 전환이나 문장 리듬이 자주 끊긴다.

2. 내 얘기처럼 읽힘
   - 5점: 이름/상황/관심사가 최소 2개 이상 섹션에서 자연스럽게 반영된다.
   - 4점: 개인화 흔적은 분명하지만 더 깊게 연결될 여지가 있다.
   - 3점 이하: 템플릿 느낌이 강하고 입력 맥락이 약하다.

3. 행동 연결
   - 5점: How to use, 다음 3년, valid_until, CTA가 하나의 행동선으로 연결된다.
   - 4점: 행동 라인은 분명하지만 섹션 간 연결감은 약간 더 다듬을 수 있다.
   - 3점 이하: 읽고 나서 바로 무엇을 할지 불명확하다.

4. 과한 jargon 억제
   - 5점: 소비자 언어 중심이고 내부 SKU/불필요 영문/전문용어 과잉이 없다.
   - 4점: 일부 용어가 남아도 독해를 크게 방해하지 않는다.
   - 3점 이하: 텍스트가 설명서처럼 느껴지거나 용어 부담이 크다.

## PASS 기준

- target gate `life_cycle_release_ok == true`
- 네 축 점수 모두 4점 이상
- exact H2 order 유지
- `life_cycle-lite` / `life_cycle-full` 외부 노출 0회
