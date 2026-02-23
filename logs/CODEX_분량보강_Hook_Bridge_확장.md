# CODEX 지시서
## 분량 부족 해결 — Hook/Bridge 확장

---

## 문제 진단 (읽고 시작할 것)

현재 챕터당 평균 분량이 **393자**입니다. 목표는 **700~1100자**입니다.

**왜 부족한가?**

방향 B 작업으로 Evidence 텍스트를 267자 → 165자로 줄였습니다.
그런데 프롬프트가 Hook/Bridge를 각각 "1문장"으로 강제하고 있어서
LLM이 Hook 32자, Bridge 17자짜리 짧은 문장만 씁니다.

결과:
```
Hook(32자) + Evidence(165자) + Bridge(17자) + Bullets(100자) = 314자  ← 목표 700자의 절반
```

**해결 방향:**

Hook과 Bridge의 글자 수 하한선을 명시해서 LLM이 충분히 쓰도록 유도합니다.
- Hook: 1문장 → **2~3문장 (80~150자)**
- Bridge: 1문장 → **2~3문장 (100~200자)**

이렇게 하면:
```
Hook(120자) + Evidence(165자) + Bridge(150자) + Bullets(100자) = 535자
```
목표 700자에 근접하고, 실무적으로 허용 가능한 수준입니다.

---

## 수정 위치

파일: `backend/llm_service.py`

수정 구간 2곳:

### 수정 1: `hybrid_output_contract` 블록 (약 2957번 라인)

**현재:**
```python
        hybrid_output_contract = f"""
HYBRID RENDER OUTPUT CONTRACT
- Actionable 챕터({", ".join(sorted(_ACTIONABLE_CHAPTER_KEYS))}) 출력 순서:
  Hook: 1문장
  <EVIDENCE_BLOCK>
  Bridge: 1문장
  - 불릿1
  - 불릿2
  - 불릿3
- 불릿 면제 챕터(Executive Summary, Life Timeline Interpretation, Final Summary) 출력 순서:
  Hook: 1문장
  <EVIDENCE_BLOCK>
  Bridge: 1문장
  Bullets 금지.
- <EVIDENCE_BLOCK> 태그는 챕터당 정확히 1회만 출력한다.
- Evidence 텍스트를 재작성/재인용/복붙하지 않는다. 태그만 출력한다.
- [근거], --- 같은 라벨/구분선 삽입 금지.
- 체크리스트식 본문 전개 금지 (마지막 action bullets 3개는 허용).
"""
```

**변경 후:**
```python
        hybrid_output_contract = f"""
HYBRID RENDER OUTPUT CONTRACT
- Actionable 챕터({", ".join(sorted(_ACTIONABLE_CHAPTER_KEYS))}) 출력 순서:
  Hook: 2~3문장 (80~150자 목표)
  <EVIDENCE_BLOCK>
  Bridge: 2~3문장 (100~200자 목표)
  - 불릿1
  - 불릿2
  - 불릿3
- 불릿 면제 챕터(Executive Summary, Life Timeline Interpretation, Final Summary) 출력 순서:
  Hook: 2~3문장 (80~150자 목표)
  <EVIDENCE_BLOCK>
  Bridge: 2~3문장 (100~200자 목표)
  Bullets 금지.
- <EVIDENCE_BLOCK> 태그는 챕터당 정확히 1회만 출력한다.
- Evidence 텍스트를 재작성/재인용/복붙하지 않는다. 태그만 출력한다.
- [근거], --- 같은 라벨/구분선 삽입 금지.
- 체크리스트식 본문 전개 금지 (마지막 action bullets 3개는 허용).
"""
```

**변경 핵심:** `1문장` → `2~3문장 (80~150자 목표)` / `2~3문장 (100~200자 목표)`

---

### 수정 2: `sales_tone_contract` 블록 (약 2973번 라인)

**현재:**
```python
        sales_tone_contract = """
[판매형 톤 계약]
- Hook은 독자가 바로 공감할 질문/감정 진술 1문장으로 시작한다.
- Bridge는 이론 설명 대신 "그래서 당신에게 어떤 의미인지" 1문장으로 연결한다.
- Hook/Bridge/Bullets에서는 점성학 전문용어를 직접 노출하지 않는다.
"""
```

**변경 후:**
```python
        sales_tone_contract = """
[판매형 톤 계약]
- Hook은 독자가 바로 공감할 질문/감정 진술로 시작하되, 2~3문장(80~150자)으로 쓴다.
  첫 문장: 공감 질문 또는 감정 진술
  이어지는 문장: 그 감정이 왜 생기는지 상황을 1~2문장으로 풀어준다.
- Bridge는 Evidence를 독자의 삶에 연결하는 2~3문장(100~200자)으로 쓴다.
  "그래서 당신에게 어떤 의미인지" 한 줄로 끝내지 않는다.
  Evidence에서 가장 중요한 포인트 1개를 골라 지금 당신의 상황에 바로 적용한다.
- Hook/Bridge/Bullets에서는 점성학 전문용어를 직접 노출하지 않는다.
"""
```

---

## 변경 전후 예시 (기대 출력)

### 변경 전 (현재)
```
## [Career & Success] 경력과 성공

책임은 늘었는데 인정은 부족한 느낌이 들지 않나요?

[Evidence 단락들 ~165자]

그래서 당신에게 어떤 의미인지.

- 불릿1
- 불릿2
- 불릿3
```

### 변경 후 (목표)
```
## [Career & Success] 경력과 성공

책임은 늘었는데 인정은 부족한 느낌이 들지 않나요? 열심히 하고 있는데
왜 이렇게 피곤한지 모르겠는 상태, 그게 지금 당신의 직업적 흐름입니다.
노력은 방향을 잡았는데 아직 결과가 따라오지 않는 구간입니다.

[Evidence 단락들 ~165자]

이 패턴이 당신에게 의미하는 건 지금 당장 크게 바꾸지 않아도 된다는 겁니다.
리더십과 책임감이 올라가는 시기이지만, 에너지 관리 없이 밀어붙이면
단기적으로 번아웃이 먼저 옵니다. 지금은 방향보다 속도 조절이 핵심입니다.

- 핵심 목표 3개만 이번 주에 고정한다
- 과로 신호 오면 권한 위임을 먼저 쓴다
- 이 에너지, 지금 어디에 쓰고 있나요? (질문형)
```

---

## 수정 후 검증 방법

```bash
LLM_HYBRID_RENDER_MODE=on \
LLM_EVIDENCE_MODE=on \
LLM_EVIDENCE_PRIORITY=evidence_only \
python scripts/cheap_validation_gate.py --profile most_balanced --allow-api 1 --force-truepath --timeout-seconds 180
```

**체크 포인트:**
1. 챕터당 평균 글자수(공백 제외) **500자 이상** 확인
2. Hook 첫 문장이 여전히 공감형인지 확인 (질문형 또는 감정 진술)
3. Bridge가 `그래서 당신에게 어떤 의미인지.` 한 줄로 끝나지 않는지 확인
4. 전문용어 Hook/Bridge에서 **0건** 유지 확인
5. `below_min_chars` 경고 감소 여부 확인

---

## 주의사항

- Hook이 길어져도 **공감형 톤은 유지**해야 합니다. 설명문이 되면 안 됩니다.
- Bridge가 길어져도 **Evidence 내용을 복붙/재정의하면 안 됩니다.**
  (기존 설명 모드 금지 규칙 유지)
- 불릿 40자 제한은 그대로 유지합니다. 불릿을 늘려서 분량을 채우지 않습니다.
