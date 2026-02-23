# CODEX 지시서
## Bridge 누락 수정 — _is_evidence_para 감지 조건 확장

---

## 문제 원인

`apply_bridge_to_all_chapters()` 안의 `_ensure_bridge_after_evidence()` 함수가
Evidence 단락을 감지하지 못해 Bridge를 삽입하지 않는 문제입니다.

**왜 감지 못하는가?**

현재 감지 조건(`_is_evidence_para`)이 `시데리얼`이 포함된 텍스트를 기준으로 만들어졌습니다.
그런데 방향 B 작업으로 Evidence 텍스트에서 `시데리얼` 헤더를 제거했기 때문에
감지 조건이 False가 되어 `last_evidence_idx = -1` → Bridge 삽입 건너뜀.

**현재 Evidence 텍스트 시작 패턴 (방향 B 이후):**
```
강점으로는 ...
강점은 ...
리스크는 ...
조언은 ...
발현은 ...
이 배치는 ...
이 구성은 ...
이 집들에 ...
이 상태는 ...
이 패턴은 ...
```

**현재 감지 조건 (놓치고 있음):**
```python
def _is_evidence_para(p: str) -> bool:
    return (
        "시데리얼" in p          ← 방향 B 이후 없음
        or "핵심 근거" in p      ← 없음
        or bool(re.match(r"^(?:\s*배치|\s*상태|\s*구성|\s*패턴|\s*근거)", p))
        # "강점", "발현은", "이 배치" 등 패턴 없음 → 누락
    )
```

---

## 수정 위치

파일: `backend/llm_service.py`
함수: `_ensure_bridge_after_evidence()` (721번 라인)
수정 대상: 내부의 `_is_evidence_para()` 중첩 함수

---

## 수정 내용

**현재 코드 (731~736번 라인):**
```python
    def _is_evidence_para(p: str) -> bool:
        return (
            "시데리얼" in p
            or "핵심 근거" in p
            or bool(re.match(r"^(?:\s*배치|\s*상태|\s*구성|\s*패턴|\s*근거)", p))
        )
```

**변경 후:**
```python
    def _is_evidence_para(p: str) -> bool:
        return (
            "시데리얼" in p
            or "핵심 근거" in p
            or bool(re.match(r"^(?:\s*배치|\s*상태|\s*구성|\s*패턴|\s*근거)", p))
            or bool(re.match(
                r"^(?:강점|리스크|조언|발현은|이\s*배치|이\s*구성|이\s*집들|이\s*상태|이\s*패턴)",
                p
            ))
        )
```

**변경 핵심:** 마지막 `or` 조건 1줄 추가.
방향 B 이후 Evidence 텍스트의 실제 시작 패턴을 감지 조건에 추가.

---

## 수정 전후 동작 비교

### 수정 전
```
[Life Timeline Interpretation]
P1 N: Hook 문장
P2 → _is_evidence_para = False  ← "이 배치는..." 감지 못함
P3 → _is_evidence_para = False
P4 → _is_evidence_para = False
last_evidence_idx = -1  → Bridge 삽입 건너뜀
```

### 수정 후
```
[Life Timeline Interpretation]
P1 N: Hook 문장
P2 → _is_evidence_para = True  ← "이 배치는..." 감지 ✅
P3 → _is_evidence_para = True
P4 → _is_evidence_para = True  ← last_evidence_idx = 3
after_evidence = []  → has_bridge = False → Bridge 삽입 ✅
"이 흐름을 실생활에 연결하면 다음과 같은 방향이 도움이 됩니다."
```

**Bridge가 삽입되는 챕터 (3개):**
- `Life Timeline Interpretation` → `"이 흐름을 실생활에 연결하면 다음과 같은 방향이 도움이 됩니다."`
- `Stability Metrics` → `"이 구조를 바탕으로 지금 단계에서 취할 수 있는 방향은 다음과 같습니다."`
- `Confidence & Forecast` → `"이 흐름을 자기확신으로 연결하려면 아래 방향을 참고하세요."`

---

## 수정 후 검증 방법

```bash
LLM_HYBRID_RENDER_MODE=on \
LLM_EVIDENCE_MODE=on \
LLM_EVIDENCE_PRIORITY=evidence_only \
python scripts/cheap_validation_gate.py --profile most_balanced --allow-api 1 --force-truepath --timeout-seconds 180
```

**체크 포인트:**
1. `Life Timeline`, `Stability Metrics`, `Confidence & Forecast` 챕터에 Bridge 문장 존재 확인
2. Bridge가 Evidence 단락 뒤, Bullets 앞에 위치하는지 확인
3. 챕터당 평균 분량 **550자 이상** 확인 (Bridge 3개 추가로 ~30~60자씩 증가)
4. 기존 Bridge 있는 챕터 7개는 영향 없는지 확인 (`has_bridge=True`면 삽입 안 함)

---

## 주의사항

- 이 수정은 `_ensure_bridge_after_evidence()` 내부 중첩 함수만 수정합니다.
- 외부의 `apply_bridge_to_all_chapters()`나 `_BRIDGE_FALLBACK_BY_CHAPTER`는 건드리지 않습니다.
- 기존에 Bridge가 있는 챕터(Career, Love, Karmic, Health, Psychological 등)는
  `has_bridge=True`로 판정되어 fallback 삽입이 발생하지 않습니다. 영향 없습니다.
