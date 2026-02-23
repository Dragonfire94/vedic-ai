# CODEX 지시서 — Evidence Pipeline 연결 + Bridge 자동 삽입 + Cross-chapter 중복 방지
## 최종 통합본 (V2 + 패치 1~2차 전부 반영)

**수정 대상 파일:** `backend/astro_engine.py`, `backend/llm_service.py`

---

## 배경

`build_evidence_packs(structured_summary, ...)` 함수는 `structured_summary`에서 아래 4개 키를 읽어 evidence item을 조립한다.

```python
source.get("detected_yogas")
source.get("pattern_flags")
source.get("lagna_lord_state")
source.get("lagna_lord_placement_group")
```

그런데 `build_structural_summary()`의 return 딕셔너리에 이 4개 키가 존재하지 않아 evidence pack의 yoga/pattern/lagna_lord 슬롯이 전부 빈 채로 돌아갔다.

추가로, evidence가 inject된 후 Bridge 문장 없이 `Hook → E → E → E → Bullets` 구조가 되는 문제와, 같은 evidence 태그가 10챕터 전부에 반복되는 문제가 확인됐다.

---

## 수정 1: `astro_engine.py` — 헬퍼 함수 추가

`build_structural_summary` 함수 **바로 위**에 추가한다.

```python
def _lagna_lord_placement_group(planets: dict[str, Any], lagna_lord: str) -> str | None:
    """Return placement group string for the lagna lord's house position.
    Priority: kendra > trikona > dusthana > upachaya > succedent.
    House 1 is both kendra and trikona; kendra takes precedence.
    """
    house = _planet_house(planets, lagna_lord)
    if house is None:
        return None
    if house in {1, 4, 7, 10}:
        return "kendra"
    if house in {5, 9}:
        return "trikona"
    if house in {6, 8, 12}:
        return "dusthana"
    if house in {3, 11}:
        return "upachaya"
    return "succedent"
```

---

## 수정 2: `astro_engine.py` — `build_structural_summary` return 블록에 4개 키 추가

`return {` 블록 안, 기존 키들 뒤에 아래를 추가한다.

```python
# ── evidence pipeline bridge keys ──────────────────────────────────
# detected_yogas: rule_key 리스트. cancelled는 detect_yogas가 이미 제외.
# active + weakened 모두 포함 (상태 필터 없이 전부).
"detected_yogas": [
    y["rule_key"]
    for y in yogas
    if isinstance(y, dict) and isinstance(y.get("rule_key"), str)
],

# pattern_flags: karmic_pattern_profile에서 primary_pattern 메타키 제외,
# 숫자형(int|float) score > 0인 항목의 key 리스트.
"pattern_flags": [
    k
    for k, v in karmic_profile.items()
    if k != "primary_pattern" and isinstance(v, (int, float)) and float(v) > 0
],

# lagna_lord_state: lagna_lord 행성의 avastha_state 문자열 (없으면 None).
"lagna_lord_state": (
    shadbala_summary.get("by_planet", {}).get(lagna_lord or "", {}).get("avastha_state")
    if lagna_lord else None
),

# lagna_lord_placement_group: lagna_lord 하우스 위치의 그룹 분류 문자열.
"lagna_lord_placement_group": (
    _lagna_lord_placement_group(planets, lagna_lord)
    if lagna_lord else None
),
```

---

## 수정 3: `llm_service.py` — 매핑 테이블 3개 추가

`_norm_yoga_id` 함수 **바로 아래**에 추가한다.

### 3-1. Yoga 매핑

엔진 `rule_key` → interpretations 키 정확한 대응 (코드 검증 완료):

```python
_ENGINE_YOGA_KEY_TO_INTERP: dict[str, str] = {
    "raja_yoga":           "yoga:RajayogaGeneral",
    "dhana_yoga":          "yoga:DhanaYogaGeneral",
    "parivartana_yoga":    "yoga:ParivartanaGeneral",
    "vipareeta_raja_yoga": "yoga:VipareetaLite",
    "gaja_kesari_yoga":    "yoga:Gajakesari",
    "neecha_bhanga":       "yoga:NeechaBhangaLite",
    "kemadruma":           "yoga:KemadrumaLite",
    # daridra_yoga: interpretations에 대응 키 없음 → 매핑 제외
}
```

### 3-2. Pattern 매핑

`financial_leak_pattern`과 `delayed_success_pattern`의 many-to-one 중복을 회피한 매핑:

```python
_ENGINE_PATTERN_KEY_TO_INTERP: dict[str, str] = {
    "authority_conflict_pattern":       "pat:malefic_overload",
    "delayed_success_pattern":          "pat:dusthana_focus",
    "financial_leak_pattern":           "pat:scattered_energy",   # dusthana_focus 중복 회피
    "obsession_public_image_pattern":   "pat:kendra_emphasis",
    "relationship_abandonment_pattern": "pat:scattered_energy",
}
```

### 3-3. Avastha → LL State 매핑

엔진 avastha state 6개와 interpretations ll:state 키 (코드 검증 완료):

```python
_ENGINE_AVASTHA_TO_LL_STATE: dict[str, str] = {
    "asta":   "ll:state:combust",      # combust
    "deepta": "ll:state:exalted",      # exalted
    "chesta": "ll:state:retrograde",   # retrograde + strong
    "yuva":   "ll:state:own",          # strong expression
    "dina":   "ll:state:debilitated",  # weak
    "madhya": "ll:state:neutral",      # balanced/neutral
}
```

### 3-4. `_norm_yoga_id`, `_norm_pat_id` 주석 추가 (삭제 금지)

두 함수 상단에 주석 1줄 추가:

```python
def _norm_yoga_id(value: str) -> str:
    # NOTE: build_evidence_packs는 _ENGINE_YOGA_KEY_TO_INTERP 매핑을 우선 사용.
    # 매핑 외부에서 직접 yoga id를 조립할 때를 위해 유지.
    ...

def _norm_pat_id(value: str) -> str:
    # NOTE: build_evidence_packs는 _ENGINE_PATTERN_KEY_TO_INTERP 매핑을 우선 사용.
    # 매핑 외부에서 직접 pattern id를 조립할 때를 위해 유지.
    ...
```

---

## 수정 4: `llm_service.py` — `build_evidence_packs` 내 id 조립 로직 교체

### 4-1. `detected_yoga_ids` 교체

**현재:**
```python
detected_yoga_ids = [_norm_yoga_id(v) for v in (source.get("detected_yogas") or []) if isinstance(v, str)]
```

**교체 후:**
```python
detected_yoga_ids = []
for v in (source.get("detected_yogas") or []):
    if not isinstance(v, str):
        continue
    mapped = _ENGINE_YOGA_KEY_TO_INTERP.get(v.strip().lower())
    if mapped:
        detected_yoga_ids.append(mapped)
    # 매핑 없는 키(daridra_yoga 등)는 스킵
```

### 4-2. `detected_pattern_ids` 교체

**현재:**
```python
detected_pattern_ids = [_norm_pat_id(v) for v in (source.get("pattern_flags") or []) if isinstance(v, str)]
```

**교체 후:**
```python
detected_pattern_ids = []
for v in (source.get("pattern_flags") or []):
    if not isinstance(v, str):
        continue
    mapped = _ENGINE_PATTERN_KEY_TO_INTERP.get(v.strip().lower())
    if mapped:
        detected_pattern_ids.append(mapped)
    # 매핑 없는 패턴은 스킵
```

### 4-3. `ll_ids` 조립 교체

**현재:**
```python
ll_state = str(source.get("lagna_lord_state") or "").strip()
ll_place = str(source.get("lagna_lord_placement_group") or "").strip()
ll_ids = [f"ll:state:{ll_state}" if ll_state else "", f"ll:placement:{ll_place}" if ll_place else ""]
ll_ids = [v for v in ll_ids if v]
```

**교체 후:**
```python
ll_state_raw = str(source.get("lagna_lord_state") or "").strip().lower()
ll_place_raw = str(source.get("lagna_lord_placement_group") or "").strip().lower()

ll_state_key = _ENGINE_AVASTHA_TO_LL_STATE.get(ll_state_raw, "")
ll_place_key = f"ll:placement:{ll_place_raw}" if ll_place_raw else ""

ll_ids = [k for k in [ll_state_key, ll_place_key] if k]
```

---

## 수정 5: `llm_service.py` — `build_evidence_packs` cross-chapter 중복 방지

`normal_chapters` 루프가 끝나고 `Final Summary` 처리 직전에 추가한다.

```python
# ── cross-chapter evidence 중복 방지 ──────────────────────────────────────
# 같은 ID가 여러 챕터에 할당된 경우, 앞 챕터가 유지하고 뒤 챕터에서 제거 후 fallback 보충.
# dedup 후 fallback_used_by_chapter는 조건문 밖에서 항상 재계산해 덮어쓴다.
_cross_seen: set[str] = set()
for chapter in normal_chapters:
    items = chapter_evidence_raw.get(chapter, [])
    deduped = []
    for item in items:
        iid = str(item.get("id", "")).strip()
        if not iid or iid not in _cross_seen:
            deduped.append(item)
            if iid:
                _cross_seen.add(iid)

    if len(deduped) < len(items):
        # 1차: 챕터 전용 fallback
        fallback_keys = _EVIDENCE_FALLBACK_PATTERNS.get(chapter, [])
        for pid in fallback_keys:
            if len(deduped) >= chapter_items_max:
                break
            seen_local = {str(x.get("id", "")) for x in deduped}
            if pid in _cross_seen or pid in seen_local:
                continue
            txt = patterns.get(pid)
            if not txt:
                continue
            t = _evidence_text_trim(txt, 360)
            if t:
                deduped.append({"id": pid, "text": t, "_kind": "fallback", "_score": 20, "_chapter": chapter})
                _cross_seen.add(pid)

        # 2차: 챕터 전용 fallback으로도 chapter_items_min 미달 시 전역 풀에서 보충
        # Life Timeline Interpretation, Confidence & Forecast 등 커버가 약한 챕터 대응.
        _GLOBAL_FALLBACK_POOL = [
            "pat:kendra_emphasis",
            "pat:trikona_emphasis",
            "pat:upachaya_emphasis",
            "pat:benefic_support",
            "pat:scattered_energy",
            "pat:strong_lagna_lord",
        ]
        if len(deduped) < max(1, chapter_items_min):
            for pid in _GLOBAL_FALLBACK_POOL:
                if len(deduped) >= max(1, chapter_items_min):
                    break
                seen_local = {str(x.get("id", "")) for x in deduped}
                if pid in _cross_seen or pid in seen_local:
                    continue
                txt = patterns.get(pid)
                if not txt:
                    continue
                t = _evidence_text_trim(txt, 360)
                if t:
                    deduped.append({"id": pid, "text": t, "_kind": "fallback", "_score": 10, "_chapter": chapter})
                    _cross_seen.add(pid)

        chapter_evidence_raw[chapter] = deduped

    # dedup 여부와 무관하게 항상 재계산 (조건문 밖).
    # "중복 제거 없이 기존 fallback이 있던 챕터"도 정확히 반영됨.
    fallback_used_by_chapter[chapter] = any(
        str(x.get("_kind", "")) == "fallback"
        for x in chapter_evidence_raw.get(chapter, [])
    )
```

---

## 수정 6: `llm_service.py` — Bridge 자동 삽입

### 6-1. 헬퍼 함수 추가

`inject_evidence_blocks` 함수 **바로 위**에 추가한다.

```python
_BRIDGE_FALLBACK_BY_CHAPTER: dict[str, str] = {
    "Career & Success":           "이 흐름을 실제 선택에 연결하면 다음과 같은 방향이 나옵니다.",
    "Stability Metrics":          "이 구조를 바탕으로 지금 단계에서 취할 수 있는 방향은 다음과 같습니다.",
    "Love & Relationships":       "이 패턴을 관계 안에서 구체적으로 다루려면 아래를 참고하세요.",
    "Karmic Patterns":            "이 반복을 알아차리는 것이 출발점이고, 실천은 여기서 시작됩니다.",
    "Health & Body Patterns":     "이 리듬을 몸에서 실제로 관리하려면 다음 방향이 유효합니다.",
    "Confidence & Forecast":      "이 흐름을 자기확신으로 연결하려면 아래 방향을 참고하세요.",
    "Psychological Architecture": "이 내적 구조를 일상에서 다루는 실천 방향은 다음과 같습니다.",
}
_BRIDGE_FALLBACK_DEFAULT = "이 흐름을 실생활에 연결하면 다음과 같은 방향이 도움이 됩니다."


def _ensure_bridge_after_evidence(body: str, chapter_key: str) -> str:
    """
    evidence 단락(시데리얼 또는 (tag:...) 로 시작) 이후에
    서사 문장(Bridge)이 없으면 중립 Bridge 1문장을 삽입한다.
    """
    body_normalized = body.replace("\r\n", "\n")
    paras = [p.strip() for p in body_normalized.split("\n\n") if p.strip()]
    if not paras:
        return body

    def _is_evidence_para(p: str) -> bool:
        return (
            "시데리얼" in p
            or bool(re.match(r"^\((yoga:|pat:|ll:|asc:|ps:)", p))
        )

    def _is_bullet_para(p: str) -> bool:
        # 시스템 전역 _BULLET_LINE_RE 재사용 — 숫자형 불릿(1. ...)도 정확히 판별
        first_line = p.split("\n")[0]
        return bool(_BULLET_LINE_RE.match(first_line))

    # 마지막 evidence 단락 위치 찾기
    last_evidence_idx = -1
    for i, p in enumerate(paras):
        if _is_evidence_para(p):
            last_evidence_idx = i

    if last_evidence_idx < 0:
        return body  # evidence 없음 → 그대로

    # evidence 이후에 서사 문장(non-evidence, non-bullet)이 있는지 확인
    after_evidence = paras[last_evidence_idx + 1:]
    has_bridge = any(
        not _is_evidence_para(p) and not _is_bullet_para(p)
        for p in after_evidence
    )

    if has_bridge:
        return body  # Bridge 이미 있음 → 그대로

    # Bridge 없음 → 마지막 evidence 단락 바로 뒤에 삽입
    bridge = _BRIDGE_FALLBACK_BY_CHAPTER.get(chapter_key, _BRIDGE_FALLBACK_DEFAULT)
    paras.insert(last_evidence_idx + 1, bridge)
    return "\n\n".join(paras)
```

### 6-2. 트리거 추가

`inject_evidence_blocks` 함수 안에서 `new_body`를 최종 확정한 직후,
`output_lines.extend(new_body.split("\n"))` **바로 앞**에 추가한다.

```python
# Bridge 자동 삽입.
# no_evidence_empty_replace(evidence 자체 없어서 태그만 제거한 챕터)만 제외하고 전부 검사.
# missing_tag_fallback_used 챕터도 포함해 구조 일관성을 높인다.
if (
    evidence_text
    and stats["evidence_items_count"] > 0
    and not stats["no_evidence_empty_replace"]
):
    new_body = _ensure_bridge_after_evidence(new_body, chapter_key)
```

---

## 검증 체크리스트

1. `python -m py_compile backend/astro_engine.py`
2. `python -m py_compile backend/llm_service.py`
3. 서버에 `LLM_HYBRID_RENDER_MODE=on`, `LLM_EVIDENCE_MODE=on` 설정 확인
4. 샘플 차트 재생성 후 확인:
   - `structured_summary`에 4개 키 존재: `detected_yogas`, `pattern_flags`, `lagna_lord_state`, `lagna_lord_placement_group`
   - `EVIDENCE_BLOCK` 잔존 0개
   - `WARN no_evidence_for_chapter` 개수 이전 대비 **증가 없음**
   - `pat:malefic_overload` 등 동일 태그가 전체 챕터 중 **최대 2~3챕터**에만 등장
   - `Life Timeline Interpretation`, `Confidence & Forecast` 챕터 `evidence_items_count >= 1`
   - 로그 `fallback_used` 값이 실제 텍스트와 일치
   - Actionable 챕터 구조: `Hook → E → Bridge → Bullets`
   - Exempt 챕터 구조: `Hook → E → Bridge`
   - Bridge가 evidence 텍스트 안에 포함되지 않고 **별도 단락**으로 존재

---

## 수정 범위 요약

| 파일 | 수정 내용 | 위험도 |
|------|-----------|--------|
| `astro_engine.py` | `_lagna_lord_placement_group` 헬퍼 추가 | 낮음 |
| `astro_engine.py` | `build_structural_summary` return에 4개 키 추가 | 낮음 |
| `llm_service.py` | yoga/pattern/avastha 매핑 테이블 3개 추가 | 낮음 |
| `llm_service.py` | `detected_yoga_ids`, `detected_pattern_ids`, `ll_ids` 조립 로직 교체 | 낮음 |
| `llm_service.py` | `build_evidence_packs` cross-chapter 중복 방지 + fallback 보충 + `fallback_used_by_chapter` 재계산 | 낮음 |
| `llm_service.py` | `_ensure_bridge_after_evidence` 함수 추가 + 트리거 삽입 | 낮음 |
