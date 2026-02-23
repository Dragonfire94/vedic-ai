# CODEX 지시서 — 최종 완성본
## Evidence Pipeline + Bridge + 품질 수정 전체 통합

**수정 대상 파일:** `backend/astro_engine.py`, `backend/llm_service.py`

---

## [astro_engine.py] 수정 1: 헬퍼 함수 추가

`build_structural_summary` 함수 **바로 위**에 추가.

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

## [astro_engine.py] 수정 2: `build_structural_summary` return 블록에 4개 키 추가

`return {` 블록 안, 기존 키들 뒤에 추가.

```python
# ── evidence pipeline bridge keys ──────────────────────────────────
"detected_yogas": [
    y["rule_key"]
    for y in yogas
    if isinstance(y, dict) and isinstance(y.get("rule_key"), str)
],
"pattern_flags": [
    k
    for k, v in karmic_profile.items()
    if k != "primary_pattern" and isinstance(v, (int, float)) and float(v) > 0
],
"lagna_lord_state": (
    shadbala_summary.get("by_planet", {}).get(lagna_lord or "", {}).get("avastha_state")
    if lagna_lord else None
),
"lagna_lord_placement_group": (
    _lagna_lord_placement_group(planets, lagna_lord)
    if lagna_lord else None
),
```

---

## [llm_service.py] 수정 3: 매핑 테이블 3개 + 주석 추가

`_norm_yoga_id` 함수 **바로 아래**에 추가.

### 3-1. Yoga 매핑
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
```python
_ENGINE_AVASTHA_TO_LL_STATE: dict[str, str] = {
    "asta":   "ll:state:combust",
    "deepta": "ll:state:exalted",
    "chesta": "ll:state:retrograde",
    "yuva":   "ll:state:own",
    "dina":   "ll:state:debilitated",
    "madhya": "ll:state:neutral",
}
```

### 3-4. `_norm_yoga_id`, `_norm_pat_id` 주석 추가 (삭제 금지)
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

## [llm_service.py] 수정 4: `build_evidence_packs` 내 id 조립 로직 교체

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

## [llm_service.py] 수정 5: `_EVIDENCE_FALLBACK_PATTERNS` 누락 챕터 추가

기존 딕셔너리에 아래 2개를 추가.

```python
"Life Timeline Interpretation": ["pat:strong_lagna_lord", "pat:kendra_emphasis", "pat:trikona_emphasis"],
"Confidence & Forecast":        ["pat:trikona_emphasis", "pat:upachaya_emphasis", "pat:benefic_support"],
```

---

## [llm_service.py] 수정 6: Cross-chapter dedup + Fallback 최대 2회 제한

`build_evidence_packs` 함수 안, `chapter_evidence_raw`, `fallback_used_by_chapter` 선언 **바로 아래**에 추가:

```python
_fallback_use_count: dict[str, int] = {}  # fallback id → 전체 사용 횟수
```

### 6-1. 초기 조립 루프의 fallback 제한

`for chapter in normal_chapters:` 안, `_EVIDENCE_FALLBACK_PATTERNS` 루프를 교체:

**현재:**
```python
for pid in _EVIDENCE_FALLBACK_PATTERNS.get(chapter, []):
    if len(items) >= chapter_items_max:
        break
    if _try_add(items, seen, chapter, pid, patterns, "fallback", 20, 360):
        used_fallback = True
```
**교체 후:**
```python
for pid in _EVIDENCE_FALLBACK_PATTERNS.get(chapter, []):
    if len(items) >= chapter_items_max:
        break
    if _fallback_use_count.get(pid, 0) >= 2:
        continue
    if _try_add(items, seen, chapter, pid, patterns, "fallback", 20, 360):
        used_fallback = True
        _fallback_use_count[pid] = _fallback_use_count.get(pid, 0) + 1
```

### 6-2. Cross-chapter dedup 루프 (`normal_chapters` 루프 끝 직후)

```python
_cross_seen: set[str] = set()
for chapter in normal_chapters:
    items = chapter_evidence_raw.get(chapter, [])
    deduped: list[dict[str, Any]] = []
    for item in items:
        iid = str(item.get("id", "")).strip()
        item_kind = str(item.get("_kind", ""))
        is_engine_signal = item_kind in ("yoga", "pattern", "lagna_lord")
        if not iid or not is_engine_signal or iid not in _cross_seen:
            deduped.append(item)
            if iid and is_engine_signal:
                _cross_seen.add(iid)

    if len(deduped) < len(items):
        # 1차: 챕터 전용 fallback
        fallback_keys = _EVIDENCE_FALLBACK_PATTERNS.get(chapter, [])
        for pid in fallback_keys:
            if len(deduped) >= chapter_items_max:
                break
            if _fallback_use_count.get(pid, 0) >= 2:
                continue
            seen_local = {str(x.get("id", "")) for x in deduped}
            if pid in seen_local:
                continue
            txt = patterns.get(pid)
            if not txt:
                continue
            t = _evidence_text_trim(txt, 360)
            if t:
                deduped.append({"id": pid, "text": t, "_kind": "fallback", "_score": 20, "_chapter": chapter})
                _fallback_use_count[pid] = _fallback_use_count.get(pid, 0) + 1

        # 2차: 전역 fallback 풀
        _GLOBAL_FALLBACK_POOL = [
            "pat:kendra_emphasis",
            "pat:trikona_emphasis",
            "pat:upachaya_emphasis",
            "pat:benefic_support",
            "pat:scattered_energy",
            "pat:strong_lagna_lord",
            "pat:strong_moon",
            "pat:strong_10th_lord",
            "pat:multi_exalted",
        ]
        if len(deduped) < max(1, chapter_items_min):
            for pid in _GLOBAL_FALLBACK_POOL:
                if len(deduped) >= max(1, chapter_items_min):
                    break
                if _fallback_use_count.get(pid, 0) >= 2:
                    continue
                seen_local = {str(x.get("id", "")) for x in deduped}
                if pid in seen_local:
                    continue
                txt = patterns.get(pid)
                if not txt:
                    continue
                t = _evidence_text_trim(txt, 360)
                if t:
                    deduped.append({"id": pid, "text": t, "_kind": "fallback", "_score": 10, "_chapter": chapter})
                    _fallback_use_count[pid] = _fallback_use_count.get(pid, 0) + 1

        chapter_evidence_raw[chapter] = deduped

    # 조건문 밖에서 항상 재계산
    fallback_used_by_chapter[chapter] = any(
        str(x.get("_kind", "")) == "fallback"
        for x in chapter_evidence_raw.get(chapter, [])
    )
```

---

## [llm_service.py] 수정 7: Evidence id 라벨 제거

evidence_text 문자열이 확정되는 지점 바로 다음에 추가:

```python
# id 라벨 프리픽스 제거: (yoga:xxx), (pat:xxx), (ll:xxx) 등 줄 앞 태그 제거
evidence_text = re.sub(r"^\s*\([a-zA-Z_]+:[^\)]+\)\s*", "", evidence_text, flags=re.MULTILINE)
```

---

## [llm_service.py] 수정 8: `_evidence_text_trim` 교체 — 문장 경계 트런케이션

```python
def _evidence_text_trim(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text.strip()
    candidate = text[:max_chars]
    last_end = -1
    for m in re.finditer(r"[.!?。](?:\s|$)", candidate):
        last_end = m.end()
    if last_end > max_chars * 0.5:
        return candidate[:last_end].strip()
    return candidate.rstrip(".,;: ").strip()
```

---

## [llm_service.py] 수정 9: Bridge 삽입 위치 안정화

### 9-1. `inject_evidence_blocks` 안의 Bridge 트리거 삭제

아래 코드를 **삭제**:

```python
# 삭제 대상 (llm_service.py:899~904 근처)
if (
    evidence_text
    and stats["evidence_items_count"] > 0
    and not stats["no_evidence_empty_replace"]
):
    new_body = _ensure_bridge_after_evidence(new_body, chapter_key)
```

### 9-2. `_ensure_bridge_after_evidence` 내 `_is_evidence_para` 교체

**현재:**
```python
def _is_evidence_para(p: str) -> bool:
    return (
        "시데리얼" in p
        or bool(re.match(r"^\((yoga:|pat:|ll:|asc:|ps:)", p))
    )
```
**교체 후:**
```python
def _is_evidence_para(p: str) -> bool:
    # id 라벨은 inject 시점에 제거되므로 한국어 키워드 패턴으로 판별
    return (
        "시데리얼" in p
        or "라히리 기준" in p
        or bool(re.match(r"^(이\s*배치|이\s*상태|이\s*구성|이\s*패턴|이\s*요가)", p))
    )
```

### 9-3. `apply_bridge_to_all_chapters` 함수 추가

`_ensure_bridge_after_evidence` 함수 **바로 아래**에 추가:

```python
def apply_bridge_to_all_chapters(text: str) -> str:
    """
    normalize_llm_layout_strict 이후에 호출.
    전체 텍스트에서 챕터별로 Bridge 누락을 검사하고 삽입한다.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    heading_positions = _extract_heading_positions(normalized)
    if not heading_positions:
        return text

    output_lines: list[str] = []
    current_idx = 0

    for sec_idx, (start, raw_heading_text) in enumerate(heading_positions):
        end = (
            heading_positions[sec_idx + 1][0]
            if sec_idx + 1 < len(heading_positions)
            else len(lines)
        )
        if current_idx < start:
            output_lines.extend(lines[current_idx:start])
        output_lines.append(lines[start])

        body = "\n".join(lines[start + 1: end])

        # 기존 코드베이스와 동일한 bracket token 추출 경로 재사용
        _m = re.match(r"^\[(.*?)\]\s*(.*)$", raw_heading_text.strip())
        token = _m.group(1).strip() if _m else raw_heading_text.strip()
        chapter_key = _normalize_chapter_key(token, raw_heading_text, sec_idx)

        new_body = _ensure_bridge_after_evidence(body, chapter_key)
        output_lines.extend(new_body.split("\n"))
        current_idx = end

    if current_idx < len(lines):
        output_lines.extend(lines[current_idx:])

    return "\n".join(output_lines)
```

### 9-4. 호출 위치 — `normalize_llm_layout_strict` 직후, `_chapter_nonspace_lengths` 직전

```python
response_text = normalize_llm_layout_strict(response_text)
# Bridge 삽입: normalize 이후, 길이 측정 이전.
# Bridge 글자 수가 최종 길이 계산에 포함되어 억울한 미달 경고 방지.
if hybrid_render_mode:
    response_text = apply_bridge_to_all_chapters(response_text)
min_chars = _resolve_min_chars_by_phase()
length_map = _chapter_nonspace_lengths(response_text)  # 기존 코드 그대로
```

---

## 검증 체크리스트

1. `python -m py_compile backend/astro_engine.py`
2. `python -m py_compile backend/llm_service.py`
3. 서버 `LLM_HYBRID_RENDER_MODE=on`, `LLM_EVIDENCE_MODE=on` 설정 확인
4. 샘플 재생성 후 확인:
   - `structured_summary`에 4개 키 존재: `detected_yogas`, `pattern_flags`, `lagna_lord_state`, `lagna_lord_placement_group`
   - `EVIDENCE_BLOCK` 잔존 **0개**
   - `WARN no_evidence_for_chapter` 이전 대비 **유지 또는 감소**
   - evidence 텍스트에 `(pat:xxx)` 라벨 **노출 없음**
   - evidence 텍스트가 `...` 로 끝나지 않고 **완전한 문장**으로 끝남
   - **전체 10챕터** `N→E→Bridge→B` 구조 일관성 확인
   - 동일 fallback 항목이 **최대 2챕터**에만 등장 확인

---

## 수정 범위 요약

| 파일 | 수정 번호 | 내용 |
|------|-----------|------|
| `astro_engine.py` | 1 | `_lagna_lord_placement_group` 헬퍼 추가 |
| `astro_engine.py` | 2 | `build_structural_summary` return에 bridge 키 4개 추가 |
| `llm_service.py` | 3 | yoga/pattern/avastha 매핑 테이블 3개 추가 |
| `llm_service.py` | 4 | `detected_yoga_ids`, `detected_pattern_ids`, `ll_ids` 조립 로직 교체 |
| `llm_service.py` | 5 | `_EVIDENCE_FALLBACK_PATTERNS` 누락 챕터 2개 추가 |
| `llm_service.py` | 6 | Cross-chapter dedup + fallback 최대 2회 제한 (초기 조립 + 보충 전체 적용) |
| `llm_service.py` | 7 | Evidence id 라벨 제거 |
| `llm_service.py` | 8 | `_evidence_text_trim` 문장 경계 트런케이션 |
| `llm_service.py` | 9 | Bridge 삽입을 normalize 이후로 이동 + `_is_evidence_para` 기준 교체 |
