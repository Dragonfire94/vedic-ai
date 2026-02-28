import os
import logging
import re
import asyncio
import hashlib
import httpx
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

from backend.report_engine import (
    _get_atomic_chart_interpretations,
    build_dasha_narrative_context,
    build_semantic_signals,
)
from backend.evidence_pipeline_v2 import (
    apply_bullet_escape_to_chapter_evidence_map,
    apply_caps_to_chapter_evidence_map,
    audit_length_density,
    audit_llm_style_only,
    pre_sanitize_global_evidence_items,
    pre_sanitize_chapter_evidence_map,
    prepatch_chapter_evidence_map,
)
from backend.vedic_lexicon import enforce_subtle_vedic_lexicon

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
LLM_RELAX_MODE = os.getenv("LLM_RELAX_MODE", "phase15").strip().lower()
logger = logging.getLogger("vedic_ai")

_SHORT_TITLE_BY_KEY = {
    "Executive Diagnosis": "핵심 진단",
    "Current Phase": "현재 흐름",
    "Core Disposition": "핵심 기질",
    "Recurring Patterns": "반복 패턴",
    "Emotional Fault Lines": "감정 구조",
    "Career & Money": "커리어/돈",
    "Love & Relationship Patterns": "관계 패턴",
    "Health & Energy Rhythm": "에너지 리듬",
    "Mid-Term Direction": "중기 흐름",
    "Risk Management Points": "리스크 관리",
    "Growth Acceleration": "성장 가속",
    "Final Integration": "마지막 통합",
}

_PREMIUM_12_KEYS = [
    "Executive Diagnosis",
    "Current Phase",
    "Core Disposition",
    "Recurring Patterns",
    "Emotional Fault Lines",
    "Career & Money",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Mid-Term Direction",
    "Risk Management Points",
    "Growth Acceleration",
    "Final Integration",
]

_ACTIONABLE_CHAPTER_KEYS = {
    "Career & Money",
    "Risk Management Points",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Core Disposition",
    "Mid-Term Direction",
    "Growth Acceleration",
}
_BULLET_EXEMPT_CHAPTER_KEYS = {
    "Executive Diagnosis",
    "Current Phase",
    "Final Integration",
}
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+).+")
_EVIDENCE_TAG_RE = re.compile(
    r'(?:<\s*EVIDENCE_BLOCK\s*>|\[EVIDENCE_BLOCK\]|EVIDENCE_BLOCK)',
    re.IGNORECASE,
)
_CONDITIONAL_REGEN_CHAPTERS_DEFAULT = {
    "Core Disposition",
    "Love & Relationship Patterns",
    "Career & Money",
}
_CHAPTER_REGEN_EVIDENCE_COUNTS: dict[str, int] = {}
_MAX_CONDITIONAL_REGEN_PER_REQUEST = 1
_INTERPRETATIONS_PATH = Path("assets/data/interpretations.kr_final.json")
_INTERPRETATIONS_INDEX_CACHE: dict[str, dict[str, str]] | None = None
_HYBRID_EVIDENCE_ESCAPE_CACHE: dict[int, dict[str, str]] = {}
_HYBRID_EVIDENCE_ESCAPE_CACHE_LIMIT = 64
_EVIDENCE_FALLBACK_PATTERNS = {
    "Executive Diagnosis": ["pat:strong_lagna_lord", "pat:kendra_emphasis", "pat:upachaya_emphasis"],
    "Current Phase": ["pat:strong_lagna_lord", "pat:kendra_emphasis", "pat:trikona_emphasis"],
    "Core Disposition": ["pat:scattered_energy", "pat:kendra_emphasis"],
    "Recurring Patterns": ["pat:dusthana_focus", "pat:scattered_energy"],
    "Emotional Fault Lines": ["pat:strong_moon", "pat:afflicted_moon", "pat:combust_emphasis"],
    "Career & Money": ["pat:strong_10th_lord", "pat:kendra_emphasis", "pat:trikona_emphasis"],
    "Love & Relationship Patterns": ["pat:benefic_support", "pat:malefic_overload"],
    "Health & Energy Rhythm": ["pat:malefic_overload", "pat:combust_emphasis"],
    "Mid-Term Direction": ["pat:trikona_emphasis", "pat:upachaya_emphasis", "pat:benefic_support"],
    "Risk Management Points": ["pat:strong_moon", "pat:afflicted_moon", "pat:combust_emphasis"],
    "Growth Acceleration": ["pat:kendra_emphasis", "pat:upachaya_emphasis"],
}
CHAPTER_EVIDENCE_RULES: dict[str, dict[str, int]] = {
    "Executive Diagnosis": {"patterns": 2, "yogas": 1, "lagna_lord": 1},
    "Current Phase": {"patterns": 2, "yogas": 0, "lagna_lord": 2},
    "Core Disposition": {"patterns": 3, "yogas": 0, "lagna_lord": 1},
    "Recurring Patterns": {"patterns": 3, "yogas": 1, "lagna_lord": 1},
    "Emotional Fault Lines": {"patterns": 2, "yogas": 0, "lagna_lord": 0},
    "Career & Money": {"patterns": 2, "yogas": 1, "lagna_lord": 1},
    "Love & Relationship Patterns": {"patterns": 2, "yogas": 1, "lagna_lord": 0},
    "Health & Energy Rhythm": {"patterns": 2, "yogas": 0, "lagna_lord": 1},
    "Mid-Term Direction": {"patterns": 2, "yogas": 1, "lagna_lord": 1},
    "Risk Management Points": {"patterns": 3, "yogas": 0, "lagna_lord": 0},
    "Growth Acceleration": {"patterns": 2, "yogas": 0, "lagna_lord": 1},
    "Final Integration": {"reuse_top": 3},
}
_SUBTLE_VEDIC_PROMPT_RULES = """
[SUBTLE VEDIC POLICY — 반드시 준수]
- 최종 출력에서 허용되는 베딕 용어는 아래 6개뿐이다:
  라후(Rahu), 케투(Ketu), 다샤(Dasha), 부크티(Bhukti), 라그나(Lagna), 나크샤트라(Nakshatra)
- 챕터당 베딕 용어 mention 최대 2회, 문서 전체 최대 8회.
- 각 용어의 첫 등장 문장은 반드시 즉시 쉬운 한국어 풀이를 같은 문장에 붙인다.
  (관장하는/보여주는/뜻하는/설명하는 스타일)
- 한 문장에 여러 베딕 용어를 나열하지 않는다. 용어 dump 금지.
- 무거운 베딕 기법 용어 금지:
  하우스 번호, 각도(°), varga(D9/D10), yoga 리스트, 산스크리트 메커닉 나열.
- 날짜/연도/분기 예언 금지:
  2026년, 상반기/하반기, 분기, Q1~Q4 같은 표현 금지.

예시(참고):
- "확장 욕구를 관장하는 라후(Rahu)는 속도를 붙이지만, 과열도 함께 부를 수 있습니다."
- "시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)는 ‘요즘 결이 바뀌는 느낌’으로 먼저 체감됩니다."
- "정리·거리두기 본능을 관장하는 케투(Ketu)가 강해지면, 설명보다 정리가 먼저 필요해질 때가 있습니다."
"""
_ENGINE_YOGA_KEY_TO_INTERP: dict[str, str] = {
    "raja_yoga": "yoga:RajayogaGeneral",
    "dhana_yoga": "yoga:DhanaYogaGeneral",
    "parivartana_yoga": "yoga:ParivartanaGeneral",
    "vipareeta_raja_yoga": "yoga:VipareetaLite",
    "gaja_kesari_yoga": "yoga:Gajakesari",
    "neecha_bhanga": "yoga:NeechaBhangaLite",
    "kemadruma": "yoga:KemadrumaLite",
}
_ENGINE_PATTERN_KEY_TO_INTERP: dict[str, str] = {
    "authority_conflict_pattern": "pat:malefic_overload",
    "delayed_success_pattern": "pat:dusthana_focus",
    "financial_leak_pattern": "pat:scattered_energy",
    "obsession_public_image_pattern": "pat:kendra_emphasis",
    "relationship_abandonment_pattern": "pat:scattered_energy",
}
_ENGINE_AVASTHA_TO_LL_STATE: dict[str, str] = {
    "asta": "ll:state:combust",
    "deepta": "ll:state:exalted",
    "chesta": "ll:state:retrograde",
    "yuva": "ll:state:own",
    "dina": "ll:state:debilitated",
    "madhya": "ll:state:neutral",
}
_EVIDENCE_LOW_PRIORITY_CHAPTERS = [
    "Executive Diagnosis",
    "Final Integration",
    "Mid-Term Direction",
]


def _active_report_chapters() -> list[str]:
    return list(_PREMIUM_12_KEYS)


def _flatten_interpretation_section(section: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(section, dict):
        return out
    for key, value in section.items():
        if isinstance(value, dict):
            text = str(value.get("text", "")).strip()
        else:
            text = str(value).strip()
        if text:
            out[str(key).strip()] = text
    return out


def _load_interpretations_index() -> dict[str, dict[str, str]]:
    global _INTERPRETATIONS_INDEX_CACHE
    if isinstance(_INTERPRETATIONS_INDEX_CACHE, dict):
        return _INTERPRETATIONS_INDEX_CACHE
    try:
        if not _INTERPRETATIONS_PATH.exists():
            logger.error("interpretations file missing: %s", _INTERPRETATIONS_PATH)
            _INTERPRETATIONS_INDEX_CACHE = {"atomic": {}, "lagna_lord": {}, "yogas": {}, "patterns": {}}
            return _INTERPRETATIONS_INDEX_CACHE
        payload = json.loads(_INTERPRETATIONS_PATH.read_text(encoding="utf-8"))
        ko = payload.get("ko", {}) if isinstance(payload, dict) else {}
        _INTERPRETATIONS_INDEX_CACHE = {
            "atomic": _flatten_interpretation_section(ko.get("atomic")),
            "lagna_lord": _flatten_interpretation_section(ko.get("lagna_lord")),
            "yogas": _flatten_interpretation_section(ko.get("yogas")),
            "patterns": _flatten_interpretation_section(ko.get("patterns")),
        }
        return _INTERPRETATIONS_INDEX_CACHE
    except Exception as e:
        logger.error("interpretations load failed path=%s error=%s", _INTERPRETATIONS_PATH, e)
        _INTERPRETATIONS_INDEX_CACHE = {"atomic": {}, "lagna_lord": {}, "yogas": {}, "patterns": {}}
        return _INTERPRETATIONS_INDEX_CACHE


def _evidence_text_trim(text: str, max_chars: int) -> str:
    if not text:
        return ""
    t = str(text).strip()
    if len(t) <= max_chars:
        return t
    candidate = t[:max_chars]
    last_end = -1
    for m in re.finditer(r"[.!?。](?:\s|$)", candidate):
        last_end = m.end()
    if last_end > max_chars * 0.5:
        return candidate[:last_end].strip()
    return candidate.rstrip(".,;: ").strip()


def _evidence_chars(items: list[dict[str, str]]) -> int:
    return sum(len(str(x.get("text", ""))) for x in items if isinstance(x, dict))


def _append_evidence_item(
    *,
    items: list[dict[str, str]],
    seen_ids: set[str],
    item_id: str,
    text: str,
    max_item_chars: int,
) -> None:
    iid = str(item_id or "").strip()
    if not iid or iid in seen_ids:
        return
    t = _evidence_text_trim(text, max_item_chars)
    if not t:
        return
    seen_ids.add(iid)
    items.append({"id": iid, "text": t})


def build_evidence_packs(
    structured_summary: dict[str, Any],
    chapter_keys: list[str],
    *,
    chapter_chars_min: int = 600,
    chapter_chars_max: int = 1200,
    global_chars_max: int = 1500,
    global_chars_hard_max: int = 2500,
    total_chars_hard_max: int = 15000,
    chapter_items_min: int = 2,
    chapter_items_max: int = 4,
) -> dict[str, Any]:
    source = structured_summary if isinstance(structured_summary, dict) else {}
    idx = _load_interpretations_index()
    atomic = idx.get("atomic", {})
    patterns = idx.get("patterns", {})
    yogas = idx.get("yogas", {})
    lagna_lord = idx.get("lagna_lord", {})

    def _norm_pat_id(value: str) -> str:
        # NOTE: build_evidence_packs prioritizes _ENGINE_PATTERN_KEY_TO_INTERP.
        # This helper is retained for direct pattern-id normalization outside mapping.
        v = str(value or "").strip()
        if not v:
            return ""
        return v if v.startswith("pat:") else f"pat:{v}"

    def _norm_yoga_id(value: str) -> str:
        # NOTE: build_evidence_packs prioritizes _ENGINE_YOGA_KEY_TO_INTERP.
        # This helper is retained for direct yoga-id normalization outside mapping.
        v = str(value or "").strip()
        if not v:
            return ""
        return v if v.startswith("yoga:") else f"yoga:{v}"

    detected_yoga_ids: list[str] = []
    for v in (source.get("detected_yogas") or []):
        if not isinstance(v, str):
            continue
        mapped = _ENGINE_YOGA_KEY_TO_INTERP.get(v.strip().lower())
        if mapped:
            detected_yoga_ids.append(mapped)

    detected_pattern_ids: list[str] = []
    for v in (source.get("pattern_flags") or []):
        if not isinstance(v, str):
            continue
        mapped = _ENGINE_PATTERN_KEY_TO_INTERP.get(v.strip().lower())
        if mapped:
            detected_pattern_ids.append(mapped)

    ll_state_raw = str(source.get("lagna_lord_state") or "").strip().lower()
    ll_place_raw = str(source.get("lagna_lord_placement_group") or "").strip().lower()

    ll_state_key = _ENGINE_AVASTHA_TO_LL_STATE.get(ll_state_raw, "")
    ll_place_key = f"ll:placement:{ll_place_raw}" if ll_place_raw else ""
    ll_ids = [k for k in [ll_state_key, ll_place_key] if k]

    global_items: list[dict[str, str]] = []
    global_seen: set[str] = set()
    missing_ids: list[str] = []

    for key in (
        (f"asc:{str(source.get('ascendant_sign') or '').strip()}" if str(source.get("ascendant_sign") or "").strip() else ""),
        (f"ps:Sun:{str(source.get('sun_sign') or '').strip()}" if str(source.get("sun_sign") or "").strip() else ""),
        (f"ps:Moon:{str(source.get('moon_sign') or '').strip()}" if str(source.get("moon_sign") or "").strip() else ""),
    ):
        if not key:
            continue
        txt = atomic.get(key)
        if txt:
            _append_evidence_item(items=global_items, seen_ids=global_seen, item_id=key, text=txt, max_item_chars=420)
        else:
            missing_ids.append(key)

    while _evidence_chars(global_items) > global_chars_max and len(global_items) > 1:
        global_items.pop()
    if _evidence_chars(global_items) > global_chars_hard_max and global_items:
        global_items = global_items[:1]
    global_ids = {str(it.get("id", "")) for it in global_items if isinstance(it, dict)}

    chapter_evidence_raw: dict[str, list[dict[str, Any]]] = {}
    fallback_used_by_chapter: dict[str, bool] = {}
    _fallback_use_count: dict[str, int] = {}
    reuse_in_final_ids: list[str] = []

    def _try_add(
        target: list[dict[str, Any]],
        seen: set[str],
        chapter: str,
        item_id: str,
        text_pool: dict[str, str],
        kind: str,
        score: int,
        max_item_chars: int,
    ) -> bool:
        iid = str(item_id or "").strip()
        if not iid:
            return False
        if iid in global_ids:
            return False
        if iid in seen:
            return False
        txt = text_pool.get(iid)
        if not txt:
            missing_ids.append(iid)
            return False
        t = _evidence_text_trim(txt, max_item_chars)
        if not t:
            return False
        seen.add(iid)
        target.append(
            {
                "id": iid,
                "text": t,
                "_kind": kind,
                "_score": score,
                "_chapter": chapter,
            }
        )
        return True

    normal_chapters = [ck for ck in chapter_keys if ck != "Final Integration"]
    for chapter in normal_chapters:
        rule = CHAPTER_EVIDENCE_RULES.get(chapter, {"patterns": 2, "yogas": 1, "lagna_lord": 1})
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        used_fallback = False

        y_added = 0
        for yid in detected_yoga_ids:
            if y_added >= int(rule.get("yogas", 0)):
                break
            if _try_add(items, seen, chapter, yid, yogas, "yoga", 100, 360):
                y_added += 1

        p_added = 0
        for pid in detected_pattern_ids:
            if p_added >= int(rule.get("patterns", 0)):
                break
            if _try_add(items, seen, chapter, pid, patterns, "pattern", 80, 360):
                p_added += 1

        ll_added = 0
        for lid in ll_ids:
            if ll_added >= int(rule.get("lagna_lord", 0)):
                break
            if _try_add(items, seen, chapter, lid, lagna_lord, "lagna_lord", 60, 320):
                ll_added += 1

        for pid in _EVIDENCE_FALLBACK_PATTERNS.get(chapter, []):
            if len(items) >= chapter_items_max:
                break
            if _fallback_use_count.get(pid, 0) >= 2:
                continue
            if _try_add(items, seen, chapter, pid, patterns, "fallback", 20, 360):
                used_fallback = True
                _fallback_use_count[pid] = _fallback_use_count.get(pid, 0) + 1

        while _evidence_chars(items) > chapter_chars_max and len(items) > 1:
            drop_idx = -1
            for idx_i, candidate in enumerate(items):
                if str(candidate.get("_kind")) == "fallback":
                    drop_idx = idx_i
                    break
            if drop_idx < 0:
                drop_idx = len(items) - 1
            items.pop(drop_idx)

        if _evidence_chars(items) > chapter_chars_max and items:
            for item in items:
                item["text"] = _evidence_text_trim(item.get("text", ""), 300)
                if _evidence_chars(items) <= chapter_chars_max:
                    break

        chapter_evidence_raw[chapter] = items
        fallback_used_by_chapter[chapter] = used_fallback

    # Cross-chapter evidence dedup: earlier chapters keep precedence.
    _cross_seen: set[str] = set()
    for chapter in normal_chapters:
        items = chapter_evidence_raw.get(chapter, [])
        deduped: list[dict[str, Any]] = []
        for item in items:
            iid = str(item.get("id", "")).strip()
            item_kind = str(item.get("_kind", ""))
            is_engine_signal = item_kind in ("yoga", "pattern", "lagna_lord", "fallback")
            if not iid or not is_engine_signal or iid not in _cross_seen:
                deduped.append(item)
                if iid and is_engine_signal:
                    _cross_seen.add(iid)

        if len(deduped) < len(items):
            # 1) chapter-specific fallback refill
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

            # 2) global fallback pool for chapters with thin/empty local fallbacks
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

        # Always recompute fallback flag from final chapter payload.
        fallback_used_by_chapter[chapter] = any(
            str(x.get("_kind", "")) == "fallback"
            for x in chapter_evidence_raw.get(chapter, [])
        )

    # Final Integration: process last using strongest evidence from other chapters.
    if "Final Integration" in chapter_keys:
        pool: list[dict[str, Any]] = []
        for chapter in normal_chapters:
            for item in chapter_evidence_raw.get(chapter, []):
                if isinstance(item, dict):
                    pool.append(item)
        pool.sort(
            key=lambda x: (
                int(x.get("_score", 0)),
                len(str(x.get("text", ""))),
            ),
            reverse=True,
        )
        final_items: list[dict[str, Any]] = []
        seen_final: set[str] = set()
        reuse_top = int(CHAPTER_EVIDENCE_RULES.get("Final Integration", {}).get("reuse_top", 2))
        for item in pool:
            iid = str(item.get("id", "")).strip()
            if not iid or iid in seen_final:
                continue
            seen_final.add(iid)
            reuse_in_final_ids.append(iid)
            final_items.append(
                {
                    "id": iid,
                    "text": str(item.get("text", "")).strip(),
                    "_kind": "reused",
                    "_score": int(item.get("_score", 0)),
                    "_chapter": "Final Integration",
                }
            )
            if len(final_items) >= reuse_top:
                break
        chapter_evidence_raw["Final Integration"] = final_items
        fallback_used_by_chapter["Final Integration"] = False

    chapter_evidence: dict[str, list[dict[str, str]]] = {}
    chapter_evidence_count: dict[str, int] = {}
    chapter_evidence_char_count: dict[str, int] = {}
    no_evidence_keys: list[str] = []
    low_density_keys: list[str] = []

    for chapter in chapter_keys:
        items = chapter_evidence_raw.get(chapter, [])
        safe_items = [{"id": str(i.get("id", "")), "text": str(i.get("text", ""))} for i in items if isinstance(i, dict)]
        chapter_evidence[chapter] = safe_items
        chapter_evidence_count[chapter] = len(safe_items)
        chapter_evidence_char_count[chapter] = _evidence_chars(safe_items)
        if len(safe_items) == 0:
            no_evidence_keys.append(chapter)
        if chapter != "Final Integration" and len(safe_items) < 3:
            low_density_keys.append(chapter)

    def _recompute_total() -> int:
        return _evidence_chars(global_items) + sum(_evidence_chars(chapter_evidence.get(k, [])) for k in chapter_keys)

    def _pop_one_by_kind(chapter: str, allowed_kinds: set[str] | None) -> bool:
        raw = chapter_evidence_raw.get(chapter, [])
        if not raw:
            return False
        for i, item in enumerate(raw):
            kind = str(item.get("_kind", ""))
            if allowed_kinds is None or kind in allowed_kinds:
                raw.pop(i)
                chapter_evidence_raw[chapter] = raw
                chapter_evidence[chapter] = [{"id": str(x.get("id", "")), "text": str(x.get("text", ""))} for x in raw]
                chapter_evidence_count[chapter] = len(chapter_evidence[chapter])
                chapter_evidence_char_count[chapter] = _evidence_chars(chapter_evidence[chapter])
                return True
        return False

    evidence_trim_level = 0
    total_chars = _recompute_total()
    while total_chars > total_chars_hard_max:
        changed = False
        # 1) remove reused evidence in Final Integration
        if _pop_one_by_kind("Final Integration", {"reused"}):
            evidence_trim_level = max(evidence_trim_level, 1)
            changed = True
        # 2) remove fallback evidence
        if not changed:
            for chapter in chapter_keys:
                if _pop_one_by_kind(chapter, {"fallback"}):
                    evidence_trim_level = max(evidence_trim_level, 2)
                    changed = True
                    break
        # 3) remove yogas
        if not changed:
            for chapter in chapter_keys:
                if _pop_one_by_kind(chapter, {"yoga"}):
                    evidence_trim_level = max(evidence_trim_level, 3)
                    changed = True
                    break
        # 4) remove lagna_lord
        if not changed:
            for chapter in chapter_keys:
                if _pop_one_by_kind(chapter, {"lagna_lord"}):
                    evidence_trim_level = max(evidence_trim_level, 4)
                    changed = True
                    break
        # 5) truncate text
        if not changed:
            truncated = False
            for chapter in chapter_keys:
                safe_items = chapter_evidence.get(chapter, [])
                for item in safe_items:
                    old = str(item.get("text", ""))
                    new = _evidence_text_trim(old, max(180, len(old) - 80))
                    if new != old:
                        item["text"] = new
                        truncated = True
                        break
                if truncated:
                    chapter_evidence_char_count[chapter] = _evidence_chars(safe_items)
                    evidence_trim_level = max(evidence_trim_level, 5)
                    changed = True
                    break
        # 6) remove one item from low-priority chapters
        if not changed:
            for chapter in _EVIDENCE_LOW_PRIORITY_CHAPTERS:
                items = chapter_evidence.get(chapter, [])
                if items:
                    items.pop()
                    chapter_evidence[chapter] = items
                    chapter_evidence_count[chapter] = len(items)
                    chapter_evidence_char_count[chapter] = _evidence_chars(items)
                    evidence_trim_level = max(evidence_trim_level, 6)
                    changed = True
                    break
        if not changed:
            break
        total_chars = _recompute_total()

    # Re-sync density keys after trimming
    no_evidence_keys = [k for k in chapter_keys if len(chapter_evidence.get(k, [])) == 0]
    low_density_keys = [k for k in chapter_keys if k != "Final Integration" and len(chapter_evidence.get(k, [])) < 3]

    return {
        "global_evidence": global_items,
        "chapter_evidence": chapter_evidence,
        "stats": {
            "global_items": len(global_items),
            "global_chars": _evidence_chars(global_items),
            "total_chars": total_chars,
            "missing_ids": missing_ids[:50],
            "chapter_evidence_count": chapter_evidence_count,
            "chapter_evidence_char_count": chapter_evidence_char_count,
            "fallback_used": fallback_used_by_chapter,
            "reused_in_final": reuse_in_final_ids[:10],
            "no_evidence_keys": no_evidence_keys,
            "low_evidence_density_keys": low_density_keys,
            "evidence_trim_level": evidence_trim_level,
        },
    }


def _escape_evidence_text(text: str) -> str:
    escaped = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    escaped = re.sub(r"(?m)^\s*\d+[.)]\s+", "• ", escaped)
    return escaped


def _apply_evidence_soft_cap(text: str, max_chars: int) -> tuple[str, bool, int]:
    if max_chars <= 0:
        return text, False, -1
    if len(text) <= max_chars:
        return text, False, -1
    paragraphs = [p for p in text.split("\n\n") if p is not None]
    if not paragraphs:
        return text[:max_chars].rstrip(), True, max_chars
    while len("\n\n".join(paragraphs)) > max_chars and len(paragraphs) > 1:
        paragraphs.pop()
    joined = "\n\n".join(paragraphs)
    if len(joined) > max_chars:
        if len(paragraphs) > 1:
            prefix = "\n\n".join(paragraphs[:-1])
            allowed = max_chars - (len(prefix) + 2) if prefix else max_chars
            allowed = max(0, allowed)
            last = paragraphs[-1][:allowed].rstrip()
            paragraphs = paragraphs[:-1] + [last]
        else:
            paragraphs = [paragraphs[0][:max_chars].rstrip()]
        joined = "\n\n".join(paragraphs)
    if not joined:
        joined = paragraphs[0][:max_chars].rstrip()
    return joined, True, len(joined)


def _get_escape_cache_for_map(target_map: dict[str, list[dict[str, str]]]) -> dict[str, str]:
    cache_key = id(target_map)
    cache = _HYBRID_EVIDENCE_ESCAPE_CACHE.get(cache_key)
    if cache is None:
        if len(_HYBRID_EVIDENCE_ESCAPE_CACHE) >= _HYBRID_EVIDENCE_ESCAPE_CACHE_LIMIT:
            _HYBRID_EVIDENCE_ESCAPE_CACHE.clear()
        cache = {}
        _HYBRID_EVIDENCE_ESCAPE_CACHE[cache_key] = cache
    return cache


def _assemble_evidence_text(
    *,
    items: list[dict[str, str]],
    cache: dict[str, str],
    cache_key: str,
    max_chars: int,
    pure_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    evidence_items_count = len(items)
    raw_lines: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        iid = str(it.get("id", "")).strip()
        txt = str(it.get("text", "")).strip()
        if not txt:
            continue
        if iid:
            raw_lines.append(f"({iid}) {txt}")
        else:
            raw_lines.append(txt)
    raw_text = "\n\n".join([line for line in raw_lines if line.strip()]).strip()
    before_len = len(raw_text)
    escape_applied = False
    if not raw_text:
        return "", {
            "evidence_items_count": evidence_items_count,
            "evidence_chars_before_escape": 0,
            "evidence_chars_after_escape": 0,
            "escape_applied": False,
            "evidence_truncated": False,
            "evidence_chars_truncated_to": -1,
        }
    if pure_mode:
        escaped_text = raw_text
        after_len = len(escaped_text)
        capped_text = escaped_text
        truncated = False
        truncated_to = -1
    else:
        if cache_key in cache:
            escaped_text = cache[cache_key]
        else:
            escaped_text = _escape_evidence_text(raw_text)
            cache[cache_key] = escaped_text
            escape_applied = True
        after_len = len(escaped_text)
        capped_text, truncated, truncated_to = _apply_evidence_soft_cap(escaped_text, max_chars)
        # Remove label-only lines while preserving paragraph boundaries between evidence items.
        capped_text = re.sub(r"^[^\S\n]*\([a-zA-Z_]+:[^\)]+\)[^\S\n]*\n?", "", capped_text, flags=re.MULTILINE)
        capped_text = re.sub(r"\n{3,}", "\n\n", capped_text).rstrip()
    return capped_text, {
        "evidence_items_count": evidence_items_count,
        "evidence_chars_before_escape": before_len,
        "evidence_chars_after_escape": after_len,
        "escape_applied": escape_applied,
        "evidence_truncated": truncated,
        "evidence_chars_truncated_to": truncated_to if truncated else -1,
    }


_BRIDGE_FALLBACK_BY_CHAPTER: dict[str, str] = {
    "Career & Money": "이 흐름을 실제 선택에 연결하면 다음과 같은 방향이 나옵니다.",
    "Risk Management Points": "이 구조를 바탕으로 지금 단계에서 취할 수 있는 방향은 다음과 같습니다.",
    "Love & Relationship Patterns": "이 패턴을 관계 안에서 구체적으로 다루려면 아래를 참고하세요.",
    "Recurring Patterns": "이 반복을 알아차리는 것이 출발점이고, 실천은 여기서 시작됩니다.",
    "Health & Energy Rhythm": "이 리듬을 몸에서 실제로 관리하려면 다음 방향이 유효합니다.",
    "Mid-Term Direction": "이 흐름을 현재 방향에 연결하려면 아래 기준을 참고하세요.",
    "Core Disposition": "이 내적 흐름을 일상에서 다루는 실천 방향은 다음과 같습니다.",
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
            or "핵심 근거" in p
            or bool(re.match(r"^(?:\s*배치|\s*상태|\s*구성|\s*패턴|\s*근거)", p))
            or bool(
                re.match(
                    r"^\s*(?:강점(?:으로는|은)?|리스크(?:는)?|조언(?:은)?|발현은|이\s*배치|이\s*구성|이\s*집들|이\s*상태|이\s*패턴)",
                    p,
                )
            )
        )

    def _is_bullet_para(p: str) -> bool:
        first_line = p.split("\n")[0]
        return bool(_BULLET_LINE_RE.match(first_line))

    last_evidence_idx = -1
    for i, p in enumerate(paras):
        if _is_evidence_para(p):
            last_evidence_idx = i

    if last_evidence_idx < 0:
        return body

    after_evidence = paras[last_evidence_idx + 1:]
    has_bridge = any(
        not _is_evidence_para(p) and not _is_bullet_para(p)
        for p in after_evidence
    )
    if has_bridge:
        return body

    bridge = _BRIDGE_FALLBACK_BY_CHAPTER.get(chapter_key, _BRIDGE_FALLBACK_DEFAULT)
    paras.insert(last_evidence_idx + 1, bridge)
    return "\n\n".join(paras)


def apply_bridge_to_all_chapters(text: str) -> str:
    """
    Run after normalize_llm_layout_strict and re-insert bridge lines chapter-wise.
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
        m = re.match(r"^\[(.*?)\]\s*(.*)$", raw_heading_text.strip())
        token = m.group(1).strip() if m else raw_heading_text.strip()
        chapter_key = _normalize_chapter_key(token, raw_heading_text, sec_idx)

        new_body = _ensure_bridge_after_evidence(body, chapter_key)
        output_lines.extend(new_body.split("\n"))
        current_idx = end

    if current_idx < len(lines):
        output_lines.extend(lines[current_idx:])

    return "\n".join(output_lines)


def inject_evidence_blocks(
    response_text: str,
    chapter_evidence_map: dict[str, list[dict[str, str]]],
    global_evidence_items: list[dict[str, str]],
    *,
    hybrid_render_mode: bool,
    pure_mode: bool = False,
    target_chapters: list[str] | None = None,
    max_evidence_chars_per_chapter: int = 1200,
) -> tuple[str, dict[str, dict]]:
    """
    target_chapters가 지정되면 해당 챕터만 치환.
    hybrid_render_mode=False면 즉시 (response_text, {}) 반환.
    Returns (modified_text, per_chapter_stats)
    """
    if not hybrid_render_mode:
        return response_text, {}
    if not isinstance(response_text, str) or not response_text.strip():
        return response_text or "", {}

    normalized_text = response_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")
    heading_positions = _extract_heading_positions(normalized_text)
    if not heading_positions:
        return response_text, {}

    per_chapter_stats: dict[str, dict] = {}
    warn_emitted: dict[str, set[str]] = {}
    evidence_cache = _get_escape_cache_for_map(chapter_evidence_map if isinstance(chapter_evidence_map, dict) else {})
    target_set = set(target_chapters) if target_chapters else None
    found_targets: set[str] = set()

    def _emit_warn(chapter_key: str, warn_type: str, message: str, *args: Any) -> None:
        key = chapter_key or "_unknown"
        bucket = warn_emitted.setdefault(key, set())
        if warn_type in bucket:
            logger.debug(message, *args)
        else:
            logger.warning(message, *args)
            bucket.add(warn_type)

    def _init_stats(chapter_key: str) -> dict[str, Any]:
        base = {
            "tag_found": False,
            "tag_count": 0,
            "tag_variant_matched": "",
            "evidence_items_count": 0,
            "evidence_chars_before_escape": 0,
            "evidence_chars_after_escape": 0,
            "escape_applied": False,
            "missing_tag_fallback_used": False,
            "no_evidence_empty_replace": False,
            "regen_reinjected": bool(target_set),
            "evidence_truncated": False,
            "evidence_chars_truncated_to": -1,
            "tag_residual_recovered": False,
        }
        per_chapter_stats[chapter_key] = base
        return base

    output_lines: list[str] = []
    current_idx = 0
    for sec_idx, (start, raw_heading_text) in enumerate(heading_positions):
        end = heading_positions[sec_idx + 1][0] if sec_idx + 1 < len(heading_positions) else len(lines)
        if current_idx < start:
            output_lines.extend(lines[current_idx:start])
        heading_line = lines[start]
        output_lines.append(heading_line)

        token = ""
        m = re.match(r"^\[(.*?)\]\s*(.*)$", raw_heading_text)
        if m:
            token = (m.group(1) or "").strip()
        chapter_key = _normalize_chapter_key(token, raw_heading_text, sec_idx)
        if target_set is not None and chapter_key not in target_set:
            block_body = "\n".join(lines[start + 1 : end])
            if block_body:
                output_lines.extend(block_body.split("\n"))
            current_idx = end
            continue

        if chapter_key:
            found_targets.add(chapter_key)
        block_body = "\n".join(lines[start + 1 : end])
        stats = _init_stats(chapter_key)

        items = chapter_evidence_map.get(chapter_key, []) if isinstance(chapter_evidence_map, dict) else []
        evidence_text, evidence_meta = _assemble_evidence_text(
            items=items if isinstance(items, list) else [],
            cache=evidence_cache,
            cache_key=chapter_key or f"sec_{sec_idx}",
            max_chars=max_evidence_chars_per_chapter,
            pure_mode=pure_mode,
        )
        stats.update(evidence_meta)

        tag_matches = list(_EVIDENCE_TAG_RE.finditer(block_body))
        tag_count = len(tag_matches)
        stats["tag_found"] = tag_count > 0
        stats["tag_count"] = tag_count
        stats["tag_variant_matched"] = tag_matches[0].group(0) if tag_count > 0 else ""

        if tag_count == 0:
            _emit_warn(
                chapter_key,
                "missing_evidence_tag",
                "WARN missing_evidence_tag chapter=%s",
                chapter_key,
            )
        elif tag_count > 1:
            _emit_warn(
                chapter_key,
                "multiple_evidence_tags",
                "WARN multiple_evidence_tags chapter=%s count=%s",
                chapter_key,
                tag_count,
            )

        if pure_mode:
            if tag_count == 0:
                logger.warning("EVIDENCE_TAG_MISSING chapter=%s", chapter_key)
                new_body = block_body
            else:
                used_first = False

                def _tag_repl_pure(match: re.Match) -> str:
                    nonlocal used_first
                    if not used_first:
                        used_first = True
                        return evidence_text
                    return ""

                new_body = _EVIDENCE_TAG_RE.sub(_tag_repl_pure, block_body)
        elif stats["evidence_items_count"] == 0:
            stats["no_evidence_empty_replace"] = True
            # TODO: global evidence fallback 정책 결정 후 구현
            _emit_warn(
                chapter_key,
                "no_evidence_for_chapter",
                "WARN no_evidence_for_chapter chapter=%s",
                chapter_key,
            )
            if tag_count > 0:
                new_body = _EVIDENCE_TAG_RE.sub("", block_body)
                new_body = re.sub(r"\n{3,}", "\n\n", new_body)
            else:
                new_body = block_body
        else:
            if stats.get("evidence_truncated"):
                _emit_warn(
                    chapter_key,
                    "evidence_truncated",
                    "WARN evidence_truncated chapter=%s before=%s after=%s",
                    chapter_key,
                    stats.get("evidence_chars_after_escape"),
                    stats.get("evidence_chars_truncated_to"),
                )
            if tag_count == 0:
                block_body = block_body.replace("\r\n", "\n")
                parts = block_body.split("\n\n", 1)
                if len(parts) == 2:
                    hook_part, rest_part = parts
                    new_body = hook_part + "\n\n" + evidence_text + "\n\n" + rest_part
                else:
                    new_body = evidence_text + "\n\n" + block_body
                stats["missing_tag_fallback_used"] = True
            else:
                used_first = False

                def _tag_repl(match: re.Match) -> str:
                    nonlocal used_first
                    if not used_first:
                        used_first = True
                        return evidence_text
                    return ""

                new_body = _EVIDENCE_TAG_RE.sub(_tag_repl, block_body)

        if new_body:
            output_lines.extend(new_body.split("\n"))
        current_idx = end

    if current_idx < len(lines):
        output_lines.extend(lines[current_idx:])

    modified_text = "\n".join(output_lines)

    if target_set is not None:
        missing_targets = [ck for ck in target_set if ck not in found_targets]
        for ck in missing_targets:
            _emit_warn(
                ck,
                "timeline_regen_block_scope_not_found",
                "WARN timeline_regen_block_scope_not_found chapter=%s",
                ck,
            )

    if (not pure_mode) and _EVIDENCE_TAG_RE.search(modified_text):
        tag_matches = list(_EVIDENCE_TAG_RE.finditer(modified_text))
        if tag_matches:
            heading_positions = _extract_heading_positions(modified_text)
            heading_positions_sorted = sorted(heading_positions, key=lambda x: x[0])

            def _chapter_for_line(line_idx: int) -> str:
                current_key = ""
                for sec_idx, (h_idx, raw_heading_text) in enumerate(heading_positions_sorted):
                    if h_idx > line_idx:
                        break
                    token = ""
                    m = re.match(r"^\[(.*?)\]\s*(.*)$", raw_heading_text)
                    if m:
                        token = (m.group(1) or "").strip()
                    current_key = _normalize_chapter_key(token, raw_heading_text, sec_idx)
                return current_key or "unknown"

            for match in tag_matches:
                line_idx = modified_text[: match.start()].count("\n")
                chapter_key = _chapter_for_line(line_idx)
                logger.error(
                    "evidence_block_tag_residual chapter=%s line=%s pos=%s",
                    chapter_key,
                    line_idx + 1,
                    match.start(),
                )
                stats = per_chapter_stats.get(chapter_key) or _init_stats(chapter_key)
                stats["tag_residual_recovered"] = True
                _emit_warn(
                    chapter_key,
                    "tag_residual_recovery_attempted",
                    "WARN tag_residual_recovery_attempted chapter=%s",
                    chapter_key,
                )

        global_text, _global_meta = _assemble_evidence_text(
            items=global_evidence_items if isinstance(global_evidence_items, list) else [],
            cache=evidence_cache,
            cache_key="__global__",
            max_chars=max_evidence_chars_per_chapter,
        )
        if not global_text:
            global_text = "이 항목에 대한 근거 데이터를 구성하는 중 오류가 발생했습니다."
        modified_text = _EVIDENCE_TAG_RE.sub(global_text, modified_text)
        if _EVIDENCE_TAG_RE.search(modified_text):
            neutral = "이 항목에 대한 근거 데이터를 구성하는 중 오류가 발생했습니다."
            modified_text = _EVIDENCE_TAG_RE.sub(neutral, modified_text)

    return modified_text, per_chapter_stats


def _strip_unresolved_evidence_tags_with_existing_re(text: str) -> tuple[str, int]:
    if not isinstance(text, str):
        return "", 0
    hits = len(_EVIDENCE_TAG_RE.findall(text))
    if hits:
        text = _EVIDENCE_TAG_RE.sub("", text)
    return text, hits


_FALLBACK_PARAGRAPH_POOL = [
    "지금은 결론을 서두르기보다 흐름을 차분히 살펴보는 편이 좋습니다.",
    "한 번에 크게 바꾸기보다 작은 확인을 쌓아가면 방향이 선명해집니다.",
    "지금 구간에서는 속도보다 리듬을 맞추는 선택이 더 안정적입니다.",
    "당장 정답을 찾기보다 반복되는 장면을 먼저 알아차리는 것이 도움이 됩니다.",
    "무리해서 밀어붙이기보다 덜 소모되는 방식으로 조정해도 충분합니다.",
    "지금은 크게 확정하기보다 흔들리는 지점을 먼저 정리하는 편이 낫습니다.",
    "한 번의 강한 결정보다 작은 조정의 누적이 더 큰 차이를 만듭니다.",
    "지금 단계에서는 확신보다 점검이 먼저일 때 흐름이 덜 흔들립니다.",
    "복잡하게 해석하기보다 지금 반복되는 선택을 가볍게 확인해 보세요.",
    "성급한 단정보다 현재의 반응 패턴을 짚어보는 것이 더 유리합니다.",
    "지금은 크고 빠른 변화보다 작은 균형 조정이 더 효과적일 수 있습니다.",
    "당장 완벽해지기보다 흔들리는 순간을 빨리 알아차리는 편이 도움이 됩니다.",
    "무리한 확정보다 현재 리듬을 읽고 맞추는 선택이 더 오래 갑니다.",
    "지금 단계에서는 정답보다 방향을 잃지 않는 것이 더 중요합니다.",
    "한 번에 해결하려 하기보다 부담이 큰 지점부터 가볍게 줄여보세요.",
    "지금은 성과보다 소모를 줄이는 쪽에서 흐름이 안정되기 쉽습니다.",
    "빠른 결론보다 반복되는 장면의 패턴을 붙잡는 편이 유리합니다.",
    "당장 바꾸기 어려운 부분은 유지하고, 바꿀 수 있는 부분부터 시작해도 됩니다.",
]

_FALLBACK_TAIL_POOL = [
    "지금은 천천히 가도 괜찮습니다.",
    "리듬을 먼저 지키는 쪽이 맞습니다.",
    "작은 점검이 오히려 멀리 갑니다.",
    "조급함보다 균형이 더 중요합니다.",
    "한 번에 바꾸려 하지 않아도 됩니다.",
    "덜 소모되는 선택이 더 유리합니다.",
    "오늘은 속도를 낮춰도 충분합니다.",
    "작게 시작해도 방향은 잡힙니다.",
    "지금은 버티는 힘을 먼저 챙기세요.",
    "무리하지 않는 선택이 오래 갑니다.",
    "흐름을 읽는 쪽이 더 안정적입니다.",
    "지금은 정리의 리듬이 우선입니다.",
]


def _select_chapter_blocks_source(chapter_blocks: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(chapter_blocks, dict):
        return {}
    v2 = chapter_blocks.get("chapter_blocks_v2")
    if isinstance(v2, dict):
        return v2
    legacy = chapter_blocks.get("chapter_blocks")
    if isinstance(legacy, dict):
        return legacy
    return chapter_blocks


def _compact_chapter_blocks_for_prompt(
    chapter_blocks: dict[str, Any],
    *,
    max_blocks_per_chapter: int = 3,
    summary_max_chars: int = 420,
    global_budget_chars: int = 12000,
) -> dict[str, Any]:
    """Build a compact but content-dense chapter_blocks snapshot for prompt assembly."""
    source = _select_chapter_blocks_source(chapter_blocks)
    if not isinstance(source, dict):
        return {}

    candidate_fields = ("summary", "analysis", "implication", "examples", "shadow_pattern")
    out: dict[str, Any] = {}
    text_budget = 0

    for key, blocks in source.items():
        if not isinstance(blocks, list) or not blocks:
            continue

        compact_items: list[dict[str, Any]] = []
        for block in blocks:
            if len(compact_items) >= max_blocks_per_chapter:
                break
            if not isinstance(block, dict):
                continue

            title = str(block.get("title", "")).strip()
            parts: list[str] = []
            for field in candidate_fields:
                raw = block.get(field)
                text = str(raw).strip() if raw is not None else ""
                if text:
                    parts.append(text)
            content = " ".join(parts).strip()
            if not content:
                continue
            if len(content) > summary_max_chars:
                content = content[:summary_max_chars].rstrip() + "..."

            item = {"title": title, "content": content}
            compact_items.append(item)
            text_budget += len(content) + len(title)

            if text_budget >= global_budget_chars:
                break

        if compact_items:
            out[str(key)] = compact_items
        if text_budget >= global_budget_chars:
            break

    return out


def _derive_narrative_mode(structural_summary: dict[str, Any]) -> str:
    stability = structural_summary.get("stability_metrics", {}) if isinstance(structural_summary, dict) else {}
    forecast = structural_summary.get("probability_forecast", {}) if isinstance(structural_summary, dict) else {}
    tension_axis = structural_summary.get("psychological_tension_axis") if isinstance(structural_summary, dict) else None

    try:
        stability_index = float(stability.get("stability_index", 50))
    except Exception:
        stability_index = 50.0

    try:
        burnout = float(forecast.get("burnout_2yr", 0))
    except Exception:
        burnout = 0.0

    try:
        career_shift = float(forecast.get("career_shift_3yr", 0))
    except Exception:
        career_shift = 0.0

    tension_strength = len(tension_axis) if isinstance(tension_axis, list) else 0

    if stability_index >= 65 and burnout <= 0.4:
        return "expansion_window"
    if burnout >= 0.7 and stability_index <= 50:
        return "pressure_window"
    if tension_strength >= 2 and stability_index < 60:
        return "high_drama"
    if career_shift >= 0.65:
        return "transition_window"
    return "measured_growth"


def _build_structural_executive_summary(structural_summary: dict[str, Any]) -> str:
    purpose = structural_summary.get("life_purpose_vector", {}) if isinstance(structural_summary, dict) else {}

    dominant = purpose.get("dominant_planet", "N/A")

    return f"""
[Executive Narrative Anchor]

- 당신은 한 번 마음이 움직이면 빠르게 실행으로 옮기는 편입니다.
- 다만 속도가 붙을수록 마음이 먼저 지칠 수 있어, 몰입과 단절이 번갈아 나타날 때가 있습니다.
- 반복되는 선택의 리듬을 먼저 보면, 지금 필요한 방향이 더 선명해집니다.
- 요즘은 시기 흐름(다샤)에서 힘의 초점이 바뀌는 구간이니, 덜 소모되는 선택을 먼저 찾는 것이 중요합니다.
- 당신의 힘이 모이는 버튼은 {dominant} 기질과 닿아 있습니다.

Use this anchor to keep the report human and resonant.
Never output metrics, indices, axes, probabilities, or percent values.
"""


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _band3(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "medium"


def _timing_state_from_vector(vector: dict[str, Any], stability_index: float) -> str:
    opportunity = _safe_float(vector.get("opportunity_factor", 0.5), 0.5)
    risk = _safe_float(vector.get("risk_factor", 0.5), 0.5)
    delta = opportunity - risk
    if stability_index < 45 or abs(delta) >= 0.3:
        return "volatile"
    if stability_index < 60 or abs(delta) >= 0.15:
        return "shifting"
    return "stable"


def _derive_cross_dynamics(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any],
) -> list[dict[str, str]]:
    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    dynamics: dict[str, dict[str, str]] = {}

    influence = source.get("engine", {}).get("influence_matrix", {}) if isinstance(source.get("engine"), dict) else {}
    risks = source.get("behavioral_risk_profile", {}) if isinstance(source.get("behavioral_risk_profile"), dict) else {}
    forecast = source.get("probability_forecast", {}) if isinstance(source.get("probability_forecast"), dict) else {}
    tension = source.get("psychological_tension_axis")
    tension_count = len(tension) if isinstance(tension, list) else (1 if tension else 0)
    saturn_conflict = _safe_float(influence.get("saturn_conflict_score", 0.0), 0.0)
    authority_conflict = _safe_float(risks.get("authority_conflict_risk", 0.0), 0.0)
    burnout = _safe_float(risks.get("burnout_risk", 0.0), 0.0)
    relationship_break = _safe_float(risks.get("relationship_break_risk", 0.0), 0.0)
    financial_instability = _safe_float(forecast.get("financial_instability_3yr", 0.0), 0.0)
    career_shift = _safe_float(forecast.get("career_shift_3yr", 0.0), 0.0)

    # Deterministic engine-derived dynamics only.
    def _priority_for_between(between: str, intensity: str) -> str:
        if between == "relationship-identity":
            return "must"
        if between == "career-identity":
            return "should"
        if between == "money-career":
            return "should" if intensity == "high" else "optional"
        return "optional"

    def upsert(between: str, pattern: str, score: float) -> None:
        existing = dynamics.get(between)
        intensity = _band3(score, 0.35, 0.65)
        priority = _priority_for_between(between, intensity)
        if existing is None:
            dynamics[between] = {"between": between, "pattern": pattern, "intensity": intensity, "priority": priority}
            return
        # Keep stronger signal deterministically.
        rank = {"low": 0, "medium": 1, "high": 2}
        if rank.get(intensity, 0) > rank.get(str(existing.get("intensity", "low")), 0):
            dynamics[between] = {"between": between, "pattern": pattern, "intensity": intensity, "priority": priority}

    # Primary deterministic interactions from engine-derived risks/signals.
    upsert(
        "career-identity",
        "authority_friction_vs_self_direction",
        max(authority_conflict / 10.0, saturn_conflict / 5.0),
    )
    upsert(
        "relationship-identity",
        "closeness_vs_self_protection",
        max(relationship_break / 10.0, float(tension_count) / 3.0),
    )
    upsert(
        "money-career",
        "income_pressure_vs_energy_recovery",
        max(financial_instability, burnout / 10.0),
    )
    if career_shift > 0.45 and relationship_break > 4.0:
        upsert(
            "career-relationship",
            "priority_conflict_under_transition",
            (career_shift + relationship_break / 10.0) / 2.0,
        )

    # Ensure minimum 3 baseline interactions (requested).
    baseline = [
        ("career-identity", "ambition_vs_self_doubt", 0.45 + min(max(saturn_conflict / 10.0, 0.0), 0.25)),
        ("relationship-identity", "connection_vs_self_protection", 0.45 + min(max(relationship_break / 20.0, 0.0), 0.25)),
        ("money-career", "security_vs_growth_timing", 0.45 + min(max(financial_instability / 2.0, 0.0), 0.25)),
    ]
    for between, pattern, score in baseline:
        if between not in dynamics:
            intensity = _band3(score, 0.35, 0.65)
            dynamics[between] = {
                "between": between,
                "pattern": pattern,
                "intensity": intensity,
                "priority": _priority_for_between(between, intensity),
            }

    ordered = [dynamics[k] for k in sorted(dynamics.keys())]
    return ordered[:6]


def _derive_internal_conflict_type(structural_summary: dict[str, Any]) -> str:
    source = structural_summary if isinstance(structural_summary, dict) else {}
    tension = source.get("psychological_tension_axis")
    vector = source.get("current_dasha_vector", {}) if isinstance(source.get("current_dasha_vector"), dict) else {}
    risk = _safe_float(vector.get("risk_factor", 0.5), 0.5)
    opportunity = _safe_float(vector.get("opportunity_factor", 0.5), 0.5)
    if isinstance(tension, list) and len(tension) >= 2:
        if risk > opportunity + 0.1:
            return "self_protection_vs_forward_drive"
        if opportunity > risk + 0.1:
            return "expansion_vs_internal_doubt"
        return "approach_avoidance_loop"
    if risk > 0.7:
        return "caution_vs_expression"
    if opportunity > 0.7:
        return "speed_vs_stability"
    return "consistency_vs_variation"


def _build_chapter_tone_hints() -> dict[str, str]:
    return {
        "executive_summary": "high-clarity high-recognition",
        "purushartha_profile": "reflective priority-balance",
        "psychological_architecture": "inner-motion plain-language",
        "behavioral_risks": "pattern-warning concise",
        "karmic_patterns": "repeat-loop emotionally-direct",
        "stability_metrics": "steadying practical",
        "personality_vector": "reaction-style grounded",
        "life_timeline_interpretation": "time-arc concrete",
        "career_and_success": "expansion-with-friction",
        "love_and_relationships": "attachment-with-boundary",
        "health_and_body_patterns": "body-rhythm calming",
        "confidence_and_forecast": "forward-clarity tempered",
        "remedies_and_program": "small-actions low-friction",
        "final_summary": "firm-close warm-depth",
        "appendix_optional": "minimal-supportive",
    }


def _build_timing_windows_safe_a(
    dasha_context: dict[str, Any],
    current_vector: dict[str, Any],
    stability_state: str,
) -> list[dict[str, str]] | None:
    ctx = dasha_context if isinstance(dasha_context, dict) else {}
    vector = current_vector if isinstance(current_vector, dict) else {}
    years = ctx.get("year_horizon")
    if not isinstance(years, list) or len(years) < 2:
        current_year = _safe_float(ctx.get("current_year", datetime.now().year), float(datetime.now().year))
        years = [int(current_year), int(current_year) + 1, int(current_year) + 2, int(current_year) + 3]
    try:
        years = [int(y) for y in years[:4]]
    except Exception:
        return None
    if len(years) < 2:
        return None

    opportunity = _safe_float(vector.get("opportunity_factor", 0.5), 0.5)
    risk = _safe_float(vector.get("risk_factor", 0.5), 0.5)
    dominant = str(vector.get("dominant_axis") or "").lower()
    dominant_theme = str(vector.get("current_theme") or "").lower()
    pressure_level = str(vector.get("pressure_level") or "").lower()
    activation = str(vector.get("activation_intensity") or "").lower()

    base_domain = "general"
    if any(k in dominant for k in ("career", "authority", "work")) or any(k in dominant_theme for k in ("career", "authority")):
        base_domain = "career"
    elif any(k in dominant for k in ("relationship", "partner", "marriage")) or any(k in dominant_theme for k in ("relationship", "partner")):
        base_domain = "relationship"
    elif any(k in dominant for k in ("money", "resource", "finance")) or any(k in dominant_theme for k in ("money", "resource", "finance")):
        base_domain = "money"
    elif any(k in dominant for k in ("health", "body", "stress")) or any(k in dominant_theme for k in ("health", "body", "stress")):
        base_domain = "health"

    if opportunity - risk > 0.15:
        primary_theme = "expansion-recalibration"
    elif risk - opportunity > 0.15:
        primary_theme = "pressure-management"
    elif stability_state == "volatile":
        primary_theme = "restructure"
    else:
        primary_theme = "stabilization"

    primary_intensity = "high" if activation == "high" or pressure_level == "elevated" else "medium"
    secondary_intensity = "medium" if primary_intensity == "high" else "low"
    secondary_theme = "recalibration" if primary_theme in {"pressure-management", "restructure"} else "stabilization"

    windows: list[dict[str, str]] = [
        {
            "window": f"{years[0]}_H1~{years[1]}_H1",
            "domain": base_domain,
            "theme": primary_theme,
            "intensity": primary_intensity,
        },
        {
            "window": f"{years[1]}_H2~{years[2]}_H1",
            "domain": "general" if base_domain != "general" else "career",
            "theme": secondary_theme,
            "intensity": secondary_intensity,
        },
    ]
    if len(years) >= 4 and stability_state in {"shifting", "volatile"}:
        windows.append(
            {
                "window": f"{years[2]}_H2~{years[3]}_H1",
                "domain": "money" if base_domain != "money" else "relationship",
                "theme": "stabilization" if risk <= opportunity else "restructure",
                "intensity": "low" if primary_intensity == "medium" else "medium",
            }
        )

    # Deterministic ordering:
    # intensity(high > medium > low) -> domain priority -> nearest window start
    intensity_rank = {"high": 0, "medium": 1, "low": 2}
    domain_rank = {"career": 0, "relationship": 1, "money": 2, "health": 3, "general": 4}

    def _start_year(window_text: str) -> int:
        m = re.search(r"(20\d{2})", window_text or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 9999
        return 9999

    windows = sorted(
        windows,
        key=lambda w: (
            intensity_rank.get(str(w.get("intensity", "low")).lower(), 9),
            domain_rank.get(str(w.get("domain", "general")).lower(), 9),
            _start_year(str(w.get("window", ""))),
        ),
    )
    return windows[:4] if windows else None


def build_relationship_signal_context(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> dict[str, Any]:
    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}

    stability = source.get("stability_metrics", {}) if isinstance(source.get("stability_metrics"), dict) else {}
    risks = source.get("behavioral_risk_profile", {}) if isinstance(source.get("behavioral_risk_profile"), dict) else {}
    forecast = source.get("probability_forecast", {}) if isinstance(source.get("probability_forecast"), dict) else {}
    purpose = source.get("life_purpose_vector", {}) if isinstance(source.get("life_purpose_vector"), dict) else {}
    vector = source.get("current_dasha_vector", {}) if isinstance(source.get("current_dasha_vector"), dict) else {}
    engine = source.get("engine", {}) if isinstance(source.get("engine"), dict) else {}
    influence = engine.get("influence_matrix", {}) if isinstance(engine.get("influence_matrix"), dict) else {}
    house_clusters = engine.get("house_clusters", {}) if isinstance(engine.get("house_clusters"), dict) else {}
    cluster_scores = house_clusters.get("cluster_scores", {}) if isinstance(house_clusters.get("cluster_scores"), dict) else {}

    stability_index = _safe_float(stability.get("stability_index", 50.0), 50.0)
    risk_factor = _safe_float(vector.get("risk_factor", 0.5), 0.5)
    opportunity_factor = _safe_float(vector.get("opportunity_factor", 0.5), 0.5)
    burnout_risk = _safe_float(risks.get("burnout_risk", 5.0), 5.0)
    emotional_volatility = _safe_float(risks.get("emotional_volatility", 5.0), 5.0)
    saturn_conflict = _safe_float(influence.get("saturn_conflict_score", 0.0), 0.0)
    dusthana_pressure = max(
        _safe_float(cluster_scores.get(6, 0.0), 0.0),
        _safe_float(cluster_scores.get(8, 0.0), 0.0),
        _safe_float(cluster_scores.get(12, 0.0), 0.0),
    )
    activation_intensity = (
        vector.get("activation_intensity")
        or ("high" if opportunity_factor > 0.75 else "moderate" if opportunity_factor > 0.55 else "low")
    )
    pressure_level = (
        vector.get("pressure_level")
        or ("elevated" if risk_factor > 0.7 else "contained")
    )
    timing_state = _timing_state_from_vector(vector, stability_index)
    timing_windows = _build_timing_windows_safe_a(timing, vector, timing_state)
    timing_axis: dict[str, Any] = {
        "current_phase": timing.get("timeframe_label") or timing.get("current_dasha_theme") or timing.get("label") or "current_cycle",
        "activation_intensity": activation_intensity,
        "pressure_level": pressure_level,
        "dominant_theme": vector.get("current_theme") or vector.get("dominant_axis") or source.get("psychological_tension_axis"),
        "stability_vs_change": timing_state,
    }
    if timing_windows:
        timing_axis["timing_windows"] = timing_windows

    context = {
        "identity_axis": {
            "dominant_force": purpose.get("dominant_planet") or source.get("planetary_dominance"),
            "tension_core": source.get("psychological_tension_axis"),
            "self_consistency": _band3(stability_index / 100.0, 0.45, 0.65),
            "internal_conflict_type": _derive_internal_conflict_type(source),
        },
        "career_axis": {
            "expansion_potential": _band3(_safe_float(forecast.get("career_shift_3yr", 0.5), 0.5), 0.4, 0.7),
            "authority_friction": _band3(_safe_float(risks.get("authority_conflict_risk", 5.0), 5.0) / 10.0, 0.35, 0.65),
            "burnout_pressure": _band3(_safe_float(risks.get("burnout_risk", 5.0), 5.0) / 10.0, 0.35, 0.65),
        },
        "relationship_axis": {
            "attachment_intensity": _band3(_safe_float(signals.get("attachment_score", 0.5), 0.5), 0.35, 0.65),
            "conflict_trigger_level": _band3(_safe_float(risks.get("relationship_break_risk", 5.0), 5.0) / 10.0, 0.35, 0.65),
            "repair_capacity": _band3(stability_index / 100.0, 0.45, 0.7),
        },
        "money_axis": {
            "instability_pressure": _band3(_safe_float(forecast.get("financial_instability_3yr", 0.5), 0.5), 0.35, 0.65),
            "control_reactivity": _band3(_safe_float(signals.get("money_control_score", 0.5), 0.5), 0.35, 0.65),
            "growth_readiness": _band3(max(opportunity_factor - risk_factor + 0.5, 0.0), 0.4, 0.7),
        },
        "stability_axis": {
            "base_strength": _band3((stability_index / 100.0 + max(0.0, 1.0 - saturn_conflict / 5.0)) / 2.0, 0.45, 0.65),
            "volatility": _band3((risk_factor + emotional_volatility / 10.0 + dusthana_pressure / 10.0) / 3.0, 0.35, 0.65),
            "recovery_speed": _band3(max(0.0, 1.0 - ((burnout_risk / 10.0 + saturn_conflict / 5.0) / 2.0)), 0.35, 0.65),
        },
        "timing_axis": timing_axis,
        "cross_dynamics": _derive_cross_dynamics(source, signals),
        "chapter_tone_hints": _build_chapter_tone_hints(),
    }
    return context


def build_relationship_compact_context(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> dict[str, Any]:
    # Backward-compatible alias.
    return build_relationship_signal_context(structural_summary, semantic_signals, dasha_context)


def audit_llm_output(response_text: str, structural_summary: dict[str, Any]) -> dict[str, Any]:
    """Korean-aware structural audit for generated LLM output (log-only)."""
    text = response_text if isinstance(response_text, str) else ""
    summary = structural_summary if isinstance(structural_summary, dict) else {}
    current_dasha = summary.get("current_dasha_vector", {}) if isinstance(summary.get("current_dasha_vector"), dict) else {}
    dominant_axis = current_dasha.get("dominant_axis")

    try:
        risk_factor = float(current_dasha.get("risk_factor", 0.0))
    except Exception:
        risk_factor = 0.0
    try:
        opportunity_factor = float(current_dasha.get("opportunity_factor", 0.0))
    except Exception:
        opportunity_factor = 0.0

    dominant_keywords = ["지배", "핵심 축", "주도 에너지", "강하게 작용", "중심 흐름", "axis"]
    stability_keywords = ["안정성", "기반", "균형", "흐름의 안정", "기초 체력"]
    risk_keywords = ["주의", "경고", "긴장", "압박", "위험"]
    optimistic_keywords = ["호재", "확장", "성장", "기회", "상승"]
    pessimistic_keywords = ["위기", "추락", "붕괴", "강한 충돌", "손실"]
    boilerplate_markers = ["AI로서", "모든 사람은 다르다", "참고용입니다", "일반적인 해석"]
    structural_refs = ["구조", "패턴", "흐름", "에너지", "축", "주기"]

    dominant_present = True
    if dominant_axis:
        dominant_present = any(keyword in text for keyword in dominant_keywords)
    stability_present = any(keyword in text for keyword in stability_keywords)

    risk_ack = True
    if risk_factor > 0.6:
        risk_ack = any(keyword in text for keyword in risk_keywords)

    missing_anchor = (not dominant_present) or (not stability_present)

    optimistic_count = sum(text.count(keyword) for keyword in optimistic_keywords)
    pessimistic_count = sum(text.count(keyword) for keyword in pessimistic_keywords)
    tone_inconsistency = False
    if opportunity_factor > 0.7 and pessimistic_count >= optimistic_count + 2:
        tone_inconsistency = True
    if risk_factor > 0.7 and optimistic_count >= pessimistic_count + 2:
        tone_inconsistency = True
    if not risk_ack:
        tone_inconsistency = True

    boilerplate_detected = any(marker in text for marker in boilerplate_markers)

    heading_count = text.count("##")
    text_length = len(text)
    structural_ref_count = sum(1 for keyword in structural_refs if keyword in text)
    low_density = (heading_count < 6) or (text_length < 2800) or (structural_ref_count < 2)

    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if s.strip()]
    sentence_counts: dict[str, int] = {}
    for sentence in sentences:
        if len(sentence) <= 20:
            continue
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
    repetition_issue = any(count > 2 for count in sentence_counts.values())

    structural_integrity_score = 100
    if missing_anchor:
        structural_integrity_score -= 40
    if not risk_ack:
        structural_integrity_score -= 20
    structural_integrity_score = max(0, min(100, structural_integrity_score))

    tone_alignment_score = 100 if not tone_inconsistency else 45
    boilerplate_score = 100 if not boilerplate_detected else 35
    density_score = 100 if not low_density else 40
    repetition_score = 100 if not repetition_issue else 45

    overall_score = int(round(
        structural_integrity_score * 0.30
        + tone_alignment_score * 0.25
        + density_score * 0.20
        + boilerplate_score * 0.15
        + repetition_score * 0.10
    ))

    return {
        "structural_integrity_score": int(structural_integrity_score),
        "tone_alignment_score": int(tone_alignment_score),
        "boilerplate_score": int(boilerplate_score),
        "density_score": int(density_score),
        "repetition_score": int(repetition_score),
        "overall_score": int(max(0, min(100, overall_score))),
        "flags": {
            "missing_anchor": bool(missing_anchor),
            "tone_inconsistency": bool(tone_inconsistency),
            "boilerplate_detected": bool(boilerplate_detected),
            "low_density": bool(low_density),
            "repetition_issue": bool(repetition_issue),
        },
    }


def _sanitize_percent_phrasing_ko(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    # Replace ratio-percent phrasing with a non-percent practical phrase.
    text = re.sub(r"\b\d{1,3}%\b", "일정 몫(예: 월 5만원부터)", text)
    return text


def _sanitize_meta_report_phrasing_ko(text: str) -> str:
    """Replace report-meta phrases with narrative-safe phrasing (no deletion)."""
    if not isinstance(text, str) or not text:
        return text
    replacements = {
        "이 리포트는": "이 흐름은",
        "본 해석은": "지금의 흐름은",
        "이 보고서는": "당신의 삶은",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _split_sentences_ko(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(
        r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+|(?<=…)\s+",
        raw,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _extract_heading_positions(text: str) -> list[tuple[int, str]]:
    """
    Returns list of (line_index, raw_heading_text).
    raw_heading_text: ## 마커 제거 후 원문 그대로 반환.
    chapter key 정규화는 호출하지 않음.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")
    positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw:
            positions.append((idx, raw))
    return positions


def _normalize_chapter_key(token: str, title: str, section_index: int) -> str:
    active = _active_report_chapters()
    token_raw = (token or "").strip()
    title_raw = (title or "").strip()
    if token_raw:
        if token_raw.isdigit():
            idx = int(token_raw) - 1
            if 0 <= idx < len(active):
                return active[idx]
        for key in active:
            if token_raw.lower() == key.lower():
                return key
    if title_raw:
        title_l = title_raw.lower()
        for key in active:
            if title_l == key.lower() or key.lower() in title_l or title_l in key.lower():
                return key
        m = re.match(r"^\s*(\d+)\.\s*", title_raw)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(active):
                return active[idx]
    if 0 <= section_index < len(active):
        return active[section_index]
    return active[-1]


def _fallback_three_paragraphs() -> list[str]:
    return [
        "지금은 이 주제를 크게 단정하기보다 흐름을 정리하는 편이 좋습니다.",
        "반복되는 패턴을 먼저 보되, 무리하지 않게 속도를 조절하세요.",
        "작게 확인하면서 쌓아가면 방향이 더 또렷해집니다.",
    ]


def _select_fallback_single_paragraph(
    *,
    chapter_key: str,
    fallback_salt: str,
    used_indices: set[int] | None,
    used_texts: set[str] | None,
) -> str:
    pool = _FALLBACK_PARAGRAPH_POOL
    if not pool:
        return "지금은 결론을 서두르기보다 흐름을 차분히 살펴보는 편이 좋습니다."
    seed = f"{fallback_salt}|{chapter_key}".encode("utf-8", errors="ignore")
    start = int(hashlib.sha256(seed).hexdigest()[:8], 16) % len(pool)
    if used_texts is None:
        used_texts = set()
    if used_indices is not None:
        for offset in range(len(pool)):
            idx = (start + offset) % len(pool)
            candidate = pool[idx]
            if idx not in used_indices and candidate not in used_texts:
                used_indices.add(idx)
                used_texts.add(candidate)
                return candidate
    # If base pool is exhausted, build deterministic two-sentence variants to avoid duplication spread.
    tail_pool = _FALLBACK_TAIL_POOL
    if tail_pool:
        mix_seed = f"{fallback_salt}|{chapter_key}|fallback_mix".encode("utf-8", errors="ignore")
        mix_start = int(hashlib.sha256(mix_seed).hexdigest()[:8], 16)
        total = len(pool) * len(tail_pool)
        for step in range(total):
            mix_idx = (mix_start + step) % total
            base = pool[mix_idx % len(pool)]
            tail = tail_pool[(mix_idx // len(pool)) % len(tail_pool)]
            candidate = f"{base} {tail}"
            if candidate not in used_texts:
                used_texts.add(candidate)
                return candidate
    candidate = pool[start]
    used_texts.add(candidate)
    return candidate


def _fallback_single_paragraph(
    *,
    chapter_key: str = "",
    fallback_salt: str = "",
    used_indices: set[int] | None = None,
    used_texts: set[str] | None = None,
) -> str:
    return _select_fallback_single_paragraph(
        chapter_key=chapter_key,
        fallback_salt=fallback_salt,
        used_indices=used_indices,
        used_texts=used_texts,
    )


def _ensure_min_paragraphs(
    body: str,
    min_paragraphs: int = 2,
    max_paragraphs: int = 4,
    *,
    chapter_key: str = "",
    fallback_salt: str = "",
    used_fallback_indices: set[int] | None = None,
    used_fallback_texts: set[str] | None = None,
) -> list[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
    if not raw_paragraphs:
        return [
            _fallback_single_paragraph(
                chapter_key=chapter_key,
                fallback_salt=fallback_salt,
                used_indices=used_fallback_indices,
                used_texts=used_fallback_texts,
            )
        ]

    paragraphs = raw_paragraphs[:max_paragraphs]
    if len(paragraphs) >= min_paragraphs:
        return paragraphs

    # If a single long paragraph exists, try sentence split once before fallback injection.
    first = paragraphs[0]
    sents = _split_sentences_ko(first)
    if len(sents) >= 2:
        pivot = max(1, min(2, len(sents) // 2))
        p1 = " ".join(sents[:pivot]).strip()
        p2 = " ".join(sents[pivot:]).strip()
        out = [p1] if p1 else []
        if p2:
            out.append(p2)
        while len(out) < min_paragraphs:
            out.append(
                _fallback_single_paragraph(
                    chapter_key=chapter_key,
                    fallback_salt=fallback_salt,
                    used_indices=used_fallback_indices,
                    used_texts=used_fallback_texts,
                )
            )
        return out[:max_paragraphs]

    # Sentence split not possible: fill up to minimum with compact fallback paragraphs.
    while len(paragraphs) < min_paragraphs:
        paragraphs.append(
            _fallback_single_paragraph(
                chapter_key=chapter_key,
                fallback_salt=fallback_salt,
                used_indices=used_fallback_indices,
                used_texts=used_fallback_texts,
            )
        )
    return paragraphs[:max_paragraphs]


def _enforce_three_paragraphs(body: str) -> list[str]:
    sentences = _split_sentences_ko(body)
    if not sentences:
        return _fallback_three_paragraphs()
    if len(sentences) > 6:
        sentences = sentences[:6]
    while len(sentences) < 3:
        sentences.append(_fallback_three_paragraphs()[len(sentences)])

    n = len(sentences)
    if n <= 3:
        groups = [[sentences[0]], [sentences[1]], [sentences[2]]]
    elif n == 4:
        groups = [sentences[:2], [sentences[2]], [sentences[3]]]
    elif n == 5:
        groups = [sentences[:2], sentences[2:4], [sentences[4]]]
    else:
        groups = [sentences[:2], sentences[2:4], sentences[4:6]]

    out: list[str] = []
    for group in groups:
        paragraph = " ".join(x.strip() for x in group if x and x.strip()).strip()
        out.append(paragraph if paragraph else _fallback_three_paragraphs()[len(out)])
    return out


def _ensure_first_paragraph_three_sentences(key: str, paragraphs: list[str]) -> list[str]:
    target_keys = {"Career & Money", "Love & Relationship Patterns", "Risk Management Points"}
    if key not in target_keys or not paragraphs:
        return paragraphs
    first = paragraphs[0] if isinstance(paragraphs[0], str) else ""
    sentences = _split_sentences_ko(first)
    if not sentences:
        return paragraphs
    if len(sentences) == 3:
        return paragraphs
    if len(sentences) > 3:
        paragraphs[0] = " ".join(sentences[:3]).strip()
        return paragraphs

    # len(sentences) < 3: keep meaning and only add short bridge sentence(s), no reinterpretation.
    bridge_by_key = {
        "Risk Management Points": [
            "불안이 올라올 때는 속도를 늦추는 선택이 도움이 됩니다.",
            "오늘 당장 가능한 작은 확인부터 시작해도 충분합니다.",
        ],
        "Love & Relationship Patterns": [
            "관계에서는 확인의 속도를 늦추면 마음이 덜 흔들립니다.",
            "이번에는 반응보다 표현을 먼저 골라보는 편이 맞습니다.",
        ],
        "Career & Money": [
            "일에서는 완벽보다 리듬을 먼저 지키는 쪽이 오래 갑니다.",
            "지금은 큰 결정보다 작은 전환을 먼저 확인해도 좋습니다.",
        ],
    }
    bridges = bridge_by_key.get(key, ["지금은 작은 확인이 큰 차이를 만듭니다."])
    while len(sentences) < 3:
        idx = len(sentences) - 1
        candidate = bridges[idx] if idx < len(bridges) else bridges[-1]
        sentences.append(candidate)
    paragraphs[0] = " ".join(sentences[:3]).strip()
    return paragraphs


def _resolve_min_chars_by_phase() -> int:
    phase = (os.getenv("LLM_LEN_PHASE", "1") or "1").strip()
    if phase == "2":
        return int(os.getenv("LLM_MIN_CHARS_PER_CHAPTER_PHASE2", "900"))
    return int(os.getenv("LLM_MIN_CHARS_PER_CHAPTER_PHASE1", "700"))


def _extract_tail_bullet_block(body_text: str, chapter_key: str) -> tuple[str, str]:
    """Extract only end-of-chapter bullet block; keep mid-body lists in prose."""
    if not isinstance(body_text, str) or not body_text.strip():
        return "", body_text or ""
    if chapter_key not in _ACTIONABLE_CHAPTER_KEYS:
        return "", body_text

    lines = body_text.splitlines()
    if not lines:
        return "", body_text
    tail_start = int(len(lines) * 0.70)  # last 30%
    tail = lines[tail_start:]
    bullet_positions = [i for i, ln in enumerate(tail) if _BULLET_LINE_RE.match((ln or "").strip())]
    if len(bullet_positions) < 3:
        return "", body_text

    start = bullet_positions[0]
    end = bullet_positions[-1]
    block_lines = tail[start : end + 1]
    bullet_lines = [ln for ln in block_lines if _BULLET_LINE_RE.match((ln or "").strip())]
    if len(bullet_lines) < 3:
        return "", body_text

    bullet_block = "\n".join([ln.rstrip() for ln in block_lines if ln.strip()]).strip()
    if not bullet_block:
        return "", body_text

    # remove extracted range from original tail, keep remaining prose
    rebuilt_tail = tail[:start] + tail[end + 1 :]
    prose_lines = lines[:tail_start] + rebuilt_tail
    prose_body = "\n".join(prose_lines).strip()
    return bullet_block, prose_body


def _chapter_nonspace_lengths(text: str) -> dict[str, int]:
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            heading_positions.append((idx, m.group(1).strip()))
    out: dict[str, int] = {}
    for i, (start, key) in enumerate(heading_positions):
        end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        out[key] = len(re.sub(r"\s+", "", body))
    return out


def _chapter_prose_nonspace_lengths(text: str) -> dict[str, int]:
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            heading_positions.append((idx, m.group(1).strip()))
    out: dict[str, int] = {}
    for i, (start, key) in enumerate(heading_positions):
        end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
        body_lines = lines[start + 1 : end]
        prose_lines = [ln for ln in body_lines if not _BULLET_LINE_RE.match((ln or "").strip())]
        body = "\n".join(prose_lines).strip()
        out[key] = len(re.sub(r"\s+", "", body))
    return out


def _length_violation_keys(text: str, min_chars: int) -> list[str]:
    lengths = _chapter_nonspace_lengths(text)
    return [k for k, v in lengths.items() if isinstance(v, int) and v < int(min_chars)]


def _actionable_bullet_coverage(text: str) -> tuple[int, int]:
    """Return (covered_actionable_chapters, total_actionable_chapters_present)."""
    if not isinstance(text, str) or not text.strip():
        return 0, 0
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            heading_positions.append((idx, m.group(1).strip()))
    covered = 0
    total = 0
    for i, (start, key) in enumerate(heading_positions):
        if key not in _ACTIONABLE_CHAPTER_KEYS:
            continue
        total += 1
        end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        _, prose = _extract_tail_bullet_block(body, key)
        if prose != body:
            covered += 1
    return covered, total


def normalize_llm_layout_strict(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""
    active_chapters = _active_report_chapters()

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")

    heading_positions = _extract_heading_positions(normalized_text)

    prologue = ""
    if heading_positions:
        prologue_raw = "\n".join(lines[: heading_positions[0][0]]).strip()
        if prologue_raw:
            prologue_sentences = _split_sentences_ko(prologue_raw)
            prologue = " ".join(prologue_sentences[:4]).strip()

    section_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    used_fallback_indices: set[int] = set()
    used_fallback_texts: set[str] = set()
    evidence_mode = os.getenv("LLM_EVIDENCE_MODE", "off").strip().lower()
    is_evidence_mode = evidence_mode == "on"
    fallback_salt = hashlib.sha256((normalized_text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
    for sec_idx, (start, raw_heading_text) in enumerate(heading_positions):
        end = heading_positions[sec_idx + 1][0] if sec_idx + 1 < len(heading_positions) else len(lines)
        token = ""
        title = raw_heading_text
        m = re.match(r"^\[(.*?)\]\s*(.*)$", raw_heading_text)
        if m:
            token = m.group(1).strip()
            title = m.group(2).strip()
        key = _normalize_chapter_key(token, raw_heading_text, sec_idx)
        if key and key not in title_map:
            title_map[key] = title or _SHORT_TITLE_BY_KEY.get(key, key)

        body_text = "\n".join(lines[start + 1 : end]).strip()
        if key in section_map and section_map[key].strip():
            section_map[key] = f"{section_map[key].strip()}\n{body_text}".strip()
        else:
            section_map[key] = body_text

    output_lines: list[str] = []
    if prologue:
        output_lines.append(prologue)
        output_lines.append("")

    for key in active_chapters:
        title = title_map.get(key) or _SHORT_TITLE_BY_KEY.get(key, key)
        output_lines.append(f"## [{key}] {title}")
        output_lines.append("")
        body_text = section_map.get(key, "")
        bullet_block, prose_body = _extract_tail_bullet_block(body_text, key)
        if LLM_RELAX_MODE == "phase15":
            dynamic_min_paragraphs = 1 if is_evidence_mode else 2
            # In evidence mode, keep parser from injecting fallback filler.
            if is_evidence_mode:
                used_fallback_indices.clear()
                used_fallback_texts.clear()
            paragraphs = _ensure_min_paragraphs(
                prose_body,
                min_paragraphs=dynamic_min_paragraphs,
                max_paragraphs=4,
                chapter_key=key,
                fallback_salt=fallback_salt,
                used_fallback_indices=used_fallback_indices,
                used_fallback_texts=used_fallback_texts,
            )
        else:
            paragraphs = _enforce_three_paragraphs(prose_body)
            paragraphs = _ensure_first_paragraph_three_sentences(key, paragraphs)
        for p_idx, paragraph in enumerate(paragraphs):
            output_lines.append(paragraph)
            if p_idx < len(paragraphs) - 1:
                output_lines.append("")
        if bullet_block and key not in _BULLET_EXEMPT_CHAPTER_KEYS:
            output_lines.append("")
            output_lines.append(bullet_block)
        output_lines.append("")

    final_text = "\n".join(output_lines).strip()
    return _dedupe_fallback_lines(final_text)


def _structural_layout_error_codes(text: str) -> list[str]:
    """Structural-only layout errors used for regeneration gating."""
    if not isinstance(text, str) or not text.strip():
        return ["chapter_boundary_error", "heading_missing", "empty_chapter"]
    active_chapters = _active_report_chapters()

    errors: list[str] = []
    lines = text.splitlines()
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            heading_positions.append((idx, m.group(1).strip()))

    if len(heading_positions) < len(active_chapters):
        errors.append("chapter_boundary_error")

    heading_keys = [k for _, k in heading_positions]
    if any(key not in heading_keys for key in active_chapters):
        errors.append("heading_missing")

    empty_found = False
    if heading_positions:
        for i, (start, _key) in enumerate(heading_positions):
            end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
            body = "\n".join(lines[start + 1 : end]).strip()
            if not body:
                empty_found = True
                break
    else:
        empty_found = True
    if empty_found:
        errors.append("empty_chapter")

    return errors


def _fallback_duplication_hits(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    hits = 0
    for line in _FALLBACK_PARAGRAPH_POOL:
        count = text.count(line)
        if count >= 2:
            hits += (count - 1)
    return hits


def _dedupe_fallback_lines(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""
    lines = text.splitlines()
    seen: set[str] = set()
    used_texts: set[str] = set()
    salt = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    out: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped in _FALLBACK_PARAGRAPH_POOL:
            if stripped in seen:
                replacement = _select_fallback_single_paragraph(
                    chapter_key=f"dedupe_{idx}",
                    fallback_salt=salt,
                    used_indices=None,
                    used_texts=used_texts,
                )
                out.append(replacement)
                seen.add(replacement)
                continue
            seen.add(stripped)
            used_texts.add(stripped)
        out.append(line)
    return "\n".join(out)



def build_life_timeline_prompt(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> str:
    import json

    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    compact = build_relationship_signal_context(source, signals, timing)
    timing_axis = compact.get("timing_axis", {}) if isinstance(compact, dict) else {}
    windows = timing_axis.get("timing_windows") if isinstance(timing_axis, dict) else None
    if not isinstance(windows, list):
        windows = None

    compressed = {
        "timing_axis": timing_axis,
        "cross_dynamics": compact.get("cross_dynamics", []) if isinstance(compact, dict) else [],
        "chapter_tone_hints": compact.get("chapter_tone_hints", {}) if isinstance(compact, dict) else {},
        "timing_windows_available": bool(windows),
        "timing_windows_count": len(windows) if windows else 0,
    }

    return f"""
Write ONLY the body content for the chapter "Current Phase" in Korean.
Do NOT output any heading or bullet labels.

Dasha Integrity + SAFE_A:
- Use provided timing_axis signals only.
- Do not invent planets, houses, or technical mechanics.
- Do not make deterministic event claims.
- If timing_axis.timing_windows exists, use only top 2 windows.
- If timing_axis.timing_windows is missing, do not force timing predictions.
- Describe windows as trend/activation/pressure, not guaranteed outcomes.
- For year ranges, start year must be >= current_year.

Output constraints:
- Use 2-4 paragraphs (no headings).
- Keep one blank line between paragraphs.
- Keep text concise and human-readable.
- Avoid Shock-style token stacking (year + planet + house in one line).

Context (read-only):
{json.dumps(compressed, ensure_ascii=False, indent=2)}
"""


def replace_life_timeline_block(full_text: str, new_block: str) -> str:
    if not isinstance(full_text, str) or not full_text.strip():
        return full_text
    if not isinstance(new_block, str) or not new_block.strip():
        return full_text

    block = re.sub(r"^\s*##\s+Current Phase\s*\n*", "", new_block.strip(), flags=re.IGNORECASE)
    if not block:
        return full_text

    pattern = re.compile(r"(?ms)^(##\s+Current Phase\s*$)(.*?)(?=^##\s+|\Z)")

    def _repl(match: re.Match) -> str:
        header = match.group(1)
        return f"{header}\n\n{block}\n\n"

    replaced, count = pattern.subn(_repl, full_text, count=1)
    return replaced if count > 0 else full_text


async def generate_life_timeline_chapter(
    *,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    selected_model: str,
    async_client: Any,
    build_payload_fn: Any,
    normalize_paragraphs_fn: Any,
) -> str:
    if async_client is None:
        raise RuntimeError("OpenAI client not initialized")

    prompt = build_life_timeline_prompt(
        structural_summary=structural_summary,
        semantic_signals=semantic_signals,
        dasha_context=dasha_context,
    )
    payload = build_payload_fn(
        model=selected_model,
        system_message="Follow the user prompt exactly.",
        user_message=prompt,
        max_completion_tokens=1200,
    )
    response = await asyncio.wait_for(
        async_client.chat.completions.create(**payload),
        timeout=90,
    )
    text = response.choices[0].message.content if response and response.choices else ""
    out = text if isinstance(text, str) else ""
    if not out.strip():
        raise RuntimeError("Current Phase generation returned empty text.")
    return normalize_paragraphs_fn(out, max_chars=300)


def _raw_timeline_paragraph_count(text: str) -> int:
    """Count Current Phase paragraphs from pre-normalize raw text (blank-line split)."""
    if not isinstance(text, str) or not text.strip():
        return 0
    match = re.search(
        r"(?ms)^##\s+Current Phase\s*$\n(.*?)(?=^##\s+|\Z)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return 0
    body = (match.group(1) or "").strip()
    if not body:
        return 0
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p and p.strip()]
    return len(paragraphs)


def build_executive_prompt(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> str:
    import json

    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    current_vector = source.get("current_dasha_vector", {})
    if not isinstance(current_vector, dict):
        current_vector = {}

    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    stability_index = _float_or_none((source.get("stability_metrics") or {}).get("stability_index"))
    tension_score = _float_or_none((source.get("psychological_tension_axis") or {}).get("score"))
    risk_index = _float_or_none((source.get("stability_metrics") or {}).get("risk_index"))

    narrative_risk_mode = "normal"
    if stability_index is not None and stability_index <= 45:
        narrative_risk_mode = "elevated"
    elif (
        stability_index is not None
        and tension_score is not None
        and tension_score >= 75
        and stability_index <= 60
    ):
        narrative_risk_mode = "elevated"
    elif risk_index is not None and risk_index >= 7.0:
        narrative_risk_mode = "elevated"

    axis_level = ""
    axis_coherence = source.get("axis_coherence")
    if isinstance(axis_coherence, dict):
        axis_level = str(axis_coherence.get("axis_coherence_level", "")).strip()
    saturation_band = ""
    saturation = source.get("structural_saturation")
    if isinstance(saturation, dict):
        saturation_band = str(saturation.get("band", "")).strip()
    state_label = ""
    structural_state = source.get("structural_state")
    if isinstance(structural_state, dict):
        state_label = str(structural_state.get("state_label", "")).strip()
    stability_anchor_line = ""
    if axis_level == "high" or saturation_band == "low_density" or state_label == "structural_equilibrium":
        stability_anchor_line = "흐름이 흩어지지 않은 상태라 큰 붕괴 위험은 낮습니다."

    executive_context = {
        "current_theme": current_vector.get("current_theme"),
        "dominant_axis": current_vector.get("dominant_axis"),
        "risk_factor": current_vector.get("risk_factor"),
        "opportunity_factor": current_vector.get("opportunity_factor"),
        "stability_index": stability_index,
        "tension_score": tension_score,
        "risk_index": risk_index,
        "narrative_risk_mode": narrative_risk_mode,
        "stability_anchor_line": stability_anchor_line,
        "semantic_signals": signals,
        "dasha_context": timing,
    }

    return f"""
Write ONLY the body content for the chapter "Executive Diagnosis" in Korean.
Do NOT output a chapter heading.

Rules:
Follow this structure exactly, with one blank line between blocks:

[Structural Diagnosis]
(one sentence only)

[Strengths]
- ...
- ...
- ...

[Structural Risks]
- ...
- ...
After the risks, add one sentence: "이 흐름은 조정이 가능한 영역입니다."

[Strategic Direction]
(one line only)
If stability_anchor_line is provided in context, place it immediately after Strategic Direction on its own line.
This structure overrides any previous narrative flow rules.

Style:
- Use abstract personality language; avoid repeating planet or zodiac names.
- Avoid repetitive structural keyword loops.
- Do not expose raw numeric values.
- Do not mention dates or prediction language.
- Strengths/risks should be noun-phrase bullets with soft descriptive tone.
- If narrative_risk_mode is "elevated", use clear diagnostic tone and avoid excessive softening phrases.
{_SUBTLE_VEDIC_PROMPT_RULES}

Context (read-only):
{json.dumps(executive_context, ensure_ascii=False, indent=2)}
"""


def build_single_chapter_prompt(
    chapter_key: str,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    chapter_blocks: dict[str, Any] | None,
) -> str:
    import json

    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    compact = build_relationship_signal_context(source, signals, timing)
    compact_blocks = {}
    if isinstance(chapter_blocks, dict):
        selected = _select_chapter_blocks_source(chapter_blocks)
        if isinstance(selected, dict) and chapter_key in selected:
            compact_blocks = _compact_chapter_blocks_for_prompt({chapter_key: selected.get(chapter_key)})

    context = {
        "chapter_key": chapter_key,
        "chapter_tone_hints": compact.get("chapter_tone_hints", {}) if isinstance(compact, dict) else {},
        "cross_dynamics": compact.get("cross_dynamics", []) if isinstance(compact, dict) else [],
        "chapter_blocks": compact_blocks,
    }

    return f"""
Write ONLY the body content for the chapter "{chapter_key}" in Korean.
Do NOT output any heading or labels.

Rules:
- 2-4 paragraphs total.
- Separate paragraphs with one blank line.
- Keep paragraphs concise and readable.
- Do not introduce new astrology claims or technical terms.
- Do not use prediction language or dates.
- Avoid meta/report phrasing.
- Limit advice to max 3 bullet points per chapter.
{_SUBTLE_VEDIC_PROMPT_RULES}

Context (read-only):
{json.dumps(context, ensure_ascii=False, indent=2)}
"""


def replace_executive_block(full_text: str, new_block: str) -> str:
    if not isinstance(full_text, str) or not full_text.strip():
        return full_text
    if not isinstance(new_block, str) or not new_block.strip():
        return full_text

    block = re.sub(r"^\s*##\s+Executive Diagnosis\s*\n*", "", new_block.strip(), flags=re.IGNORECASE)
    if not block:
        return full_text

    pattern = re.compile(r"(?ms)^(##\s+Executive Diagnosis\s*$)(.*?)(?=^##\s+|\Z)")

    def _repl(match: re.Match) -> str:
        header = match.group(1)
        return f"{header}\n\n{block}\n\n"

    replaced, count = pattern.subn(_repl, full_text, count=1)
    return replaced if count > 0 else full_text


def replace_chapter_block(full_text: str, chapter_key: str, new_block: str) -> str:
    if not isinstance(full_text, str) or not full_text.strip():
        return full_text
    if not isinstance(new_block, str) or not new_block.strip():
        return full_text
    key = re.escape(str(chapter_key or "").strip())
    if not key:
        return full_text

    block = re.sub(rf"^\s*##\s*(?:\[{key}\]|{key}).*\n*", "", new_block.strip(), flags=re.IGNORECASE)
    if not block:
        return full_text

    patterns = [
        re.compile(rf"(?ms)^(##\s+\[{key}\].*$)(.*?)(?=^##\s+|\Z)", re.IGNORECASE),
        re.compile(rf"(?ms)^(##\s+{key}\s*$)(.*?)(?=^##\s+|\Z)", re.IGNORECASE),
    ]
    for pattern in patterns:
        def _repl(match: re.Match) -> str:
            header = match.group(1)
            return f"{header}\n\n{block}\n\n"

        replaced, count = pattern.subn(_repl, full_text, count=1)
        if count > 0:
            return replaced
    return full_text


async def generate_executive_chapter(
    *,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    selected_model: str,
    async_client: Any,
    build_payload_fn: Any,
    normalize_paragraphs_fn: Any,
) -> str | None:
    if async_client is None:
        return None
    try:
        prompt = build_executive_prompt(
            structural_summary=structural_summary,
            semantic_signals=semantic_signals,
            dasha_context=dasha_context,
        )
        payload = build_payload_fn(
            model=selected_model,
            system_message="Follow the user prompt exactly.",
            user_message=prompt,
            max_completion_tokens=1000,
        )
        response = await asyncio.wait_for(
            async_client.chat.completions.create(**payload),
            timeout=90,
        )
        text = response.choices[0].message.content if response and response.choices else ""
        out = text if isinstance(text, str) else ""
        if not out.strip():
            return None
        return normalize_paragraphs_fn(out, max_chars=300)
    except Exception:
        return None


async def generate_single_chapter(
    *,
    chapter_key: str,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    chapter_blocks: dict[str, Any] | None,
    selected_model: str,
    async_client: Any,
    build_payload_fn: Any,
    normalize_paragraphs_fn: Any,
    max_tokens: int = 900,
) -> str | None:
    if async_client is None:
        return None
    prompt = build_single_chapter_prompt(
        chapter_key=chapter_key,
        structural_summary=structural_summary,
        semantic_signals=semantic_signals,
        dasha_context=dasha_context,
        chapter_blocks=chapter_blocks,
    )
    payload = build_payload_fn(
        model=selected_model,
        system_message="Follow the user prompt exactly.",
        user_message=prompt,
        max_completion_tokens=max_tokens,
    )
    response = await asyncio.wait_for(
        async_client.chat.completions.create(**payload),
        timeout=90,
    )
    text = response.choices[0].message.content if response and response.choices else ""
    out = text if isinstance(text, str) else ""
    if not out.strip():
        return None
    return normalize_paragraphs_fn(out, max_chars=300)


async def refine_reading_with_llm(
    *,
    async_client: Any,
    chapter_blocks: dict[str, Any],
    structural_summary: dict[str, Any],
    language: str,
    request_id: str,
    chart_hash: str,
    endpoint: str,
    max_tokens: int,
    model: str = OPENAI_MODEL,
    validate_blocks_fn: Any = None,
    build_ai_input_fn: Any = None,
    candidate_models_fn: Any = None,
    build_payload_fn: Any = None,
    emit_audit_fn: Any = None,
    normalize_paragraphs_fn: Any = None,
    compute_hash_fn: Any = None,
) -> str:
    if async_client is None:
        raise RuntimeError("OpenAI client not initialized")

    validated = validate_blocks_fn(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    structural_payload = build_ai_input_fn({"structural_summary": structural_summary})
    raw_signals = build_semantic_signals(structural_summary if isinstance(structural_summary, dict) else {})
    semantic_signals = dict(raw_signals) if isinstance(raw_signals, dict) else {}
    narrative_mode = _derive_narrative_mode(structural_summary if isinstance(structural_summary, dict) else {})
    if narrative_mode in ("expansion_window", "transition_window"):
        semantic_signals["amplification_bias"] = "positive"
    elif narrative_mode == "pressure_window":
        semantic_signals["amplification_bias"] = "cautionary"
    elif narrative_mode == "high_drama":
        semantic_signals["amplification_bias"] = "intense"
    else:
        semantic_signals["amplification_bias"] = "balanced"
    executive_summary = _build_structural_executive_summary(structural_summary if isinstance(structural_summary, dict) else {})
    dasha_context = build_dasha_narrative_context(structural_summary if isinstance(structural_summary, dict) else {})
    current_year = int(datetime.now().year)
    if not isinstance(dasha_context, dict):
        dasha_context = {}
    dasha_context = {
        **dasha_context,
        "current_year": current_year,
        "year_plus_1": current_year + 1,
        "year_plus_2": current_year + 2,
        "year_plus_3": current_year + 3,
        "year_horizon": [current_year, current_year + 1, current_year + 2, current_year + 3],
    }
    atomic_interpretations = _get_atomic_chart_interpretations(structural_summary if isinstance(structural_summary, dict) else {})
    evidence_mode = (os.getenv("LLM_EVIDENCE_MODE", "off") or "").strip().lower()
    evidence_global_chars_max = int(os.getenv("LLM_EVIDENCE_GLOBAL_CHARS_MAX", "1500"))
    evidence_global_chars_hard_max = int(os.getenv("LLM_EVIDENCE_GLOBAL_CHARS_HARD_MAX", "2500"))
    evidence_chapter_chars_min = int(os.getenv("LLM_EVIDENCE_CHAPTER_CHARS_MIN", "600"))
    evidence_chapter_chars_max = int(os.getenv("LLM_EVIDENCE_CHAPTER_CHARS_MAX", "1200"))
    evidence_total_chars_hard_max = int(os.getenv("LLM_EVIDENCE_TOTAL_CHARS_HARD_MAX", "15000"))
    hybrid_render_mode = (os.getenv("LLM_HYBRID_RENDER_MODE", "off") or "").strip().lower() == "on"
    max_evidence_chars_per_chapter = int(os.getenv("LLM_HYBRID_EVIDENCE_CHAPTER_MAX_CHARS", "1200"))
    use_evidence_pipeline_v2 = (os.getenv("EVIDENCE_PIPELINE_V2", "0") or "").strip() == "1"

    evidence_pack = {"global_evidence": [], "chapter_evidence": {}, "stats": {}}
    if evidence_mode == "on":
        evidence_pack = build_evidence_packs(
            structural_summary if isinstance(structural_summary, dict) else {},
            _active_report_chapters(),
            chapter_chars_min=evidence_chapter_chars_min,
            chapter_chars_max=evidence_chapter_chars_max,
            global_chars_max=evidence_global_chars_max,
            global_chars_hard_max=evidence_global_chars_hard_max,
            total_chars_hard_max=evidence_total_chars_hard_max,
        )
    global_evidence_items = evidence_pack.get("global_evidence", []) if isinstance(evidence_pack, dict) else []
    chapter_evidence_map = evidence_pack.get("chapter_evidence", {}) if isinstance(evidence_pack, dict) else {}
    global_evidence_items_sanitized = global_evidence_items
    if use_evidence_pipeline_v2 and isinstance(global_evidence_items, list):
        global_evidence_items_sanitized = pre_sanitize_global_evidence_items(global_evidence_items, global_cap=1500)
        logger.info(
            "GLOBAL_EVIDENCE_SANITIZED items=%s chars=%s request_id=%s",
            len(global_evidence_items_sanitized),
            _evidence_chars(global_evidence_items_sanitized),
            request_id,
        )
    if use_evidence_pipeline_v2 and isinstance(chapter_evidence_map, dict):
        chapter_evidence_map = pre_sanitize_chapter_evidence_map(chapter_evidence_map)
        chapter_evidence_map = apply_caps_to_chapter_evidence_map(
            chapter_evidence_map,
            chapter_cap=900,
            global_cap=1500,
        )
        chapter_evidence_map = apply_bullet_escape_to_chapter_evidence_map(chapter_evidence_map)
        chapter_evidence_map = await prepatch_chapter_evidence_map(
            async_client,
            chapter_evidence_map,
            logger,
            max_sentences=3,
        )
    # Request-scope guard: main prompt may include global evidence once;
    # individual regen prompts should not re-inject it.
    global_evidence_injected_count = 0
    system_message = "Follow the user prompt exactly."
    user_message = build_llm_structural_prompt(
        structural_payload,
        language=language,
        atomic_interpretations=atomic_interpretations,
        chapter_blocks=validated,
        semantic_signals=semantic_signals,
        narrative_mode=narrative_mode,
        executive_summary=executive_summary,
        dasha_context=dasha_context,
        global_evidence_items=global_evidence_items_sanitized if isinstance(global_evidence_items_sanitized, list) else None,
        global_evidence_injected_count=global_evidence_injected_count,
    )
    logger.info(
        "LLM prompt assembled request_id=%s prompt_char_length=%s context_mode=%s prompt_mode=%s",
        request_id,
        len(user_message or ""),
        (os.getenv("LLM_CONTEXT_MODE", "relationship_compact") or "").strip().lower(),
        (os.getenv("PROMPT_MODE", "analyzer_first") or "").strip().lower(),
    )
    chapter_blocks_hash = compute_hash_fn(validated)
    selected_model = str(model or OPENAI_MODEL).strip() or OPENAI_MODEL
    candidate_models = candidate_models_fn(selected_model)
    last_error: Optional[Exception] = None

    timeout_retries_left = 1
    stage = "pdf" if str(endpoint or "").strip().lower() == "/pdf" else "ai_reading"

    for candidate_model in candidate_models:
        inject_called = False
        timeline_regen_count = 0
        executive_regen_count = 0
        conditional_regen_count = 0
        conditional_regen_chapter = None
        payload = build_payload_fn(
            model=candidate_model,
            system_message=system_message,
            user_message=user_message,
            max_completion_tokens=max_tokens,
        )
        try:
            while True:
                logger.info(
                    "LLM API call started request_id=%s selected_model=%s chapter_blocks_hash=%s",
                    request_id,
                    candidate_model,
                    chapter_blocks_hash,
                )
                try:
                    response = await async_client.chat.completions.create(**payload)
                except Exception as call_err:
                    is_timeout = isinstance(call_err, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException))
                    if (not is_timeout) and "timeout" in type(call_err).__name__.lower():
                        is_timeout = True
                    if is_timeout:
                        logger.warning(
                            "TIMEOUT_OCCURRED stage=%s request_id=%s selected_model=%s chapter_blocks_hash=%s error_type=%s error=%s",
                            stage,
                            request_id,
                            candidate_model,
                            chapter_blocks_hash,
                            type(call_err).__name__,
                            str(call_err),
                        )
                        if timeout_retries_left > 0:
                            timeout_retries_left -= 1
                            logger.info(
                                "TIMEOUT_RETRY stage=%s request_id=%s selected_model=%s chapter_blocks_hash=%s retries_left=%s",
                                stage,
                                request_id,
                                candidate_model,
                                chapter_blocks_hash,
                                timeout_retries_left,
                            )
                            continue
                    raise
                break
            text = response.choices[0].message.content if response and response.choices else ""
            response_text = text if isinstance(text, str) else ""
            logger.debug(
                "LLM API response received request_id=%s selected_model=%s chapter_blocks_hash=%s response_length=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
                len(response_text),
            )
            if not response_text or not response_text.strip():
                logger.debug(
                    "LLM raw response dump request_id=%s selected_model=%s chapter_blocks_hash=%s response=%r",
                    request_id,
                    candidate_model,
                    chapter_blocks_hash,
                    response,
                )
                raise RuntimeError(
                    "LLM returned empty refinement. Model: "
                    f"{candidate_model}, finish_reason: "
                    f"{response.choices[0].finish_reason if response and response.choices else 'N/A'}"
                )
            raw_llm_text = response_text
            response_text = normalize_paragraphs_fn(response_text, max_chars=300)
            response_text = _sanitize_percent_phrasing_ko(response_text)
            response_text = _sanitize_meta_report_phrasing_ko(response_text)
            if use_evidence_pipeline_v2:
                style_audit = audit_llm_style_only(raw_llm_text)
                logger.info(
                    "[LLM STYLE V2] fail=%s reasons=%s explain_trigger_count=%s banned_set_count=%s triad_chapter_count=%s",
                    style_audit.get("fail"),
                    style_audit.get("fail_reasons"),
                    style_audit.get("explain_trigger_count"),
                    style_audit.get("banned_set_count"),
                    style_audit.get("triad_chapter_count"),
                )
            if hybrid_render_mode:
                if inject_called:
                    logger.error("inject_double_call_prevented request_id=%s", request_id)
                else:
                    response_text, _inject_stats = inject_evidence_blocks(
                        response_text,
                        chapter_evidence_map if isinstance(chapter_evidence_map, dict) else {},
                        global_evidence_items_sanitized if isinstance(global_evidence_items_sanitized, list) else [],
                        hybrid_render_mode=hybrid_render_mode,
                        pure_mode=use_evidence_pipeline_v2,
                        target_chapters=None,
                        max_evidence_chars_per_chapter=max_evidence_chars_per_chapter,
                    )
                    if use_evidence_pipeline_v2:
                        response_text, tag_hits = _strip_unresolved_evidence_tags_with_existing_re(response_text)
                        if tag_hits:
                            logger.warning(
                                "UNRESOLVED_EVIDENCE_TAGS_REMOVED count=%s request_id=%s",
                                tag_hits,
                                request_id,
                            )
                    inject_called = True
            has_timeline = ("## Current Phase" in response_text)
            timeline_raw_paragraphs = _raw_timeline_paragraph_count(response_text)
            timeline_structural_errors = _structural_layout_error_codes(response_text)
            timeline_should_regen = has_timeline and (
                any(code in {"chapter_boundary_error", "empty_chapter"} for code in timeline_structural_errors)
                or timeline_raw_paragraphs < 2
            )
            if timeline_should_regen:
                try:
                    timeline_regen_count += 1
                    timeline_text = await generate_life_timeline_chapter(
                        structural_summary=structural_summary,
                        semantic_signals=semantic_signals,
                        dasha_context=dasha_context,
                        selected_model=candidate_model,
                        async_client=async_client,
                        build_payload_fn=build_payload_fn,
                        normalize_paragraphs_fn=normalize_paragraphs_fn,
                    )
                    response_text = replace_life_timeline_block(response_text, timeline_text)
                    if hybrid_render_mode:
                        response_text, _regen_stats = inject_evidence_blocks(
                            response_text,
                            chapter_evidence_map if isinstance(chapter_evidence_map, dict) else {},
                            global_evidence_items_sanitized if isinstance(global_evidence_items_sanitized, list) else [],
                            hybrid_render_mode=hybrid_render_mode,
                            pure_mode=use_evidence_pipeline_v2,
                            target_chapters=["Current Phase"],
                            max_evidence_chars_per_chapter=max_evidence_chars_per_chapter,
                        )
                        if use_evidence_pipeline_v2:
                            response_text, tag_hits = _strip_unresolved_evidence_tags_with_existing_re(response_text)
                            if tag_hits:
                                logger.warning(
                                    "UNRESOLVED_EVIDENCE_TAGS_REMOVED count=%s request_id=%s",
                                    tag_hits,
                                    request_id,
                                )
                except Exception as timeline_err:
                    logger.warning(
                        "Current Phase isolation fallback to base text request_id=%s selected_model=%s error_type=%s error=%s",
                        request_id,
                        candidate_model,
                        type(timeline_err).__name__,
                        str(timeline_err),
                    )
            response_text = _sanitize_percent_phrasing_ko(response_text)
            response_text = _sanitize_meta_report_phrasing_ko(response_text)
            response_text = normalize_llm_layout_strict(response_text)
            if hybrid_render_mode and not use_evidence_pipeline_v2:
                response_text = apply_bridge_to_all_chapters(response_text)
            final_text = response_text
            if use_evidence_pipeline_v2:
                density_audit = audit_length_density(final_text)
                logger.info(
                    "[LLM LENGTH DENSITY V2] warn=%s warnings=%s heading_count=%s text_length=%s structural_ref_count=%s",
                    density_audit.get("warn"),
                    density_audit.get("warnings"),
                    density_audit.get("heading_count"),
                    density_audit.get("text_length"),
                    density_audit.get("structural_ref_count"),
                )
            min_chars = _resolve_min_chars_by_phase()
            length_map = _chapter_nonspace_lengths(final_text)
            prose_length_map = _chapter_prose_nonspace_lengths(final_text)
            below_min = _length_violation_keys(final_text, min_chars)
            if below_min:
                logger.warning(
                    "[LLM LENGTH] below_min_chars=%s min_chars=%s request_id=%s selected_model=%s lengths=%s prose_lengths=%s",
                    below_min,
                    min_chars,
                    request_id,
                    candidate_model,
                    length_map,
                    prose_length_map,
                )
            covered, total = _actionable_bullet_coverage(final_text)
            if total > 0:
                coverage_ratio = covered / max(total, 1)
                if coverage_ratio < 0.7:
                    logger.warning(
                        "[LLM BULLETS] warn_action_bullets_coverage_low covered=%s total=%s ratio=%.2f request_id=%s selected_model=%s",
                        covered,
                        total,
                        coverage_ratio,
                        request_id,
                        candidate_model,
                    )
            fallback_dup_hits = _fallback_duplication_hits(final_text)
            if fallback_dup_hits > 0:
                logger.warning(
                    "[LLM LAYOUT] duplicated_fallback_spread hits=%s request_id=%s selected_model=%s",
                    fallback_dup_hits,
                    request_id,
                    candidate_model,
                )
            audit_report = audit_llm_output(final_text, structural_summary)
            structural_errors = _structural_layout_error_codes(final_text)
            if (not use_evidence_pipeline_v2) and int(audit_report.get("overall_score", 0)) < 65 and structural_errors and "## Executive Diagnosis" in final_text:
                try:
                    executive_regen_count += 1
                    new_exec = await generate_executive_chapter(
                        structural_summary=structural_summary,
                        semantic_signals=semantic_signals,
                        dasha_context=dasha_context,
                        selected_model=candidate_model,
                        async_client=async_client,
                        build_payload_fn=build_payload_fn,
                        normalize_paragraphs_fn=normalize_paragraphs_fn,
                    )
                    if isinstance(new_exec, str) and new_exec.strip():
                        final_text = replace_executive_block(final_text, new_exec)
                        final_text = _sanitize_percent_phrasing_ko(final_text)
                        final_text = _sanitize_meta_report_phrasing_ko(final_text)
                        final_text = normalize_llm_layout_strict(final_text)
                        if hybrid_render_mode and not use_evidence_pipeline_v2:
                            final_text = apply_bridge_to_all_chapters(final_text)
                        audit_report = audit_llm_output(final_text, structural_summary)
                except Exception as exec_err:
                    logger.warning(
                        "Executive isolation fallback to base text request_id=%s selected_model=%s error_type=%s error=%s",
                        request_id,
                        candidate_model,
                        type(exec_err).__name__,
                        str(exec_err),
                    )
            # Conditional regen for additional chapters based on accumulated evidence.
            try:
                evidence_threshold = int(os.getenv("LLM_REGEN_EVIDENCE_THRESHOLD", "3"))
            except Exception:
                evidence_threshold = 3
            regen_env = os.getenv("LLM_REGEN_CHAPTERS", "")
            if regen_env.strip():
                allowlist = {c.strip() for c in regen_env.split(",") if c.strip()}
            else:
                allowlist = set(_CONDITIONAL_REGEN_CHAPTERS_DEFAULT)

            if below_min and conditional_regen_count < _MAX_CONDITIONAL_REGEN_PER_REQUEST:
                for key in below_min:
                    if key not in allowlist:
                        continue
                    _CHAPTER_REGEN_EVIDENCE_COUNTS[key] = _CHAPTER_REGEN_EVIDENCE_COUNTS.get(key, 0) + 1
                    if _CHAPTER_REGEN_EVIDENCE_COUNTS[key] == evidence_threshold:
                        logger.info(
                            "[LLM REGEN EVIDENCE] threshold_reached chapter=%s count=%s",
                            key,
                            _CHAPTER_REGEN_EVIDENCE_COUNTS[key],
                        )

                eligible = [
                    key
                    for key in below_min
                    if key in allowlist and _CHAPTER_REGEN_EVIDENCE_COUNTS.get(key, 0) >= evidence_threshold
                ]
                if eligible:
                    candidate_key = min(
                        eligible,
                        key=lambda k: prose_length_map.get(k, length_map.get(k, 0)),
                    )
                    new_block = await generate_single_chapter(
                        chapter_key=candidate_key,
                        structural_summary=structural_summary,
                        semantic_signals=semantic_signals,
                        dasha_context=dasha_context,
                        chapter_blocks=chapter_blocks,
                        selected_model=candidate_model,
                        async_client=async_client,
                        build_payload_fn=build_payload_fn,
                        normalize_paragraphs_fn=normalize_paragraphs_fn,
                    )
                    if isinstance(new_block, str) and new_block.strip():
                        conditional_regen_count += 1
                        conditional_regen_chapter = candidate_key
                        final_text = replace_chapter_block(final_text, candidate_key, new_block)
                        final_text = _sanitize_percent_phrasing_ko(final_text)
                        final_text = _sanitize_meta_report_phrasing_ko(final_text)
                        final_text = normalize_llm_layout_strict(final_text)
                        if hybrid_render_mode and not use_evidence_pipeline_v2:
                            final_text = apply_bridge_to_all_chapters(final_text)
                        audit_report = audit_llm_output(final_text, structural_summary)
            if conditional_regen_count:
                logger.info(
                    "[LLM REGEN EXTRA] chapter=%s count=%s request_id=%s selected_model=%s",
                    conditional_regen_chapter,
                    conditional_regen_count,
                    request_id,
                    candidate_model,
                )
            logger.info("[LLM AUDIT] score=%s flags=%s", audit_report.get("overall_score"), audit_report.get("flags"))
            logger.info(
                "[LLM REGEN] timeline=%s executive=%s conditional=%s request_id=%s selected_model=%s",
                timeline_regen_count,
                executive_regen_count,
                conditional_regen_count,
                request_id,
                candidate_model,
            )
            if int(audit_report.get("overall_score", 0)) < 75:
                logger.warning("[LLM AUDIT WARNING] Quality below threshold.")
            model_used = f"openai/{candidate_model}"
            emit_audit_fn(
                request_id=request_id,
                chart_hash=chart_hash,
                chapter_blocks_hash=chapter_blocks_hash,
                model_used=model_used,
                endpoint=endpoint,
            )
            logger.info(
                "LLM refinement executed request_id=%s selected_model=%s model_used=%s chapter_blocks_hash=%s",
                request_id,
                candidate_model,
                model_used,
                chapter_blocks_hash,
            )
            final_text = enforce_subtle_vedic_lexicon(
                final_text,
                allow_zero_term_injection=True,
            )
            return final_text
        except Exception as e:
            last_error = e
            logger.warning(
                "LLM model attempt failed request_id=%s selected_model=%s chapter_blocks_hash=%s error_type=%s error=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
                type(e).__name__,
                str(e),
            )

    raise RuntimeError(
        "LLM refinement failed for all candidate models "
        f"{candidate_models}. last_error={type(last_error).__name__ if last_error else 'N/A'}: {last_error}"
    ) from last_error


def build_llm_structural_prompt(
    structural_summary: dict,
    language: str,
    atomic_interpretations: dict[str, str] | None = None,
    chapter_blocks: dict | None = None,
    semantic_signals: dict[str, Any] | None = None,
    narrative_mode: str | None = None,
    executive_summary: str | None = None,
    dasha_context: dict[str, Any] | None = None,
    global_evidence_items: list[dict[str, Any]] | None = None,
    global_evidence_injected_count: int = 0,
) -> str:
    import json

    atomic = atomic_interpretations if isinstance(atomic_interpretations, dict) else {}
    signals = dict(semantic_signals) if isinstance(semantic_signals, dict) else {}
    timing = dict(dasha_context) if isinstance(dasha_context, dict) else {}
    source = structural_summary if isinstance(structural_summary, dict) else {}
    mode = str(narrative_mode).strip() if isinstance(narrative_mode, str) and narrative_mode.strip() else "measured_growth"
    prompt_style = (os.getenv("PROMPT_STYLE", "") or "").strip().lower()
    style_run151158 = prompt_style == "run151158_like"
    prompt_mode = (os.getenv("PROMPT_MODE", "analyzer_first") or "").strip().lower()
    context_mode = (os.getenv("LLM_CONTEXT_MODE", "relationship_compact") or "").strip().lower()
    evidence_mode = (os.getenv("LLM_EVIDENCE_MODE", "off") or "").strip().lower()
    evidence_priority = (os.getenv("LLM_EVIDENCE_PRIORITY", "evidence_only") or "").strip().lower()
    evidence_only = evidence_mode == "on" and evidence_priority == "evidence_only"
    hybrid_render_mode = (os.getenv("LLM_HYBRID_RENDER_MODE", "off") or "").strip().lower() == "on"
    min_chars = _resolve_min_chars_by_phase()
    target_chars = int(os.getenv("LLM_TARGET_CHARS_PER_CHAPTER", "1100"))
    min_anchors = int(os.getenv("LLM_MIN_ANCHORS_PER_CHAPTER", "4"))
    overview = executive_summary if isinstance(executive_summary, str) else ""
    asc_text = str(atomic.get("asc", "")).strip()
    sun_text = str(atomic.get("sun", "")).strip()
    moon_text = str(atomic.get("moon", "")).strip()
    compact_context = build_relationship_signal_context(
        structural_summary=source,
        semantic_signals=signals,
        dasha_context=timing,
    )
    compact_context_json = json.dumps(compact_context, indent=2, ensure_ascii=False)
    chapter_blocks_included = (context_mode in {"legacy", "hybrid_compact"}) and not evidence_only
    compact_mode = os.getenv("LLM_GATE_COMPACT", "0").strip() == "1"
    source_blocks = _select_chapter_blocks_source(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    active_chapters = _active_report_chapters()
    evidence_global_chars_max = int(os.getenv("LLM_EVIDENCE_GLOBAL_CHARS_MAX", "1500"))
    evidence_global_chars_hard_max = int(os.getenv("LLM_EVIDENCE_GLOBAL_CHARS_HARD_MAX", "2500"))
    evidence_chapter_chars_min = int(os.getenv("LLM_EVIDENCE_CHAPTER_CHARS_MIN", "600"))
    evidence_chapter_chars_max = int(os.getenv("LLM_EVIDENCE_CHAPTER_CHARS_MAX", "1200"))
    evidence_total_chars_hard_max = int(os.getenv("LLM_EVIDENCE_TOTAL_CHARS_HARD_MAX", "15000"))

    evidence_pack = {"global_evidence": [], "chapter_evidence": {}, "stats": {}}
    if evidence_mode == "on":
        evidence_pack = build_evidence_packs(
            source,
            active_chapters,
            chapter_chars_min=evidence_chapter_chars_min,
            chapter_chars_max=evidence_chapter_chars_max,
            global_chars_max=evidence_global_chars_max,
            global_chars_hard_max=evidence_global_chars_hard_max,
            total_chars_hard_max=evidence_total_chars_hard_max,
        )

    prompt_global_evidence_items = (
        global_evidence_items
        if isinstance(global_evidence_items, list)
        else (evidence_pack.get("global_evidence", []) if isinstance(evidence_pack, dict) else [])
    )
    chapter_evidence_map = evidence_pack.get("chapter_evidence", {}) if isinstance(evidence_pack, dict) else {}
    evidence_stats = evidence_pack.get("stats", {}) if isinstance(evidence_pack, dict) else {}
    global_evidence_text = "\n".join(
        f"- ({it.get('id','')}) {it.get('text','')}" for it in prompt_global_evidence_items if isinstance(it, dict)
    ).strip()
    chapter_evidence_lines: list[str] = []
    if isinstance(chapter_evidence_map, dict):
        for ck in active_chapters:
            items = chapter_evidence_map.get(ck, [])
            if not isinstance(items, list) or not items:
                continue
            chapter_evidence_lines.append(f"EVIDENCE[{ck}]")
            for it in items:
                if isinstance(it, dict):
                    chapter_evidence_lines.append(f"- ({it.get('id','')}) {it.get('text','')}")
            chapter_evidence_lines.append("")
    chapter_evidence_text = "\n".join(chapter_evidence_lines).strip()
    if chapter_blocks_included:
        if compact_mode:
            compact_blocks = _compact_chapter_blocks_for_prompt(source_blocks)
            blocks_json = json.dumps(compact_blocks, indent=2, ensure_ascii=False) if compact_blocks else "{}"
        else:
            blocks_json = json.dumps(source_blocks, indent=2, ensure_ascii=False) if source_blocks else "{}"
    else:
        blocks_json = "{}"
    context_blocks_chars = len(blocks_json) if chapter_blocks_included else 0
    chapter_key_lines = "\n".join(f"- {key}" for key in active_chapters)
    intensity_dist: dict[str, int] = {}
    for item in compact_context.get("cross_dynamics", []) if isinstance(compact_context, dict) else []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("intensity", "")).strip() or "unknown"
        intensity_dist[level] = intensity_dist.get(level, 0) + 1
    timing_windows = None
    if isinstance(compact_context, dict):
        timing_axis = compact_context.get("timing_axis")
        if isinstance(timing_axis, dict):
            timing_windows = timing_axis.get("timing_windows")
    timing_windows_count = len(timing_windows) if isinstance(timing_windows, list) else 0
    effective_global_injected = int(global_evidence_injected_count or (1 if (evidence_mode == "on" and global_evidence_text) else 0))
    logger.info(
        "LLM prompt context mode=%s prompt_mode=%s evidence_mode=%s priority=%s blocks_injected=%s context_blocks_chars=%s relationship_signal_context_length=%s cross_dynamics_count=%s cross_dynamics_intensity=%s timing_windows_count=%s global_evidence_injected_count=%s",
        context_mode,
        prompt_mode,
        evidence_mode,
        evidence_priority,
        chapter_blocks_included,
        context_blocks_chars,
        len(compact_context_json),
        len(compact_context.get("cross_dynamics", [])) if isinstance(compact_context, dict) else 0,
        intensity_dist,
        timing_windows_count,
        effective_global_injected,
    )
    if effective_global_injected > 1:
        logger.warning("[LLM EVIDENCE] global_evidence_injected_count_exceeded=%s", effective_global_injected)
    if evidence_mode == "on":
        logger.info(
            "LLM evidence stats mode=%s priority=%s global_items=%s global_chars=%s total_chars=%s missing_ids=%s chapter_evidence_count=%s chapter_evidence_char_count=%s fallback_used=%s reused_in_final=%s evidence_trim_level=%s",
            evidence_mode,
            evidence_priority,
            len(prompt_global_evidence_items),
            _evidence_chars(prompt_global_evidence_items),
            int(evidence_stats.get("total_chars", 0)) if isinstance(evidence_stats, dict) else 0,
            (evidence_stats.get("missing_ids", []) if isinstance(evidence_stats, dict) else [])[:10],
            evidence_stats.get("chapter_evidence_count", {}) if isinstance(evidence_stats, dict) else {},
            evidence_stats.get("chapter_evidence_char_count", {}) if isinstance(evidence_stats, dict) else {},
            evidence_stats.get("fallback_used", {}) if isinstance(evidence_stats, dict) else {},
            evidence_stats.get("reused_in_final", []) if isinstance(evidence_stats, dict) else [],
            evidence_stats.get("evidence_trim_level", 0) if isinstance(evidence_stats, dict) else 0,
        )
        if _evidence_chars(prompt_global_evidence_items) < 400:
            logger.warning("[LLM EVIDENCE] warn_evidence_global_chars_low chars=%s", _evidence_chars(prompt_global_evidence_items))
        if int(evidence_stats.get("total_chars", 0)) > evidence_total_chars_hard_max:
            logger.warning("[LLM EVIDENCE] warn_evidence_total_chars_overflow_guard_applied total=%s hard=%s", int(evidence_stats.get("total_chars", 0)), evidence_total_chars_hard_max)
        if isinstance(chapter_evidence_map, dict):
            low_char_keys = []
            for ck, items in chapter_evidence_map.items():
                if _evidence_chars(items if isinstance(items, list) else []) < evidence_chapter_chars_min:
                    low_char_keys.append(ck)
            if low_char_keys:
                logger.warning(
                    "[LLM EVIDENCE] warn_evidence_chapter_chars_low keys=%s min=%s",
                    low_char_keys,
                    evidence_chapter_chars_min,
                )
        no_keys = evidence_stats.get("no_evidence_keys", []) if isinstance(evidence_stats, dict) else []
        low_density_keys = evidence_stats.get("low_evidence_density_keys", []) if isinstance(evidence_stats, dict) else []
        if no_keys:
            logger.warning("[LLM EVIDENCE] no_evidence_injected keys=%s", no_keys)
        if low_density_keys:
            logger.warning("[LLM EVIDENCE] low_evidence_density_injected keys=%s", low_density_keys)
    style_override_block = ""
    if style_run151158:
        style_override_block = """
STYLE OVERRIDE (run151158_like)
- 목표: 대중형/즉시공감형. 해석보다 즉시 이해를 우선한다.
- 고정 수사 순서를 강제하지 않는다. (관찰->공감->패턴->통찰->선택지 반복 금지)
- 조언 문장을 기본 종결로 쓰지 않는다. 일부 챕터는 통찰형/선언형으로 끝내도 된다.
- 라벨성 전환 문구를 피한다:
  "공감하자면", "패턴적으로", "진술적으로", "문장으로 말하자면", "선택지:"
- 챕터 본문에서 다음과 같은 콜론 라벨 문장을 만들지 않는다:
  "공감:", "통찰:", "패턴:", "권장:", "요약:"
- 설명형 메타문장보다 장면형/상황형 문장을 우선한다.
- 문장을 어렵게 압축하지 말고, 처음 읽을 때 바로 이해되게 쓴다.
- 문단 길이는 과도하게 늘리지 않는다. 짧고 분명한 문단을 우선한다.
- 같은 완화형 종결을 반복하지 않는다:
  "지금은 ... 편이 좋습니다", "무리하지 말고 ...", "당장은 ..."
- 본문에서 "선택지:" 라벨을 사용하지 않는다. 조언이 필요하면 일반 문장으로 짧게 포함한다.
- 조언형 종결(~해보세요/~좋습니다/~유리합니다)을 연속 챕터에서 반복하지 않는다.
- 챕터 시작부에서 "지금은...", "당장..."으로 여는 문장을 반복하지 않는다.
"""

    hybrid_output_contract = ""
    sales_tone_contract = ""
    jargon_transform_rules = """
[전문용어 변환 규칙 — 반드시 준수]
※ 적용 범위: Hook / Bridge / Bullets 에만 적용.
※ 점성학 앵커(행성/하우스/라시/낙샤트라/다샤/요가)는 <EVIDENCE_BLOCK>에서만 사용.
※ 앵커 최소 4개 요건은 Evidence 단락에서 충족하며 Hook/Bridge/Bullets에서 별도 충족 불필요.

- 라그나 로드 / 상승궁 지배성 -> "당신의 핵심 에너지", "삶을 이끄는 힘"
- 켄드라(1,4,7,10하우스) -> "삶의 주요 무대", "외부로 드러나는 영역"
- 트리코나(1,5,9하우스) -> "타고난 흐름", "자연스러운 재능의 방향"
- 우파차야(3,6,10,11하우스) -> "시간이 지날수록 강해지는 구조"
- 두스타나(6,8,12하우스) -> "반복되는 위기 패턴", "숨겨진 긴장"
- 연소(Combust) -> "에너지가 눌린 상태", "잠시 빛이 가려진 시기"
- 흉성 과다 / 악성 행성 -> "외부 압박이 집중되는 구조"
- 다샤 전환 -> "삶의 흐름이 바뀌는 구간"
- 시데리얼 / 라히리 기준 -> Hook/Bridge/Bullets에서는 금지(Evidence에서만 허용)
- 아바스타 -> 사용 금지
- 요가(Yoga) -> "특정 결합이 만드는 패턴"처럼 풀어쓰기
"""
    chapter_hook_hint_block = """
[챕터별 Hook 감정 힌트]
- Career & Money: "책임은 늘었는데 인정은 부족한 느낌", "잘하고 있는데 왜 불안한지 모르는 상태"
- Risk Management Points: "안정을 원하면서도 변화가 두려운 역설", "기반을 다지려 할수록 흔들리는 느낌"
- Love & Relationship Patterns: "가까워질수록 오히려 어색해지는 패턴", "관계에서 반복되는 같은 상처"
- Recurring Patterns: "분명히 알면서도 또 같은 선택을 하는 자신", "끊고 싶은데 끊기지 않는 반복"
- Health & Energy Rhythm: "머리는 괜찮다고 하는데 몸이 먼저 신호를 보내는 상황", "에너지가 갑자기 바닥나는 패턴"
- Mid-Term Direction: "잘 될 것 같으면서도 확신이 없는 상태", "준비는 됐는데 시작을 못 하는 느낌"
- Core Disposition: "겉으로는 괜찮아 보이지만 안에서 다른 목소리가 들리는 상태"
- Executive Diagnosis: "내가 어떤 사람인지 알 것 같으면서도 모르는 느낌"
- Current Phase: "지금 이 시기가 전환점인 것 같은 막연한 감각"
"""
    anti_repeat_rules = """
[반복 구조 금지]
- 각 챕터에서 "정의 -> 강점 -> 리스크 -> 조언" 순서를 반복하지 않는다.
- Hook은 매 챕터마다 다른 감정/상황에서 시작한다. 같은 도입 문장 패턴 반복 금지.
- "~할 수 있다", "~가능성이 있다", "~경향이 있다" 종결을 연속 2회 이상 쓰지 않는다.
- Bullets 3개가 모두 명령형으로 끝나는 구조 금지. 최소 1개는 질문형 또는 관찰형.
"""
    explanation_mode_ban = """
[설명 모드 금지 — 반드시 준수]
- Evidence 내용을 다시 정의하거나 이론 설명하지 않는다.
- "~는 ~을 의미한다", "~를 가리킨다", "~라고 본다", "~는 ~한 결합이다" 형태 금지.
- Evidence를 요약하지 말고 독자의 현재 삶에 바로 연결한다.
- "이 패턴은 당신에게…", "지금 당신의 상황에서 이것은…" 형태로 적용 중심으로 쓴다.
- 점성학 이론을 가르치지 말고 독자의 현재 상황을 해석한다.
- Hook/Bridge에서 <EVIDENCE_BLOCK>의 내용을 미리 언급하거나 복붙하지 않는다.
"""
    bullet_compaction_rules = """
[Bullets 압축 규칙]
- 각 불릿은 40자 이내(한국어 기준)로 작성한다.
- 불릿 1개 = 행동 1개. 여러 행동을 한 불릿에 묶지 않는다.
- "~하고, ~하며, ~하십시오" 같은 나열형 불릿 금지.
- 권장 형식: "~할 때 -> ~한다" 또는 "~을 위해 ~을 먼저 한다"
"""
    chapter_rhythm_line = "- 챕터 리듬(Hook/요약/주의/실행팁)은 권장이지 강제가 아니다."
    chapter_paragraph_line = "- 각 챕터는 2~4문단(2문단도 허용), 문단은 가독성 있게 분리한다."
    actionable_bullet_line = f"- Actionable 챕터({', '.join(sorted(_ACTIONABLE_CHAPTER_KEYS))})는 마지막에 행동 팁 불릿 최소 3개를 둔다."
    bullet_exempt_line = "- Executive Diagnosis/Current Phase/Final Integration는 불릿 강제를 적용하지 않는다."
    if hybrid_render_mode:
        hybrid_output_contract = f"""
HYBRID RENDER OUTPUT CONTRACT
- Actionable 챕터({", ".join(sorted(_ACTIONABLE_CHAPTER_KEYS))}) 출력 순서:
  Hook: 2~3문장 (80~150자 목표)
  <EVIDENCE_BLOCK>
  Bridge: 2~3문장 (100~200자 목표)
  - 불릿1
  - 불릿2
  - 불릿3
- 불릿 면제 챕터(Executive Diagnosis, Current Phase, Final Integration) 출력 순서:
  Hook: 2~3문장 (80~150자 목표)
  <EVIDENCE_BLOCK>
  Bridge: 2~3문장 (100~200자 목표)
  Bullets 금지.
- <EVIDENCE_BLOCK> 태그는 챕터당 정확히 1회만 출력한다.
- Evidence 텍스트를 재작성/재인용/복붙하지 않는다. 태그만 출력한다.
- [근거], --- 같은 라벨/구분선 삽입 금지.
- 체크리스트식 본문 전개 금지 (마지막 action bullets 3개는 허용).
"""
        sales_tone_contract = """
[판매형 톤 계약]
- Hook은 독자가 바로 공감할 질문/감정 진술로 시작하되, 2~3문장(80~150자)으로 쓴다.
  첫 문장: 공감 질문 또는 감정 진술
  이어지는 문장: 그 감정이 왜 생기는지 상황을 1~2문장으로 풀어준다.
- Bridge는 Evidence를 독자의 삶에 연결하는 2~3문장(100~200자)으로 쓴다.
  "그래서 당신에게 어떤 의미인지" 한 줄로 끝내지 않는다.
  Evidence에서 가장 중요한 포인트 1개를 골라 지금 당신의 상황에 바로 적용한다.
- Hook+Bridge 합산은 최소 220자 이상을 권장한다.
- Bridge는 Evidence의 핵심 포인트 1개만 확장해 설명한다.
- Hook/Bridge/Bullets에서는 점성학 전문용어를 직접 노출하지 않는다.
"""
        chapter_rhythm_line = "- 챕터 리듬은 아래 HYBRID RENDER OUTPUT CONTRACT를 반드시 따른다."
        chapter_paragraph_line = "- 각 챕터는 Hook/Bridge 흐름이 명확한 짧은 문단 구성을 유지한다."
        actionable_bullet_line = "- Actionable 챕터는 HYBRID RENDER OUTPUT CONTRACT의 3개 불릿 규칙을 따른다."
        bullet_exempt_line = "- Bullet-exempt 챕터는 HYBRID RENDER OUTPUT CONTRACT의 불릿 금지 규칙을 따른다."

    if prompt_mode == "analyzer_first":
        return f"""
ROLE
- 당신은 엔진이 만든 신호를 조합해 서술하는 조립기(assembler)다.
- 새로운 원인/원천 데이터를 만들지 않는다.

OUTPUT CONTRACT (STRICT)
- 정확히 {len(active_chapters)}개 챕터를 작성한다.
- 모든 챕터 헤딩은 `## [<chapter_key>] <한국어 제목>` 형식으로 시작한다.
- 챕터 순서/경계를 절대 바꾸지 않는다.
- 챕터를 병합/누락하지 않는다.
- 문단 사이는 반드시 빈 줄(Blank line) 1개로 구분한다.
- 챕터 본문은 공백 제외 최소 {min_chars}자, 권장 {min_chars}~{target_chars}자를 목표로 한다.
- 분량이 부족하면 새 사실을 만들지 말고 주어진 근거를 더 구체화해 확장한다.
- 각 챕터는 최소 {min_anchors}개의 구체 앵커(행성/하우스/라시/낙샤트라/다샤/요가)를 포함한다.
- 앵커는 나열하지 말고 문장 안에서 인과적으로 연결한다.
{hybrid_output_contract}
{sales_tone_contract}

CHAPTER KEY ORDER
{chapter_key_lines}

ANALYSIS RULES
- 각 챕터는 최소 1개 이상의 매핑 신호를 반영한다.
- 분석 라벨(원인/표현/영향) 표기 금지.
- 모순 신호가 있으면 모순을 숨기지 말고 그대로 설명한다.
- 문장은 바로 이해 가능하게, 생활어 중심으로 작성한다.
- {chapter_rhythm_line[2:] if chapter_rhythm_line.startswith('- ') else chapter_rhythm_line}
- {actionable_bullet_line[2:] if actionable_bullet_line.startswith('- ') else actionable_bullet_line}
- {bullet_exempt_line[2:] if bullet_exempt_line.startswith('- ') else bullet_exempt_line}
- 같은 조언형 종결(~도움됩니다/~유리합니다/~좋습니다) 반복을 피한다.
- Evidence에 없는 새로운 점성 요소/사실은 생성하지 않는다.
- 근거가 부족하면 일반론을 최소화하고, 중립적/제한적 문장으로 처리한다.
{_SUBTLE_VEDIC_PROMPT_RULES}
{jargon_transform_rules}
{chapter_hook_hint_block}
{anti_repeat_rules}
{explanation_mode_ban}
{bullet_compaction_rules}

SAFETY RULES
- 내부 메타 용어를 출력하지 말 것:
  activation intensity, dominant axis, psychological tension axis,
  stability index, risk_factor, opportunity_factor, vector, modifier, amplification.
- 수치/퍼센트/점수/지표 직접 노출 금지.
- 내부 메타 라벨/지표명은 앵커로 쓰지 않는다.
- 사건 확정 예언 금지:
  결혼, 이직, 합격, 당첨, 임신, 수술, 이혼, 파산, 대박, 확정 수익 등 결과 확정형 사건 단정 금지.
- 단정 강화 표현 금지: 반드시, 무조건, 확정, 틀림없이.
- dasha_context에 없는 값은 만들지 않는다.
- 연도/반기/분기/Q1~Q4 표기 금지.
- timing_axis.timing_windows가 없으면 시기 문장을 억지로 만들지 않는다.

DASHA INTEGRITY
- 시기 흐름은 dasha_context를 기반으로 해석한다.
- 고전 lords 정보가 없으면 중립적 시기 프레이밍을 사용한다.
- "시기 흐름(다샤)" 표기는 최초 1회만 사용 가능하다.
- 이미 시작된 시기는 "현재 진행 중"으로, 이미 종료된 시기는 제외한다.
- 점성학 용어는 은은하게 사용하고 과잉 노출 금지.
- Timing windows 우선순위: intensity(high>medium>low) -> domain(career>relationship>money>health>general) -> nearest start.

Narrative Mode:
{mode}

Structural Executive Overview:
{overview}

Timing Context (internal cue):
{json.dumps(timing, indent=2, ensure_ascii=False)}

Relationship Compact Context (JSON):
{compact_context_json}

[GLOBAL CHART EVIDENCE]
{global_evidence_text if global_evidence_text else "- (none)"}

[CHAPTER SPECIFIC EVIDENCE]
{chapter_evidence_text if chapter_evidence_text else "(none)"}

Core Chart Identity (internal cue only):
{asc_text} / {sun_text} / {moon_text}

Chapter Blocks (JSON):
{blocks_json if chapter_blocks_included else "{}"}
"""

    return f"""
PERSONA
- 당신은 엔진 신호를 조립해 읽히는 한국어 서사로 바꾸는 조립기다.
- 문장은 따뜻하고 명확하게, 생활어 중심으로 쓴다.

OUTPUT CONTRACT (STRICT)
- 정확히 {len(active_chapters)}개 챕터를 작성한다.
- 모든 챕터 헤딩은 `## [<chapter_key>] <한국어 제목>` 형식으로 시작한다.
- 아래 chapter_key 순서/경계를 절대 바꾸지 않는다.
- 챕터를 병합/누락하지 않는다.
- {chapter_paragraph_line[2:] if chapter_paragraph_line.startswith('- ') else chapter_paragraph_line}
- 문단 사이는 반드시 빈 줄(Blank line) 1개로 구분한다.
- 메타 라벨 출력 금지: "중심 주제:", "내적 줄다리기:", "전략 제안:" 등.
- 챕터 본문은 공백 제외 최소 {min_chars}자, 권장 {min_chars}~{target_chars}자를 목표로 한다.
- 분량이 부족하면 새 사실을 만들지 말고 주어진 근거를 더 구체화해 확장한다.
- 각 챕터는 최소 {min_anchors}개의 구체 앵커(행성/하우스/라시/낙샤트라/다샤/요가)를 포함한다.
- 앵커는 나열하지 말고 문장 안에서 인과적으로 연결한다.
{hybrid_output_contract}
{sales_tone_contract}

CHAPTER KEY ORDER
{chapter_key_lines}

HARD BANS
- 내부 메타 용어를 출력하지 말 것:
  activation intensity, dominant axis, psychological tension axis,
  stability index, risk_factor, opportunity_factor, vector, modifier, amplification.
- 수치/퍼센트/점수/지표 직접 노출 금지.
- 연도/반기/분기/Q1~Q4 표기 금지.
- 사건 확정 예언 금지:
  결혼, 이직, 합격, 당첨, 임신, 수술, 이혼, 파산, 대박, 확정 수익 등 결과 확정형 사건 단정 금지.
- 단정 강화 표현 금지: 반드시, 무조건, 확정, 틀림없이.
- 공포 마케팅 문장 금지.
- (원인), (표현), (영향) 같은 분석 라벨 직접 표기 금지.

DASHA INTEGRITY
- 시기 흐름은 dasha_context를 따르되, 없는 값을 만들어내지 말 것.
- 고전 lords 정보가 없으면 중립적 시기 프레이밍을 사용.
- "시기 흐름(다샤)" 표기는 최초 1회만 사용 가능.
- timing_axis.timing_windows가 없으면 시기 언급을 억지로 만들지 않는다(추론 금지).
- 점성학 용어는 은은하게만 사용하고, 생활어 중심으로 설명한다.
- 제공된 신호에서 타이밍 강조가 반복되면 강한 신호로 간주한다.
- Future Timing 섹션이 리포트 전체 분량을 지배하지 않게 유지한다.
- Timing windows 우선순위: intensity(high>medium>low) -> domain(career>relationship>money>health>general) -> nearest start.
- 이미 시작된 구간은 "현재 진행 중"으로, 이미 종료된 구간은 제외한다.

CORE WRITING GUIDANCE
- 구조 신호는 내부적으로만 쓰고, 출력은 사람의 경험 언어로 번역한다.
- 각 챕터는 독립적으로 충분한 설명을 갖추되, 장문 반복으로 늘려 쓰지 않는다.
- 동일 문형/클로징을 반복하지 않는다.
- Money / Relationship / Career는 서로 다른 문제의식과 감정 결로 구분해 작성한다.
- 챕터를 각각 별도의 분석 보고서처럼 분리하지 말고, 미세한 세계관 연속성을 유지한다.
- 단계별 매뉴얼형 전략 나열을 줄이고, 통찰 중심 문장을 우선한다.
- 모든 챕터를 조언으로 끝내지 않는다.
- Not every chapter needs a concluding instruction.
- {actionable_bullet_line[2:] if actionable_bullet_line.startswith('- ') else actionable_bullet_line}
- {bullet_exempt_line[2:] if bullet_exempt_line.startswith('- ') else bullet_exempt_line}
- Avoid repeatedly using similar softening or mitigating phrases across multiple chapters (e.g., "지금은...", "무리하지 말고...", "당장은...").
- Allow at least a few sentences per report that feel emotionally decisive rather than explanatory.
- HOT 섹션(Executive Diagnosis, Recurring Patterns, Love & Relationship Patterns, Mid-Term Direction)에서는 긴장이 자연스럽게 존재할 때만, 섹션당 sharp line을 최대 1회 허용한다.
- Evidence에 없는 새로운 점성 요소/사실은 생성하지 않는다.
- 근거가 부족하면 일반론을 최소화하고, 중립적/제한적 문장으로 처리한다.
{_SUBTLE_VEDIC_PROMPT_RULES}
{jargon_transform_rules}
{chapter_hook_hint_block}
{anti_repeat_rules}
{explanation_mode_ban}
{bullet_compaction_rules}

{style_override_block}

META REPORT-VOICE BAN
- 본문에서 아래 메타 문장을 쓰지 말 것:
  - "이 리포트는"
  - "본 해석은"
  - "이 보고서는"
- 챕터 제목/키 표기는 예외다. (예: Final Integration, 표기 자체는 허용)
- 본문에서 메타 설명 라벨(예: "보충 메모:")은 금지한다.

STANDALONE LINE CAP
- 단독 문장은 최대 4개까지 허용한다.
- 단독 문장을 연속으로 배치하지 않는다.

FINAL SUMMARY MINIMUM INSIGHTS
- Final Integration에는 아래 통찰 2개를 반드시 포함:
  1) 영역 간 연결 통찰 1개
  2) 반복 패턴 통찰 1개
- 강제 체크리스트 문구는 쓰지 않는다.

Narrative Mode:
{mode}

Structural Executive Overview:
{overview}

Timing Context (internal cue):
{json.dumps(timing, indent=2, ensure_ascii=False)}

Core Chart Identity:
Ascendant: {asc_text}
Sun: {sun_text}
Moon: {moon_text}

Relationship Compact Context (JSON):
{compact_context_json}

[GLOBAL CHART EVIDENCE]
{global_evidence_text if global_evidence_text else "- (none)"}

[CHAPTER SPECIFIC EVIDENCE]
{chapter_evidence_text if chapter_evidence_text else "(none)"}

Chapter Blocks (JSON):
{blocks_json if chapter_blocks_included else "{}"}
"""
