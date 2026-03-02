from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any

from backend.commercial_quality_constants import (
    ACTIONABLE_BULLET_RE,
    ACTIONABLE_HINT_RE,
    ACTION_STEPS_INLINE_HEADING_RE,
    ACTION_STEPS_INLINE_SPLIT_RE,
    ACTION_BOUNDARY_BASE_WINDOW,
    ACTION_BOUNDARY_EXPAND_MAX_LEN,
    ACTION_BOUNDARY_MAX_WINDOW,
    CODE_FENCE_TOGGLE_RE,
    DASHA_DEFINITION_CANONICAL_LINE,
    DASHA_DEFINITION_RE,
    DASHA_DEFINITION_REDUNDANCY_RE,
    DASHA_DEFINITION_REDUNDANT_TEMPLATE_RE,
    DASHA_DEFINITION_SHRINK_LINE,
    DASHA_NESTED_PHRASE_RE,
    EXPLANATORY_BULLET_RE,
    INLINE_ACTION_CHAIN_EXPLANATION_RE,
    INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE,
    INLINE_ACTION_CHAIN_LIST_START_RE,
    INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE,
    INLINE_ACTION_CHAIN_SLASH_SPLIT_RE,
    CORE_ACTION_TOOLKIT,
    ACTION_STEPS_GENERIC_BANNED_FPS,
    ONE_PAGE_SUMMARY_START_RE,
    TOP_LEVEL_H1_RE,
    RESIDUAL_MIN_CHARS,
    RESIDUAL_SENTENCE_END_RE,
    RESIDUAL_VALID_CHAR_RE,
    TRAIL_CONNECTOR_RE,
    TRAIL_MIN_LEN,
    TRAIL_SENTENCE_END_RE,
    TRAIL_SPLIT_RE,
)
from backend.front_contract_constants import (
    ACTIONABLE_CHAPTER_KEY_ALIASES,
    PLAYBOOK_END_RE,
    PLAYBOOK_LABELS,
    PLAYBOOK_LINE_CAUTION_PREFIX,
    PLAYBOOK_LINE_REASON_PREFIX,
    PLAYBOOK_LINE_RULE_PREFIX,
    PLAYBOOK_START_RE,
    RULE_SEPARATOR,
)
from backend.vedic_lexicon import extract_timing_map_span

logger = logging.getLogger("vedic_ai")

_CHAPTER_SPLIT_RE = re.compile(r"(?=^##\s)", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^\s*#{2,3}\s+")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_LIST_CONTINUATION_RE = re.compile(r"^\s{2,}\S+")
_TIMING_MAP_H3_RE = re.compile(r"(?im)^[ \t]*###[ \t]*Timing Map[ \t]*$")
_MID_TERM_HEADING_RE = re.compile(
    r"^\s*##\s*(?:\[\s*Mid-Term Direction\s*\]|Mid-Term Direction)(?:\s|$)",
    re.IGNORECASE,
)
_INLINE_BULLET_RE = re.compile(r"([.!?…])\s+-\s+")
_BROKEN_RISK_H2_RE = re.compile(r"(?m)^##\s*\[Risk Management Points\]\s*Timing Map\s*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_ANY_HEADING_LINE_RE = re.compile(r"(?m)^\s*#{2,6}\s+")

_DASHA_DUP_RE = re.compile(
    r"다샤\s*[\(（]\s*Dasha\s*[\)）]\s*,?\s*즉\s*시기 흐름\(인생의 큰 시즌\)을 보여주는\s*다샤\s*[\(（]\s*Dasha\s*[\)）]"
)
_RAHU_REDUNDANT_RE = re.compile(
    r"(확장 욕구를 관장하는 라후\s*[\(（]\s*Rahu\s*[\)）]\s*는)\s*확장 욕구를 관장하는 요소로서,?\s*"
)

_GARBLED_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"회복 시간을\s*중심\s*흐름소하는"),
        "회복 시간을 축소하는",
    ),
    (
        re.compile(r"반복 가능한\s*흐름로"),
        "반복 가능한 구조로",
    ),
    (
        re.compile(r"눈여겨볼한"),
        "눈여겨볼 만한",
    ),
    (
        re.compile(r"눈여겨볼하며"),
        "눈여겨볼 만하며",
    ),
    (
        re.compile(r"압중심\s*흐름해"),
        "압축해",
    ),
    (
        re.compile(r"우선순위를 하나로\s*압중심\s*흐름해두는"),
        "우선순위를 하나로 압축해두는",
    ),
    (
        re.compile(r"이러한 내부\s*흐름는"),
        "이러한 내부 구조는",
    ),
    (
        re.compile(r"흐름가"),
        "흐름이",
    ),
    (
        re.compile(r"방향성가"),
        "방향성이",
    ),
    (
        re.compile(r"흐름는"),
        "흐름은",
    ),
    (
        re.compile(r"부담를"),
        "부담을",
    ),
    (
        re.compile(r"흐름로"),
        "흐름으로",
    ),
    (
        re.compile(r"압중심"),
        "압축",
    ),
)
_GARBLED_REMAIN_RE = re.compile(
    r"압중심|흐름해두는|이러한 내부 흐름는|흐름가|방향성가|흐름는|중심\s*흐름소하는|부담를|눈여겨볼한|눈여겨볼하며|흐름로"
)

_CALENDAR_ENTRY_RE = re.compile(r"^\s*(?:-\s*)?[^:\n]*~[^:\n]*:\s*.+$")
_DEJARGON_SIGN_RISE_RE = re.compile(
    r"시데리얼\s*\(항성황도\)\s*,?\s*라히리\s*기준의\s*([가-힣]+)자리\s*상승"
)
_DEJARGON_STANDALONE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"시데리얼\s*\(항성황도\)"), ""),
    (re.compile(r"라히리\s*기준"), ""),
    (re.compile(r"\bsidereal\b", re.IGNORECASE), ""),
    (re.compile(r"\blahiri\b", re.IGNORECASE), ""),
    (re.compile(r"\bayanamsa\b", re.IGNORECASE), ""),
    (re.compile(r"시데리얼"), ""),
    (re.compile(r"항성황도"), ""),
    (re.compile(r"라히리"), ""),
)
_COMMERCIAL_HEADING_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Shadbala\s*&\s*Avastha\s*Snapshot", re.IGNORECASE), "강약 스냅샷"),
    (re.compile(r"Remedy\s*Priority\s*by\s*Shadbala", re.IGNORECASE), "보완 우선순위"),
    (re.compile(r"Final\s*Synthesis\s*:\s*Strength\s*Axis", re.IGNORECASE), "최종 종합: 강약 축"),
)
_COMMERCIAL_TOKEN_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z])shadbala(?![A-Za-z])", re.IGNORECASE), "강약 지표"),
    (re.compile(r"(?<![A-Za-z])avastha(?![A-Za-z])", re.IGNORECASE), "상태 지표"),
    (re.compile(r"\bstrength\s+axis\b", re.IGNORECASE), "강약 축"),
    (re.compile(r"Śadbala", re.IGNORECASE), "강약 지표"),
    (re.compile(r"Avasthā", re.IGNORECASE), "상태 지표"),
)
_INTERNAL_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^\s*<!--\s*chapter_key:.*?-->\s*$"),
    re.compile(r"(?m)^[A-Za-z0-9 \[\]_&.-]+-\s*해석 블록\s*\d+\s*$"),
)
_H2_SECTION_RE = re.compile(r"(?m)^##\s+.*$")
_H2_KEY_RE = re.compile(r"^\s*##\s*(?:\[\s*([^\]]+)\s*\]\s*(.*)|(.+))$")
_ACTION_STEPS_H3_RE = re.compile(r"(?im)^[ \t]*###[ \t]*Action Steps[ \t]*$")
_STRICT_BULLET_LINE_RE = re.compile(r"(?m)^\s*-\s+\S")
_ORDERED_BULLET_LINE_RE = re.compile(r"^\s*\d+\.\s+\S")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CONNECTOR_PREFIX_RE = re.compile(r"^\s*(?:다만|그리고)\s*")
_DENSITY_SHORT_NORMALIZE_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_ACTIONABLE_CHAPTER_KEYS = [
    "Current Phase",
    "Career & Money",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Mid-Term Direction",
    "Risk Management Points",
    "Growth Acceleration",
]
_ACTION_BULLET_CUE_WORDS = ("좋습니다", "유리합니다", "필요", "주의", "권장", "먼저", "피하기")
_FALLBACK_DENSITY_SENTENCE = (
    "작은 루틴(수면·식사·일정)을 먼저 고정하고, 오늘 할 일을 1~2개로 줄이면 리듬이 더 안정적으로 굴러갑니다."
)
_GENERIC_ACTION_BULLETS = (
    "- 결정을 서두르지 말고, 오늘의 우선순위 1개만 정해 마무리하세요.",
    "- 수면·식사·일정 루틴을 먼저 고정해 리듬을 안정시키세요.",
)
_CHAPTER_H2_WITH_KEY_RE = re.compile(r"(?m)^\s*##\s*\[")
_FRONT_CHECKBOX_LINE_RE = re.compile(r"^\s*-\s*\[[ xX]\]\s+")
_FRONT_START_SENTINEL = "<!-- FRONT_START -->"
_FRONT_END_SENTINEL = "<!-- FRONT_END -->"
_CHAPTERS_START_SENTINEL = "<!-- CHAPTERS_START -->"
_CHAPTERS_END_SENTINEL = "<!-- CHAPTERS_END -->"
_SENTINEL_LINE_RE = re.compile(r"(?m)^\s*<!--\s*(?:FRONT_START|FRONT_END|CHAPTERS_START|CHAPTERS_END)\s*-->\s*$")
_FRONT_SEGMENT_RE = re.compile(
    r"<!--\s*FRONT_START\s*-->\s*(?P<front>.*?)\s*<!--\s*FRONT_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_CHAPTER_SEGMENT_RE = re.compile(
    r"<!--\s*CHAPTERS_START\s*-->\s*(?P<chapters>.*?)\s*<!--\s*CHAPTERS_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_PLAYBOOK_INLINE_LABEL_RE = re.compile(r"^\s*(이번 달|다음 달|그다음 달)\s*-\s*주의:\s*(.*)$")
_PLAYBOOK_CAUTION_CAPTURE_RE = re.compile(r"-\s*주의:\s*(.*?)(?=\s*-\s*규칙:|\s*-\s*이유:|$)")
_PLAYBOOK_RULE_CAPTURE_RE = re.compile(r"-\s*규칙:\s*(.*?)(?=\s*-\s*이유:|$)")
_PLAYBOOK_REASON_CAPTURE_RE = re.compile(r"-\s*이유:\s*(.*)$")
_INLINE_ADDITIONAL_BULLET_RE = re.compile(r"\s+(?:-\s+|\d+\.\s+)")
_INLINE_CHAIN_LEADING_PREFIX_RE = re.compile(r"^\s*(?:-\s+|•\s+|\d+\.\s+)")
_OVERFLOW_PREFIX = "추가 제안:"
_PLAYBOOK_SAFE_SLOT_BY_LABEL: dict[str, dict[str, str]] = {
    "이번 달": {
        "caution": "감정이 먼저 올라오면 결론을 서두르기 쉬워 손실이 커질 수 있습니다.",
        "rule_left": "확인 질문 1개 먼저 하기",
        "rule_right": "중요 대화 12시간 보류하기",
        "reason": "감정 반응 속도가 빨라질수록 작은 오해가 커져 확인 절차가 손실을 줄입니다.",
    },
    "다음 달": {
        "caution": "검증 없이 진행하면 작은 누락도 비용과 신뢰 손실로 번질 수 있습니다.",
        "rule_left": "결정 24시간 보류하기",
        "rule_right": "조건 한 줄 문서화하기",
        "reason": "주의 구간은 누락 비용이 커서 속도보다 검증이 결과를 지켜줍니다.",
    },
    "그다음 달": {
        "caution": "한 번에 크게 확장하면 유지 비용이 늘어 기회가 약해질 수 있습니다.",
        "rule_left": "파일럿 1개 먼저 시작하기",
        "rule_right": "되는 것만 확대하기",
        "reason": "기회 구간은 작은 테스트가 확장보다 성공 확률을 안정적으로 올립니다.",
    },
}

_DEFINITION_DASHA_SPAN_RE = re.compile(
    r"다샤\s*[\(（]\s*Dasha\s*[\)）][^.\n]*(?:\.[\s]*)?",
    re.IGNORECASE,
)
_DEFINITION_SPAN_MAX_CHARS = 600
_QUALITY_METRICS_DEFAULT: dict[str, int | bool] = {
    "action_steps_inline_heading_repairs": 0,
    "action_boundary_window_extended_count": 0,
    "action_steps_contaminated_line_repairs": 0,
    "action_steps_tail_move_count": 0,
    "action_steps_tail_move_dedup_skips": 0,
    "inline_action_chain_migrations": 0,
    "inline_execution_lines_moved": 0,
    "inline_action_chain_overflow_summaries": 0,
    "action_steps_non_actionable_rewrites": 0,
    "action_steps_duplicate_replacements": 0,
    "definition_dasha_occurrences_after": 0,
    "definition_nested_phrase_violations_after": 0,
    "dasha_definition_nested_pattern_repairs": 0,
    "dasha_definition_redundancy_repairs": 0,
    "one_page_summary_dedup_repairs": 0,
    "dasha_insert_retry_blocked": False,
    "commercial_quality_metrics_valid": True,
}


def _fix_broken_risk_h2(text: str) -> str:
    return _BROKEN_RISK_H2_RE.sub("## [Risk Management Points] 리스크 관리 요점", text)


def _looks_like_calendar_entry(line: str) -> bool:
    return bool(_CALENDAR_ENTRY_RE.match(line.strip()))


def _split_multi_calendar_bullets(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    body = stripped[2:].strip()
    parts = [part.strip() for part in re.split(r"\s+-\s+", body) if part and part.strip()]
    if len(parts) <= 1:
        return None
    if not all(_looks_like_calendar_entry(part) for part in parts):
        return None
    return [f"- {part}" for part in parts]


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _is_heading_only_block(lines: list[str]) -> bool:
    return bool(lines) and all(_HEADING_LINE_RE.match(line) for line in lines)


def _is_list_only_block(lines: list[str]) -> bool:
    if not lines:
        return False
    saw_list_line = False
    for line in lines:
        if _LIST_LINE_RE.match(line):
            saw_list_line = True
            continue
        if saw_list_line and _LIST_CONTINUATION_RE.match(line):
            continue
        return False
    return saw_list_line


def _is_code_fence_block(block: str) -> bool:
    stripped = block.strip()
    return stripped.startswith("```") and stripped.endswith("```")


def _is_html_comment_block(block: str) -> bool:
    return "<!--" in block and "-->" in block


def _soft_wrap_join(text: str) -> str:
    blocks = [block for block in re.split(r"\n\n+", text) if block and block.strip()]
    if not blocks:
        return text.strip()

    out_blocks: list[str] = []
    for block in blocks:
        stripped_block = block.strip()
        lines = [line for line in stripped_block.splitlines() if line.strip()]
        if not lines:
            continue
        if _is_heading_only_block(lines):
            out_blocks.append(stripped_block)
            continue
        if _is_list_only_block(lines):
            out_blocks.append(stripped_block)
            continue
        if _is_code_fence_block(stripped_block):
            out_blocks.append(stripped_block)
            continue
        if _is_html_comment_block(stripped_block):
            out_blocks.append(stripped_block)
            continue

        merged = re.sub(r"\s*\n\s*", " ", stripped_block)
        merged = re.sub(r"[ \t]{2,}", " ", merged).strip()
        out_blocks.append(merged)
    return "\n\n".join(out_blocks).strip()


def _promote_timing_map_heading_mid_term_only(text: str) -> str:
    chapters = [chunk for chunk in _CHAPTER_SPLIT_RE.split(text) if chunk]
    if not chapters:
        return text

    changed = False
    for idx, chapter in enumerate(chapters):
        first_line = chapter.splitlines()[0] if chapter.splitlines() else ""
        if not _MID_TERM_HEADING_RE.match(first_line):
            continue
        if _TIMING_MAP_H3_RE.search(chapter):
            continue

        replaced = chapter
        promoted = False
        line_label_count = 0
        replaced, count = re.subn(
            r"(?im)^\s*Timing Map\s*:\s*$",
            "### Timing Map",
            replaced,
        )
        line_label_count += count
        promoted = promoted or count > 0
        replaced, count = re.subn(
            r"(?im)^\s*Timing Map\s*:\s*(\S.+)$",
            r"### Timing Map\n\n\1",
            replaced,
        )
        line_label_count += count
        promoted = promoted or count > 0

        if line_label_count == 0:
            replaced, count = re.subn(
                r"(?i)\s+Timing Map\s*:\s*(?=\S)",
                "\n\n### Timing Map\n\n",
                replaced,
                count=1,
            )
            promoted = promoted or count > 0
            if count == 0:
                replaced, count = re.subn(
                    r"(?i)\s+Timing Map\s*:\s*(?:$|\n)",
                    "\n\n### Timing Map\n\n",
                    replaced,
                    count=1,
                )
                promoted = promoted or count > 0
        if not promoted:
            continue

        replaced = re.sub(
            r"[ \t]*\n[ \t]*###\s*Timing Map\s*[ \t]*\n*",
            "\n\n### Timing Map\n\n",
            replaced,
            flags=re.IGNORECASE,
        )
        chapters[idx] = replaced
        changed = True

    return "".join(chapters) if changed else text


def _normalize_timing_map_body(text: str) -> str:
    span = extract_timing_map_span(text)
    if span is None:
        return text
    start, end = span
    timing = text[start:end]
    timing = re.sub(r"_\s*[Hh]\s*1\b", " (초반)", timing)
    timing = re.sub(r"_\s*[Hh]\s*2\b", " (후반)", timing)
    timing = timing.replace("_", "")
    timing = re.sub(r"\s*실행 팁:\s*", "\n\n실행 팁:\n\n", timing)

    out_lines: list[str] = []
    in_tip_block = False
    for raw in timing.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue

        if _TIMING_MAP_H3_RE.match(stripped):
            out_lines.append("### Timing Map")
            in_tip_block = False
            continue

        if stripped == "실행 팁:":
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append("실행 팁:")
            out_lines.append("")
            in_tip_block = True
            continue

        split_entries = _split_multi_calendar_bullets(stripped)
        if split_entries is not None:
            out_lines.extend(split_entries)
            in_tip_block = False
            continue

        if _looks_like_calendar_entry(stripped):
            bullet = stripped if stripped.startswith("- ") else f"- {stripped.lstrip('-').strip()}"
            out_lines.append(bullet)
            in_tip_block = False
            continue

        if in_tip_block and not _LIST_PREFIX_RE.match(stripped):
            out_lines.append(f"- {stripped}")
            continue

        out_lines.append(stripped)

    timing = "\n".join(out_lines)
    timing = re.sub(r"[ \t]{2,}", " ", timing)
    return f"{text[:start]}{timing}{text[end:]}"


def _cleanup_vedic_redundancy(text: str) -> str:
    out = _DASHA_DUP_RE.sub("시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)", text)
    out = DASHA_DEFINITION_REDUNDANT_TEMPLATE_RE.sub(DASHA_DEFINITION_CANONICAL_LINE, out)
    out = DASHA_NESTED_PHRASE_RE.sub("시기 흐름(", out)
    out = re.sub(r"(?:추가 제안:\s*){2,}", "추가 제안: ", out)
    out = _RAHU_REDUNDANT_RE.sub(r"\1 ", out)
    return out


def _cleanup_garbled_tokens(text: str) -> str:
    out = text
    for pattern, repl in _GARBLED_REPLACEMENTS:
        out = pattern.sub(repl, out)

    remain = _GARBLED_REMAIN_RE.search(out)
    if remain is not None:
        left = max(0, remain.start() - 40)
        right = min(len(out), remain.end() + 40)
        sample = out[left:right].replace("\n", " ")
        logger.warning(
            "warn_remediation_garble token=%s context=%s",
            remain.group(0),
            sample,
        )
    return out


def _apply_commercial_surface_firewall(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pattern, replacement in _COMMERCIAL_HEADING_REWRITES:
        out = pattern.sub(replacement, out)
    for pattern, replacement in _COMMERCIAL_TOKEN_REWRITES:
        out = pattern.sub(replacement, out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _split_h2_sections(text: str) -> list[dict[str, str]]:
    normalized = _normalize_newlines(text).strip()
    if not normalized:
        return []
    matches = list(_H2_SECTION_RE.finditer(normalized))
    if not matches:
        return [{"heading": "## Document", "body": normalized}]

    sections: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        section_text = normalized[start:end].strip()
        lines = section_text.splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        sections.append({"heading": heading, "body": body})
    return sections


def _join_h2_sections(sections: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for section in sections:
        heading = str(section.get("heading", "")).strip()
        body = str(section.get("body", "")).strip()
        if not heading:
            continue
        if body:
            chunks.append(f"{heading}\n\n{body}")
        else:
            chunks.append(heading)
    out = "\n\n".join(chunks)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def extract_h2_chapter_key(h2_line: str) -> str | None:
    line = str(h2_line or "").strip()
    if not line:
        return None
    match = _H2_KEY_RE.match(line)
    if not match:
        plain = line.removeprefix("##").strip()
        return plain or None
    bracket_key = str(match.group(1) or "").strip()
    plain_key = str(match.group(3) or "").strip()
    return bracket_key or plain_key or None


def _normalize_chapter_key_for_match(value: str) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[\[\]\(\){}]", " ", token)
    token = re.sub(r"[^\w&가-힣\s-]", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _resolve_actionable_canonical_key(raw_key: str) -> str | None:
    normalized = _normalize_chapter_key_for_match(raw_key)
    if not normalized:
        return None
    for canonical, aliases in ACTIONABLE_CHAPTER_KEY_ALIASES.items():
        alias_norms = {_normalize_chapter_key_for_match(canonical)}
        alias_norms.update(_normalize_chapter_key_for_match(alias) for alias in aliases)
        if normalized in alias_norms:
            return canonical
    return None


def _actionable_key_set_from_aliases() -> set[str]:
    return set(ACTIONABLE_CHAPTER_KEY_ALIASES.keys())


def _next_heading_boundary(text: str, start: int) -> int:
    next_heading = _ANY_HEADING_LINE_RE.search(text, start)
    return next_heading.start() if next_heading else len(text)


def _find_midterm_timing_map_spans(chapter_key: str, chapter_body: str) -> list[tuple[int, int]]:
    if chapter_key != "Mid-Term Direction":
        return []
    spans: list[tuple[int, int]] = []
    for match in _TIMING_MAP_H3_RE.finditer(chapter_body):
        end = _next_heading_boundary(chapter_body, match.end())
        spans.append((match.start(), end))
    return spans


def _index_in_spans(index: int, spans: list[tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= index < end:
            return True
    return False


def _replace_inline_bullets_outside_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not text:
        return text
    out: list[str] = []
    cursor = 0
    for match in _INLINE_BULLET_RE.finditer(text):
        out.append(text[cursor:match.start()])
        if _index_in_spans(match.start(), spans):
            out.append(match.group(0))
        else:
            out.append(match.expand(r"\1\n\n- "))
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out)


def _find_valid_action_steps_span(chapter_body: str, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    for match in _ACTION_STEPS_H3_RE.finditer(chapter_body):
        if _index_in_spans(match.start(), spans):
            continue
        block_start = match.end()
        block_end = _next_heading_boundary(chapter_body, block_start)
        bullets = _STRICT_BULLET_LINE_RE.findall(chapter_body[block_start:block_end])
        if len(bullets) >= 2:
            return (match.start(), block_end)
    return None


def _split_paragraph_blocks_for_density(text: str) -> list[str]:
    normalized = _normalize_newlines(text)
    return [block.strip() for block in re.split(r"\n\n+", normalized) if block and block.strip()]


def _is_short_caption_block(block: str, threshold: int = 40) -> bool:
    normalized = _DENSITY_SHORT_NORMALIZE_RE.sub("", block or "")
    return len(normalized) < threshold


def _count_body_paragraphs_for_density(chapter_body: str) -> int:
    body_count = 0
    for block in _split_paragraph_blocks_for_density(chapter_body):
        lines = [line for line in block.splitlines() if line.strip()]
        if _is_heading_only_block(lines):
            continue
        if _is_list_only_block(lines):
            continue
        if _is_short_caption_block(block, threshold=40):
            continue
        body_count += 1
    return body_count


def _normalize_for_exact_dedupe(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _previous_non_empty_line(blocks: list[str], idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        lines = [line.strip() for line in blocks[j].splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return ""


def _is_body_paragraph_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return False
    if _is_heading_only_block(lines):
        return False
    if _is_list_only_block(lines):
        return False
    if _is_code_fence_block(block):
        return False
    if _is_html_comment_block(block):
        return False
    return True


def ensure_min_body_paragraphs_per_chapter(md: str) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    sections = _split_h2_sections(md)
    if not sections:
        return md

    changed = False
    for section in sections:
        body = str(section.get("body", ""))
        if _count_body_paragraphs_for_density(body) > 0:
            continue

        blocks = _split_paragraph_blocks_for_density(body)
        kept_blocks: list[str] = []
        short_blocks: list[str] = []
        heading_descriptor = re.sub(r"^\s*##\s*", "", str(section.get("heading", "") or "")).strip()
        if heading_descriptor:
            short_blocks.append(_normalize_for_exact_dedupe(heading_descriptor))
        for block in blocks:
            lines = [line for line in block.splitlines() if line.strip()]
            if _is_heading_only_block(lines) or _is_list_only_block(lines):
                kept_blocks.append(block)
                if _is_heading_only_block(lines):
                    heading_text = " ".join(re.sub(r"^\s*#{2,3}\s*", "", line).strip() for line in lines if line.strip())
                    heading_text = _normalize_for_exact_dedupe(heading_text)
                    if heading_text:
                        short_blocks.append(heading_text)
                continue
            if _is_short_caption_block(block, threshold=40):
                short_blocks.append(_normalize_for_exact_dedupe(block))
                continue
            kept_blocks.append(block)

        merged = _normalize_for_exact_dedupe(" ".join(part for part in short_blocks if part))
        if merged:
            kept_blocks.append(merged)
        rebuilt = "\n\n".join(block for block in kept_blocks if block and block.strip()).strip()
        if _count_body_paragraphs_for_density(rebuilt) == 0:
            rebuilt_blocks = [block for block in _split_paragraph_blocks_for_density(rebuilt)]
            if _FALLBACK_DENSITY_SENTENCE not in rebuilt:
                rebuilt_blocks.append(_FALLBACK_DENSITY_SENTENCE)
            rebuilt = "\n\n".join(block for block in rebuilt_blocks if block and block.strip()).strip()
        rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
        if rebuilt != body:
            changed = True
            section["body"] = rebuilt

    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def dedupe_exact_paragraphs(md: str) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    sections = _split_h2_sections(md)
    if not sections:
        return md

    global_seen: set[str] = set()
    changed = False
    for section in sections:
        body = str(section.get("body", ""))
        blocks = _split_paragraph_blocks_for_density(body)
        if not blocks:
            continue

        kept: list[str] = []
        chapter_seen: set[str] = set()
        first_body_block: str | None = None
        body_idx = 0
        in_action_steps = False

        for idx, block in enumerate(blocks):
            stripped = block.strip()
            lines = [line for line in stripped.splitlines() if line.strip()]
            if not lines:
                continue

            if _is_heading_only_block(lines):
                heading_line = lines[0].strip()
                in_action_steps = bool(_ACTION_STEPS_H3_RE.match(heading_line))
                kept.append(stripped)
                continue

            if in_action_steps and _is_list_only_block(lines):
                kept.append(stripped)
                continue

            if _is_code_fence_block(stripped) or _is_html_comment_block(stripped):
                kept.append(stripped)
                continue

            if not _is_body_paragraph_block(stripped):
                kept.append(stripped)
                continue

            norm = _normalize_for_exact_dedupe(stripped)
            if not norm:
                continue

            if first_body_block is None:
                first_body_block = stripped

            chapter_key = norm
            if chapter_key in chapter_seen:
                changed = True
                body_idx += 1
                continue
            chapter_seen.add(chapter_key)

            heading_adjacent = bool(re.match(r"^###\s+", _previous_non_empty_line(blocks, idx)))
            global_key = f"{int(heading_adjacent)}||{norm}"
            if body_idx > 0 and len(norm) >= 120 and global_key in global_seen:
                changed = True
                body_idx += 1
                continue

            kept.append(stripped)
            if len(norm) >= 120 and body_idx > 0:
                global_seen.add(global_key)
            body_idx += 1

        if not any(_is_body_paragraph_block(block) for block in kept) and first_body_block:
            kept.append(first_body_block)
            changed = True

        rebuilt = "\n\n".join(block for block in kept if block and block.strip()).strip()
        if rebuilt != body:
            section["body"] = rebuilt

    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def _extract_candidate_action_sentences(chapter_body: str) -> list[str]:
    filtered_lines: list[str] = []
    for raw in _normalize_newlines(chapter_body).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if _STRICT_BULLET_LINE_RE.match(line):
            continue
        filtered_lines.append(line)
    prose = " ".join(filtered_lines).strip()
    if not prose:
        return []

    raw_sentences = _SENTENCE_SPLIT_RE.split(prose)
    if len(raw_sentences) <= 1:
        raw_sentences = [s.strip() for s in re.split(r"(?<=[다요])\s+", prose) if s and s.strip()]

    selected: list[str] = []
    seen: set[str] = set()
    for sentence in raw_sentences:
        cleaned = _CONNECTOR_PREFIX_RE.sub("", sentence.strip())
        if not cleaned:
            continue
        if not any(token in cleaned for token in _ACTION_BULLET_CUE_WORDS):
            continue
        if len(cleaned) < 20:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        selected.append(cleaned)
        if len(selected) >= 3:
            break
    return selected


def _collect_and_strip_non_protected_bullets(chapter_body: str, spans: list[tuple[int, int]]) -> tuple[str, list[str]]:
    lines = _normalize_newlines(chapter_body).split("\n")
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1

    cleaned_lines: list[str] = []
    collected_bullets: list[str] = []
    seen_bullets: set[str] = set()
    for idx, line in enumerate(lines):
        line_start = offsets[idx]
        protected = _index_in_spans(line_start, spans)
        stripped = line.strip()

        if _ACTION_STEPS_H3_RE.match(stripped) and not protected:
            # Remove stale/invalid Action Steps headings and rebuild deterministically at chapter end.
            continue

        if _STRICT_BULLET_LINE_RE.match(line) and not protected:
            bullet_text = re.sub(r"^\s*-\s+", "", line).strip()
            if bullet_text:
                canonical = f"- {bullet_text}"
                normalized = _normalize_for_exact_dedupe(canonical)
                if normalized not in seen_bullets:
                    seen_bullets.add(normalized)
                    collected_bullets.append(canonical)
            continue

        cleaned_lines.append(line)

    cleaned_body = "\n".join(cleaned_lines)
    cleaned_body = re.sub(r"[ \t]+\n", "\n", cleaned_body)
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body).strip()
    return cleaned_body, collected_bullets


def _find_action_steps_blocks(chapter_body: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for match in _ACTION_STEPS_H3_RE.finditer(chapter_body):
        if _index_in_spans(match.start(), spans):
            continue
        block_start = match.start()
        block_end = _next_heading_boundary(chapter_body, match.end())
        blocks.append((block_start, block_end))
    return blocks


def _split_inline_action_steps_heading_line(line: str) -> list[str]:
    raw = str(line or "")
    match = ACTION_STEPS_INLINE_HEADING_RE.match(raw)
    if match is None:
        return [line]
    rest = str(match.group("rest") or "").strip()
    if not rest:
        return [line]
    if re.match(r"^\s*(?:-\s+|•\s+|\d+\.\s+)$", rest):
        return [line]

    content = _INLINE_CHAIN_LEADING_PREFIX_RE.sub("", rest, count=1)
    splits = [part.strip() for part in ACTION_STEPS_INLINE_SPLIT_RE.split(content) if part and part.strip()]
    if len(splits) < 2:
        return [line]
    bullets: list[str] = []
    for part in splits:
        token = _normalize_inline_action_item(part)
        if not token:
            continue
        bullets.append(f"- {token}")
    if len(bullets) < 2:
        return [line]
    return ["### Action Steps", ""] + bullets


def _split_inline_action_steps_heading_in_chapters(chapter_body: str) -> tuple[str, int]:
    if not isinstance(chapter_body, str) or not chapter_body.strip():
        return chapter_body, 0
    lines = _normalize_newlines(chapter_body).splitlines()
    rebuilt: list[str] = []
    in_code_fence = False
    in_comment = False
    repairs = 0

    for line in lines:
        stripped = line.strip()
        if CODE_FENCE_TOGGLE_RE.match(stripped):
            in_code_fence = not in_code_fence
            rebuilt.append(line)
            continue
        if not in_code_fence:
            if "<!--" in stripped and "-->" not in stripped:
                in_comment = True
            if in_comment:
                rebuilt.append(line)
                if "-->" in stripped:
                    in_comment = False
                continue
            if stripped.startswith(">"):
                rebuilt.append(line)
                continue
            split_lines = _split_inline_action_steps_heading_line(line)
            if len(split_lines) > 1:
                repairs += 1
            rebuilt.extend(split_lines)
            continue
        rebuilt.append(line)

    out = "\n".join(rebuilt)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, repairs


def _split_inline_action_bullet_line(line: str) -> list[str]:
    stripped = str(line or "").strip()
    prefix_match = re.match(r"^\s*(?:-\s+|\d+\.\s+)(.+)$", stripped)
    if prefix_match is None:
        return [line]
    content = str(prefix_match.group(1) or "").strip()
    additional_matches = list(_INLINE_ADDITIONAL_BULLET_RE.finditer(content))
    if len(additional_matches) < 2:
        return [line]
    segments = [segment.strip() for segment in _INLINE_ADDITIONAL_BULLET_RE.split(content) if segment and segment.strip()]
    normalized_segments: list[str] = []
    for segment in segments:
        candidate = _normalize_for_exact_dedupe(segment)
        if len(candidate) < 4:
            return [line]
        normalized_segments.append(candidate)
    if len(normalized_segments) < 2:
        return [line]
    return [f"- {segment}" for segment in normalized_segments]


def _repair_inline_bullets_in_action_steps(chapter_body: str, spans: list[tuple[int, int]]) -> tuple[str, int]:
    blocks = _find_action_steps_blocks(chapter_body, spans)
    if not blocks:
        return chapter_body, 0

    rebuilt_parts: list[str] = []
    cursor = 0
    repairs = 0
    for block_start, block_end in blocks:
        rebuilt_parts.append(chapter_body[cursor:block_start])
        block_text = chapter_body[block_start:block_end]
        lines = _normalize_newlines(block_text).splitlines()
        if not lines:
            rebuilt_parts.append(block_text)
            cursor = block_end
            continue
        kept_lines: list[str] = [lines[0].strip() or "### Action Steps"]
        for line in lines[1:]:
            split_lines = _split_inline_action_bullet_line(line)
            if len(split_lines) > 1:
                repairs += 1
            kept_lines.extend(split_lines)
        rebuilt_block = "\n".join(line.rstrip() for line in kept_lines if line is not None).strip()
        rebuilt_parts.append(rebuilt_block)
        # Preserve a stable block boundary so adjacent Action Steps headers never collapse inline.
        rebuilt_parts.append("\n\n")
        cursor = block_end
    rebuilt_parts.append(chapter_body[cursor:])
    out = "".join(rebuilt_parts)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, repairs


def _line_offsets(text: str) -> list[int]:
    lines = _normalize_newlines(text).splitlines()
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1
    return offsets


def _action_boundary_end(content: str, metrics: dict[str, int | bool]) -> int | None:
    raw = str(content or "")
    if not raw.strip():
        return None
    window = raw[:ACTION_BOUNDARY_BASE_WINDOW]
    matches = list(ACTIONABLE_HINT_RE.finditer(window))
    if not matches and len(raw) < ACTION_BOUNDARY_EXPAND_MAX_LEN:
        expanded = raw[:ACTION_BOUNDARY_MAX_WINDOW]
        matches = list(ACTIONABLE_HINT_RE.finditer(expanded))
        if matches:
            metrics["action_boundary_window_extended_count"] = int(metrics.get("action_boundary_window_extended_count", 0)) + 1
    if not matches:
        return None
    return int(matches[-1].end())


def _is_tail_contaminated(tail: str) -> bool:
    token = _normalize_for_exact_dedupe(tail)
    if not token:
        return False
    end_hits = TRAIL_SENTENCE_END_RE.findall(token)
    if not end_hits:
        return False
    sentences = [s.strip() for s in TRAIL_SPLIT_RE.split(token) if s and s.strip()]
    sentence_count = len(sentences) if sentences else 1
    has_connector = bool(TRAIL_CONNECTOR_RE.search(token))
    if sentence_count >= 2:
        return True
    if len(token) >= TRAIL_MIN_LEN:
        return True
    if len(end_hits) >= 2:
        return True
    # Connector alone is not sufficient; require a sentence ending hit as above.
    return bool(has_connector and len(token) >= 30 and len(end_hits) >= 1)


def _repair_action_steps_contaminated_lines(chapter_key: str, chapter_body: str) -> tuple[str, dict[str, int | bool]]:
    metrics: dict[str, int | bool] = {
        "action_steps_contaminated_line_repairs": 0,
        "action_steps_tail_move_count": 0,
        "action_steps_tail_move_dedup_skips": 0,
        "action_boundary_window_extended_count": 0,
    }
    if not isinstance(chapter_body, str) or not chapter_body.strip():
        return chapter_body, metrics

    spans = _find_midterm_timing_map_spans(chapter_key, chapter_body)
    repaired_inline, inline_repairs = _repair_inline_bullets_in_action_steps(chapter_body, spans)
    if inline_repairs:
        metrics["action_steps_contaminated_line_repairs"] = int(metrics["action_steps_contaminated_line_repairs"]) + int(inline_repairs)

    blocks = _find_action_steps_blocks(repaired_inline, spans)
    if not blocks:
        return repaired_inline, metrics

    offsets = _line_offsets(repaired_inline)
    lines = _normalize_newlines(repaired_inline).splitlines()
    moved_tail_fingerprints: set[str] = set()

    rebuilt_parts: list[str] = []
    cursor = 0
    for block_start, block_end in blocks:
        rebuilt_parts.append(repaired_inline[cursor:block_start])
        block_text = repaired_inline[block_start:block_end]
        block_lines = _normalize_newlines(block_text).splitlines()
        if not block_lines:
            rebuilt_parts.append(block_text)
            cursor = block_end
            continue

        heading_line = block_lines[0].strip() or "### Action Steps"
        rebuilt_bullets: list[str] = []
        moved_tails_for_block: list[str] = []

        block_abs_line_index = 0
        for i, full_line in enumerate(lines):
            if offsets[i] >= block_start:
                block_abs_line_index = i
                break

        for rel_idx, line in enumerate(block_lines[1:], start=1):
            stripped = line.strip()
            if not stripped:
                continue
            is_bullet = bool(_STRICT_BULLET_LINE_RE.match(stripped) or _ORDERED_BULLET_LINE_RE.match(stripped))
            if not is_bullet:
                continue

            bullet_text = re.sub(r"^\s*(?:-\s+|\d+\.\s+)", "", stripped).strip()
            boundary_end = _action_boundary_end(bullet_text, metrics)
            if boundary_end is None:
                tail_text = bullet_text
                safe = _GENERIC_ACTION_BULLETS[0]
                rebuilt_bullets.append(safe)
                abs_idx = block_abs_line_index + rel_idx
                source_offset = offsets[abs_idx] if 0 <= abs_idx < len(offsets) else 0
                fp = hashlib.sha256(
                    f"{chapter_key}|{source_offset}|{_normalize_for_exact_dedupe(stripped)}".encode("utf-8")
                ).hexdigest()
                if fp in moved_tail_fingerprints:
                    metrics["action_steps_tail_move_dedup_skips"] = int(metrics["action_steps_tail_move_dedup_skips"]) + 1
                else:
                    moved_tail_fingerprints.add(fp)
                    moved_tails_for_block.append(tail_text)
                    metrics["action_steps_tail_move_count"] = int(metrics["action_steps_tail_move_count"]) + 1
                metrics["action_steps_contaminated_line_repairs"] = int(metrics["action_steps_contaminated_line_repairs"]) + 1
                continue

            head = bullet_text[:boundary_end].strip()
            tail = bullet_text[boundary_end:].strip()
            if not tail or not _is_tail_contaminated(tail):
                rebuilt_bullets.append(f"- {bullet_text}")
                continue

            abs_idx = block_abs_line_index + rel_idx
            source_offset = offsets[abs_idx] if 0 <= abs_idx < len(offsets) else 0
            fp = hashlib.sha256(
                f"{chapter_key}|{source_offset}|{_normalize_for_exact_dedupe(stripped)}".encode("utf-8")
            ).hexdigest()
            if fp in moved_tail_fingerprints:
                metrics["action_steps_tail_move_dedup_skips"] = int(metrics["action_steps_tail_move_dedup_skips"]) + 1
            else:
                moved_tail_fingerprints.add(fp)
                moved_tails_for_block.append(tail)
                metrics["action_steps_tail_move_count"] = int(metrics["action_steps_tail_move_count"]) + 1
            repaired_bullet = f"- {head}" if head else _GENERIC_ACTION_BULLETS[0]
            rebuilt_bullets.append(repaired_bullet)
            metrics["action_steps_contaminated_line_repairs"] = int(metrics["action_steps_contaminated_line_repairs"]) + 1

        if len(rebuilt_bullets) > 3:
            # Truncate with no data loss: tails have already been moved to body.
            rebuilt_bullets = rebuilt_bullets[:3]

        block_chunks: list[str] = []
        if moved_tails_for_block:
            block_chunks.append("\n\n".join(t.strip() for t in moved_tails_for_block if t and t.strip()))
        block_chunks.append(heading_line)
        if rebuilt_bullets:
            block_chunks.append("\n".join(rebuilt_bullets))
        rebuilt_block = "\n\n".join(part for part in block_chunks if part and part.strip())
        rebuilt_parts.append(rebuilt_block)
        rebuilt_parts.append("\n\n")
        cursor = block_end

    rebuilt_parts.append(repaired_inline[cursor:])
    out = "".join(rebuilt_parts)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, metrics


def _extract_action_bullets_from_block(block_text: str) -> list[str]:
    bullets: list[str] = []
    seen: set[str] = set()
    lines = _normalize_newlines(block_text).splitlines()
    for line in lines[1:]:
        stripped = line.strip()
        if _STRICT_BULLET_LINE_RE.match(stripped):
            candidate = re.sub(r"^\s*-\s+", "- ", stripped)
        elif _ORDERED_BULLET_LINE_RE.match(stripped):
            candidate = re.sub(r"^\s*\d+\.\s+", "- ", stripped)
        else:
            continue
        norm = _normalize_for_exact_dedupe(candidate)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        bullets.append(candidate)
    return bullets


def _is_actionable_bullet_text(text: str) -> bool:
    token = _normalize_for_exact_dedupe(text)
    if len(token) < 8:
        return False
    if ACTIONABLE_BULLET_RE.search(token) is None:
        return False
    action_shape = re.search(
        r"(?:하기\b|하세요|해보세요|보류하기|고정하기|기록하기|확인하기|점검하기|실행하기|예약하기|문서화하기|분해하기|선택하기|작성하기|"
        r"\d+\s*(?:분|시간|회|개|줄)|오늘|내일|이번\s*주|이번\s*달)",
        token,
        re.IGNORECASE,
    )
    if action_shape is not None:
        return True
    # Bare explanatory lines are not actionable even if they contain abstract hint nouns.
    if EXPLANATORY_BULLET_RE.search(token):
        return False
    return True


def _is_explanatory_bullet_text(text: str) -> bool:
    token = _normalize_for_exact_dedupe(text)
    if not token:
        return False
    return EXPLANATORY_BULLET_RE.search(token) is not None


def _normalize_action_bullet_fingerprint(text: str) -> str:
    token = _normalize_newlines(str(text or "")).strip()
    token = re.sub(r"^\s*(?:-\s+|\d+\.\s+)", "", token)
    token = _normalize_for_exact_dedupe(token)
    token = token.rstrip(".,;:!?")
    return token


def _toolkit_for_chapter(chapter_key: str) -> list[str]:
    canonical = _resolve_actionable_canonical_key(chapter_key)
    if not canonical:
        return []
    return list(CORE_ACTION_TOOLKIT.get(canonical, []))


def _pick_toolkit_bullet(
    chapter_key: str,
    used_fps: set[str],
    global_counts: dict[str, int],
) -> str | None:
    for template in _toolkit_for_chapter(chapter_key):
        fp = _normalize_action_bullet_fingerprint(template)
        if not fp:
            continue
        if fp in used_fps:
            continue
        if fp in ACTION_STEPS_GENERIC_BANNED_FPS:
            continue
        if global_counts.get(fp, 0) >= 2:
            continue
        return f"- {template.strip()}"
    return None


def _collect_action_bullets_in_block(block_text: str) -> list[str]:
    out: list[str] = []
    for line in _normalize_newlines(block_text).splitlines()[1:]:
        stripped = line.strip()
        if _STRICT_BULLET_LINE_RE.match(stripped):
            out.append(re.sub(r"^\s*-\s+", "- ", stripped))
        elif _ORDERED_BULLET_LINE_RE.match(stripped):
            out.append(re.sub(r"^\s*\d+\.\s+", "- ", stripped))
    return out


def _append_unique_bullet(target: list[str], candidate: str) -> None:
    canonical = re.sub(r"^\s*-\s+", "- ", str(candidate or "").strip())
    if len(canonical.strip()) < 20:
        return
    if not _is_actionable_bullet_text(canonical):
        return
    candidate_norm = _normalize_for_exact_dedupe(canonical)
    for existing in target:
        if _normalize_for_exact_dedupe(existing) == candidate_norm:
            return
    target.append(canonical)


def _normalize_inline_action_item(text: str) -> str:
    token = _normalize_newlines(str(text or "")).strip()
    if not token:
        return ""
    token = re.sub(r"\s+", " ", token)
    token = re.sub(r"[ \t]+([,.;:!?])", r"\1", token)
    token = token.rstrip(".,;:!? ")
    return token.strip()


def _normalize_overflow_item_for_compare(item: str) -> str:
    token = _normalize_for_exact_dedupe(str(item or ""))
    while token.startswith(_OVERFLOW_PREFIX):
        token = token[len(_OVERFLOW_PREFIX):].strip()
    return token.rstrip(".,;:!? ").strip()


def _extract_overflow_item_set_from_line(line: str) -> frozenset[str] | None:
    stripped = str(line or "").strip()
    if not stripped.startswith(_OVERFLOW_PREFIX):
        return None
    payload = stripped[len(_OVERFLOW_PREFIX):].strip()
    if not payload:
        return None
    parts = [_normalize_overflow_item_for_compare(part) for part in payload.split(";")]
    items = {part for part in parts if part}
    if not items:
        return None
    return frozenset(items)


def _existing_overflow_item_sets(text: str) -> set[frozenset[str]]:
    sets: set[frozenset[str]] = set()
    for line in _normalize_newlines(str(text or "")).splitlines():
        item_set = _extract_overflow_item_set_from_line(line)
        if item_set:
            sets.add(item_set)
    return sets


def _build_overflow_line(items: list[str]) -> str:
    cleaned: list[str] = []
    for item in items:
        token = _normalize_inline_action_item(item)
        while token.startswith(_OVERFLOW_PREFIX):
            token = token[len(_OVERFLOW_PREFIX):].strip()
        token = token.strip()
        if token:
            cleaned.append(token)
    return f"{_OVERFLOW_PREFIX} {'; '.join(cleaned)}".strip()


def _is_valid_residual_text(text: str) -> bool:
    token = _normalize_inline_action_item(text)
    if not token:
        return False
    if RESIDUAL_SENTENCE_END_RE.search(token):
        return True
    return len(RESIDUAL_VALID_CHAR_RE.findall(token)) >= int(RESIDUAL_MIN_CHARS)


def _select_inline_chain_splitter(content: str) -> tuple[re.Pattern[str], str] | None:
    token = str(content or "")
    if len(INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE.findall(token)) >= 2:
        return INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE, " - "
    if len(INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE.findall(token)) >= 2:
        return INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE, "; "
    if len(INLINE_ACTION_CHAIN_SLASH_SPLIT_RE.findall(token)) >= 2:
        return INLINE_ACTION_CHAIN_SLASH_SPLIT_RE, " / "
    return None


def _parse_inline_action_chain_line(line: str) -> tuple[list[str], str] | None:
    raw = str(line or "")
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.startswith(_OVERFLOW_PREFIX):
        return None
    if INLINE_ACTION_CHAIN_EXPLANATION_RE.search(stripped):
        return None
    if not (INLINE_ACTION_CHAIN_LIST_START_RE.match(stripped) or "->" in stripped):
        return None

    content = _INLINE_CHAIN_LEADING_PREFIX_RE.sub("", stripped, count=1)
    splitter = _select_inline_chain_splitter(content)
    if splitter is None:
        return None
    split_re, join_sep = splitter
    raw_parts = [part.strip() for part in split_re.split(content) if part and part.strip()]
    if len(raw_parts) < 2:
        return None

    actions: list[str] = []
    residual_parts: list[str] = []
    metrics_probe: dict[str, int | bool] = {"action_boundary_window_extended_count": 0}

    for part in raw_parts:
        boundary_end = _action_boundary_end(part, metrics_probe)
        if boundary_end is None:
            fallback = _normalize_inline_action_item(part)
            if "->" in part and len(fallback) >= 4:
                actions.append(fallback)
                continue
            residual_parts.append(part)
            continue

        head = _normalize_inline_action_item(part[:boundary_end])
        tail = _normalize_inline_action_item(part[boundary_end:])
        if not head or ACTIONABLE_HINT_RE.search(head) is None:
            residual_parts.append(part)
            continue
        actions.append(head)
        if tail:
            residual_parts.append(tail)

    normalized_actions: list[str] = []
    seen_action_keys: set[str] = set()
    for action in actions:
        key = _normalize_overflow_item_for_compare(action)
        if not key or key in seen_action_keys:
            continue
        seen_action_keys.add(key)
        normalized_actions.append(action)
    if len(normalized_actions) < 2:
        return None

    residual = _normalize_inline_action_item(" ".join(part for part in residual_parts if part))
    if residual:
        residual = residual.lstrip("-;:/ ").strip()
    if residual and not _is_valid_residual_text(residual):
        residual = ""

    # Keep deterministic shape for migration; chain separators are renderer-owned.
    _ = join_sep
    return normalized_actions, residual


def _append_unique_action_item(target: list[str], candidate: str) -> None:
    normalized = _normalize_inline_action_item(candidate)
    if not normalized:
        return
    key = _normalize_overflow_item_for_compare(normalized)
    for existing in target:
        if _normalize_overflow_item_for_compare(existing) == key:
            return
    target.append(normalized)


def _migrate_inline_action_chains_to_action_steps(chapter_key: str, chapter_body: str) -> tuple[str, dict[str, int | bool]]:
    metrics: dict[str, int | bool] = {
        "inline_action_chain_migrations": 0,
        "inline_action_chain_overflow_summaries": 0,
    }
    if not isinstance(chapter_body, str) or not chapter_body.strip():
        return chapter_body, metrics

    normalized = _normalize_newlines(chapter_body)
    timing_spans = _find_midterm_timing_map_spans(chapter_key, normalized)
    action_spans = _find_action_steps_blocks(normalized, timing_spans)
    protected_spans = list(timing_spans) + list(action_spans)

    lines = normalized.splitlines()
    offsets = _line_offsets(normalized)
    migrated_actions: list[str] = []
    rebuilt_lines: list[str] = []
    migrated_any = False

    for idx, line in enumerate(lines):
        start_offset = offsets[idx] if idx < len(offsets) else 0
        if _index_in_spans(start_offset, protected_spans):
            rebuilt_lines.append(line)
            continue
        parsed = _parse_inline_action_chain_line(line)
        if parsed is None:
            rebuilt_lines.append(line)
            continue
        actions, residual = parsed
        if len(actions) < 2:
            rebuilt_lines.append(line)
            continue
        migrated_any = True
        metrics["inline_action_chain_migrations"] = int(metrics["inline_action_chain_migrations"]) + 1
        for action in actions:
            _append_unique_action_item(migrated_actions, action)
        if residual:
            rebuilt_lines.append(residual)

    if not migrated_any:
        return normalized.strip(), metrics

    base_body = "\n".join(rebuilt_lines)
    base_body = re.sub(r"[ \t]+\n", "\n", base_body)
    base_body = re.sub(r"\n{3,}", "\n\n", base_body).strip()

    canonical = _resolve_actionable_canonical_key(chapter_key)
    actionable = _actionable_key_set_from_aliases()
    is_actionable = bool(canonical and canonical in actionable)

    spans_after = _find_midterm_timing_map_spans(chapter_key, base_body)
    action_blocks_after = _find_action_steps_blocks(base_body, spans_after)
    existing_action_items: list[str] = []
    if action_blocks_after:
        first_start, first_end = action_blocks_after[0]
        for bullet in _extract_action_bullets_from_block(base_body[first_start:first_end]):
            token = re.sub(r"^\s*-\s+", "", bullet).strip()
            _append_unique_action_item(existing_action_items, token)

    stripped_body = _remove_action_steps_sections(base_body, spans_after)
    existing_overflow_sets = _existing_overflow_item_sets(stripped_body)

    combined: list[str] = []
    for item in existing_action_items:
        _append_unique_action_item(combined, item)
    for item in migrated_actions:
        _append_unique_action_item(combined, item)

    kept_for_action = combined[:3] if is_actionable else []
    overflow_candidates = combined[3:] if is_actionable else combined

    overflow_line = ""
    if overflow_candidates:
        candidate_line = _build_overflow_line(overflow_candidates)
        candidate_set = _extract_overflow_item_set_from_line(candidate_line)
        if candidate_set and candidate_set not in existing_overflow_sets:
            overflow_line = candidate_line
            metrics["inline_action_chain_overflow_summaries"] = int(metrics["inline_action_chain_overflow_summaries"]) + 1

    action_block = ""
    if kept_for_action:
        action_block = "### Action Steps\n\n" + "\n".join(f"- {_normalize_inline_action_item(item)}" for item in kept_for_action)

    rebuilt_parts: list[str] = []
    core = stripped_body.strip()
    if core:
        rebuilt_parts.append(core)
    if overflow_line:
        rebuilt_parts.append(overflow_line)
    if action_block:
        rebuilt_parts.append(action_block)

    out = "\n\n".join(part for part in rebuilt_parts if part and part.strip()).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out, metrics


def _remove_action_steps_sections(chapter_body: str, spans: list[tuple[int, int]]) -> str:
    if not chapter_body:
        return chapter_body
    body = chapter_body
    while True:
        removed = False
        for match in _ACTION_STEPS_H3_RE.finditer(body):
            if _index_in_spans(match.start(), spans):
                continue
            block_end = _next_heading_boundary(body, match.end())
            body = (body[:match.start()] + body[block_end:]).strip()
            removed = True
            break
        if not removed:
            break
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def _dedupe_definition_span_in_block(block: str, *, dasha_seen: bool) -> tuple[str, bool, bool]:
    if not isinstance(block, str) or not block.strip():
        return block, dasha_seen, False
    matches = list(_DEFINITION_DASHA_SPAN_RE.finditer(block))
    if not matches:
        return block, dasha_seen, False

    # Keep first definition occurrence in the document untouched.
    if not dasha_seen:
        return block, True, False

    # Overly broad matches fallback to shrink mode only.
    if len(block) > _DEFINITION_SPAN_MAX_CHARS:
        return _DEFINITION_SHRINK_LINE, True, True

    # Remove only matched definition spans, preserving personalized remainder.
    parts: list[str] = []
    cursor = 0
    for match in matches:
        span = block[match.start():match.end()]
        if len(span) > _DEFINITION_SPAN_MAX_CHARS:
            return _DEFINITION_SHRINK_LINE, True, True
        parts.append(block[cursor:match.start()])
        cursor = match.end()
    parts.append(block[cursor:])
    candidate = _normalize_for_exact_dedupe(" ".join(part.strip() for part in parts if part and part.strip()))
    # Preserve personalized tails; shrink only when almost empty after definition removal.
    if len(candidate) < 12:
        return _DEFINITION_SHRINK_LINE, True, True
    return candidate, True, candidate != _normalize_for_exact_dedupe(block)


def dedupe_definition_blocks(md: str) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    sections = _split_h2_sections(md)
    if not sections:
        return _normalize_newlines(md).strip()

    changed = False
    dasha_seen = False
    for section in sections:
        body = str(section.get("body", ""))
        blocks = _split_paragraph_blocks_for_density(body)
        if not blocks:
            continue
        kept: list[str] = []
        for block in blocks:
            lines = [line for line in block.splitlines() if line.strip()]
            if _is_heading_only_block(lines) or _is_list_only_block(lines) or _is_code_fence_block(block) or _is_html_comment_block(block):
                kept.append(block)
                continue
            rebuilt, dasha_seen, block_changed = _dedupe_definition_span_in_block(block, dasha_seen=dasha_seen)
            if block_changed:
                changed = True
            kept.append(rebuilt)
        rebuilt_body = "\n\n".join(part for part in kept if part and str(part).strip()).strip()
        if rebuilt_body != body:
            section["body"] = rebuilt_body
            changed = True
    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def normalize_section_structure(md: str) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    sections = _split_h2_sections(md)
    if not sections:
        return _normalize_newlines(md).strip()

    changed = False
    for section in sections:
        heading = str(section.get("heading", ""))
        body = str(section.get("body", ""))

        if re.search(r"(?i)\baction\s+steps\b", heading):
            normalized_heading = re.sub(r"(?i)\s*action\s+steps\s*$", "", heading).rstrip()
            if normalized_heading != heading:
                section["heading"] = normalized_heading
                changed = True

        spans = _find_midterm_timing_map_spans(str(extract_h2_chapter_key(str(section.get("heading", ""))) or "").strip(), body)
        action_blocks = _find_action_steps_blocks(body, spans)
        if not action_blocks:
            continue
        first_start, first_end = action_blocks[0]
        first_block_text = body[first_start:first_end].strip()
        prefix = body[:first_start].strip()
        suffix = body[first_end:].strip()
        moved_body_parts: list[str] = []
        if prefix:
            moved_body_parts.append(prefix)
        if suffix:
            moved_body_parts.append(suffix)
        moved_body_parts.append(first_block_text)
        rebuilt = "\n\n".join(part for part in moved_body_parts if part and part.strip()).strip()
        rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
        if rebuilt != body:
            section["body"] = rebuilt
            changed = True

    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def enforce_action_steps_chapter_policy(md: str, actionable_chapter_keys: list[str]) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    del actionable_chapter_keys
    actionable = _actionable_key_set_from_aliases()
    if not actionable:
        return _normalize_newlines(md).strip()

    sections = _split_h2_sections(md)
    if not sections:
        return md

    changed = False
    for section in sections:
        heading = str(section.get("heading", ""))
        key = str(extract_h2_chapter_key(heading) or "").strip()
        canonical = _resolve_actionable_canonical_key(key)
        if not key or canonical in actionable:
            continue
        body = str(section.get("body", ""))
        spans = _find_midterm_timing_map_spans(key, body)
        stripped = _remove_action_steps_sections(body, spans)
        if stripped != body:
            section["body"] = stripped
            changed = True

    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def ensure_action_bullets(md: str, actionable_chapter_keys: list[str]) -> str:
    if not isinstance(md, str) or not md.strip():
        return md
    del actionable_chapter_keys
    actionable = _actionable_key_set_from_aliases()
    if not actionable:
        return _normalize_newlines(md).strip()

    sections = _split_h2_sections(md)
    if not sections:
        return md

    changed = False
    for section in sections:
        heading = str(section.get("heading", ""))
        key = str(extract_h2_chapter_key(heading) or "").strip()
        canonical = _resolve_actionable_canonical_key(key)
        if canonical not in actionable:
            continue

        body = str(section.get("body", ""))
        timing_protected_spans = _find_midterm_timing_map_spans(key, body)
        repaired_body, _inline_repairs = _repair_inline_bullets_in_action_steps(body, timing_protected_spans)
        action_blocks = _find_action_steps_blocks(repaired_body, timing_protected_spans)
        if action_blocks:
            first_start, first_end = action_blocks[0]
            first_block_text = repaired_body[first_start:first_end]
            first_bullets = _extract_action_bullets_from_block(first_block_text)[:3]
            if len(first_bullets) >= 2:
                trimmed_block = "### Action Steps\n\n" + "\n".join(first_bullets)
                stripped_body = _remove_action_steps_sections(repaired_body, timing_protected_spans)
                rebuilt_body = stripped_body.strip()
                if rebuilt_body:
                    rebuilt_body = f"{rebuilt_body}\n\n{trimmed_block}"
                else:
                    rebuilt_body = trimmed_block
                rebuilt_body = re.sub(r"\n{3,}", "\n\n", rebuilt_body).strip()
                if rebuilt_body != body:
                    section["body"] = rebuilt_body
                    changed = True
                continue

        timing_protected_spans = _find_midterm_timing_map_spans(key, repaired_body)
        stripped_body, collected_bullets = _collect_and_strip_non_protected_bullets(repaired_body, timing_protected_spans)

        bullets: list[str] = []
        for bullet in collected_bullets:
            _append_unique_bullet(bullets, bullet)
            if len(bullets) >= 3:
                break

        if len(bullets) < 3:
            for sentence in _extract_candidate_action_sentences(stripped_body):
                _append_unique_bullet(bullets, f"- {sentence}")
                if len(bullets) >= 3:
                    break

        if len(bullets) < 2:
            for toolkit_item in _toolkit_for_chapter(key):
                _append_unique_bullet(bullets, f"- {toolkit_item}")
                if len(bullets) >= 2:
                    break
        if len(bullets) < 2:
            for fallback in _GENERIC_ACTION_BULLETS:
                _append_unique_bullet(bullets, fallback)
                if len(bullets) >= 2:
                    break

        if len(bullets) < 2:
            toolkit = _toolkit_for_chapter(key)
            if len(toolkit) >= 2:
                bullets = [f"- {toolkit[0]}", f"- {toolkit[1]}"]
            else:
                bullets = list(_GENERIC_ACTION_BULLETS[:2])
        else:
            bullets = bullets[:3]

        append_block = "### Action Steps\n\n" + "\n".join(bullets)
        rebuilt_body = stripped_body.strip()
        if rebuilt_body:
            rebuilt_body = f"{rebuilt_body}\n\n{append_block}"
        else:
            rebuilt_body = append_block
        section["body"] = rebuilt_body.strip()
        changed = True

    return _join_h2_sections(sections) if changed else _normalize_newlines(md).strip()


def _diversify_action_steps(md: str) -> tuple[str, dict[str, int]]:
    metrics: dict[str, int] = {
        "action_steps_non_actionable_rewrites": 0,
        "action_steps_duplicate_replacements": 0,
    }
    if not isinstance(md, str) or not md.strip():
        return md, metrics

    sections = _split_h2_sections(md)
    if not sections:
        return md, metrics

    actionable = _actionable_key_set_from_aliases()
    global_counts: dict[str, int] = {}
    changed = False

    for section in sections:
        heading = str(section.get("heading", ""))
        chapter_key = str(extract_h2_chapter_key(heading) or "").strip()
        canonical = _resolve_actionable_canonical_key(chapter_key)
        if not canonical or canonical not in actionable:
            continue

        body = str(section.get("body", ""))
        spans = _find_midterm_timing_map_spans(chapter_key, body)
        blocks = _find_action_steps_blocks(body, spans)
        if not blocks:
            continue

        first_start, first_end = blocks[0]
        block_text = body[first_start:first_end]
        raw_bullets = _collect_action_bullets_in_block(block_text)
        if not raw_bullets:
            continue

        used_fps: set[str] = set()
        rebuilt_bullets: list[str] = []

        for bullet in raw_bullets:
            candidate = re.sub(r"^\s*-\s+", "- ", bullet.strip())
            content = re.sub(r"^\s*-\s+", "", candidate).strip()
            fp = _normalize_action_bullet_fingerprint(content)
            actionable_ok = _is_actionable_bullet_text(content)
            explanatory = _is_explanatory_bullet_text(content)
            duplicate_or_banned = bool(fp and (fp in ACTION_STEPS_GENERIC_BANNED_FPS or global_counts.get(fp, 0) >= 2))
            needs_rewrite = bool((explanatory and not actionable_ok) or duplicate_or_banned)

            if needs_rewrite:
                replacement = _pick_toolkit_bullet(chapter_key, used_fps, global_counts)
                if replacement:
                    if explanatory and not actionable_ok:
                        metrics["action_steps_non_actionable_rewrites"] = int(metrics["action_steps_non_actionable_rewrites"]) + 1
                    if duplicate_or_banned:
                        metrics["action_steps_duplicate_replacements"] = int(metrics["action_steps_duplicate_replacements"]) + 1
                    candidate = replacement
                    content = re.sub(r"^\s*-\s+", "", candidate).strip()
                    fp = _normalize_action_bullet_fingerprint(content)
                    actionable_ok = _is_actionable_bullet_text(content)
                    changed = True

            if not actionable_ok:
                replacement = _pick_toolkit_bullet(chapter_key, used_fps, global_counts) or _GENERIC_ACTION_BULLETS[0]
                candidate = re.sub(r"^\s*-\s+", "- ", replacement.strip())
                content = re.sub(r"^\s*-\s+", "", candidate).strip()
                fp = _normalize_action_bullet_fingerprint(content)
                actionable_ok = _is_actionable_bullet_text(content)
                metrics["action_steps_non_actionable_rewrites"] = int(metrics["action_steps_non_actionable_rewrites"]) + 1
                changed = True

            if not actionable_ok:
                continue

            if fp:
                if fp in used_fps:
                    continue
                used_fps.add(fp)
                global_counts[fp] = int(global_counts.get(fp, 0)) + 1
            rebuilt_bullets.append(candidate)
            if len(rebuilt_bullets) >= 3:
                break

        while len(rebuilt_bullets) < 2:
            replacement = _pick_toolkit_bullet(chapter_key, used_fps, global_counts)
            if replacement is None:
                replacement = _GENERIC_ACTION_BULLETS[len(rebuilt_bullets) % len(_GENERIC_ACTION_BULLETS)]
            candidate = re.sub(r"^\s*-\s+", "- ", replacement.strip())
            content = re.sub(r"^\s*-\s+", "", candidate).strip()
            fp = _normalize_action_bullet_fingerprint(content)
            if fp and fp in used_fps:
                break
            rebuilt_bullets.append(candidate)
            if fp:
                used_fps.add(fp)
                global_counts[fp] = int(global_counts.get(fp, 0)) + 1
            changed = True

        if not rebuilt_bullets:
            continue

        rebuilt_block = "### Action Steps\n\n" + "\n".join(rebuilt_bullets[:3])
        stripped_body = _remove_action_steps_sections(body, spans)
        rebuilt_body = stripped_body.strip()
        if rebuilt_body:
            rebuilt_body = f"{rebuilt_body}\n\n{rebuilt_block}"
        else:
            rebuilt_body = rebuilt_block
        rebuilt_body = re.sub(r"\n{3,}", "\n\n", rebuilt_body).strip()
        if rebuilt_body != body:
            section["body"] = rebuilt_body
            changed = True

    return (_join_h2_sections(sections) if changed else _normalize_newlines(md).strip()), metrics


def _count_dasha_definition_occurrences_in_chapters(text: str) -> int:
    return int(len(DASHA_DEFINITION_RE.findall(str(text or ""))))


def _insert_dasha_definition_once(chapters_text: str) -> tuple[str, bool]:
    sections = _split_h2_sections(chapters_text)
    if not sections:
        return chapters_text, False
    first = sections[0]
    body = str(first.get("body", ""))
    lines = _normalize_newlines(body).splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.strip():
            insert_at = idx
            break
    if lines and lines[insert_at].strip().startswith("### Action Steps"):
        insert_at = 0
    prefix = lines[:insert_at]
    suffix = lines[insert_at:]
    inserted_lines = prefix + [DASHA_DEFINITION_CANONICAL_LINE, ""] + suffix
    first["body"] = "\n".join(inserted_lines).strip()
    return _join_h2_sections(sections), True


def _insert_dasha_shrink_once(chapters_text: str) -> str:
    sections = _split_h2_sections(chapters_text)
    if not sections:
        return chapters_text
    first = sections[0]
    body = str(first.get("body", ""))
    lines = _normalize_newlines(body).splitlines()
    if lines:
        first["body"] = "\n".join([DASHA_DEFINITION_SHRINK_LINE, ""] + lines).strip()
    else:
        first["body"] = DASHA_DEFINITION_SHRINK_LINE
    return _join_h2_sections(sections)


def _count_nested_dasha_phrase_violations(text: str) -> int:
    return int(len(DASHA_NESTED_PHRASE_RE.findall(str(text or ""))))


def _count_dasha_definition_redundancy_violations(text: str) -> int:
    token = str(text or "")
    if not token.strip():
        return 0
    count = 0
    for line in _normalize_newlines(token).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Evaluate redundancy only inside definition-template-like lines.
        if DASHA_DEFINITION_RE.search(stripped) is None and "다샤" not in stripped:
            continue
        if DASHA_DEFINITION_REDUNDANCY_RE.search(stripped):
            count += 1
    return int(count)


def postprocess_commercial_quality_with_metrics(text: str) -> tuple[str, dict[str, int | bool]]:
    metrics = dict(_QUALITY_METRICS_DEFAULT)
    if not isinstance(text, str) or not text:
        metrics["commercial_quality_metrics_valid"] = False
        return text, metrics
    out = _normalize_newlines(text)
    # Stage 1 (CHAPTERS-only): structural normalization.
    out = _apply_commercial_surface_firewall(out)
    out = ensure_min_body_paragraphs_per_chapter(out)
    out = dedupe_exact_paragraphs(out)
    out = normalize_section_structure(out)
    sections = _split_h2_sections(out)
    if not sections:
        metrics["commercial_quality_metrics_valid"] = False
        metrics["definition_dasha_occurrences_after"] = 0
        return out.strip(), metrics
    if len(sections) == 1 and str(sections[0].get("heading", "")).strip() == "## Document":
        metrics["commercial_quality_metrics_valid"] = False
        metrics["definition_dasha_occurrences_after"] = 0
        return out.strip(), metrics
    repaired_sections: list[dict[str, str]] = []
    for section in sections:
        heading = str(section.get("heading", ""))
        body = str(section.get("body", ""))
        chapter_key = str(extract_h2_chapter_key(heading) or "").strip()
        split_body, heading_repairs = _split_inline_action_steps_heading_in_chapters(body)
        if heading_repairs:
            metrics["action_steps_inline_heading_repairs"] = int(metrics.get("action_steps_inline_heading_repairs", 0)) + int(heading_repairs)
        # Stage 2 (CHAPTERS-only): contaminated Action Steps line repair.
        repaired_body, local_metrics = _repair_action_steps_contaminated_lines(chapter_key, split_body)
        # Stage 3 (CHAPTERS-only): inline execution-chain migration.
        migrated_body, migration_metrics = _migrate_inline_action_chains_to_action_steps(chapter_key, repaired_body)
        for k in (
            "action_boundary_window_extended_count",
            "action_steps_contaminated_line_repairs",
            "action_steps_tail_move_count",
            "action_steps_tail_move_dedup_skips",
        ):
            metrics[k] = int(metrics.get(k, 0)) + int(local_metrics.get(k, 0))
        for k in ("inline_action_chain_migrations", "inline_action_chain_overflow_summaries"):
            metrics[k] = int(metrics.get(k, 0)) + int(migration_metrics.get(k, 0))
        metrics["inline_execution_lines_moved"] = int(metrics.get("inline_execution_lines_moved", 0)) + int(
            migration_metrics.get("inline_action_chain_migrations", 0)
        )
        repaired_sections.append({"heading": heading, "body": migrated_body})
    out = _join_h2_sections(repaired_sections)
    # Stage 4 (CHAPTERS-only): enforce chapter policy.
    out = enforce_action_steps_chapter_policy(out, _ACTIONABLE_CHAPTER_KEYS)
    # Stage 5 (CHAPTERS-only): actionable filter + chapter-wise diversification.
    out, diversity_metrics = _diversify_action_steps(out)
    metrics["action_steps_non_actionable_rewrites"] = int(metrics.get("action_steps_non_actionable_rewrites", 0)) + int(
        diversity_metrics.get("action_steps_non_actionable_rewrites", 0)
    )
    metrics["action_steps_duplicate_replacements"] = int(metrics.get("action_steps_duplicate_replacements", 0)) + int(
        diversity_metrics.get("action_steps_duplicate_replacements", 0)
    )
    # Stage 6 (CHAPTERS-only): ensure minimum Action Steps bullets.
    out = ensure_action_bullets(out, _ACTIONABLE_CHAPTER_KEYS)
    # Stage 7 (CHAPTERS-only): re-enforce chapter policy after filling.
    out = enforce_action_steps_chapter_policy(out, _ACTIONABLE_CHAPTER_KEYS)
    # Stage 8 (CHAPTERS-only): definition dedupe/cleanup.
    out = dedupe_definition_blocks(out)
    before_insert = out
    dasha_count = _count_dasha_definition_occurrences_in_chapters(out)
    if dasha_count == 0:
        inserted_text, inserted = _insert_dasha_definition_once(out)
        if inserted:
            post_insert_count = _count_dasha_definition_occurrences_in_chapters(inserted_text)
            if post_insert_count == 1:
                out = inserted_text
                dasha_count = 1
            else:
                # Rollback + fail-safe; do not retry insertion in this run.
                out = before_insert
                metrics["dasha_insert_retry_blocked"] = True
                if DASHA_DEFINITION_SHRINK_LINE not in out:
                    out = _insert_dasha_shrink_once(out)
                dasha_count = _count_dasha_definition_occurrences_in_chapters(out)
    nested_before = _count_nested_dasha_phrase_violations(out)
    redundancy_before = _count_dasha_definition_redundancy_violations(out)
    # Stage 10 (CHAPTERS-only): final cleanup/firewall.
    out = _apply_commercial_surface_firewall(out)
    out = _cleanup_vedic_redundancy(out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()
    nested_after = _count_nested_dasha_phrase_violations(out)
    redundancy_after = _count_dasha_definition_redundancy_violations(out)
    metrics["definition_dasha_occurrences_after"] = int(dasha_count)
    metrics["definition_nested_phrase_violations_after"] = int(nested_after)
    metrics["dasha_definition_nested_pattern_repairs"] = int(max(0, nested_before - nested_after))
    metrics["dasha_definition_redundancy_repairs"] = int(max(0, redundancy_before - redundancy_after))
    return out, metrics


def postprocess_commercial_quality(text: str) -> str:
    out, _metrics = postprocess_commercial_quality_with_metrics(text)
    return out


def _split_front_and_chapters(text: str) -> tuple[str, str, bool, bool, int]:
    raw = _normalize_newlines(text)
    front_match = _FRONT_SEGMENT_RE.search(raw)
    chapters_match = _CHAPTER_SEGMENT_RE.search(raw)
    if front_match and chapters_match and front_match.start() < chapters_match.start():
        front_start = front_match.start("front")
        front_end = front_match.end("front")
        # Boundary offset is defined as the end of FRONT segment in source text.
        return (
            str(front_match.group("front") or ""),
            str(chapters_match.group("chapters") or ""),
            True,
            False,
            int(front_end),
        )
    if front_match and not chapters_match:
        logger.warning("front_sentinel_invalid missing_chapter_segment")
        return str(front_match.group("front") or ""), "", False, True, int(front_match.end("front"))
    if chapters_match and not front_match:
        logger.warning("front_sentinel_invalid missing_front_segment")
        return "", str(chapters_match.group("chapters") or ""), False, True, int(chapters_match.start("chapters"))

    # Sentinel fallback: parse using first H2 chapter boundary.
    match = _CHAPTER_H2_WITH_KEY_RE.search(raw)
    if match is None:
        return raw, "", False, True, len(raw)
    front = raw[:match.start()]
    chapters = raw[match.start():]
    return front, chapters, False, True, int(match.start())


def _preserve_front_checkbox_layout(front_text: str) -> str:
    if not front_text:
        return ""
    lines_out: list[str] = []
    for line in _normalize_newlines(front_text).splitlines():
        if _FRONT_CHECKBOX_LINE_RE.match(line):
            lines_out.append(line.rstrip())
            continue
        lines_out.append(line.rstrip())
    # Keep front layout stable: do not compress block spacing aggressively.
    return "\n".join(lines_out).strip()


def _extract_front_playbook_span(front_text: str) -> tuple[int, int] | None:
    raw = _normalize_newlines(str(front_text or ""))
    start_match = PLAYBOOK_START_RE.search(raw)
    if start_match is None:
        return None
    start = start_match.start()
    end = len(raw)
    for end_match in PLAYBOOK_END_RE.finditer(raw, start_match.end()):
        if end_match.start() > start:
            end = end_match.start()
            break
    return start, end


def _extract_one_page_summary_span(front_text: str) -> tuple[int, int] | None:
    raw = _normalize_newlines(str(front_text or ""))
    start_match = ONE_PAGE_SUMMARY_START_RE.search(raw)
    if start_match is None:
        return None
    start = start_match.start()
    end = len(raw)
    for end_match in TOP_LEVEL_H1_RE.finditer(raw, start_match.end()):
        if end_match.start() > start:
            end = end_match.start()
            break
    return start, end


def _collapse_immediate_repeat_phrase(text: str) -> tuple[str, int]:
    token = str(text or "")
    repairs = 0
    patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"결정이\s*빨라져\s*결정\s*속도가\s*빨라져"), "결정 속도가 빨라져"),
        (
            re.compile(r"(?P<seg>[가-힣A-Za-z0-9]{2,}(?:\s+[가-힣A-Za-z0-9]{2,}){1,3})\s+(?P=seg)"),
            r"\g<seg>",
        ),
    )
    for pattern, replacement in patterns:
        while True:
            token_new, n = pattern.subn(replacement, token, count=1)
            if n == 0:
                break
            token = token_new
            repairs += n
    return token, repairs


def _dedupe_one_page_summary_block(summary_text: str) -> tuple[str, int]:
    lines = _normalize_newlines(str(summary_text or "")).splitlines()
    out_lines: list[str] = []
    repairs = 0
    prev_sentence_norm = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            prev_sentence_norm = ""
            continue
        if re.match(r"^\s*#", stripped):
            out_lines.append(stripped)
            prev_sentence_norm = ""
            continue
        if re.match(r"^\s*(?:-\s+|•\s+|\d+\.\s+)", stripped):
            out_lines.append(stripped)
            prev_sentence_norm = ""
            continue

        collapsed, phrase_repairs = _collapse_immediate_repeat_phrase(stripped)
        repairs += int(phrase_repairs)

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(collapsed) if s and s.strip()]
        if not sentences:
            out_lines.append(collapsed.strip())
            prev_sentence_norm = ""
            continue

        kept_sentences: list[str] = []
        for sentence in sentences:
            norm = _normalize_for_exact_dedupe(sentence)
            if norm and norm == prev_sentence_norm:
                repairs += 1
                continue
            kept_sentences.append(sentence)
            prev_sentence_norm = norm
        rebuilt = " ".join(kept_sentences).strip()
        if not rebuilt:
            # Entire line was duplicate of immediately previous sentence.
            continue
        out_lines.append(rebuilt)

    rebuilt_text = "\n".join(out_lines)
    rebuilt_text = re.sub(r"\n{3,}", "\n\n", rebuilt_text).strip()
    return rebuilt_text, repairs


def _dedupe_one_page_summary_front(front_text: str) -> tuple[str, int]:
    raw = _normalize_newlines(str(front_text or ""))
    span = _extract_one_page_summary_span(raw)
    if span is None:
        return raw.strip(), 0
    start, end = span
    block = raw[start:end]
    deduped, repairs = _dedupe_one_page_summary_block(block)
    if repairs <= 0:
        return raw.strip(), 0
    prefix = raw[:start].rstrip()
    suffix = raw[end:].lstrip()
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(deduped)
    if suffix:
        parts.append(suffix)
    out = "\n\n".join(parts).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out, repairs


def _normalize_playbook_rule_text(rule_text: str, *, label: str) -> str:
    token = _normalize_for_exact_dedupe(rule_text)
    parts = [part.strip() for part in re.split(r"\s*/\s*", token) if part and part.strip()]
    if len(parts) != 2:
        safe = _PLAYBOOK_SAFE_SLOT_BY_LABEL.get(label, _PLAYBOOK_SAFE_SLOT_BY_LABEL["이번 달"])
        parts = [safe["rule_left"], safe["rule_right"]]
    left = _normalize_for_exact_dedupe(parts[0]).replace("/", " ")
    right = _normalize_for_exact_dedupe(parts[1]).replace("/", " ")
    return f"{left}{RULE_SEPARATOR}{right}"


def _build_safe_playbook_slot_lines(label: str) -> list[str]:
    safe = _PLAYBOOK_SAFE_SLOT_BY_LABEL.get(label, _PLAYBOOK_SAFE_SLOT_BY_LABEL["이번 달"])
    rule = f"{safe['rule_left']}{RULE_SEPARATOR}{safe['rule_right']}"
    return [
        label,
        f"{PLAYBOOK_LINE_CAUTION_PREFIX} {safe['caution']}",
        f"{PLAYBOOK_LINE_RULE_PREFIX} {rule}",
        f"{PLAYBOOK_LINE_REASON_PREFIX} {safe['reason']}",
    ]


def _extract_playbook_slot_parts(label: str, raw_lines: list[str]) -> list[str]:
    joined = _normalize_for_exact_dedupe(" ".join(line.strip() for line in raw_lines if line and line.strip()))
    if not joined:
        return _build_safe_playbook_slot_lines(label)
    if PLAYBOOK_LINE_CAUTION_PREFIX not in joined and "주의:" in joined:
        joined = joined.replace("주의:", PLAYBOOK_LINE_CAUTION_PREFIX, 1)
    if PLAYBOOK_LINE_RULE_PREFIX not in joined and "규칙:" in joined:
        joined = joined.replace("규칙:", PLAYBOOK_LINE_RULE_PREFIX, 1)
    if PLAYBOOK_LINE_REASON_PREFIX not in joined and "이유:" in joined:
        joined = joined.replace("이유:", PLAYBOOK_LINE_REASON_PREFIX, 1)

    caution_match = _PLAYBOOK_CAUTION_CAPTURE_RE.search(joined)
    rule_match = _PLAYBOOK_RULE_CAPTURE_RE.search(joined)
    reason_match = _PLAYBOOK_REASON_CAPTURE_RE.search(joined)
    if caution_match is None or rule_match is None or reason_match is None:
        return _build_safe_playbook_slot_lines(label)

    caution = _normalize_for_exact_dedupe(caution_match.group(1))
    rule = _normalize_playbook_rule_text(str(rule_match.group(1) or ""), label=label)
    reason = _normalize_for_exact_dedupe(reason_match.group(1))
    if not caution or not reason:
        return _build_safe_playbook_slot_lines(label)
    return [
        label,
        f"{PLAYBOOK_LINE_CAUTION_PREFIX} {caution}",
        f"{PLAYBOOK_LINE_RULE_PREFIX} {rule}",
        f"{PLAYBOOK_LINE_REASON_PREFIX} {reason}",
    ]


def _build_strict_playbook_slot_block(label: str, raw_lines: list[str]) -> list[str]:
    slot_lines = _extract_playbook_slot_parts(label, raw_lines)
    if not _validate_playbook_slot_lines(slot_lines):
        return _build_safe_playbook_slot_lines(label)
    return slot_lines


def _parse_playbook_slots_from_span(span_text: str) -> dict[str, list[str]]:
    lines = _normalize_newlines(span_text).splitlines()
    slot_raw: dict[str, list[str]] = {label: [] for label in PLAYBOOK_LABELS}
    current_label: str | None = None

    # Skip the playbook heading line and parse slot content only.
    for raw in lines[1:]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped in PLAYBOOK_LABELS:
            current_label = stripped
            continue
        inline = _PLAYBOOK_INLINE_LABEL_RE.match(stripped)
        if inline is not None:
            current_label = str(inline.group(1))
            rest = _normalize_for_exact_dedupe(str(inline.group(2) or ""))
            if rest:
                slot_raw[current_label].append(f"{PLAYBOOK_LINE_CAUTION_PREFIX} {rest}")
            continue
        if current_label is not None:
            slot_raw[current_label].append(stripped)
    return slot_raw


def _validate_playbook_slot_lines(slot_lines: list[str]) -> bool:
    if len(slot_lines) != 4:
        return False
    if slot_lines[0].strip() not in PLAYBOOK_LABELS:
        return False
    caution = slot_lines[1].strip()
    rule = slot_lines[2].strip()
    reason = slot_lines[3].strip()
    if not caution.startswith(PLAYBOOK_LINE_CAUTION_PREFIX):
        return False
    if not reason.startswith(PLAYBOOK_LINE_REASON_PREFIX):
        return False
    if not rule.startswith(PLAYBOOK_LINE_RULE_PREFIX):
        return False
    rule_text = rule[len(PLAYBOOK_LINE_RULE_PREFIX):].strip()
    return rule_text.count("/") == 1 and RULE_SEPARATOR in rule_text


def _validate_front_playbook_contract(playbook_text: str) -> tuple[bool, int]:
    lines = _normalize_newlines(playbook_text).splitlines()
    slot_ok_count = 0
    line_count = len(lines)
    for idx, line in enumerate(lines):
        label = line.strip()
        if label not in PLAYBOOK_LABELS:
            continue
        block = [label]
        for look_ahead in range(1, 4):
            if idx + look_ahead >= line_count:
                break
            block.append(lines[idx + look_ahead].strip())
        if _validate_playbook_slot_lines(block):
            slot_ok_count += 1
    return slot_ok_count == 3, slot_ok_count


def _build_safe_playbook_span() -> str:
    out: list[str] = ["# 3개월 플레이북", ""]
    for idx, label in enumerate(PLAYBOOK_LABELS):
        out.extend(_build_safe_playbook_slot_lines(label))
        if idx < len(PLAYBOOK_LABELS) - 1:
            out.append("")
    return "\n".join(out).strip()


def enforce_front_playbook_contract(front_text: str) -> tuple[str, bool, bool, int]:
    raw_front = _normalize_newlines(str(front_text or ""))
    span = _extract_front_playbook_span(raw_front)
    if span is None:
        return raw_front.strip(), False, False, 0

    start, end = span
    playbook_span = raw_front[start:end]
    slots_raw = _parse_playbook_slots_from_span(playbook_span)

    rebuilt_lines: list[str] = ["# 3개월 플레이북", ""]
    for idx, label in enumerate(PLAYBOOK_LABELS):
        slot_lines = _build_strict_playbook_slot_block(label, slots_raw.get(label, []))
        if not _validate_playbook_slot_lines(slot_lines):
            slot_lines = _build_safe_playbook_slot_lines(label)
        rebuilt_lines.extend(slot_lines)
        if idx < len(PLAYBOOK_LABELS) - 1:
            rebuilt_lines.append("")
    rebuilt_span = "\n".join(rebuilt_lines).strip()
    valid, slot_ok_count = _validate_front_playbook_contract(rebuilt_span)
    if not valid:
        rebuilt_span = _build_safe_playbook_span()
        valid, slot_ok_count = _validate_front_playbook_contract(rebuilt_span)

    prefix = _normalize_newlines(raw_front[:start]).strip()
    suffix = _normalize_newlines(raw_front[end:]).strip()
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(rebuilt_span.strip())
    if suffix:
        parts.append(suffix)
    new_front = "\n\n".join(parts)
    new_front = re.sub(r"\n{3,}", "\n\n", new_front).strip()
    repaired = _normalize_newlines(raw_front).strip() != new_front
    return new_front, repaired, True, slot_ok_count


def _build_run_id(text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    run_id = f"{stamp}_{digest}"
    run_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
    return run_id


def _write_front_debug_if_enabled(text_with_sentinels: str) -> None:
    save_enabled = str(os.getenv("FRONT_DEBUG_SAVE_SENTINELS", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
    archive_enabled = str(os.getenv("FRONT_DEBUG_ARCHIVE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
    if not save_enabled:
        return
    debug_dir = Path("logs") / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    latest_path = debug_dir / "front_with_sentinels.latest.md"
    latest_path.write_text(text_with_sentinels, encoding="utf-8")
    if archive_enabled:
        run_path = debug_dir / f"front_with_sentinels.{_build_run_id(text_with_sentinels)}.md"
        run_path.write_text(text_with_sentinels, encoding="utf-8")


def _strip_surface_sentinels(text: str) -> str:
    out = _SENTINEL_LINE_RE.sub("", _normalize_newlines(text))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sanitize_commercial_surface_with_front_protection_with_metrics(text: str) -> tuple[str, dict[str, int | bool]]:
    metrics: dict[str, int | bool] = {
        "one_page_summary_dedup_repairs": 0,
        "chapters_boundary_fallback_used": False,
        "front_byte_equal_after_b_pass": True,
        "front_end_offset_stable": True,
    }
    if not isinstance(text, str) or not text.strip():
        return str(text or ""), metrics
    raw = _normalize_newlines(text).strip()
    front_original, chapters, split_ok, fallback_used, front_end_offset = _split_front_and_chapters(raw)
    metrics["chapters_boundary_fallback_used"] = bool(fallback_used)
    # Keep FRONT byte-identical to source text across chapter-only B passes.
    front_clean = front_original
    if front_clean.strip():
        # Stage 9 (FRONT-only): one-page summary dedupe. Playbook/system blocks remain immutable.
        front_clean, summary_repairs = _dedupe_one_page_summary_front(front_clean)
        metrics["one_page_summary_dedup_repairs"] = int(summary_repairs)
    metrics["front_byte_equal_after_b_pass"] = bool(_normalize_newlines(front_original) == _normalize_newlines(front_clean))
    metrics["front_end_offset_stable"] = True
    if not chapters:
        cleaned = front_clean
        with_sentinels = f"{_FRONT_START_SENTINEL}\n{cleaned}\n{_FRONT_END_SENTINEL}".strip() if cleaned else cleaned
        if fallback_used:
            logger.warning("chapters_boundary_fallback_used=true front_end_offset=%s", front_end_offset)
        _write_front_debug_if_enabled(with_sentinels)
        strip_enabled = str(os.getenv("FRONT_SENTINEL_STRIP", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
        return (_strip_surface_sentinels(with_sentinels) if strip_enabled else with_sentinels), metrics

    chapter_clean = _normalize_newlines(chapters)
    if split_ok:
        chapter_clean = postprocess_commercial_quality(chapter_clean)
    else:
        if not (front_clean or "").strip() and chapter_clean.strip():
            # Chapters-only payload is safe to process even when sentinel split falls back.
            chapter_clean = postprocess_commercial_quality(chapter_clean)
        else:
            # On boundary fallback with non-empty FRONT, skip chapter-level B transforms to prevent FRONT contamination.
            logger.warning("chapters_boundary_fallback_used=true front_end_offset=%s", front_end_offset)
            chapter_clean = chapter_clean.strip()
    if front_clean:
        combined = (
            f"{_FRONT_START_SENTINEL}\n{front_clean}\n{_FRONT_END_SENTINEL}\n\n"
            f"{_CHAPTERS_START_SENTINEL}\n{chapter_clean}\n{_CHAPTERS_END_SENTINEL}"
        ).strip()
    else:
        combined = f"{_CHAPTERS_START_SENTINEL}\n{chapter_clean}\n{_CHAPTERS_END_SENTINEL}".strip()
    _write_front_debug_if_enabled(combined)
    strip_enabled = str(os.getenv("FRONT_SENTINEL_STRIP", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
    return (_strip_surface_sentinels(combined) if strip_enabled else combined), metrics


def sanitize_commercial_surface_with_front_protection(text: str) -> str:
    out, _metrics = sanitize_commercial_surface_with_front_protection_with_metrics(text)
    return out


def commercial_dejargonize(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    out = _apply_commercial_surface_firewall(out)
    out = _DEJARGON_SIGN_RISE_RE.sub(r"\1자리 라그나(Lagna)", out)
    for pattern, repl in _DEJARGON_STANDALONE_REPLACEMENTS:
        out = pattern.sub(repl, out)
    out = _apply_commercial_surface_firewall(out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_internal_artifacts(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pattern in _INTERNAL_ARTIFACT_PATTERNS:
        out = pattern.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def postprocess_reading_markdown_surface(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    out = _normalize_newlines(text)
    out = _soft_wrap_join(out)
    out = _fix_broken_risk_h2(out)
    # Step2 must run before Step3 because timing-map span extraction expects H3 heading.
    out = _promote_timing_map_heading_mid_term_only(out)
    out = _normalize_timing_map_body(out)
    out = _fix_broken_risk_h2(out)
    out = _cleanup_vedic_redundancy(out)
    out = _cleanup_garbled_tokens(out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
