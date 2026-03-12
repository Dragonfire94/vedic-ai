from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.astro_engine import build_structural_summary
from backend.golden_sample_runner import _candidate_metrics, generate_golden_charts, select_profiles
from backend.llm_output_scanner import scan_forbidden_patterns
from backend.llm_service import _derive_narrative_mode, build_llm_structural_prompt, normalize_llm_layout_strict
from backend.main import (
    _apply_style_remediation,
    _compute_body_paragraph_density_metrics,
    _reading_style_error_codes,
    _render_chapter_blocks_deterministic,
    app,
    get_chart,
)
from backend.output_surface_postprocess import (
    postprocess_commercial_quality,
    postprocess_commercial_quality_with_metrics,
    sanitize_commercial_surface_with_front_protection_with_metrics,
)
from backend.commercial_quality_constants import (
    ACTIONABLE_HINT_RE,
    ACTION_STEPS_INLINE_HEADING_RE,
    DASHA_DEFINITION_RE,
    DASHA_DEFINITION_REDUNDANCY_RE,
    DASHA_NESTED_PHRASE_RE,
    INLINE_ACTION_CHAIN_EXPLANATION_RE,
    INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE,
    INLINE_ACTION_CHAIN_LIST_START_RE,
    INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE,
    INLINE_ACTION_CHAIN_SLASH_SPLIT_RE,
    TRAIL_SENTENCE_END_RE,
)
from backend.commercial_surface_renderer import render_fallback_front_modules
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
from backend.report_engine import build_dasha_narrative_context, build_report_payload, build_semantic_signals
from backend.vedic_lexicon import YEAR_QUARTER_PATTERNS, scan_timing_map_contract, scan_vedic_term_budget


OUT_DIR = Path("logs/cheap_validation_gate")
HASH_GUARD_PATH = OUT_DIR / "prompt_hash_guard.json"
logger = logging.getLogger("cheap_validation_gate")
_ACTIONABLE_CHAPTER_KEYS = {
    "Current Phase",
    "Career & Money",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Mid-Term Direction",
    "Risk Management Points",
    "Growth Acceleration",
}
_H2_HEADING_RE = re.compile(r"(?m)^\s*##\s+(?:\[\s*([^\]]+)\s*\]\s*(?:.*)?|(.+))$")
_BULLET_LINE_RE = re.compile(r"(?m)^\s*-\s+\S")
_ACTION_STEPS_H3_RE = re.compile(r"(?im)^\s*###\s*Action Steps\s*$")
_ANY_HEADING_LINE_RE = re.compile(r"(?m)^\s*#{2,6}\s+")
_FRONT_H1_RE = re.compile(r"(?m)^#\s+(.+?)\s*$")
_FRONT_SCENE_DOMAIN_RE = re.compile(r"(?m)^\s*-\s*\[(일|돈|관계)\]\s+")
_FRONT_CHECKBOX_RE = re.compile(r"(?m)^\s*-\s*\[[ xX]\]\s+")
_FRONT_TRIGGER_TERMS = ("피곤", "감정", "압박", "갈등", "모호", "기대치", "버틴", "올라온", "상황", "날")
_FRONT_START_SENTINEL = "<!-- FRONT_START -->"
_FRONT_END_SENTINEL = "<!-- FRONT_END -->"
_CHAPTERS_START_SENTINEL = "<!-- CHAPTERS_START -->"
_CHAPTERS_END_SENTINEL = "<!-- CHAPTERS_END -->"
_FRONT_SEGMENT_RE = re.compile(
    r"<!--\s*FRONT_START\s*-->\s*(?P<front>.*?)\s*<!--\s*FRONT_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_CHAPTER_SEGMENT_RE = re.compile(
    r"<!--\s*CHAPTERS_START\s*-->\s*(?P<chapters>.*?)\s*<!--\s*CHAPTERS_END\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_FRONT_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]+")
_SYNTHETIC_ACTION_H2_RE = re.compile(r"(?mi)^\s*##\s+.*\bAction Steps\b\s*$")
_DEFINITION_SHRINK_LINE = "(용어 설명은 상단 참조)"
_INLINE_CHAIN_LEADING_PREFIX_RE = re.compile(r"^\s*(?:-\s+|•\s+|\d+\.\s+)")
_LIFECYCLE_REQUIRED_H2 = [
    "cover/meta",
    "How to use 1p",
    "인생 구조 한 장 요약",
    "4단계 인생 구조",
    "현재 위치",
    "마하다샤 단계 목록",
    "방법론 카드",
    "valid_until 설명",
    "CTA-lite",
    "면책/윤리/데이터 보호",
]
_LIFECYCLE_FORBIDDEN_H2 = {
    "인생 고점/저점 지도",
    "반복 패턴 분석",
    "다음 3년 구체화",
}
_LIFECYCLE_TARGET_REQUIRED_H2 = [
    "cover/meta",
    "How to use 1p",
    "인생 구조 한 장 요약",
    "4단계 인생 구조",
    "현재 위치",
    "마하다샤 단계 목록",
    "인생 고점/저점 지도",
    "반복 패턴 분석",
    "다음 3년 구체화",
    "방법론 카드",
    "valid_until 설명",
    "CTA-lite",
    "면책/윤리/데이터 보호",
]
_LIFECYCLE_TARGET_ACTION_REQUIRED_HEADERS = {
    "How to use 1p",
    "다음 3년 구체화",
    "valid_until 설명",
    "CTA-lite",
}
_LIFECYCLE_HF11_EXCLUDE_HEADERS = {
    "cover/meta",
    "How to use 1p",
    "방법론 카드",
    "valid_until 설명",
    "CTA-lite",
    "면책/윤리/데이터 보호",
}
_LIFECYCLE_ACTION_REQUIRED_HEADERS = {
    "How to use 1p",
    "valid_until 설명",
    "CTA-lite",
}
_LIFECYCLE_ACTION_LINE_RE = re.compile(r"(?m)^\s*(?:오늘의 행동|행동|Action)\s*:")
_LIFECYCLE_BULLET_LINE_RE = re.compile(r"(?m)^\s*-\s+.+$")
_LIFECYCLE_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_LIFECYCLE_SECONDARY_CONTROL_RE = re.compile(r"(동의 버튼|동의\b|checkbox|check box|consent|체크박스|secondary control)", re.IGNORECASE)
_LIFECYCLE_EMPATHY_RE = re.compile(r"(지금|요즘|흐름|흔들|망설|버겁|불안|막막|정리)")
_LIFECYCLE_BUTTON_LABEL_RE = re.compile(r"(?m)^\s*버튼\s*:\s*(.+)$")
_LIFECYCLE_TARGET_NEXT_ACTION_RE = re.compile(r"(?m)^\s*(?:행동|지금 메모할 질문|Action)\s*:")
_FORBIDDEN_YEAR_QUARTER_PATTERN_KEYS = {pattern.pattern for pattern in YEAR_QUARTER_PATTERNS}


def _normalize_chapter_key_for_match(value: str) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[\[\]\(\){}]", " ", token)
    token = re.sub(r"[^\w&가-힣\s-]", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _resolve_actionable_key(raw_key: str) -> str | None:
    normalized = _normalize_chapter_key_for_match(raw_key)
    if not normalized:
        return None
    for canonical, aliases in ACTIONABLE_CHAPTER_KEY_ALIASES.items():
        alias_norms = {_normalize_chapter_key_for_match(canonical)}
        alias_norms.update(_normalize_chapter_key_for_match(alias) for alias in aliases)
        if normalized in alias_norms:
            return canonical
    return None


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in msg or "timeout" in msg


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_nonempty_text(text: str | None) -> bool:
    return text is not None and str(text).strip() != ""


def _write_text_lf(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(normalized)


def _sha256_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_chapter_bodies(text: str) -> dict[str, str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(_H2_HEADING_RE.finditer(normalized))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = str(match.group(1) or match.group(2) or "").strip()
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        out[key] = normalized[start:end].strip()
    return out


def _compute_core_action_chapter_match_miss_count(text: str) -> int:
    chapters = _split_chapter_bodies(text)
    matched: set[str] = set()
    for key in chapters.keys():
        canonical = _resolve_actionable_key(key)
        if canonical:
            matched.add(canonical)
    expected = set(ACTIONABLE_CHAPTER_KEY_ALIASES.keys())
    return int(len(expected - matched))


def _compute_actionable_bullet_coverage(text: str) -> dict[str, Any]:
    chapters = _split_chapter_bodies(text)
    covered: list[str] = []
    missing: list[str] = []
    for key in sorted(_ACTIONABLE_CHAPTER_KEYS):
        body = chapters.get(key, "")
        if _has_valid_action_steps_block(body):
            covered.append(key)
        else:
            missing.append(key)
    return {
        "covered": len(covered),
        "total": len(_ACTIONABLE_CHAPTER_KEYS),
        "missing": missing,
    }


def _has_valid_action_steps_block(chapter_body: str) -> bool:
    text = str(chapter_body or "")
    for match in _ACTION_STEPS_H3_RE.finditer(text):
        start = match.end()
        next_heading = _ANY_HEADING_LINE_RE.search(text, start)
        end = next_heading.start() if next_heading else len(text)
        bullets = _BULLET_LINE_RE.findall(text[start:end])
        if len(bullets) >= 2:
            return True
    return False


def _compute_exact_duplicate_metrics(text: str) -> dict[str, int]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for block in re.split(r"\n\n+", normalized):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if _BULLET_LINE_RE.match(stripped):
            continue
        key = re.sub(r"\s+", " ", stripped).strip()
        if len(key) < 120:
            continue
        paragraphs.append(key)
    counts: dict[str, int] = {}
    for para in paragraphs:
        counts[para] = counts.get(para, 0) + 1
    duplicates = [count for count in counts.values() if count > 1]
    return {
        "exact_duplicate_paragraphs_removed": int(sum(max(0, count - 1) for count in duplicates)),
        "max_repeat_count": int(max(duplicates) if duplicates else 1),
    }


def _compute_action_steps_contract_metrics(text: str) -> dict[str, int | bool]:
    chapters = _split_chapter_bodies(text)
    block_count_violations = 0
    inline_repairs_needed = 0
    inline_heading_violations = 0
    contaminated_line_violations = 0
    non_actionable_line_violations = 0
    total_action_step_bullets = 0
    actionable_action_step_bullets = 0
    contract_ok = True
    for _key, body in chapters.items():
        for line in str(body or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
            match = ACTION_STEPS_INLINE_HEADING_RE.match(line.strip())
            if match is None:
                continue
            rest = str(match.group("rest") or "").strip()
            if not rest:
                continue
            inline_heading_violations += 1
            contract_ok = False

        action_matches = list(_ACTION_STEPS_H3_RE.finditer(body))
        if len(action_matches) > 1:
            block_count_violations += 1
            contract_ok = False
        if not action_matches:
            continue
        first = action_matches[0]
        next_heading = _ANY_HEADING_LINE_RE.search(body, first.end())
        block_end = next_heading.start() if next_heading else len(body)
        lines = [line.strip() for line in body[first.end():block_end].splitlines() if line.strip()]
        bullet_lines = [line for line in lines if re.match(r"^\s*(?:-\s+|\d+\.\s+)\S", line)]
        if len(bullet_lines) > 3:
            contract_ok = False
        for line in bullet_lines:
            total_action_step_bullets += 1
            if len(re.findall(r"\s+(?:-\s+|\d+\.\s+)", line)) >= 2:
                inline_repairs_needed += 1
                contract_ok = False
            normalized = re.sub(r"^\s*(?:-\s+|\d+\.\s+)", "", line).strip()
            if ACTIONABLE_HINT_RE.search(normalized) is None:
                non_actionable_line_violations += 1
                contract_ok = False
            else:
                actionable_action_step_bullets += 1
            end_hits = len(TRAIL_SENTENCE_END_RE.findall(normalized))
            if end_hits >= 2 and len(normalized) >= 110:
                contaminated_line_violations += 1
                contract_ok = False
    actionable_ratio = (
        float(actionable_action_step_bullets / total_action_step_bullets)
        if total_action_step_bullets > 0
        else 0.0
    )
    return {
        "action_steps_contract_ok": bool(contract_ok),
        "action_steps_block_count_violations": int(block_count_violations),
        "action_steps_inline_heading_violations": int(inline_heading_violations),
        "action_steps_inline_repairs": int(inline_repairs_needed),
        "action_steps_contaminated_line_violations": int(contaminated_line_violations),
        "action_steps_non_actionable_line_violations": int(non_actionable_line_violations),
        "action_steps_total_bullets": int(total_action_step_bullets),
        "action_steps_actionable_bullets": int(actionable_action_step_bullets),
        "action_steps_actionable_ratio": float(actionable_ratio),
    }


def _split_h2_sections_ordered(text: str) -> list[tuple[str, str]]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(_H2_HEADING_RE.finditer(normalized))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        key = str(match.group(1) or match.group(2) or "").strip()
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        out.append((key, normalized[start:end].strip()))
    return out

def _extract_lifecycle_overlap_tokens(text: str) -> set[str]:
    tokens = {token.strip().lower() for token in _LIFECYCLE_TOKEN_RE.findall(str(text or "")) if token and len(token.strip()) >= 2}
    return {token for token in tokens if token not in {"today", "action", "valid", "until"}}


def _has_lifecycle_action_line(body: str) -> bool:
    return _LIFECYCLE_ACTION_LINE_RE.search(str(body or "")) is not None


def _compute_life_cycle_lite_release_metrics(text: str, *, subject_name: str | None = None, valid_until_fallback: bool | None = None) -> dict[str, Any]:
    sections = _split_h2_sections_ordered(text)
    headers = [header for header, _body in sections]
    header_counts: dict[str, int] = {}
    for header in headers:
        header_counts[header] = header_counts.get(header, 0) + 1

    missing_headers = [header for header in _LIFECYCLE_REQUIRED_H2 if header_counts.get(header, 0) == 0]
    duplicate_headers = [header for header in _LIFECYCLE_REQUIRED_H2 if header_counts.get(header, 0) > 1]
    unexpected_headers = [header for header in headers if header not in _LIFECYCLE_REQUIRED_H2]
    forbidden_headers_present = [header for header in headers if header in _LIFECYCLE_FORBIDDEN_H2]
    exact_order_ok = headers == _LIFECYCLE_REQUIRED_H2
    front_contract_ok = not missing_headers and not duplicate_headers and not unexpected_headers and not forbidden_headers_present and exact_order_ok

    section_map = {header: body for header, body in sections}
    how_to_use = section_map.get("How to use 1p", "")
    valid_until_body = section_map.get("valid_until 설명", "")
    cta_body = section_map.get("CTA-lite", "")
    summary_body = section_map.get("인생 구조 한 장 요약", "")

    how_to_use_bullets = _LIFECYCLE_BULLET_LINE_RE.findall(how_to_use)
    how_to_use_ok = len(how_to_use_bullets) >= 5 and "복구 플랜" in how_to_use and _has_lifecycle_action_line(how_to_use)
    valid_until_action_lines = len(_LIFECYCLE_ACTION_LINE_RE.findall(valid_until_body))
    valid_until_ok = valid_until_action_lines >= 1
    cta_action_lines = len(_LIFECYCLE_ACTION_LINE_RE.findall(cta_body))
    cta_secondary_controls = len(_LIFECYCLE_SECONDARY_CONTROL_RE.findall(cta_body))
    cta_button_labels = [str(match.group(1) or "").strip() for match in _LIFECYCLE_BUTTON_LABEL_RE.finditer(cta_body)]
    cta_button_too_long = any(len(label) > 20 for label in cta_button_labels)
    cta_ok = cta_action_lines == 1 and cta_secondary_controls == 0 and not cta_button_too_long

    hf12_ok = how_to_use_ok and valid_until_ok and cta_ok
    hf14_overlap = sorted(_extract_lifecycle_overlap_tokens(valid_until_body) & _extract_lifecycle_overlap_tokens(cta_body))
    hf14_ok = exact_order_ok and len(hf14_overlap) > 0

    narrative_body = ""
    narrative_header = None
    for header, body in sections:
        if header in _LIFECYCLE_HF11_EXCLUDE_HEADERS:
            continue
        narrative_header = header
        narrative_body = body
        break
    subject_name_norm = str(subject_name or "").strip()
    hf11_ok = bool(narrative_header and narrative_body and (_LIFECYCLE_EMPATHY_RE.search(narrative_body) or (subject_name_norm and subject_name_norm in narrative_body)))

    if not subject_name_norm or subject_name_norm == "당신":
        hf16_ok = True
        hf16_skipped = True
    else:
        hf16_ok = subject_name_norm in summary_body
        hf16_skipped = False

    action_steps_contract_ok = front_contract_ok and hf12_ok and hf14_ok
    release_ok = front_contract_ok and action_steps_contract_ok and hf11_ok and hf12_ok and hf14_ok and hf16_ok

    return {
        "release_mode": "life_cycle_lite",
        "front_contract_ok": bool(front_contract_ok),
        "front_contract_detail": {
            "required_headers": list(_LIFECYCLE_REQUIRED_H2),
            "headers": headers,
            "missing_headers": missing_headers,
            "duplicate_headers": duplicate_headers,
            "unexpected_headers": unexpected_headers,
            "forbidden_headers_present": forbidden_headers_present,
            "exact_order_ok": bool(exact_order_ok),
        },
        "action_steps_contract_ok": bool(action_steps_contract_ok),
        "action_steps_block_count_violations": int(0 if how_to_use_ok and valid_until_ok and cta_ok else 1),
        "life_cycle_how_to_use_ok": bool(how_to_use_ok),
        "life_cycle_valid_until_ok": bool(valid_until_ok),
        "life_cycle_cta_ok": bool(cta_ok),
        "life_cycle_hf11_ok": bool(hf11_ok),
        "life_cycle_hf11_target_header": narrative_header,
        "life_cycle_hf12_ok": bool(hf12_ok),
        "life_cycle_hf14_ok": bool(hf14_ok),
        "life_cycle_hf14_overlap_tokens": hf14_overlap[:20],
        "life_cycle_hf16_ok": bool(hf16_ok),
        "life_cycle_hf16_skipped": bool(hf16_skipped),
        "life_cycle_release_ok": bool(release_ok),
        "valid_until_fallback": valid_until_fallback if isinstance(valid_until_fallback, bool) else None,
    }


def _compute_life_cycle_target_release_metrics(text: str, *, subject_name: str | None = None, valid_until_fallback: bool | None = None) -> dict[str, Any]:
    sections = _split_h2_sections_ordered(text)
    headers = [header for header, _body in sections]
    header_counts: dict[str, int] = {}
    for header in headers:
        header_counts[header] = header_counts.get(header, 0) + 1

    missing_headers = [header for header in _LIFECYCLE_TARGET_REQUIRED_H2 if header_counts.get(header, 0) == 0]
    duplicate_headers = [header for header in _LIFECYCLE_TARGET_REQUIRED_H2 if header_counts.get(header, 0) > 1]
    unexpected_headers = [header for header in headers if header not in _LIFECYCLE_TARGET_REQUIRED_H2]
    exact_order_ok = headers == _LIFECYCLE_TARGET_REQUIRED_H2
    front_contract_ok = not missing_headers and not duplicate_headers and not unexpected_headers and exact_order_ok

    section_map = {header: body for header, body in sections}
    how_to_use = section_map.get("How to use 1p", "")
    current_position_body = section_map.get("현재 위치", "")
    valid_until_body = section_map.get("valid_until 설명", "")
    cta_body = section_map.get("CTA-lite", "")
    summary_body = section_map.get("인생 구조 한 장 요약", "")
    high_low_body = section_map.get("인생 고점/저점 지도", "")
    repeat_pattern_body = section_map.get("반복 패턴 분석", "")
    next_three_years_body = section_map.get("다음 3년 구체화", "")

    how_to_use_bullets = _LIFECYCLE_BULLET_LINE_RE.findall(how_to_use)
    how_to_use_ok = len(how_to_use_bullets) >= 5 and "복구 플랜" in how_to_use and _has_lifecycle_action_line(how_to_use)
    valid_until_action_lines = len(_LIFECYCLE_ACTION_LINE_RE.findall(valid_until_body))
    valid_until_ok = valid_until_action_lines >= 1
    cta_action_lines = len(_LIFECYCLE_ACTION_LINE_RE.findall(cta_body))
    cta_secondary_controls = len(_LIFECYCLE_SECONDARY_CONTROL_RE.findall(cta_body))
    cta_button_labels = [str(match.group(1) or "").strip() for match in _LIFECYCLE_BUTTON_LABEL_RE.finditer(cta_body)]
    cta_button_too_long = any(len(label) > 20 for label in cta_button_labels)
    cta_ok = cta_action_lines == 1 and cta_secondary_controls == 0 and not cta_button_too_long

    high_low_ok = (
        "🔺 가장 상승 가능성 높은 3구간" in high_low_body
        and "⚡ 인생 전환점 5개" in high_low_body
        and "⚠ 경계해야 할 구간 3개" in high_low_body
        and len(re.findall(r"(?m)^\s*\d+\.\s", high_low_body)) >= 3
    )
    repeat_patterns_ok = (
        bool(repeat_pattern_body.strip())
        and "👉" in repeat_pattern_body
        and any(token in repeat_pattern_body for token in ("반복 시기", "2회 이상", "반복 패턴"))
    )
    next_three_years_has_slots = bool(re.search(r"(?m)^\s*\d+\.\s+\d{4}-\d{2}-\d{2}\s+~\s+\d{4}-\d{2}-\d{2}\s+—", next_three_years_body))
    next_three_years_has_closing = (
        "새로운 챕터로 넘어갑니다" in next_three_years_body
        and "업데이트된 지도를 확인해보세요" in next_three_years_body
    )
    next_three_years_has_action = _LIFECYCLE_TARGET_NEXT_ACTION_RE.search(next_three_years_body) is not None
    next_three_years_ok = next_three_years_has_closing and (next_three_years_has_action or not next_three_years_has_slots)

    hf12_ok = how_to_use_ok and valid_until_ok and cta_ok and next_three_years_ok
    hf14_overlap = sorted(_extract_lifecycle_overlap_tokens(valid_until_body) & _extract_lifecycle_overlap_tokens(cta_body))
    hf14_ok = exact_order_ok and len(hf14_overlap) > 0

    narrative_body = ""
    narrative_header = None
    for header, body in sections:
        if header in _LIFECYCLE_HF11_EXCLUDE_HEADERS:
            continue
        narrative_header = header
        narrative_body = body
        break
    subject_name_norm = str(subject_name or "").strip()
    hf11_ok = bool(narrative_header and narrative_body and (_LIFECYCLE_EMPATHY_RE.search(narrative_body) or (subject_name_norm and subject_name_norm in narrative_body)))

    if not subject_name_norm or subject_name_norm == "당신":
        hf16_ok = True
        hf16_skipped = True
    else:
        hf16_ok = subject_name_norm in summary_body
        hf16_skipped = False

    personalization_sections = []
    if subject_name_norm:
        if subject_name_norm in summary_body:
            personalization_sections.append("인생 구조 한 장 요약")
        if subject_name_norm in current_position_body:
            personalization_sections.append("현재 위치")
        if subject_name_norm in next_three_years_body:
            personalization_sections.append("다음 3년 구체화")

    action_steps_contract_ok = front_contract_ok and hf12_ok and hf14_ok and high_low_ok and repeat_patterns_ok and next_three_years_ok
    release_ok = action_steps_contract_ok and hf11_ok and hf12_ok and hf14_ok and hf16_ok

    return {
        "release_mode": "life_cycle_target",
        "front_contract_ok": bool(front_contract_ok),
        "front_contract_detail": {
            "required_headers": list(_LIFECYCLE_TARGET_REQUIRED_H2),
            "headers": headers,
            "missing_headers": missing_headers,
            "duplicate_headers": duplicate_headers,
            "unexpected_headers": unexpected_headers,
            "exact_order_ok": bool(exact_order_ok),
        },
        "action_steps_contract_ok": bool(action_steps_contract_ok),
        "action_steps_block_count_violations": int(0 if how_to_use_ok and valid_until_ok and cta_ok and next_three_years_ok else 1),
        "life_cycle_target_high_low_ok": bool(high_low_ok),
        "life_cycle_target_repeat_patterns_ok": bool(repeat_patterns_ok),
        "life_cycle_target_next_three_years_ok": bool(next_three_years_ok),
        "life_cycle_target_next_three_years_has_slots": bool(next_three_years_has_slots),
        "life_cycle_target_personalization_sections": personalization_sections,
        "life_cycle_how_to_use_ok": bool(how_to_use_ok),
        "life_cycle_valid_until_ok": bool(valid_until_ok),
        "life_cycle_cta_ok": bool(cta_ok),
        "life_cycle_hf11_ok": bool(hf11_ok),
        "life_cycle_hf11_target_header": narrative_header,
        "life_cycle_hf12_ok": bool(hf12_ok),
        "life_cycle_hf14_ok": bool(hf14_ok),
        "life_cycle_hf14_overlap_tokens": hf14_overlap[:20],
        "life_cycle_hf16_ok": bool(hf16_ok),
        "life_cycle_hf16_skipped": bool(hf16_skipped),
        "life_cycle_release_ok": bool(release_ok),
        "valid_until_fallback": valid_until_fallback if isinstance(valid_until_fallback, bool) else None,
    }


def _filter_forbidden_hits_for_release_mode(findings: list[dict[str, str]], release_mode: str | None) -> list[dict[str, str]]:
    if str(release_mode or "").strip() not in {"life_cycle_lite", "life_cycle_target"}:
        return findings
    return [
        hit
        for hit in findings
        if str(hit.get("pattern") or "") not in _FORBIDDEN_YEAR_QUARTER_PATTERN_KEYS
    ]


def _compute_inline_action_chain_violations(text: str) -> int:
    chapters = _split_chapter_bodies(text)
    violations = 0
    for _key, body in chapters.items():
        action_matches = list(_ACTION_STEPS_H3_RE.finditer(body))
        action_ranges: list[tuple[int, int]] = []
        for match in action_matches:
            next_heading = _ANY_HEADING_LINE_RE.search(body, match.end())
            end = next_heading.start() if next_heading else len(body)
            action_ranges.append((match.start(), end))

        lines = body.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        offsets: list[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line) + 1

        for idx, line in enumerate(lines):
            start = offsets[idx] if idx < len(offsets) else 0
            if any(s <= start < e for s, e in action_ranges):
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("추가 제안:"):
                continue
            if INLINE_ACTION_CHAIN_EXPLANATION_RE.search(stripped):
                continue
            if not (INLINE_ACTION_CHAIN_LIST_START_RE.match(stripped) or "->" in stripped):
                continue
            content = _INLINE_CHAIN_LEADING_PREFIX_RE.sub("", stripped, count=1)

            parts: list[str] = []
            if len(INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE.findall(content)) >= 2:
                parts = [p.strip() for p in INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE.split(content) if p and p.strip()]
            elif len(INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE.findall(content)) >= 2:
                parts = [p.strip() for p in INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE.split(content) if p and p.strip()]
            elif len(INLINE_ACTION_CHAIN_SLASH_SPLIT_RE.findall(content)) >= 2:
                parts = [p.strip() for p in INLINE_ACTION_CHAIN_SLASH_SPLIT_RE.split(content) if p and p.strip()]

            if len(parts) < 2:
                continue

            actionable_count = 0
            for part in parts:
                if ACTIONABLE_HINT_RE.search(part):
                    actionable_count += 1
            if actionable_count >= 2:
                violations += 1
    return int(violations)


def _compute_structure_and_definition_metrics(text: str) -> dict[str, int]:
    heading_violations = len(_SYNTHETIC_ACTION_H2_RE.findall(str(text or "")))
    definition_hits = len(DASHA_DEFINITION_RE.findall(str(text or "")))
    nested_hits = len(DASHA_NESTED_PHRASE_RE.findall(str(text or "")))
    redundancy_hits = len(DASHA_DEFINITION_REDUNDANCY_RE.findall(str(text or "")))
    shrink_hits = str(text or "").count(_DEFINITION_SHRINK_LINE)
    core_miss = _compute_core_action_chapter_match_miss_count(text)
    return {
        "header_structure_violations": int(heading_violations),
        "synthetic_heading_rewrites": int(heading_violations),
        "definition_dedup_removed_count": int(max(0, definition_hits - 1)),
        "definition_dedup_shrink_count": int(shrink_hits),
        "definition_nested_phrase_violations": int(nested_hits),
        "dasha_definition_nested_pattern_violations": int(nested_hits),
        "dasha_definition_redundancy_violations": int(redundancy_hits),
        "definition_dasha_occurrences_after": int(definition_hits),
        "core_action_chapter_match_miss_count": int(core_miss),
    }


def _extract_chapters_text_for_quality(text: str) -> tuple[str, bool]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"(?m)^\s*##\s*\[", normalized)
    if start is None:
        return "", True
    return normalized[start.start():].strip(), False


def _extract_h1_sections(text: str) -> dict[str, str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(_FRONT_H1_RE.finditer(normalized))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = str(match.group(1) or "").strip()
        if not title:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        out[title] = normalized[start:end].strip()
    return out


def _ensure_front_sentinels_for_metrics(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    front_match = _FRONT_SEGMENT_RE.search(normalized)
    chapters_match = _CHAPTER_SEGMENT_RE.search(normalized)
    if front_match and chapters_match:
        return normalized
    if front_match and not chapters_match:
        return normalized
    chapter_start = re.search(r"(?m)^\s*##\s*\[", normalized)
    if chapter_start is None:
        return f"{_FRONT_START_SENTINEL}\n{normalized}\n{_FRONT_END_SENTINEL}"
    front = normalized[:chapter_start.start()].strip()
    chapters = normalized[chapter_start.start():].strip()
    if front:
        return (
            f"{_FRONT_START_SENTINEL}\n{front}\n{_FRONT_END_SENTINEL}\n\n"
            f"{_CHAPTERS_START_SENTINEL}\n{chapters}\n{_CHAPTERS_END_SENTINEL}"
        ).strip()
    return f"{_CHAPTERS_START_SENTINEL}\n{chapters}\n{_CHAPTERS_END_SENTINEL}".strip()


def _extract_front_segment(text: str) -> str:
    normalized = _ensure_front_sentinels_for_metrics(text)
    match = _FRONT_SEGMENT_RE.search(normalized)
    if match is None:
        return ""
    return str(match.group("front") or "").strip()


def _extract_chapter_segment(text: str) -> str:
    normalized = _ensure_front_sentinels_for_metrics(text)
    match = _CHAPTER_SEGMENT_RE.search(normalized)
    if match is None:
        return ""
    return str(match.group("chapters") or "").strip()


def _compute_front_english_token_hits(text: str) -> dict[str, Any]:
    front = _extract_front_segment(text)
    tokens = [token for token in _FRONT_ASCII_ALPHA_RE.findall(front) if token]
    return {
        "count": len(tokens),
        "samples": tokens[:20],
        "ok": len(tokens) == 0,
    }


def _compute_front_markdown_integrity(text: str) -> dict[str, Any]:
    front = _extract_front_segment(text)
    headings = _extract_h1_sections(front)
    heading_count = len(headings)
    has_required_headings = all(key in headings for key in ("한 장 요약", "3개월 플레이북", "7일 시스템"))
    bullet_lines = [line for line in front.splitlines() if line.strip().startswith("- ")]
    broken_bullets = [line for line in bullet_lines if line.strip() == "-"]
    front_has_h2 = bool(re.search(r"(?m)^\s*##\s*\[", front))
    return {
        "heading_count": heading_count,
        "required_headings_ok": has_required_headings,
        "broken_list_lines": len(broken_bullets),
        "front_has_h2_with_key": front_has_h2,
        "ok": has_required_headings and len(broken_bullets) == 0 and not front_has_h2,
    }


def _slot_contract_ok(playbook: str, label: str) -> bool:
    lines = str(playbook or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    target_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == label:
            target_idx = idx
            break
    if target_idx < 0:
        return False
    if target_idx + 3 >= len(lines):
        return False
    caution_line = lines[target_idx + 1].strip()
    rule_line = lines[target_idx + 2].strip()
    reason_line = lines[target_idx + 3].strip()
    if not caution_line.startswith(PLAYBOOK_LINE_CAUTION_PREFIX):
        return False
    if not reason_line.startswith(PLAYBOOK_LINE_REASON_PREFIX):
        return False
    if not rule_line.startswith(PLAYBOOK_LINE_RULE_PREFIX):
        return False
    rule_text = rule_line[len(PLAYBOOK_LINE_RULE_PREFIX):].strip()
    return rule_text.count("/") == 1 and RULE_SEPARATOR in rule_text


def _extract_front_playbook_section(front_text: str) -> tuple[str, bool]:
    front = str(front_text or "").replace("\r\n", "\n").replace("\r", "\n")
    start_match = PLAYBOOK_START_RE.search(front)
    if start_match is None:
        return "", False
    start = start_match.start()
    end = len(front)
    for end_match in PLAYBOOK_END_RE.finditer(front, start_match.end()):
        if end_match.start() > start:
            end = end_match.start()
            break
    return front[start:end].strip(), True


def _compute_front_contract(text: str) -> dict[str, Any]:
    front = _extract_front_segment(text)
    sections = _extract_h1_sections(front)
    summary = sections.get("한 장 요약", "")
    playbook, section_found = _extract_front_playbook_section(front)
    system = sections.get("7일 시스템", "")
    summary_tokens = ("핵심 패턴 3개", "금지·권장 3개", "상황 예시", "미니 템플릿 3개")
    summary_ok = all(token in summary for token in summary_tokens)
    slots_ok = all(_slot_contract_ok(playbook, label) for label in PLAYBOOK_LABELS)
    slot_ok_count = sum(1 for label in PLAYBOOK_LABELS if _slot_contract_ok(playbook, label))
    checklist_ok = len(_FRONT_CHECKBOX_RE.findall(system)) >= 4
    operating_ok = "운영법(하루 10분)" in system
    return {
        "ok": bool(summary_ok and slots_ok and checklist_ok and operating_ok and section_found),
        "summary_ok": summary_ok,
        "slots_ok": slots_ok,
        "slot_ok_count": slot_ok_count,
        "checklist_ok": checklist_ok,
        "operating_ok": operating_ok,
        "section_found": section_found,
    }


def _normalize_metric_text(text: str) -> str:
    out = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()


def _is_front_contract_fallback_applied(text: str) -> bool:
    current_front = _extract_front_segment(text)
    if not current_front:
        return False
    fallback_front = _extract_front_segment(render_fallback_front_modules())
    if not fallback_front:
        return False
    return _normalize_metric_text(current_front) == _normalize_metric_text(fallback_front)


def _compute_conversion_metrics(text: str) -> dict[str, Any]:
    front = _extract_front_segment(text) or str(text or "")
    sections = _extract_h1_sections(front)
    summary = sections.get("한 장 요약", "")
    playbook, _playbook_found = _extract_front_playbook_section(front)
    system = sections.get("7일 시스템", "")

    pattern_lines: list[str] = []
    if summary:
        for line in summary.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                pattern_lines.append(stripped[2:].strip())
        pattern_lines = pattern_lines[:3]

    trigger_hits = 0
    for line in pattern_lines:
        if any(term in line for term in _FRONT_TRIGGER_TERMS):
            trigger_hits += 1
    empathy_ratio = float(trigger_hits / max(len(pattern_lines), 1)) if pattern_lines else 0.0

    checklist_count = len(_FRONT_CHECKBOX_RE.findall(system))

    slot_labels = PLAYBOOK_LABELS
    slot_ok = sum(1 for label in slot_labels if _slot_contract_ok(playbook, label))

    scene_domains = sorted(set(_FRONT_SCENE_DOMAIN_RE.findall(summary)))

    return {
        "empathy_hook_presence": {
            "pattern_count": len(pattern_lines),
            "trigger_hits": trigger_hits,
            "ratio": round(empathy_ratio, 4),
        },
        "action_system_presence": {
            "checkbox_count": checklist_count,
            "ok": checklist_count >= 4,
        },
        "playbook_presence": {
            "slot_labels_found": sum(1 for label in slot_labels if label in playbook),
            "slot_contract_ok": slot_ok,
            "ok": slot_ok == 3,
        },
        "unique_scene_coverage": {
            "domains": scene_domains,
            "count": len(scene_domains),
            "ok": len(scene_domains) >= 2,
        },
    }


def _select_commercial_scan_surface(ai_reading: dict[str, Any]) -> tuple[str, str]:
    polished_text = str(ai_reading.get("polished_reading") or "")
    if polished_text.strip():
        return polished_text, "polished_reading"
    reading_text = str(ai_reading.get("reading") or "")
    return reading_text, "reading"


def _build_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for payload in generate_golden_charts():
        chart = get_chart(
            year=payload["year"],
            month=payload["month"],
            day=payload["day"],
            hour=payload["hour"],
            lat=payload["lat"],
            lon=payload["lon"],
            house_system=payload["house_system"],
            include_nodes=payload["include_nodes"],
            include_d9=payload["include_d9"],
            include_vargas=payload["include_vargas"],
            gender=payload["gender"],
            timezone=payload["timezone"],
        )
        structural_summary = build_structural_summary(chart, analysis_mode=payload["analysis_mode"])
        candidates.append(_candidate_metrics(payload, chart, structural_summary))
    return candidates


def _pick_profile(selected: list[tuple[str, dict[str, Any]]], profile: str) -> tuple[str, dict[str, Any]]:
    for name, cand in selected:
        if name == profile:
            return name, cand
    # Fallback to most_balanced-like last profile when requested one is missing.
    return selected[-1]


def _build_prompt_for_candidate(candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    structural_summary = candidate["structural_summary"]
    narrative_mode = _derive_narrative_mode(structural_summary)
    semantic_signals = build_semantic_signals(structural_summary)
    dasha_context = build_dasha_narrative_context(structural_summary)
    report_payload = build_report_payload({"structural_summary": structural_summary, "language": "ko"})
    chapter_blocks = report_payload.get("chapter_blocks", {}) or {}
    prompt = build_llm_structural_prompt(
        structural_summary=structural_summary,
        language="ko",
        chapter_blocks=chapter_blocks,
        semantic_signals=semantic_signals,
        narrative_mode=narrative_mode,
        dasha_context=dasha_context,
    )
    return prompt, chapter_blocks


def _static_prompt_check(prompt: str) -> dict[str, Any]:
    required_markers = [
        "OUTPUT CONTRACT (STRICT)",
        "DASHA INTEGRITY",
        "HARD BANS",
        "SHOCK ARCHITECTURE",
    ]
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "prompt_hash": prompt_hash,
        "prompt_length": len(prompt),
        "required_markers": {m: (m in prompt) for m in required_markers},
    }


def _strict_vedic_failed(scan: dict[str, Any] | None, timing_map_contract: dict[str, Any] | None = None) -> bool:
    if not isinstance(scan, dict):
        return bool(isinstance(timing_map_contract, dict) and timing_map_contract.get("over"))
    if bool(scan.get("doc_over")):
        return True
    for chapter in scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []:
        if bool(chapter.get("over")):
            return True
        if int(chapter.get("stacking_hits", 0)) > 0:
            return True
    if isinstance(timing_map_contract, dict) and bool(timing_map_contract.get("over")):
        return True
    return False


def _strict_vedic_violations(scan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scan, dict):
        return {
            "doc_over": False,
            "chapter_budget_over": [],
            "chapter_stacking_over": [],
        }
    chapters = scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []
    chapter_budget_over = [c for c in chapters if bool(c.get("over"))]
    chapter_stacking_over = [c for c in chapters if int(c.get("stacking_hits", 0)) > 0]
    return {
        "doc_over": bool(scan.get("doc_over")),
        "chapter_budget_over": chapter_budget_over,
        "chapter_stacking_over": chapter_stacking_over,
    }


def _log_strict_vedic_fail(
    stage: str,
    scan: dict[str, Any] | None,
    timing_map_contract: dict[str, Any] | None = None,
) -> None:
    if not isinstance(scan, dict):
        logger.error("[STRICT_VEDIC] stage=%s missing_scan_payload", stage)
        if isinstance(timing_map_contract, dict) and bool(timing_map_contract.get("over")):
            logger.error(
                "[STRICT_VEDIC] stage=%s timing_map_present=%s calendar_lines=%s max_calendar_lines=%s over=%s",
                stage,
                timing_map_contract.get("timing_map_present"),
                timing_map_contract.get("calendar_lines"),
                timing_map_contract.get("max_calendar_lines"),
                timing_map_contract.get("over"),
            )
        return
    logger.error(
        "[STRICT_VEDIC] stage=%s doc_total=%s max_total=%s doc_over=%s",
        stage,
        scan.get("doc_total"),
        scan.get("max_terms_total"),
        scan.get("doc_over"),
    )
    for chapter in scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []:
        if not bool(chapter.get("over")) and int(chapter.get("stacking_hits", 0)) <= 0:
            continue
        logger.error(
            "[STRICT_VEDIC] stage=%s chapter=%s chapter_total=%s max_per_chapter=%s terms=%s axis_tokens=%s stacking_hits=%s",
            stage,
            chapter.get("chapter_heading"),
            chapter.get("chapter_total"),
            chapter.get("max_terms_per_chapter"),
            chapter.get("terms"),
            chapter.get("axis_tokens"),
            chapter.get("stacking_hits"),
        )
    if isinstance(timing_map_contract, dict) and bool(timing_map_contract.get("over")):
        logger.error(
            "[STRICT_VEDIC] stage=%s timing_map_present=%s calendar_lines=%s max_calendar_lines=%s over=%s samples=%s",
            stage,
            timing_map_contract.get("timing_map_present"),
            timing_map_contract.get("calendar_lines"),
            timing_map_contract.get("max_calendar_lines"),
            timing_map_contract.get("over"),
            timing_map_contract.get("line_samples"),
        )


def _dry_structure_check(chapter_blocks: dict[str, Any], *, strict_vedic: bool = False) -> dict[str, Any]:
    deterministic = _render_chapter_blocks_deterministic(chapter_blocks, language="ko")
    normalized = normalize_llm_layout_strict(deterministic)
    remediated = _apply_style_remediation(normalized, allow_zero_term_injection=False)
    style_errors = _reading_style_error_codes(remediated)
    forbidden_hits = _filter_forbidden_hits_for_release_mode(
        scan_forbidden_patterns(
            remediated,
            allow_year_quarter_in_timing_map=bool(strict_vedic),
        ),
        release_mode_norm,
    )
    vedic_scan = scan_vedic_term_budget(remediated) if strict_vedic else None
    timing_map_contract = scan_timing_map_contract(remediated) if strict_vedic else None

    density_metrics = _compute_body_paragraph_density_metrics(remediated)
    warn_only = {"label_pattern_detected", "paragraph_too_long", "warn_paragraph_density_low"}
    if release_mode_norm in {"life_cycle_lite", "life_cycle_target"}:
        warn_only = set(warn_only)
        warn_only.update({"paragraph_density_low", "paragraph_too_long"})
    hard_style_errors = [e for e in style_errors if e not in warn_only]
    warn_style_errors = [e for e in style_errors if e in warn_only]

    return {
        "text_length": len(remediated),
        "heading_count": remediated.count("\n## ") + (1 if remediated.startswith("## ") else 0),
        "hard_style_errors": hard_style_errors,
        "warn_style_errors": warn_style_errors,
        "forbidden_hits": len(forbidden_hits),
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
        "timing_map_contract": timing_map_contract,
        "paragraph_density_metrics": density_metrics,
        **life_cycle_release_metrics,
    }


def _load_hash_guard() -> dict[str, Any]:
    if not HASH_GUARD_PATH.exists():
        return {}
    try:
        return json.loads(HASH_GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_hash_guard(profile_name: str, prompt_hash: str) -> None:
    payload = {
        "profile_name": profile_name,
        "prompt_hash": prompt_hash,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json_dump(HASH_GUARD_PATH, payload)


def _run_single_true_path(
    candidate: dict[str, Any],
    timeout_seconds: int = 180,
    *,
    strict_vedic: bool = False,
) -> dict[str, Any]:
    payload = candidate["input"]
    params = {
        "year": payload["year"],
        "month": payload["month"],
        "day": payload["day"],
        "hour": payload["hour"],
        "lat": payload["lat"],
        "lon": payload["lon"],
        "house_system": payload["house_system"],
        "include_nodes": payload["include_nodes"],
        "include_d9": payload["include_d9"],
        "include_vargas": payload["include_vargas"],
        "language": "ko",
        "gender": payload["gender"],
        "analysis_mode": payload["analysis_mode"],
        "detail_level": "full",
        # During tuning, force cache-off to avoid stale fallback/polished reuse.
        "use_cache": 0,
        "debug_payload": 0,
    }
    with TestClient(app) as client:
        try:
            resp = client.get("/ai_reading", params=params, timeout=timeout_seconds)
        except Exception as e:
            if _is_timeout_error(e):
                logger.error(
                    "TIMEOUT_OCCURRED stage=ai_reading timeout=%s profile=%s",
                    timeout_seconds,
                    candidate.get("profile_name", "unknown"),
                )
            return {
                "ok": False,
                "status_code": 0,
                "error": str(e)[:400],
            }
    if resp.status_code != 200:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text[:400],
        }
    data = resp.json()
    ai_cache_key = data.get("ai_cache_key")
    reading_text = str(data.get("reading") or "")
    polished_text = str(data.get("polished_reading") or "")
    scan_source, scan_surface_source = _select_commercial_scan_surface(data)
    # True-path strict scoring must use the exact commercial surface returned to users.
    # Only normalize newlines for stable file output and deterministic scans.
    scan_surface_text = str(scan_source or "").replace("\r\n", "\n").replace("\r", "\n")

    post_surface_from_reading, reading_surface_metrics = sanitize_commercial_surface_with_front_protection_with_metrics(reading_text)
    post_surface_from_reading = post_surface_from_reading.replace("\r\n", "\n").replace("\r", "\n")
    post_surface_from_polished = ""
    polished_surface_metrics: dict[str, int | bool] = {
        "one_page_summary_dedup_repairs": 0,
        "chapters_boundary_fallback_used": False,
        "front_byte_equal_after_b_pass": True,
        "front_end_offset_stable": True,
    }
    if _is_nonempty_text(polished_text):
        post_surface_from_polished, polished_surface_metrics = sanitize_commercial_surface_with_front_protection_with_metrics(polished_text)
        post_surface_from_polished = post_surface_from_polished.replace("\r\n", "\n").replace("\r", "\n")
    if _is_nonempty_text(polished_text):
        scored_surface_name = "polished_reading"
        scored_surface_text = (
            post_surface_from_polished
            if _is_nonempty_text(post_surface_from_polished)
            else polished_text.replace("\r\n", "\n").replace("\r", "\n")
        )
        scored_surface_metrics = polished_surface_metrics
    elif _is_nonempty_text(post_surface_from_reading):
        scored_surface_name = "reading_post_remediation"
        scored_surface_text = post_surface_from_reading
        scored_surface_metrics = reading_surface_metrics
    else:
        scored_surface_name = "reading_scan_surface"
        scored_surface_text = scan_surface_text
        scored_surface_metrics = reading_surface_metrics

    chapters_text_for_quality, chapters_boundary_fallback_used = _extract_chapters_text_for_quality(scored_surface_text)
    if chapters_boundary_fallback_used:
        quality_metrics: dict[str, int | bool] = {
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
            "commercial_quality_metrics_valid": False,
        }
    else:
        _quality_text, quality_metrics = postprocess_commercial_quality_with_metrics(chapters_text_for_quality)
    quality_metrics["one_page_summary_dedup_repairs"] = int(
        scored_surface_metrics.get("one_page_summary_dedup_repairs", 0)
    )
    actionable_bullet_coverage = _compute_actionable_bullet_coverage(scored_surface_text)
    duplicate_metrics = _compute_exact_duplicate_metrics(scored_surface_text)
    action_steps_metrics = _compute_action_steps_contract_metrics(scored_surface_text)
    inline_action_chain_violations = _compute_inline_action_chain_violations(scored_surface_text)
    structure_definition_metrics = _compute_structure_and_definition_metrics(scored_surface_text)
    conversion_metrics = _compute_conversion_metrics(scored_surface_text)
    front_english_token_hits = _compute_front_english_token_hits(scored_surface_text)
    front_markdown_integrity = _compute_front_markdown_integrity(scored_surface_text)
    front_contract = _compute_front_contract(scored_surface_text)
    front_contract_fallback_applied = _is_front_contract_fallback_applied(scored_surface_text)
    front_playbook_repaired = bool(not front_contract.get("slots_ok", False))
    front_playbook_section_found = bool(front_contract.get("section_found", False))
    front_playbook_slot_contract_ok_count = int(front_contract.get("slot_ok_count", 0))
    front_byte_equal_after_b_pass = bool(scored_surface_metrics.get("front_byte_equal_after_b_pass", True))
    front_end_offset_stable = bool(scored_surface_metrics.get("front_end_offset_stable", True))
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    summary_payload = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    structured_summary_payload = summary_payload.get("structured_summary") if isinstance(summary_payload.get("structured_summary"), dict) else {}
    release_mode = None
    life_cycle_release_metrics: dict[str, Any] = {}
    render_profile = str(meta.get("render_profile") or "").strip()
    if render_profile == "life_cycle_target_v1":
        release_mode = "life_cycle_target"
        life_cycle_release_metrics = _compute_life_cycle_target_release_metrics(
            scored_surface_text,
            subject_name=str(structured_summary_payload.get("subject_name") or "").strip() or None,
            valid_until_fallback=meta.get("valid_until_fallback") if isinstance(meta.get("valid_until_fallback"), bool) else None,
        )
        front_contract = life_cycle_release_metrics.get("front_contract_detail", {})
        action_steps_metrics["action_steps_contract_ok"] = bool(life_cycle_release_metrics.get("action_steps_contract_ok", False))
        action_steps_metrics["action_steps_block_count_violations"] = int(life_cycle_release_metrics.get("action_steps_block_count_violations", 0))
    elif str(data.get("product_type") or meta.get("product_type") or "").strip().lower() == "life_cycle" or render_profile == "life_cycle_lite_v1":
        release_mode = "life_cycle_lite"
        life_cycle_release_metrics = _compute_life_cycle_lite_release_metrics(
            scored_surface_text,
            subject_name=str(structured_summary_payload.get("subject_name") or "").strip() or None,
            valid_until_fallback=meta.get("valid_until_fallback") if isinstance(meta.get("valid_until_fallback"), bool) else None,
        )
        front_contract = life_cycle_release_metrics.get("front_contract_detail", {})
        action_steps_metrics["action_steps_contract_ok"] = bool(life_cycle_release_metrics.get("action_steps_contract_ok", False))
        action_steps_metrics["action_steps_block_count_violations"] = int(life_cycle_release_metrics.get("action_steps_block_count_violations", 0))
    forbidden_findings = _filter_forbidden_hits_for_release_mode(
        scan_forbidden_patterns(
            scored_surface_text,
            allow_year_quarter_in_timing_map=bool(strict_vedic),
        ),
        release_mode or "generic",
    )
    vedic_scan = scan_vedic_term_budget(scored_surface_text) if strict_vedic else None
    timing_map_contract = scan_timing_map_contract(scored_surface_text) if strict_vedic else None
    density_metrics = _compute_body_paragraph_density_metrics(scored_surface_text)
    audit = data.get("audit") or {}
    debug_info = data.get("debug_info") if isinstance(data.get("debug_info"), dict) else {}
    selected_model = (
        data.get("selected_model")
        or debug_info.get("model_requested")
        or data.get("model")
    )
    model_used = (
        data.get("model_used")
        or debug_info.get("model_used")
        or data.get("model")
    )
    llm_input_source = (
        data.get("llm_input_source")
        or debug_info.get("llm_input_source")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / "truepath_runs" / f"{candidate.get('profile_name', 'profile')}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist reading output for direct inspection.
    (run_dir / "reading.md").write_text(reading_text, encoding="utf-8")
    scan_surface_path = run_dir / "reading_scan_surface.md"
    post_surface_path = run_dir / "reading_post_remediation.md"
    scored_surface_path = run_dir / "scored_surface.md"
    _write_text_lf(scan_surface_path, scan_surface_text)
    _write_text_lf(post_surface_path, post_surface_from_reading)
    _write_text_lf(scored_surface_path, scored_surface_text)
    _json_dump(run_dir / "ai_reading_response.json", data)

    scan_surface_sha256 = _sha256_text(scan_surface_text)
    post_sha256 = _sha256_text(post_surface_from_reading)
    scored_surface_sha256 = _sha256_text(scored_surface_text)
    scan_surface_file_sha256 = _sha256_bytes(scan_surface_path.read_bytes())
    post_file_sha256 = _sha256_bytes(post_surface_path.read_bytes())
    scored_surface_file_sha256 = _sha256_bytes(scored_surface_path.read_bytes())
    postprocess_applied = scan_surface_sha256 != post_sha256

    # Persist PDF output in same run folder (unless explicitly disabled).
    # NOTE: include_ai defaults to 1 so PDF reflects the same AI narrative path.
    pdf_feature_disabled = str(os.getenv("PDF_DISABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    pdf_include_ai = 1 if str(os.getenv("CHEAP_GATE_PDF_INCLUDE_AI", "1")).strip() not in {"0", "false", "no"} else 0
    pdf_status = None
    pdf_path = None
    pdf_error = None
    if pdf_feature_disabled:
        pdf_status = 503
        pdf_error = "PDF disabled by PDF_DISABLED flag"
    else:
        try:
            pdf_params = {
                **params,
                "include_ai": pdf_include_ai,
                # Reuse the same cached ai_reading payload to keep JSON/PDF consistent
                # and prevent duplicate LLM API calls.
                "ai_cache_key": ai_cache_key,
            }
            # Keep PDF timeout aligned with policy; never below true-path timeout.
            pdf_timeout = max(int(timeout_seconds), 180)
            pdf_resp = client.get("/pdf", params=pdf_params, timeout=pdf_timeout)
            pdf_status = pdf_resp.status_code
            if pdf_resp.status_code == 200:
                pdf_path = run_dir / "report.pdf"
                pdf_path.write_bytes(pdf_resp.content)
            else:
                pdf_error = (pdf_resp.text or "")[:400]
        except Exception as e:
            if _is_timeout_error(e):
                logger.error(
                    "TIMEOUT_OCCURRED stage=pdf timeout=%s profile=%s",
                    pdf_timeout,
                    candidate.get("profile_name", "unknown"),
                )
            pdf_error = str(e)

    return {
        "ok": True,
        "status_code": resp.status_code,
        "fallback": bool(data.get("fallback", True)),
        "llm_input_source": llm_input_source,
        "selected_model": selected_model,
        "model_used": model_used,
        "ai_cache_key": ai_cache_key,
        "chapter_blocks_hash": data.get("chapter_blocks_hash"),
        "audit_overall": int((audit.get("overall_score")) or 0),
        "audit_flags": audit.get("flags") if isinstance(audit, dict) else None,
        "scan_surface_source": scan_surface_source,
        "scored_surface_name": scored_surface_name,
        "reading_length": len(scored_surface_text),
        "heading_count": scored_surface_text.count("\n## ") + (1 if scored_surface_text.startswith("## ") else 0),
        "forbidden_hits": len(forbidden_findings),
        "actionable_bullet_coverage": actionable_bullet_coverage,
        "conversion_metrics": conversion_metrics,
        "front_english_token_hits": front_english_token_hits,
        "front_markdown_integrity": front_markdown_integrity,
        "release_mode": release_mode or "generic",
        "front_contract_ok": bool(life_cycle_release_metrics.get("front_contract_ok")) if release_mode in {"life_cycle_lite", "life_cycle_target"} else bool(front_contract.get("ok")),
        "front_contract_detail": front_contract,
        "front_contract_fallback_applied": bool(front_contract_fallback_applied),
        "front_playbook_repaired": front_playbook_repaired,
        "front_playbook_section_found": front_playbook_section_found,
        "front_playbook_slot_contract_ok_count": front_playbook_slot_contract_ok_count,
        "chapters_boundary_fallback_used": bool(chapters_boundary_fallback_used),
        "front_byte_equal_after_b_pass": bool(front_byte_equal_after_b_pass),
        "front_end_offset_stable": bool(front_end_offset_stable),
        **action_steps_metrics,
        **life_cycle_release_metrics,
        "inline_action_chain_violations": int(inline_action_chain_violations),
        **structure_definition_metrics,
        **quality_metrics,
        **duplicate_metrics,
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
        "timing_map_contract": timing_map_contract,
        "paragraph_density_metrics": density_metrics,
        "error": data.get("error"),
        "analysis_mode_fallback": data.get("analysis_mode_fallback"),
        "run_dir": str(run_dir),
        "reading_path": str(run_dir / "reading.md"),
        "scan_surface_path": str(scan_surface_path),
        "reading_scan_surface_path": str(scan_surface_path),
        "reading_post_remediation_path": str(post_surface_path),
        "scored_surface_path": str(scored_surface_path),
        "scan_surface_sha256": scan_surface_sha256,
        "post_sha256": post_sha256,
        "scored_surface_sha256": scored_surface_sha256,
        "scan_surface_file_sha256": scan_surface_file_sha256,
        "post_file_sha256": post_file_sha256,
        "scored_surface_file_sha256": scored_surface_file_sha256,
        "postprocess_applied": bool(postprocess_applied),
        "pdf_status": pdf_status,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "pdf_include_ai": int(pdf_include_ai),
        "pdf_disabled": bool(pdf_feature_disabled),
        "pdf_error": pdf_error,
        "fallback_reason_hint": (
            "fallback=true with missing llm markers"
            if bool(data.get("fallback", True))
            and not selected_model
            and not model_used
            else None
        ),
    }


async def run_cheap_validation(
    profile: str,
    force_truepath: bool,
    skip_truepath_on_same_hash: bool,
    allow_api: bool,
    timeout_seconds: int,
    strict_vedic: bool,
) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _build_candidates()
    selected = select_profiles(candidates)
    profile_name, candidate = _pick_profile(selected, profile)

    prompt, chapter_blocks = _build_prompt_for_candidate(candidate)
    static = _static_prompt_check(prompt)
    dry = _dry_structure_check(chapter_blocks, strict_vedic=strict_vedic)
    guard = _load_hash_guard()
    same_hash = static["prompt_hash"] == str(guard.get("prompt_hash", ""))

    should_run_truepath = bool(allow_api) and (force_truepath or not (skip_truepath_on_same_hash and same_hash))
    if not allow_api:
        truepath = {"skipped": True, "reason": "api_disabled"}
    elif not should_run_truepath:
        truepath = {"skipped": True, "reason": "prompt_hash_unchanged"}
    else:
        truepath = _run_single_true_path(
            candidate,
            timeout_seconds=timeout_seconds,
            strict_vedic=strict_vedic,
        )

    summary = {
        "profile_name": profile_name,
        "selected_model_env": os.getenv("OPENAI_MODEL", ""),
        "static": static,
        "dry": dry,
        "hash_guard": {
            "previous_hash": guard.get("prompt_hash"),
            "same_hash": same_hash,
            "should_run_truepath": should_run_truepath,
        },
        "truepath": truepath,
        "strict_vedic": bool(strict_vedic),
    }
    _json_dump(OUT_DIR / "cheap_validation_summary.json", summary)
    dry_density = dry.get("paragraph_density_metrics", {}) if isinstance(dry.get("paragraph_density_metrics"), dict) else {}
    dry_excluded = dry_density.get("excluded_blocks", {}) if isinstance(dry_density.get("excluded_blocks"), dict) else {}
    density_summary = (
        "avg_body_paragraphs_per_chapter="
        f"{float(dry_density.get('avg_body_paragraphs_per_chapter', 0.0)):.2f} "
        f"zero_body_chapter_count={int(dry_density.get('zero_body_chapter_count', 0))} "
        f"excluded={{heading:{int(dry_excluded.get('heading', 0))},list:{int(dry_excluded.get('list', 0))},short:{int(dry_excluded.get('short', 0))}}}"
    )

    print(
        "CHEAP_VALIDATION "
        f"profile={profile_name} "
        f"same_hash={same_hash} "
        f"truepath={'run' if should_run_truepath else 'skip'} "
        f"dry_forbidden={dry['forbidden_hits']} "
        f"dry_hard_style={len(dry['hard_style_errors'])} "
        f"strict_vedic={int(bool(strict_vedic))} "
        f"{density_summary}"
    )

    # Persist prompt hash after successful static+dry pass (whether true-path ran or skipped).
    _save_hash_guard(profile_name, static["prompt_hash"])

    if dry["forbidden_hits"] > 0 or len(dry["hard_style_errors"]) > 0:
        return 1
    if strict_vedic and _strict_vedic_failed(dry.get("vedic_scan"), dry.get("timing_map_contract")):
        _log_strict_vedic_fail("dry", dry.get("vedic_scan"), dry.get("timing_map_contract"))
        return 1
    if should_run_truepath:
        if not truepath.get("ok"):
            return 1
        if bool(truepath.get("fallback", True)):
            return 1
        if int(truepath.get("forbidden_hits", 0)) > 0:
            return 1
        if str(truepath.get("release_mode") or "") in {"life_cycle_lite", "life_cycle_target"} and not bool(truepath.get("life_cycle_release_ok", False)):
            return 1
        if strict_vedic and _strict_vedic_failed(truepath.get("vedic_scan"), truepath.get("timing_map_contract")):
            _log_strict_vedic_fail("truepath", truepath.get("vedic_scan"), truepath.get("timing_map_contract"))
            return 1
    return 0


def run_strict_vedic_scan(path: str, strict_vedic: bool, release_mode: str = "generic", subject_name: str = "") -> int:
    input_path = Path(path)
    if not input_path.exists() or not input_path.is_file():
        logger.error("input file not found: %s", input_path)
        return 1

    raw_text = input_path.read_text(encoding="utf-8")
    release_mode_norm = str(release_mode or "").strip()
    if release_mode_norm in {"life_cycle_lite", "life_cycle_target"}:
        normalized = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        remediated = normalized
        chapters_boundary_fallback_used = False
        quality_metrics: dict[str, int | bool] = {
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
            "commercial_quality_metrics_valid": False,
        }
    else:
        normalized = normalize_llm_layout_strict(raw_text)
        remediated = _apply_style_remediation(normalized, allow_zero_term_injection=False)
        remediated = postprocess_commercial_quality(remediated)
        chapters_text_for_quality, chapters_boundary_fallback_used = _extract_chapters_text_for_quality(remediated)
        if chapters_boundary_fallback_used:
            quality_metrics = {
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
                "commercial_quality_metrics_valid": False,
            }
        else:
            _quality_text, quality_metrics = postprocess_commercial_quality_with_metrics(chapters_text_for_quality)
    actionable_bullet_coverage = _compute_actionable_bullet_coverage(remediated)
    duplicate_metrics = _compute_exact_duplicate_metrics(remediated)
    style_errors = _reading_style_error_codes(remediated)
    forbidden_hits = _filter_forbidden_hits_for_release_mode(
        scan_forbidden_patterns(
            remediated,
            allow_year_quarter_in_timing_map=bool(strict_vedic),
        ),
        release_mode_norm,
    )
    vedic_scan = scan_vedic_term_budget(remediated) if strict_vedic else None
    timing_map_contract = scan_timing_map_contract(remediated) if strict_vedic else None

    density_metrics = _compute_body_paragraph_density_metrics(remediated)
    warn_only = {"label_pattern_detected", "paragraph_too_long", "warn_paragraph_density_low"}
    if release_mode_norm in {"life_cycle_lite", "life_cycle_target"}:
        warn_only = set(warn_only)
        warn_only.update({"paragraph_density_low", "paragraph_too_long"})
    hard_style_errors = [e for e in style_errors if e not in warn_only]
    life_cycle_release_metrics: dict[str, Any] = {}
    if release_mode_norm == "life_cycle_lite":
        life_cycle_release_metrics = _compute_life_cycle_lite_release_metrics(
            remediated,
            subject_name=str(subject_name or "").strip() or None,
            valid_until_fallback=None,
        )
    elif release_mode_norm == "life_cycle_target":
        life_cycle_release_metrics = _compute_life_cycle_target_release_metrics(
            remediated,
            subject_name=str(subject_name or "").strip() or None,
            valid_until_fallback=None,
        )

    payload = {
        "input_path": str(input_path),
        "strict_vedic": bool(strict_vedic),
        "release_mode": release_mode_norm or "generic",
        "text_length": len(remediated),
        "hard_style_errors": hard_style_errors,
        "warn_style_errors": [e for e in style_errors if e in warn_only],
        "forbidden_hits": len(forbidden_hits),
        "actionable_bullet_coverage": actionable_bullet_coverage,
        **duplicate_metrics,
        "inline_action_chain_violations": int(_compute_inline_action_chain_violations(remediated)),
        **quality_metrics,
        "chapters_boundary_fallback_used": bool(chapters_boundary_fallback_used),
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
        "timing_map_contract": timing_map_contract,
        "paragraph_density_metrics": density_metrics,
        **life_cycle_release_metrics,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _json_dump(OUT_DIR / f"strict_vedic_scan_{stamp}.json", payload)

    density_excluded = density_metrics.get("excluded_blocks", {}) if isinstance(density_metrics.get("excluded_blocks"), dict) else {}
    print(
        "STRICT_VEDIC_SCAN "
        f"input={input_path} "
        f"strict_vedic={int(bool(strict_vedic))} "
        f"forbidden={len(forbidden_hits)} "
        f"hard_style={len(hard_style_errors)} "
        f"avg_body_paragraphs_per_chapter={float(density_metrics.get('avg_body_paragraphs_per_chapter', 0.0)):.2f} "
        f"zero_body_chapter_count={int(density_metrics.get('zero_body_chapter_count', 0))} "
        f"excluded={{heading:{int(density_excluded.get('heading', 0))},list:{int(density_excluded.get('list', 0))},short:{int(density_excluded.get('short', 0))}}}"
    )

    if len(forbidden_hits) > 0 or len(hard_style_errors) > 0:
        return 1
    if release_mode_norm in {"life_cycle_lite", "life_cycle_target"} and not bool(payload.get("life_cycle_release_ok", False)):
        return 1
    if strict_vedic and _strict_vedic_failed(vedic_scan, timing_map_contract):
        _log_strict_vedic_fail("input", vedic_scan, timing_map_contract)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_timeout_env = os.getenv("DEFAULT_COMMAND_TIMEOUT")
    if default_timeout_env is None:
        default_timeout = 180
        logger.info("DEFAULT_COMMAND_TIMEOUT not set, using fallback=%s", default_timeout)
    else:
        default_timeout = int(default_timeout_env)
    parser.add_argument(
        "--profile",
        default="most_balanced",
        help="Selected golden profile for static/dry and optional single true-path run.",
    )
    parser.add_argument(
        "--force-truepath",
        action="store_true",
        help="Run one true-path call even when prompt hash is unchanged.",
    )
    parser.add_argument(
        "--skip-truepath-on-same-hash",
        type=int,
        default=1,
        help="Skip true-path when prompt hash is unchanged (1=yes, 0=no).",
    )
    parser.add_argument(
        "--allow-api",
        type=int,
        default=0,
        help="Allow true-path API call (1=yes, 0=no). Default is 0 for token-safe validation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=max(default_timeout, 120),
        help="Per-command timeout seconds. Default from DEFAULT_COMMAND_TIMEOUT (min 120, recommended 180).",
    )
    parser.add_argument(
        "--strict-vedic",
        type=int,
        default=0,
        help="Enable strict Vedic budget/stacking hard-fail checks (1=yes, 0=no).",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Direct markdown/text path for synchronous strict scan. Bypasses profile pipeline.",
    )
    parser.add_argument(
        "--release-mode",
        default="generic",
        choices=["generic", "life_cycle_lite", "life_cycle_target"],
        help="Optional product-specific release semantics for direct input scans.",
    )
    parser.add_argument(
        "--subject-name",
        default="",
        help="Optional subject name used by life_cycle release-mode HF16 checks.",
    )
    args = parser.parse_args()
    if isinstance(args.input, str) and args.input.strip():
        raise SystemExit(
            run_strict_vedic_scan(
                path=args.input.strip(),
                strict_vedic=bool(args.strict_vedic),
                release_mode=str(args.release_mode or "generic"),
                subject_name=str(args.subject_name or ""),
            )
        )
    raise SystemExit(
        asyncio.run(
            run_cheap_validation(
                profile=args.profile,
                force_truepath=bool(args.force_truepath),
                skip_truepath_on_same_hash=bool(args.skip_truepath_on_same_hash),
                allow_api=bool(args.allow_api),
                timeout_seconds=int(args.timeout_seconds),
                strict_vedic=bool(args.strict_vedic),
            )
        )
    )
