"""Report assembly helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from backend.report_engine import normalize_recommendation_tone
from backend.report_config import (
    CHAPTER_DISPLAY_NAME_KO,
    PREMIUM_12_CHAPTER_ORDER,
    STRONG_META_LINE_PATTERNS,
    MODERATE_META_LINE_PATTERNS,
    CHAPTER_NARRATIVE_LINES_KO,
    FALLBACK_NARRATIVE_LINES_KO,
    FORBIDDEN_OUTPUT_REGEXES,
)

logger = logging.getLogger("vedic_ai")


def _active_chapter_order_for_style() -> list[str]:
    return list(PREMIUM_12_CHAPTER_ORDER)


def _stable_pick(lines: list[str], chapter_key: str, salt: str) -> list[str]:
    if not lines:
        return []
    digest = hashlib.sha256(f"{chapter_key}::{salt}".encode("utf-8")).hexdigest()
    start = int(digest[:8], 16) % len(lines)
    k = min(3, len(lines))
    return [lines[(start + i) % len(lines)] for i in range(k)]


def _contains_forbidden_output(text: str) -> bool:
    if not text:
        return False
    return any(rx.search(text) for rx in FORBIDDEN_OUTPUT_REGEXES)


def _sanitize_deterministic_text_ko(chapter_key: str, text: str, *, salt: str) -> tuple[str, int]:
    if not text or not text.strip():
        return text, 0

    patterns = STRONG_META_LINE_PATTERNS + MODERATE_META_LINE_PATTERNS
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!--") and "chapter_key:" in stripped:
            kept.append(line)
            continue
        if any(p.search(line) for p in patterns):
            removed += 1
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    if removed > 0 and len(cleaned) < 120:
        candidate = CHAPTER_NARRATIVE_LINES_KO.get(chapter_key, FALLBACK_NARRATIVE_LINES_KO)
        bridge = _stable_pick(candidate, chapter_key, salt=salt)
        if bridge:
            cleaned = (cleaned + "\n\n" if cleaned else "") + "\n".join(bridge)
    return cleaned, removed


def _render_chapter_blocks_deterministic(chapter_blocks: dict[str, Any], language: str = "ko") -> str:
    chapter_name_ko = {
        "Executive Diagnosis": "Executive Diagnosis",
        "Current Phase": "Current Phase",
        "Core Disposition": "Core Disposition",
        "Recurring Patterns": "Recurring Patterns",
        "Emotional Fault Lines": "Emotional Fault Lines",
        "Career & Money": "Career & Money",
        "Love & Relationship Patterns": "Love & Relationship Patterns",
        "Health & Energy Rhythm": "Health & Energy Rhythm",
        "Mid-Term Direction": "Mid-Term Direction",
        "Risk Management Points": "Risk Management Points",
        "Growth Acceleration": "Growth Acceleration",
        "Final Integration": "Final Integration",
    }
    ordered_fields = [
        "title",
        "summary",
        "analysis",
        "implication",
        "examples",
        "shadow_pattern",
        "defense_mechanism",
        "emotional_trigger",
        "repetition_cycle",
        "integration_path",
        "micro_scenario",
        "long_term_projection",
    ]
    out: list[str] = []
    qa_removed_lines_total = 0
    qa_forbidden_hits = 0
    lang_norm = str(language or "ko").strip().lower()
    for idx, chapter in enumerate(_active_chapter_order_for_style(), start=1):
        title_legacy = chapter_name_ko.get(chapter, chapter)
        display_title = (
            CHAPTER_DISPLAY_NAME_KO.get(chapter, title_legacy or chapter)
            if lang_norm.startswith("ko")
            else (title_legacy or chapter)
        )
        out.append(f"# {idx}. {display_title}")
        out.append(f"<!-- chapter_key: {chapter} -->")
        blocks = chapter_blocks.get(chapter, []) if isinstance(chapter_blocks, dict) else []
        if not blocks:
            out.append("Insufficient chapter fragments were available for this section.")
            out.append("")
            continue

        for block_idx, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            if "spike_text" in block:
                spike_text = str(block.get("spike_text", "")).strip()
                if spike_text:
                    if lang_norm.startswith("ko"):
                        spike_text, removed_count = _sanitize_deterministic_text_ko(
                            chapter, spike_text, salt=f"spike:{block_idx}"
                        )
                        qa_removed_lines_total += removed_count
                        if _contains_forbidden_output(spike_text):
                            qa_forbidden_hits += 1
                    out.append(f"[Insight Spike {block_idx}] {spike_text}")
                continue

            if not lang_norm.startswith("ko"):
                out.append(f"## Fragment {block_idx}")
            for field in ordered_fields:
                raw = block.get(field)
                if not isinstance(raw, str):
                    continue
                value = raw.strip()
                if not value:
                    continue
                if lang_norm.startswith("ko"):
                    value, removed_count = _sanitize_deterministic_text_ko(
                        chapter, value, salt=f"{field}:{block_idx}"
                    )
                    qa_removed_lines_total += removed_count
                    if not value:
                        continue
                    if _contains_forbidden_output(value):
                        qa_forbidden_hits += 1
                    out.append(value)
                else:
                    out.append(f"{field.replace('_', ' ').title()}:")
                    out.append(value)
                out.append("")

            choice_fork = block.get("choice_fork")
            if isinstance(choice_fork, dict):
                if lang_norm.startswith("ko"):
                    bridge = _stable_pick(
                        CHAPTER_NARRATIVE_LINES_KO.get(chapter, FALLBACK_NARRATIVE_LINES_KO),
                        chapter,
                        salt=f"choice_fork:{block_idx}",
                    )
                    if bridge:
                        out.extend(bridge)
                else:
                    out.append("Choice Fork:")
                    out.append(json.dumps(choice_fork, ensure_ascii=False, indent=2))
                out.append("")

            predictive = block.get("predictive_compression")
            if isinstance(predictive, dict):
                if lang_norm.startswith("ko"):
                    bridge = _stable_pick(
                        CHAPTER_NARRATIVE_LINES_KO.get(chapter, FALLBACK_NARRATIVE_LINES_KO),
                        chapter,
                        salt=f"predictive_compression:{block_idx}",
                    )
                    if bridge:
                        out.extend(bridge)
                else:
                    out.append("Predictive Compression:")
                    out.append(json.dumps(predictive, ensure_ascii=False, indent=2))
                out.append("")
        out.append("")
    rendered = "\n".join(out).strip()
    if lang_norm.startswith("ko") and (qa_removed_lines_total > 0 or qa_forbidden_hits > 0):
        logger.debug(
            "deterministic_sanitizer_summary removed_lines=%s forbidden_hits=%s",
            qa_removed_lines_total,
            qa_forbidden_hits,
        )
    return rendered


def _apply_recommendation_tone_normalization(text: str | None, language: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return text
    return normalize_recommendation_tone(
        text,
        language=language,
        allowed_chapters=set(_active_chapter_order_for_style()),
    )
