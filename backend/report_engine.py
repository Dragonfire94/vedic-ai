"""Deterministic report block selector and GPT payload builder."""

from __future__ import annotations

import json
import operator as op
from pathlib import Path
from typing import Any

REPORT_CHAPTERS = [
    "Executive Summary",
    "Purushartha Profile",
    "Psychological Architecture",
    "Behavioral Risks",
    "Karmic Patterns",
    "Stability Metrics",
    "Personality Vector",
    "Life Timeline Interpretation",
    "Career & Success",
    "Love & Relationships",
    "Health & Body Patterns",
    "Confidence & Forecast",
    "Remedies & Program",
    "Final Summary",
    "Appendix (Optional)",
]

SYSTEM_PROMPT = """You are a master narrative editor.
Provided with structured interpretation blocks for each chapter,
your job is not to compute astrology.
You are only to stitch the provided text fragments into a cohesive narrative.

Constraints:
- Do NOT invent new astrology interpretations.
- Do NOT add new causes or facts.
- Write in a professional, analytical and coherent style.
- Each chapter should have:
    Title
    Intro paragraph
    At least 2 paragraphs discussing the block content
    A concluding sentence tying it to the person’s journey.

Output must be plain text (no JSON) with explicit chapter boundaries marked.

Chapters to include in exact order:
Executive Summary
Purushartha Profile
Psychological Architecture
Behavioral Risks
Karmic Patterns
Stability Metrics
Personality Vector
Life Timeline Interpretation
Career & Success
Love & Relationships
Health & Body Patterns
Confidence & Forecast
Remedies & Program
Final Summary
Appendix (Optional)"""

_TEMPLATE_FILES = [
    "purushartha_patterns.json",
    "karmic_patterns.json",
    "psychological_axis_patterns.json",
    "behavioral_risk_patterns.json",
    "stability_grade_patterns.json",
]
_DEFAULT_TEMPLATE_FILE = "default_patterns.json"

OP_MAP = {
    "==": op.eq,
    "!=": op.ne,
    "<": op.lt,
    "<=": op.le,
    ">": op.gt,
    ">=": op.ge,
}

TEMPLATES: list[dict[str, Any]] = []
DEFAULT_BLOCKS: dict[str, list[dict[str, Any]]] = {}


def _load_template_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _load_templates() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    base = Path(__file__).resolve().parent / "report_templates"
    templates: list[dict[str, Any]] = []
    for filename in _TEMPLATE_FILES:
        templates.extend(_load_template_file(base / filename))

    defaults: dict[str, list[dict[str, Any]]] = {chapter: [] for chapter in REPORT_CHAPTERS}
    for block in _load_template_file(base / _DEFAULT_TEMPLATE_FILE):
        chapter = block.get("chapter")
        if chapter in defaults:
            defaults[chapter].append(block)

    for chapter in defaults:
        defaults[chapter].sort(key=lambda b: b.get("priority", 0), reverse=True)

    return templates, defaults


def _ensure_loaded() -> None:
    global TEMPLATES, DEFAULT_BLOCKS
    if not TEMPLATES and not DEFAULT_BLOCKS:
        TEMPLATES, DEFAULT_BLOCKS = _load_templates()


def get_template_libraries() -> dict[str, Any]:
    _ensure_loaded()
    return {"templates": TEMPLATES, "defaults": DEFAULT_BLOCKS}


def _get_structural_value(structural_summary: dict[str, Any], field_path: str) -> Any:
    parts = field_path.split(".")
    val: Any = structural_summary
    for part in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(part)
        if val is None:
            return None
    return val


def _evaluate_condition(structural_summary: dict[str, Any], condition: dict[str, Any]) -> bool:
    field_val = _get_structural_value(structural_summary, str(condition.get("field", "")))
    if field_val is None:
        return False

    operator_fn = OP_MAP.get(condition.get("operator"))
    if operator_fn is None:
        return False

    try:
        return bool(operator_fn(field_val, condition.get("value")))
    except Exception:
        return False


def select_template_blocks(structural_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _ensure_loaded()
    selected: dict[str, list[dict[str, Any]]] = {chapter: [] for chapter in REPORT_CHAPTERS}

    for block in TEMPLATES:
        conditions = block.get("conditions", [])
        logic = str(block.get("logic", "AND")).upper()

        if not isinstance(conditions, list) or not conditions:
            continue

        results = [_evaluate_condition(structural_summary, cond) for cond in conditions if isinstance(cond, dict)]
        if not results:
            continue

        if logic == "AND":
            passed = all(results)
        elif logic == "OR":
            passed = any(results)
        else:
            passed = False

        if passed:
            chapter = block.get("chapter")
            if chapter in selected:
                selected[chapter].append(block)

    for chapter in selected:
        selected[chapter].sort(key=lambda b: b.get("priority", 0), reverse=True)

    return selected


def build_report_payload(rectified_structural_summary: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    structural = rectified_structural_summary.get("structural_summary") if isinstance(rectified_structural_summary, dict) else None
    if not isinstance(structural, dict):
        structural = rectified_structural_summary if isinstance(rectified_structural_summary, dict) else {}

    raw_blocks = select_template_blocks(structural)

    chapter_blocks: dict[str, list[dict[str, str]]] = {}
    for chapter in REPORT_CHAPTERS:
        blocks = raw_blocks.get(chapter, [])
        if not blocks:
            blocks = DEFAULT_BLOCKS.get(chapter, [])

        chapter_blocks[chapter] = [
            {
                "title": str(content.get("title", chapter)),
                "summary": str(content.get("summary", "")),
                "analysis": str(content.get("analysis", "")),
                "implication": str(content.get("implication", "")),
            }
            for content in [block.get("content", {}) for block in blocks[:5]]
        ]

    return {"chapter_blocks": chapter_blocks}


def build_gpt_user_content(payload: dict[str, Any]) -> str:
    return "<BEGIN STRUCTURED BLOCKS>\n" + json.dumps(payload.get("chapter_blocks", {}), ensure_ascii=False, indent=2) + "\n<END STRUCTURED BLOCKS>"
