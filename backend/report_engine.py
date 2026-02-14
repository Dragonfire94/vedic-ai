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

REINFORCE_RULES = [
    {
        "if_block_ids": ["high_tension_risk_aggro", "high_stability_fall"],
        "add_block_id": "tension_stability_interaction",
        "target_chapter": "Psychological Architecture",
    },
    {
        "if_block_ids": ["career_conflict_karma", "career_purushartha"],
        "add_block_id": "career_karma_pattern_reinforcement",
        "target_chapter": "Executive Summary",
    },
]


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


def compute_block_intensity(block: dict[str, Any], features: dict[str, Any]) -> float:
    """Compute a normalized [0.0, 1.0] narrative intensity score for a block."""

    del block  # intensity depends only on structural_summary features in this phase.

    base = 0.0

    tension = features.get("psychological_tension_axis", 0)
    if isinstance(tension, dict):
        tension = tension.get("score", 0)
    if isinstance(tension, (int, float)):
        base += tension / 100

    risk_profile = features.get("behavioral_risk_profile", {})
    if isinstance(risk_profile, dict) and risk_profile:
        numeric_risks = [
            value / 100 for value in risk_profile.values() if isinstance(value, (int, float))
        ]
        if numeric_risks:
            base += sum(numeric_risks) / len(numeric_risks)

    stability = features.get("stability_metrics", {})
    stability_index: Any = 0
    if isinstance(stability, dict):
        stability_index = stability.get("stability_index", 0)
    if isinstance(stability_index, (int, float)):
        base += (100 - stability_index) / 100

    return min(max(base / 3.0, 0.0), 1.0)


def _lookup_template_by_id(block_id: str) -> dict[str, Any] | None:
    for block in TEMPLATES:
        if block.get("id") == block_id:
            return block
    return None


def _sort_selected_blocks(selected: dict[str, list[dict[str, Any]]]) -> None:
    for chapter in selected:
        selected[chapter].sort(
            key=lambda b: (b.get("priority", 0), b.get("_intensity", 0), b.get("_match_index", -1)),
            reverse=True,
        )


def _apply_cross_chapter_reinforcement(selected: dict[str, list[dict[str, Any]]], structural_summary: dict[str, Any]) -> None:
    selected_ids = {block.get("id") for blocks in selected.values() for block in blocks}
    for rule in REINFORCE_RULES:
        if all(block_id in selected_ids for block_id in rule.get("if_block_ids", [])):
            add_id = rule.get("add_block_id")
            target_chapter = rule.get("target_chapter")
            if not isinstance(add_id, str) or target_chapter not in selected:
                continue
            add_block = _lookup_template_by_id(add_id)
            if not add_block:
                continue

            already_present = any(b.get("id") == add_id for b in selected[target_chapter])
            if already_present:
                continue

            reinforced = dict(add_block)
            reinforced["_intensity"] = compute_block_intensity(reinforced, structural_summary)
            reinforced["_match_index"] = len(selected[target_chapter])
            selected[target_chapter].append(reinforced)


def select_template_blocks(structural_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _ensure_loaded()
    selected: dict[str, list[dict[str, Any]]] = {chapter: [] for chapter in REPORT_CHAPTERS}

    matched_index = 0
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
                matched = dict(block)
                matched["_intensity"] = compute_block_intensity(matched, structural_summary)
                matched["_match_index"] = matched_index
                matched_index += 1
                selected[chapter].append(matched)

    for chapter in selected:
        seen_ids = {b.get("id") for b in selected[chapter]}
        idx = 0
        while idx < len(selected[chapter]):
            followups = selected[chapter][idx].get("chain_followups", [])
            for followup_id in followups if isinstance(followups, list) else []:
                followup = _lookup_template_by_id(str(followup_id))
                if not followup or followup.get("chapter") != chapter:
                    continue
                if followup.get("id") in seen_ids:
                    continue
                chained = dict(followup)
                chained["_intensity"] = compute_block_intensity(chained, structural_summary)
                chained["_match_index"] = matched_index
                matched_index += 1
                selected[chapter].append(chained)
                seen_ids.add(chained.get("id"))
            idx += 1

    _apply_cross_chapter_reinforcement(selected, structural_summary)
    _sort_selected_blocks(selected)

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

        chapter_payload: list[dict[str, str]] = []
        for block in blocks[:5]:
            content = block.get("content", {})
            intensity = block.get("_intensity", 0)

            if intensity > 0.8:
                include_fields = ["title", "summary", "analysis", "implication", "examples"]
            elif intensity > 0.5:
                include_fields = ["title", "summary", "analysis", "implication"]
            else:
                include_fields = ["title", "summary", "analysis"]

            payload_block: dict[str, str] = {}
            for field in include_fields:
                default_val = chapter if field == "title" else ""
                payload_block[field] = str(content.get(field, default_val))
            chapter_payload.append(payload_block)

        chapter_blocks[chapter] = chapter_payload

    return {"chapter_blocks": chapter_blocks}


def build_gpt_user_content(payload: dict[str, Any]) -> str:
    return "<BEGIN STRUCTURED BLOCKS>\n" + json.dumps(payload.get("chapter_blocks", {}), ensure_ascii=False, indent=2) + "\n<END STRUCTURED BLOCKS>"
