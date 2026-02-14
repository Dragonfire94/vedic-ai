"""Deterministic report block selector and GPT payload builder."""

from __future__ import annotations

import json
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

_TEMPLATE_FILES = {
    "purushartha": "purushartha_patterns.json",
    "karmic": "karmic_patterns.json",
    "psychological": "psychological_axis_patterns.json",
    "behavioral": "behavioral_risk_patterns.json",
    "stability": "stability_grade_patterns.json",
}

_LIBRARY_TO_CHAPTER = {
    "purushartha": "Purushartha Profile",
    "karmic": "Karmic Patterns",
    "psychological": "Psychological Architecture",
    "behavioral": "Behavioral Risks",
    "stability": "Stability Metrics",
}

_DEFAULT_BLOCKS = {
    "Executive Summary": {
        "title": "Executive Summary",
        "summary": "This report synthesizes deterministic psychological-astrology signals into a coherent personal narrative.",
        "insight": "The goal is clarity, decision quality, and long-term stability through structured interpretation.",
    },
    "Personality Vector": {
        "title": "Personality Vector",
        "summary": "Your personality vector indicates a consistent pattern of adaptive strengths and pressure sensitivities.",
        "insight": "Sustained growth depends on leveraging strengths while managing repeating stress signatures.",
    },
    "Life Timeline Interpretation": {
        "title": "Life Timeline Interpretation",
        "summary": "The timeline chapter frames development as phases of activation, consolidation, and recalibration.",
        "insight": "Viewing timing as cycles supports better planning and lower emotional overreaction.",
    },
    "Career & Success": {
        "title": "Career & Success",
        "summary": "Career outcomes are favored by deliberate role alignment, strategic pacing, and accountability loops.",
        "insight": "Execution quality improves when goals are anchored to values and realistic delivery windows.",
    },
    "Love & Relationships": {
        "title": "Love & Relationships",
        "summary": "Relationship patterns highlight the balance between autonomy, emotional availability, and trust-building.",
        "insight": "Relational stability grows through transparent expectations and repeatable communication habits.",
    },
    "Health & Body Patterns": {
        "title": "Health & Body Patterns",
        "summary": "Body-pattern guidance emphasizes stress regulation, recovery hygiene, and rhythm consistency.",
        "insight": "Preventive routines are more protective than reactive correction under high-load periods.",
    },
    "Confidence & Forecast": {
        "title": "Confidence & Forecast",
        "summary": "Forecast confidence should be interpreted as directional guidance rather than deterministic certainty.",
        "insight": "Planning with ranges and contingencies preserves resilience while maintaining momentum.",
    },
    "Remedies & Program": {
        "title": "Remedies & Program",
        "summary": "Recommended program design focuses on small, enforceable routines that compound over time.",
        "insight": "Consistency in daily practice is the central mechanism for long-term correction and growth.",
    },
    "Final Summary": {
        "title": "Final Summary",
        "summary": "Your profile points to meaningful potential when structure and self-awareness are applied together.",
        "insight": "The next stage is disciplined integration of insight into everyday execution.",
    },
    "Appendix (Optional)": {
        "title": "Appendix",
        "summary": "Appendix notes can include glossary, methodology, and implementation checklist items.",
        "insight": "Use this section as a practical reference layer for sustained application.",
    },
}

_TEMPLATE_LIBRARIES: dict[str, list[dict[str, Any]]] = {}


def _load_template_libraries() -> dict[str, list[dict[str, Any]]]:
    base = Path(__file__).resolve().parent / "report_templates"
    libraries: dict[str, list[dict[str, Any]]] = {}
    for key, filename in _TEMPLATE_FILES.items():
        path = base / filename
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        libraries[key] = payload if isinstance(payload, list) else []
    return libraries


def get_template_libraries() -> dict[str, list[dict[str, Any]]]:
    global _TEMPLATE_LIBRARIES
    if not _TEMPLATE_LIBRARIES:
        _TEMPLATE_LIBRARIES = _load_template_libraries()
    return _TEMPLATE_LIBRARIES


def _get_path_value(data: dict[str, Any], path: str) -> Any:
    cursor: Any = data
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _condition_matches(structural_summary: dict[str, Any], conditions: dict[str, Any]) -> bool:
    for key, expected in conditions.items():
        actual = _get_path_value(structural_summary, key)
        if actual is None and "." not in key and key in structural_summary:
            actual = structural_summary.get(key)
        if actual != expected:
            return False
    return True


def select_template_blocks(structural_summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for library_key, blocks in get_template_libraries().items():
        chapter = _LIBRARY_TO_CHAPTER[library_key]
        for block in blocks:
            conditions = block.get("conditions", {})
            template = block.get("template", {})
            if isinstance(conditions, dict) and _condition_matches(structural_summary, conditions):
                selected[chapter] = {
                    "title": str(template.get("title", chapter)),
                    "summary": str(template.get("summary", "")),
                    "insight": str(template.get("insight", "")),
                    "source_block_id": str(block.get("id", "")),
                }
                break
    return selected


def _chapter_fallback(chapter: str) -> dict[str, str]:
    default = _DEFAULT_BLOCKS.get(
        chapter,
        {
            "title": chapter,
            "summary": f"{chapter} is synthesized from deterministic signal combinations.",
            "insight": "This chapter is constrained to provided interpretation blocks only.",
        },
    )
    return {
        "title": default["title"],
        "summary": default["summary"],
        "insight": default["insight"],
        "source_block_id": "default",
    }


def build_report_payload(rectified_structural_summary: dict[str, Any]) -> dict[str, Any]:
    structural = rectified_structural_summary.get("structural_summary") if isinstance(rectified_structural_summary, dict) else None
    if not isinstance(structural, dict):
        structural = rectified_structural_summary if isinstance(rectified_structural_summary, dict) else {}

    selected = select_template_blocks(structural)
    chapter_blocks: dict[str, dict[str, str]] = {}
    for chapter in REPORT_CHAPTERS:
        chapter_blocks[chapter] = selected.get(chapter, _chapter_fallback(chapter))

    return {"chapter_blocks": chapter_blocks}


def build_gpt_user_content(payload: dict[str, Any]) -> str:
    return "<BEGIN STRUCTURED BLOCKS>\n" + json.dumps(payload.get("chapter_blocks", {}), ensure_ascii=False, indent=2) + "\n<END STRUCTURED BLOCKS>"
