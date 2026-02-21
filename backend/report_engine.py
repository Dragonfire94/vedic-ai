"""Deterministic report block selector and GPT payload builder."""

from __future__ import annotations

import json
import logging
import os
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
- You must only refine and improve readability of the provided deterministic astrology report. Do not add, infer, or invent new astrological interpretation.
- Write in a professional, analytical and coherent style.
- Produce full-length, publication-grade detail in every chapter.
- Do not compress chapters into short summaries.
- Each chapter should have:
    Title
    Intro paragraph
    At least 4 substantial paragraphs discussing the block content
    Practical implications and application guidance
    A concluding sentence tying it to the person's journey.

Output format contract (deterministic):
- Output must be Markdown text (no JSON).
- Preserve deterministic chapter boundaries using level-2 markdown headings exactly as `## <Chapter Name>`.
- Use the chapter heading list below in exact order with no omissions or renaming.
- Within each chapter, include semantic emphasis markers where appropriate (e.g., `**Key Insight**`, `*Caution*`, `**Action**`) while keeping claims grounded only in provided blocks.

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
_TEMPLATES_BY_LANG: dict[str, list[dict[str, Any]]] = {}
_DEFAULTS_BY_LANG: dict[str, dict[str, list[dict[str, Any]]]] = {}
_ACTIVE_LANGUAGE = "en"
logger = logging.getLogger("report_engine")
REPORT_MAPPING_DEBUG = str(os.getenv("REPORT_MAPPING_DEBUG", "1")).strip().lower() not in {"0", "false", "no", "off"}

INTERPRETATIONS_KR_FILE = Path(__file__).resolve().parent.parent / "assets" / "data" / "interpretations.kr_final.json"
INTERPRETATIONS_KR: dict[str, Any] = {}
INTERPRETATIONS_KR_ATOMIC: dict[str, Any] = {}
INTERPRETATIONS_KR_ENTRIES_COUNT = 0
ATOMIC_RUNTIME_AUDIT = {
    "atomic_keys_generated": 0,
    "atomic_lookup_hits": 0,
    "atomic_lookup_misses": 0,
    "atomic_text_applied_count": 0,
}

MIN_DEPTH_KO_CHARS = 0
MIN_DEPTH_EN_WORDS = 0
BASE_CHAPTER_LIMIT = 5
MAX_DYNAMIC_CHAPTER_LIMIT = 12

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

EMOTIONAL_ESCALATION_RULE = {
    "tension_threshold": 75,
    "stability_threshold": 45,
    "inject_block_id": "high_pressure_identity_fragmentation",
    "target_chapter": "Psychological Architecture",
}

CHOICE_FORK_RULES = [
    {
        "conditions": {
            "psychological_tension_axis.score": {">=": 70},
        },
        "inject_into_chapter": "Psychological Architecture",
        "fork_id": "tension_choice_fork",
    },
    {
        "conditions": {
            "behavioral_risk_profile.primary_risk": "impulsivity",
        },
        "inject_into_chapter": "Behavioral Risks",
        "fork_id": "impulsivity_choice_fork",
    },
    {
        "conditions": {
            "stability_metrics.stability_index": {"<=": 45},
        },
        "inject_into_chapter": "Stability Metrics",
        "fork_id": "stability_choice_fork",
    },
]

SCENARIO_COMPRESSION_RULES = [
    {
        "id": "structural_transition_window",
        "conditions": {
            "probability_forecast.career_shift_3yr": {">=": 0.6},
            "probability_forecast.marriage_5yr": {">=": 0.6},
        },
        "chapter": "Final Summary",
        "priority": 98,
    },
    {
        "id": "burnout_risk_window",
        "conditions": {
            "probability_forecast.burnout_2yr": {">=": 0.7},
            "stability_metrics.stability_index": {"<=": 50},
        },
        "chapter": "Stability Metrics",
        "priority": 97,
    },
    {
        "id": "financial_instability_window",
        "conditions": {
            "probability_forecast.financial_instability_3yr": {">=": 0.65},
        },
        "chapter": "Final Summary",
        "priority": 96,
    },
]


def _extract_interpretation_text(entry: Any) -> str | None:
    if isinstance(entry, dict):
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    elif isinstance(entry, str) and entry.strip():
        return entry.strip()
    return None


def _new_mapping_audit() -> dict[str, Any]:
    return {
        "total_signals_processed": 0,
        "mapping_hits": 0,
        "mapping_misses": 0,
        "used_mapping_keys": {},
        "atomic_inputs_available": {
            "ascendant": False,
            "sun": False,
            "moon": False,
        },
        "atomic_usage": {
            "ascendant": {"hits": 0, "misses": 0},
            "sun": {"hits": 0, "misses": 0},
            "moon": {"hits": 0, "misses": 0},
        },
    }


def _load_interpretations_kr() -> tuple[dict[str, Any], dict[str, Any], int]:
    if not INTERPRETATIONS_KR_FILE.exists():
        logger.warning("interpretations.kr_final.json not found at %s", INTERPRETATIONS_KR_FILE)
        return {}, {}, 0
    try:
        with open(INTERPRETATIONS_KR_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("interpretations.kr_final.json load failed: %s", exc)
        return {}, {}, 0

    try:
        ko_payload = data["ko"]
        atomic_payload = ko_payload["atomic"]
    except Exception as exc:
        logger.warning("interpretations.kr_final.json missing ko/atomic structure: %s", exc)
        return {}, {}, 0

    if not isinstance(ko_payload, dict) or not isinstance(atomic_payload, dict):
        logger.warning("interpretations.kr_final.json invalid ko/atomic types")
        return {}, {}, 0

    entries = 0
    for section_name in ("atomic", "lagna_lord", "yogas", "patterns"):
        section = ko_payload.get(section_name)
        if isinstance(section, dict):
            entries += len(section)

    logger.info("interpretations.kr_final.json loaded successfully, entries count=%s", entries)
    logger.info("interpretations.kr_final.json loaded: atomic_count=%d", len(atomic_payload))
    return ko_payload, atomic_payload, entries


INTERPRETATIONS_KR, INTERPRETATIONS_KR_ATOMIC, INTERPRETATIONS_KR_ENTRIES_COUNT = _load_interpretations_kr()


def _load_template_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _localized_ko_content(chapter: str, block_id: str) -> dict[str, Any]:
    return {
        "title": f"{chapter} - {block_id}",
        "summary": f"현재 `{block_id}` 블록에 대한 한국어 템플릿이 없어 기본 요약을 표시합니다.",
        "analysis": "세부 해석 템플릿이 준비되지 않아 일반 분석 문구로 대체되었습니다. 템플릿 파일을 보강하면 보다 정밀한 표현으로 자동 대체됩니다.",
        "implication": "이 문단은 임시 대체 텍스트이며, 실제 서비스 품질을 위해 대응되는 한국어 블록을 report_templates_ko에 추가하는 것을 권장합니다.",
        "examples": "예시: 템플릿 누락 시 기본 설명이 표시됩니다. 템플릿 추가 후에는 해당 문단이 맞춤 해석으로 교체됩니다.",
    }


def _localize_block_ko(base_block: dict[str, Any]) -> dict[str, Any]:
    localized = dict(base_block)
    content = localized.get("content")
    if isinstance(content, dict):
        localized_content = dict(content)
        localized_content.update(_localized_ko_content(str(localized.get("chapter", "")), str(localized.get("id", "block"))))
        localized["content"] = localized_content
    else:
        localized["content"] = _localized_ko_content(str(localized.get("chapter", "")), str(localized.get("id", "block")))
    return localized


def _load_templates_for_language(language: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    lang = str(language or "en").strip().lower()
    base = Path(__file__).resolve().parent / "report_templates"
    ko_base = Path(__file__).resolve().parent / "report_templates_ko"
    templates: list[dict[str, Any]] = []
    ko_templates_by_id: dict[str, dict[str, Any]] = {}
    if lang == "ko":
        for filename in _TEMPLATE_FILES:
            for block in _load_template_file(ko_base / filename):
                block_id = str(block.get("id", "")).strip()
                if block_id:
                    ko_templates_by_id[block_id] = block

    for filename in _TEMPLATE_FILES:
        for block in _load_template_file(base / filename):
            if lang == "ko":
                block_id = str(block.get("id", "")).strip()
                if block_id in ko_templates_by_id:
                    block = dict(ko_templates_by_id[block_id])
                else:
                    block = _localize_block_ko(block)
            chapter = block.get("chapter")
            if chapter not in REPORT_CHAPTERS:
                logger.warning("Template block has unknown chapter '%s': id=%s", chapter, block.get("id"))
            templates.append(block)

    defaults: dict[str, list[dict[str, Any]]] = {chapter: [] for chapter in REPORT_CHAPTERS}
    ko_defaults_by_id: dict[str, dict[str, Any]] = {}
    if lang == "ko":
        for block in _load_template_file(ko_base / _DEFAULT_TEMPLATE_FILE):
            block_id = str(block.get("id", "")).strip()
            if block_id:
                ko_defaults_by_id[block_id] = block
    for block in _load_template_file(base / _DEFAULT_TEMPLATE_FILE):
        if lang == "ko":
            block_id = str(block.get("id", "")).strip()
            if block_id in ko_defaults_by_id:
                block = dict(ko_defaults_by_id[block_id])
            else:
                block = _localize_block_ko(block)
        chapter = block.get("chapter")
        if chapter in defaults:
            defaults[chapter].append(block)
        else:
            logger.warning("Default block has unknown chapter '%s': id=%s", chapter, block.get("id"))

    for chapter in defaults:
        defaults[chapter].sort(key=lambda b: b.get("priority", 0), reverse=True)

    return templates, defaults


def _validate_rule_chapters() -> None:
    for rule in SCENARIO_COMPRESSION_RULES:
        chapter = rule.get("chapter")
        if chapter not in REPORT_CHAPTERS:
            logger.warning(
                "Scenario compression rule references unknown chapter '%s': id=%s",
                chapter,
                rule.get("id"),
            )


def _set_active_language(language: str) -> None:
    global TEMPLATES, DEFAULT_BLOCKS, _ACTIVE_LANGUAGE
    lang = "ko" if str(language or "").strip().lower().startswith("ko") else "en"
    _ensure_loaded(lang)
    TEMPLATES = _TEMPLATES_BY_LANG.get(lang, [])
    DEFAULT_BLOCKS = _DEFAULTS_BY_LANG.get(lang, {chapter: [] for chapter in REPORT_CHAPTERS})
    _ACTIVE_LANGUAGE = lang


def _ensure_loaded(language: str = "en") -> None:
    lang = "ko" if str(language or "").strip().lower().startswith("ko") else "en"
    if lang not in _TEMPLATES_BY_LANG or lang not in _DEFAULTS_BY_LANG:
        templates, defaults = _load_templates_for_language(lang)
        _TEMPLATES_BY_LANG[lang] = templates
        _DEFAULTS_BY_LANG[lang] = defaults
        _validate_rule_chapters()
    global TEMPLATES, DEFAULT_BLOCKS, _ACTIVE_LANGUAGE
    if not TEMPLATES and not DEFAULT_BLOCKS:
        TEMPLATES = _TEMPLATES_BY_LANG.get("en", [])
        DEFAULT_BLOCKS = _DEFAULTS_BY_LANG.get("en", {chapter: [] for chapter in REPORT_CHAPTERS})
        _ACTIVE_LANGUAGE = "en"


def get_template_libraries(language: str = "en") -> dict[str, Any]:
    _set_active_language(language)
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


def _evaluate_conditions(structural_summary: dict[str, Any], conditions: dict[str, Any]) -> bool:
    if not isinstance(conditions, dict):
        return False

    for field_path, expected in conditions.items():
        field_val = _get_structural_value(structural_summary, str(field_path))
        if field_val is None:
            return False

        if isinstance(expected, dict):
            for operator_text, operand in expected.items():
                operator_fn = OP_MAP.get(str(operator_text))
                if operator_fn is None:
                    return False
                try:
                    if not bool(operator_fn(field_val, operand)):
                        return False
                except Exception:
                    return False
        else:
            if field_val != expected:
                return False

    return True


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


def _lookup_template_by_id(block_id: str, chapter: str | None = None) -> dict[str, Any] | None:
    for block in TEMPLATES:
        if block.get("id") == block_id and (chapter is None or block.get("chapter") == chapter):
            return block

    for blocks in DEFAULT_BLOCKS.values():
        for block in blocks:
            if block.get("id") == block_id and (chapter is None or block.get("chapter") == chapter):
                return block

    return None


def _find_block_by_id(block_id: str, chapter: str | None = None) -> dict[str, Any] | None:
    return _lookup_template_by_id(block_id, chapter=chapter)


def _dedup_append(blocks: list[dict[str, Any]], block: dict[str, Any]) -> bool:
    block_id = block.get("id")
    if any(existing.get("id") == block_id for existing in blocks):
        return False
    blocks.append(block)
    return True


def _render_payload_fragment(block: dict[str, Any], chapter: str, intensity: float) -> dict[str, Any]:
    content = block.get("content", {})
    if not isinstance(content, dict):
        return {}

    # Preserve full narrative detail regardless of intensity band.
    include_fields = [
        "title",
        "summary",
        "key_forecast",
        "analysis",
        "implication",
        "examples",
        "shadow_pattern",
        "defense_mechanism",
        "emotional_trigger",
        "repetition_cycle",
        "integration_path",
        "choice_fork",
        "predictive_compression",
    ]

    payload_block: dict[str, Any] = {}
    for field in include_fields:
        if field == "title":
            payload_block[field] = str(content.get(field, chapter))
            continue
        if field not in content:
            continue

        value = content.get(field)
        if field == "choice_fork":
            if isinstance(value, dict):
                payload_block["choice_fork"] = value
        elif field == "predictive_compression":
            if isinstance(value, dict):
                payload_block["predictive_compression"] = value
                probability = value.get("probability_strength")
                dominant_theme = value.get("dominant_theme")
                window = value.get("window")
                if isinstance(probability, str) and probability.strip() and isinstance(dominant_theme, str) and dominant_theme.strip():
                    forecast_window = f" ({window})" if isinstance(window, str) and window.strip() else ""
                    payload_block.setdefault("key_forecast", f"{dominant_theme}{forecast_window} 쨌 probability {probability}")
        else:
            payload_block[field] = str(value)

    variant = None
    if intensity >= 0.85:
        variant = "high"
    elif intensity >= 0.65:
        variant = "moderate"

    scaling = block.get("scaling_variants", {})
    if variant and isinstance(scaling, dict) and variant in scaling:
        extensions = scaling[variant]
        if isinstance(extensions, dict):
            for field, ext_text in extensions.items():
                if not isinstance(ext_text, str) or not ext_text.strip():
                    continue
                base_field = field.replace("_extension", "")
                if base_field == "example":
                    base_field = "examples"
                if base_field in payload_block and isinstance(payload_block[base_field], str):
                    payload_block[base_field] += "\n\n" + ext_text

            if variant == "high":
                micro_scenario = extensions.get("micro_scenario")
                if isinstance(micro_scenario, str) and micro_scenario.strip():
                    payload_block["micro_scenario"] = micro_scenario

                long_term_projection = extensions.get("long_term_projection")
                if isinstance(long_term_projection, str) and long_term_projection.strip():
                    payload_block["long_term_projection"] = long_term_projection

    return payload_block


def _inject_choice_forks(
    structural_summary: dict[str, Any],
    chapter_blocks: dict[str, list[dict[str, Any]]],
    chapter_meta: dict[str, list[dict[str, Any]]],
    chapter_limits: dict[str, int],
) -> None:
    for rule in CHOICE_FORK_RULES:
        if not _evaluate_conditions(structural_summary, rule.get("conditions", {})):
            continue

        fork_id = rule.get("fork_id")
        if not isinstance(fork_id, str):
            continue

        fork_template = _find_block_by_id(fork_id)
        if not fork_template:
            continue

        chapter_name = rule.get("inject_into_chapter")
        if chapter_name not in chapter_blocks or chapter_name not in chapter_meta:
            continue

        fork_block = dict(fork_template)
        intensity = compute_block_intensity(fork_block, structural_summary)
        if intensity < 0.75:
            continue

        fork_block["_intensity"] = intensity
        meta = chapter_meta[chapter_name]
        existing = chapter_blocks[chapter_name]

        if any(b.get("id") == fork_block.get("id") for b in meta):
            continue

        rendered = _render_payload_fragment(fork_block, str(chapter_name), intensity)
        if not rendered:
            continue

        chapter_limit = max(0, int(chapter_limits.get(str(chapter_name), 5)))
        if chapter_limit == 0:
            continue

        if len(meta) < chapter_limit:
            meta.insert(0, fork_block)
            existing.insert(0, rendered)
            continue

        lowest_idx = min(range(len(meta)), key=lambda i: meta[i].get("priority", 0))
        meta[lowest_idx] = fork_block
        existing[lowest_idx] = rendered


def _inject_scenario_compression(
    structural_summary: dict[str, Any],
    chapter_blocks: dict[str, list[dict[str, Any]]],
    chapter_meta: dict[str, list[dict[str, Any]]],
    chapter_limits: dict[str, int],
) -> None:
    for rule in SCENARIO_COMPRESSION_RULES:
        if not _evaluate_conditions(structural_summary, rule.get("conditions", {})):
            continue

        block_id = rule.get("id")
        if not isinstance(block_id, str):
            continue

        block = _find_block_by_id(block_id)
        if not block:
            continue

        chapter = rule.get("chapter")
        if chapter not in chapter_blocks or chapter not in chapter_meta:
            continue

        if any(existing.get("id") == block.get("id") for existing in chapter_meta[chapter]):
            continue

        intensity = compute_block_intensity(block, structural_summary)
        if intensity < 0.6:
            continue

        scenario_block = dict(block)
        scenario_block["_intensity"] = intensity
        rendered = _render_payload_fragment(scenario_block, str(chapter), intensity)
        if not rendered:
            continue

        existing = chapter_blocks[chapter]
        meta = chapter_meta[chapter]
        chapter_limit = max(0, int(chapter_limits.get(str(chapter), 5)))

        if chapter_limit == 0:
            continue

        if len(existing) < chapter_limit:
            existing.insert(0, rendered)
            meta.insert(0, scenario_block)
        else:
            lowest_idx = min(range(len(meta)), key=lambda i: meta[i].get("priority", 0))
            existing[lowest_idx] = rendered
            meta[lowest_idx] = scenario_block


def _build_shadbala_insight_block(structural_summary: dict[str, Any], chapter: str) -> dict[str, Any] | None:
    shadbala_summary = structural_summary.get("shadbala_summary")
    if not isinstance(shadbala_summary, dict):
        return None

    by_planet = shadbala_summary.get("by_planet")
    if not isinstance(by_planet, dict) or not by_planet:
        return None

    top3 = shadbala_summary.get("top3_planets")
    if not isinstance(top3, list):
        top3 = []

    lines: list[str] = []
    for planet in top3[:3]:
        entry = by_planet.get(planet, {})
        if not isinstance(entry, dict):
            continue
        band = str(entry.get("band", "medium"))
        total = entry.get("total", 0.5)
        avastha = str(entry.get("avastha_state", "madhya"))
        tags = entry.get("evidence_tags", [])
        tags_text = ", ".join([str(t) for t in tags if isinstance(t, str) and t.strip()][:3]) or "No dominant tag"
        try:
            total_text = f"{float(total):.2f}"
        except (TypeError, ValueError):
            total_text = "0.50"
        lines.append(f"{planet}: {band.upper()} ({total_text}) | Avastha: {avastha} | Evidence: {tags_text}")

    if not lines:
        return None

    top_planets_text = ", ".join([str(p) for p in top3[:3] if isinstance(p, str) and p]) or "No clear top cluster"
    analysis = (
        "Shadbala approximate metrics and Avastha state viewed together:\n"
        + "\n".join(lines)
    )

    if chapter == "Stability Metrics":
        return {
            "id": "shadbala_avastha_snapshot_stability",
            "chapter": chapter,
            "priority": 95,
            "_intensity": 0.82,
            "content": {
                "title": "Shadbala & Avastha Snapshot",
                "summary": f"Top stabilizing planets: {top_planets_text}.",
                "analysis": analysis,
                "implication": (
                    "Use stronger planets as execution anchors for timing cycles, and treat weaker-planet domains "
                    "with stabilization and recovery before expansion."
                ),
                "examples": (
                    "Evidence tags include Directional Strength, Exalted, Own Sign, Combust, and Retrograde. "
                    "Interpretation emphasizes relative planetary hierarchy over single scalar scores."
                ),
            },
        }
    if chapter == "Final Summary":
        return {
            "id": "shadbala_avastha_snapshot_final",
            "chapter": chapter,
            "priority": 94,
            "_intensity": 0.8,
            "content": {
                "title": "Final Synthesis: Strength Axis",
                "summary": f"The primary strength axis converges on {top_planets_text}.",
                "analysis": (
                    "Top planets act as the center of execution and resilience, while weak-planet domains become volatile "
                    "under overextension. Recommendations should be prioritized by this strong-weak structure."
                ),
                "implication": "Align major decisions to strong axes first, and approach weaker axes with staged correction.",
            },
        }
    if chapter == "Remedies & Program":
        return {
            "id": "shadbala_avastha_snapshot_remedy",
            "chapter": chapter,
            "priority": 94,
            "_intensity": 0.8,
            "content": {
                "title": "Remedy Priority by Shadbala",
                "summary": "Set remedy priority as weak-planet stabilization first, then strong-planet overload prevention.",
                "analysis": (
                    "Start weak-planet correction with sleep, rhythm, and behavioral consistency. For strong planets, "
                    "distribute excessive responsibility concentration to preserve global chart balance."
                ),
                "examples": "Begin with a two-week check cycle and adjust intensity according to observed band shifts.",
            },
        }
    return None


def _inject_shadbala_insight(
    structural_summary: dict[str, Any],
    chapter_blocks: dict[str, list[dict[str, Any]]],
    chapter_meta: dict[str, list[dict[str, Any]]],
    chapter_limits: dict[str, int],
) -> None:
    for chapter in ["Stability Metrics", "Final Summary", "Remedies & Program"]:
        block = _build_shadbala_insight_block(structural_summary, chapter)
        if not block:
            continue
        if chapter not in chapter_blocks or chapter not in chapter_meta:
            continue
        if any(existing.get("id") == block.get("id") for existing in chapter_meta[chapter]):
            continue

        chapter_limit = max(0, int(chapter_limits.get(chapter, 5)))
        if chapter_limit == 0:
            continue

        intensity = block.get("_intensity", 0.0)
        rendered = _render_payload_fragment(block, chapter, float(intensity))
        if not rendered:
            continue

        existing = chapter_blocks[chapter]
        meta = chapter_meta[chapter]
        if len(existing) < chapter_limit:
            existing.insert(0, rendered)
            meta.insert(0, block)
            continue

        lowest_idx = min(range(len(meta)), key=lambda idx: float(meta[idx].get("_intensity", 0.0)))
        if float(intensity) > float(meta[lowest_idx].get("_intensity", 0.0)):
            existing[lowest_idx] = rendered
            meta[lowest_idx] = block


def _append_unique_block(
    selected: dict[str, list[dict[str, Any]]],
    chapter: str,
    block: dict[str, Any] | None,
    structural_summary: dict[str, Any],
) -> bool:
    if not block or chapter not in selected:
        return False

    block_id = block.get("id")
    if any(existing.get("id") == block_id for existing in selected[chapter]):
        return False

    expanded = dict(block)
    expanded["_intensity"] = compute_block_intensity(expanded, structural_summary)
    expanded["_match_index"] = len(selected[chapter])
    selected[chapter].append(expanded)
    return True


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

            _append_unique_block(selected, target_chapter, add_block, structural_summary)


def _apply_emotional_escalation(selected: dict[str, list[dict[str, Any]]], structural_summary: dict[str, Any]) -> None:
    tension = _get_structural_value(structural_summary, "psychological_tension_axis.score")
    stability = _get_structural_value(structural_summary, "stability_metrics.stability_index")

    if not isinstance(tension, (int, float)) or not isinstance(stability, (int, float)):
        return

    if tension < EMOTIONAL_ESCALATION_RULE["tension_threshold"]:
        return
    if stability > EMOTIONAL_ESCALATION_RULE["stability_threshold"]:
        return

    add_block = _lookup_template_by_id(
        str(EMOTIONAL_ESCALATION_RULE["inject_block_id"]),
        chapter=str(EMOTIONAL_ESCALATION_RULE["target_chapter"]),
    )
    _append_unique_block(
        selected,
        str(EMOTIONAL_ESCALATION_RULE["target_chapter"]),
        add_block,
        structural_summary,
    )


def _apply_recursive_correction(selected: dict[str, list[dict[str, Any]]], structural_summary: dict[str, Any]) -> None:
    karma_pattern = _get_structural_value(structural_summary, "karmic_pattern_profile.primary_pattern")
    primary_risk = _get_structural_value(structural_summary, "behavioral_risk_profile.primary_risk")

    if karma_pattern != "correction" or primary_risk != "impulsivity":
        return

    block_id = "recursive_correction_loop"
    target_chapter = "Life Timeline Interpretation"
    add_block = _lookup_template_by_id(block_id, chapter=target_chapter)
    _append_unique_block(selected, target_chapter, add_block, structural_summary)


def _apply_psychological_echo(selected: dict[str, list[dict[str, Any]]], structural_summary: dict[str, Any]) -> None:
    psych_ids = {block.get("id") for block in selected.get("Psychological Architecture", []) if block.get("id")}

    for psych_id in psych_ids:
        summary_match = _lookup_template_by_id(str(psych_id), chapter="Final Summary")
        _append_unique_block(selected, "Final Summary", summary_match, structural_summary)


def _is_korean_language(structural_summary: dict[str, Any]) -> bool:
    lang = str(structural_summary.get("language", "ko")).strip().lower()
    return not lang.startswith("en")


def _canonical_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(_canonical_value(v) for v in value[:6])
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value.keys())
        return "{ " + ", ".join(keys[:8]) + " }"
    return str(value)


def _flatten_deterministic_signals(structural_summary: dict[str, Any]) -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value.keys(), key=lambda x: str(x)):
                walk(f"{prefix}.{key}" if prefix else str(key), value.get(key))
            return
        if isinstance(value, list):
            if value and all(isinstance(v, (str, int, float, bool)) for v in value):
                flattened.append((prefix, list(value[:8])))
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened.append((prefix, value))

    allowed = [
        "personality_vector",
        "stability_metrics",
        "psychological_tension_axis",
        "life_purpose_vector",
        "planet_power_ranking",
        "dominant_house_cluster",
        "purushartha_profile",
        "behavioral_risk_profile",
        "interaction_risks",
        "enhanced_behavioral_risks",
        "probability_forecast",
        "karmic_pattern_profile",
        "varga_alignment",
    ]
    engine = structural_summary.get("engine")
    if isinstance(engine, dict):
        for engine_key in ["influence_matrix", "house_clusters", "personality_vector", "stability_metrics", "varga_alignment"]:
            if engine_key in engine:
                walk(f"engine.{engine_key}", engine.get(engine_key))

    for key in allowed:
        if key in structural_summary:
            walk(key, structural_summary.get(key))

    dedup: dict[str, Any] = {}
    for path, value in flattened:
        if path and path not in dedup:
            dedup[path] = value
    return [(path, dedup[path]) for path in sorted(dedup.keys())]


def _chapter_signal_priorities(chapter: str) -> list[str]:
    mapping = {
        "Executive Summary": ["life_purpose_vector", "planet_power_ranking", "stability_metrics", "psychological_tension_axis"],
        "Purushartha Profile": ["purushartha_profile", "life_purpose_vector", "dominant_house_cluster"],
        "Psychological Architecture": ["psychological_tension_axis", "personality_vector", "engine.influence_matrix", "interaction_risks"],
        "Behavioral Risks": ["behavioral_risk_profile", "enhanced_behavioral_risks", "interaction_risks", "stability_metrics"],
        "Karmic Patterns": ["karmic_pattern_profile", "life_purpose_vector", "engine.influence_matrix", "varga_alignment"],
        "Stability Metrics": ["stability_metrics", "engine.stability_metrics", "engine.influence_matrix", "behavioral_risk_profile"],
        "Personality Vector": ["personality_vector", "engine.personality_vector", "stability_metrics", "psychological_tension_axis"],
        "Life Timeline Interpretation": ["probability_forecast", "karmic_pattern_profile", "stability_metrics", "varga_alignment"],
        "Career & Success": ["probability_forecast", "varga_alignment.career_alignment", "life_purpose_vector", "engine.house_clusters"],
        "Love & Relationships": ["varga_alignment.relationship_alignment", "personality_vector.emotional_regulation", "behavioral_risk_profile", "karmic_pattern_profile"],
        "Health & Body Patterns": ["stability_metrics", "behavioral_risk_profile", "personality_vector.discipline_index", "engine.house_clusters"],
        "Confidence & Forecast": ["probability_forecast", "stability_metrics", "psychological_tension_axis", "personality_vector"],
        "Remedies & Program": ["behavioral_risk_profile", "stability_metrics", "personality_vector", "varga_alignment"],
        "Final Summary": ["life_purpose_vector", "stability_metrics", "personality_vector", "varga_alignment", "probability_forecast"],
        "Appendix (Optional)": ["planet_power_ranking", "varga_alignment", "stability_metrics", "personality_vector", "engine.influence_matrix"],
    }
    return mapping.get(chapter, ["stability_metrics", "personality_vector", "psychological_tension_axis"])


def _collect_chapter_signals(chapter: str, structural_summary: dict[str, Any], limit: int = 20) -> list[tuple[str, Any]]:
    flattened = _flatten_deterministic_signals(structural_summary)
    priorities = _chapter_signal_priorities(chapter)
    weighted: list[tuple[int, str, Any]] = []
    for path, value in flattened:
        rank = len(priorities) + 2
        for idx, prefix in enumerate(priorities):
            if path == prefix or path.startswith(prefix + "."):
                rank = idx
                break
        weighted.append((rank, path, value))
    weighted.sort(key=lambda item: (item[0], item[1]))
    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for _, path, value in weighted:
        if path in seen:
            continue
        out.append((path, value))
        seen.add(path)
        if len(out) >= max(1, limit):
            break
    return out


def _dynamic_chapter_limit(structural_summary: dict[str, Any], spike_count: int = 0) -> int:
    richness = 0
    for _, value in _flatten_deterministic_signals(structural_summary):
        if isinstance(value, (str, int, float, bool)):
            richness += 1
        elif isinstance(value, list):
            richness += max(1, len(value))
    dynamic = BASE_CHAPTER_LIMIT + min(7, richness // 30)
    bounded = min(MAX_DYNAMIC_CHAPTER_LIMIT, max(BASE_CHAPTER_LIMIT, dynamic))
    return max(1, bounded - max(0, spike_count))


def _fragment_text_length(fragment: dict[str, Any]) -> tuple[int, int]:
    parts: list[str] = []
    for field in ("title", "summary", "analysis", "implication", "examples"):
        value = fragment.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    combined = "\n".join(parts)
    return len(combined), len(combined.split())


def _chapter_depth_stats(chapter_blocks: list[dict[str, Any]]) -> tuple[int, int]:
    total_chars = 0
    total_words = 0
    for fragment in chapter_blocks:
        if not isinstance(fragment, dict):
            continue
        chars, words = _fragment_text_length(fragment)
        total_chars += chars
        total_words += words
    return total_chars, total_words


def _planet_meaning_text(planet: str, ko_mode: bool) -> str:
    ko = {
        "Sun": "Sun 우세는 자아 의식과 리더십을 강화하며 스스로 방향을 정해 움직이려는 성향을 높입니다.",
        "Moon": "Moon 우세는 정서 감수성과 공감 반응을 강화해 관계와 분위기에 민감하게 반응하게 만듭니다.",
        "Mars": "Mars 우세는 추진력과 독립성을 강화해 목표 달성을 위해 빠르게 행동하는 경향을 만듭니다.",
        "Mercury": "Mercury 우세는 분석력과 언어적 판단력을 강화해 전략적 의사결정 정확도를 높입니다.",
        "Jupiter": "Jupiter 우세는 확장성과 성장 지향성을 강화해 장기 기회 포착 능력을 높입니다.",
        "Venus": "Venus 우세는 조화 감각과 관계 품질을 강화해 균형 중심의 선택을 선호하게 만듭니다.",
        "Saturn": "Saturn 우세는 규율과 책임감을 강화하며 성과는 늦더라도 축적형 결과로 수렴하게 합니다.",
        "Rahu": "Rahu 우세는 급진적 변화 추구를 강화해 빠른 성장 가능성과 변동 리스크를 함께 키웁니다.",
        "Ketu": "Ketu 우세는 분리와 내적 재구성을 강화해 단순화와 본질 추구를 촉진합니다.",
    }
    en = {
        "Sun": "Sun dominance strengthens identity and leadership, increasing self-directed agency.",
        "Moon": "Moon dominance heightens emotional sensitivity and relational responsiveness.",
        "Mars": "Mars dominance amplifies initiative and independence, driving fast goal-oriented action.",
        "Mercury": "Mercury dominance strengthens analytical and verbal judgment in complex decisions.",
        "Jupiter": "Jupiter dominance supports expansion and long-range developmental growth.",
        "Venus": "Venus dominance reinforces harmony and relational quality in value-based choices.",
        "Saturn": "Saturn dominance emphasizes discipline and responsibility with delayed but durable outcomes.",
        "Rahu": "Rahu dominance increases disruptive ambition with both opportunity and volatility.",
        "Ketu": "Ketu dominance encourages detachment and inner restructuring priorities.",
    }
    return (ko if ko_mode else en).get(
        planet,
        f"{planet} {'우세는 해당 행성의 주제를 삶의 중심 동력으로 끌어올립니다.' if ko_mode else 'dominance elevates that planet theme into a core life driver.'}",
    )


def _signal_numeric(signal_value: Any) -> float | None:
    if isinstance(signal_value, (int, float)):
        value = float(signal_value)
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return value
    return None


def _interpretation_lookup(section: str, key: str) -> str | None:
    if section == "atomic":
        hit = key in INTERPRETATIONS_KR_ATOMIC
        if hit:
            ATOMIC_RUNTIME_AUDIT["atomic_lookup_hits"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_hits", 0)) + 1
        else:
            ATOMIC_RUNTIME_AUDIT["atomic_lookup_misses"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_misses", 0)) + 1
        logger.info("atomic_lookup key=%s hit=%s", key, hit)
        return _extract_interpretation_text(INTERPRETATIONS_KR_ATOMIC.get(key))

    section_payload = INTERPRETATIONS_KR.get(section) if isinstance(INTERPRETATIONS_KR, dict) else None
    if not isinstance(section_payload, dict):
        return None
    return _extract_interpretation_text(section_payload.get(key))


def _get_atomic_chart_interpretations(structural_summary: dict[str, Any]) -> dict[str, str]:
    chart_signature = structural_summary.get("chart_signature") if isinstance(structural_summary.get("chart_signature"), dict) else {}
    ascendant_sign = chart_signature.get("ascendant_sign") or structural_summary.get("ascendant_sign") or structural_summary.get("asc_sign")
    sun_sign = chart_signature.get("sun_sign") or structural_summary.get("sun_sign")
    moon_sign = chart_signature.get("moon_sign") or structural_summary.get("moon_sign")

    asc_key = f"asc:{str(ascendant_sign).strip()}" if isinstance(ascendant_sign, str) and ascendant_sign.strip() else ""
    sun_key = f"ps:Sun:{str(sun_sign).strip()}" if isinstance(sun_sign, str) and sun_sign.strip() else ""
    moon_key = f"ps:Moon:{str(moon_sign).strip()}" if isinstance(moon_sign, str) and moon_sign.strip() else ""

    logger.info("atomic_keys_generated asc=%s sun=%s moon=%s", asc_key, sun_key, moon_key)
    if any((asc_key, sun_key, moon_key)):
        ATOMIC_RUNTIME_AUDIT["atomic_keys_generated"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_keys_generated", 0)) + 1

    out = {"asc": "", "sun": "", "moon": ""}
    for label, key in (("asc", asc_key), ("sun", sun_key), ("moon", moon_key)):
        if not key:
            continue
        hit = key in INTERPRETATIONS_KR_ATOMIC
        if hit:
            ATOMIC_RUNTIME_AUDIT["atomic_lookup_hits"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_hits", 0)) + 1
        else:
            ATOMIC_RUNTIME_AUDIT["atomic_lookup_misses"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_misses", 0)) + 1
        logger.info("atomic_lookup key=%s hit=%s", key, hit)
        text = _extract_interpretation_text(INTERPRETATIONS_KR_ATOMIC.get(key))
        if isinstance(text, str) and text.strip():
            out[label] = text.strip()
    return out


def _atomic_key_for_signal_path(signal_path: str, structural_summary: dict[str, Any]) -> str:
    chart_signature = structural_summary.get("chart_signature") if isinstance(structural_summary.get("chart_signature"), dict) else {}
    ascendant_sign = chart_signature.get("ascendant_sign") or structural_summary.get("ascendant_sign") or structural_summary.get("asc_sign")
    sun_sign = chart_signature.get("sun_sign") or structural_summary.get("sun_sign")
    moon_sign = chart_signature.get("moon_sign") or structural_summary.get("moon_sign")
    p = str(signal_path or "").lower()
    if ("moon" in p) or ("emotional" in p) or ("relationship" in p) or ("tension" in p):
        if isinstance(moon_sign, str) and moon_sign.strip():
            return f"ps:Moon:{moon_sign.strip()}"
    if ("sun" in p) or ("career" in p) or ("authority" in p) or ("leadership" in p):
        if isinstance(sun_sign, str) and sun_sign.strip():
            return f"ps:Sun:{sun_sign.strip()}"
    if isinstance(ascendant_sign, str) and ascendant_sign.strip():
        return f"asc:{ascendant_sign.strip()}"
    if isinstance(sun_sign, str) and sun_sign.strip():
        return f"ps:Sun:{sun_sign.strip()}"
    if isinstance(moon_sign, str) and moon_sign.strip():
        return f"ps:Moon:{moon_sign.strip()}"
    return ""


def _integrate_atomic_with_signals(atomic_text: str, structural_summary: dict[str, Any]) -> str:
    base = str(atomic_text or "").strip()
    if not base:
        return ""
    stability = structural_summary.get("stability_metrics") if isinstance(structural_summary.get("stability_metrics"), dict) else {}
    personality = structural_summary.get("personality_vector") if isinstance(structural_summary.get("personality_vector"), dict) else {}
    purpose = structural_summary.get("life_purpose_vector") if isinstance(structural_summary.get("life_purpose_vector"), dict) else {}
    tension = structural_summary.get("psychological_tension_axis") if isinstance(structural_summary.get("psychological_tension_axis"), dict) else {}

    extensions: list[str] = []
    dominant_planet = purpose.get("dominant_planet")
    if isinstance(dominant_planet, str) and dominant_planet.strip():
        extensions.append(f"현재 구조에서는 {dominant_planet.strip()} 성향이 실행 방식과 선택 우선순위에 강하게 영향을 줍니다.")
    stability_grade = stability.get("stability_grade") or stability.get("grade")
    if isinstance(stability_grade, str) and stability_grade.strip():
        extensions.append(f"안정성 등급({stability_grade.strip()})은 변화 구간에서 반응 폭과 회복 속도를 함께 규정합니다.")
    tension_axis = tension.get("axis")
    if isinstance(tension_axis, str) and tension_axis.strip():
        extensions.append(f"심리 긴장 축({tension_axis.strip()})과 결합되면 관계 및 의사결정 국면에서 반복 패턴이 강화될 수 있습니다.")
    discipline = personality.get("discipline_index")
    if isinstance(discipline, (int, float)):
        extensions.append("규율 지수와 결합된 구조에서는 단기 반응보다 누적 성과를 만들도록 행동 리듬을 고정하는 것이 유리합니다.")

    if not extensions:
        return base
    return base + "\n\n" + " ".join(extensions[:2])


def _mapping_keys_for_signal(signal_path: str, signal_value: Any, structural_summary: dict[str, Any]) -> list[tuple[str, str]]:
    p = signal_path.lower()
    candidates: list[tuple[str, str]] = []
    chart_signature = structural_summary.get("chart_signature") if isinstance(structural_summary.get("chart_signature"), dict) else {}
    asc = chart_signature.get("ascendant_sign")
    sun = chart_signature.get("sun_sign")
    moon = chart_signature.get("moon_sign")

    # Global chart identity context: always include atomic sign keys before signal-specific rules.
    global_atomic_candidates: list[tuple[str, str]] = []
    asc_key = ""
    sun_key = ""
    moon_key = ""
    if isinstance(asc, str) and asc.strip():
        asc_key = f"asc:{asc.strip()}"
        global_atomic_candidates.append(("atomic", asc_key))
    if isinstance(sun, str) and sun.strip():
        sun_key = f"ps:Sun:{sun.strip()}"
        global_atomic_candidates.append(("atomic", sun_key))
    if isinstance(moon, str) and moon.strip():
        moon_key = f"ps:Moon:{moon.strip()}"
        global_atomic_candidates.append(("atomic", moon_key))

    if any((asc_key, sun_key, moon_key)):
        ATOMIC_RUNTIME_AUDIT["atomic_keys_generated"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_keys_generated", 0)) + 1
        logger.info("atomic_keys_generated asc=%s sun=%s moon=%s", asc_key, sun_key, moon_key)

    # Deterministic prioritization so Sun/Moon keys are actively used where semantically aligned.
    if ("moon" in p) or ("emotional" in p) or ("relationship" in p) or ("tension" in p):
        priority_order = ("ps:Moon:", "ps:Sun:", "asc:")
    elif ("sun" in p) or ("career" in p) or ("authority" in p) or ("leadership" in p):
        priority_order = ("ps:Sun:", "ps:Moon:", "asc:")
    else:
        priority_order = ("asc:", "ps:Sun:", "ps:Moon:")

    for prefix in priority_order:
        for section, key in global_atomic_candidates:
            if key.startswith(prefix):
                candidates.append((section, key))

    personality = structural_summary.get("personality_vector") if isinstance(structural_summary.get("personality_vector"), dict) else {}
    stability = structural_summary.get("stability_metrics") if isinstance(structural_summary.get("stability_metrics"), dict) else {}
    forecast = structural_summary.get("probability_forecast") if isinstance(structural_summary.get("probability_forecast"), dict) else {}
    varga = structural_summary.get("varga_alignment") if isinstance(structural_summary.get("varga_alignment"), dict) else {}
    house_strengths = structural_summary.get("house_strengths") if isinstance(structural_summary.get("house_strengths"), dict) else {}
    if not house_strengths:
        engine = structural_summary.get("engine") if isinstance(structural_summary.get("engine"), dict) else {}
        clusters = engine.get("house_clusters") if isinstance(engine.get("house_clusters"), dict) else {}
        cluster_scores = clusters.get("cluster_scores")
        if isinstance(cluster_scores, dict):
            house_strengths = cluster_scores

    if "dominant_planet" in p or "planet_power_ranking" in p:
        dominant = str(signal_value[0] if isinstance(signal_value, list) and signal_value else signal_value or "").strip()
        planet_strategy = {
            "Mars": "ll:strategy:direct_action",
            "Sun": "ll:strategy:direct_action",
            "Rahu": "ll:strategy:direct_action",
            "Saturn": "ll:strategy:long_game",
            "Jupiter": "ll:strategy:long_game",
            "Mercury": "ll:strategy:skill_compounding",
            "Venus": "ll:strategy:relationship_leverage",
            "Moon": "ll:strategy:relationship_leverage",
            "Ketu": "ll:strategy:structure_routine",
        }
        if dominant in planet_strategy:
            candidates.append(("lagna_lord", planet_strategy[dominant]))
        candidates.append(("patterns", "pat:strong_lagna_lord"))

    if "personality_vector" in p:
        discipline = _signal_numeric(personality.get("discipline_index"))
        risk_appetite = _signal_numeric(personality.get("risk_appetite"))
        emotional_regulation = _signal_numeric(personality.get("emotional_regulation"))
        if isinstance(discipline, float):
            candidates.append(("lagna_lord", "ll:strategy:structure_routine" if discipline >= 60 else "ll:strategy:skill_compounding"))
        if isinstance(risk_appetite, float):
            candidates.append(("lagna_lord", "ll:strategy:direct_action" if risk_appetite >= 65 else "ll:strategy:avoid_overextension"))
        if isinstance(emotional_regulation, float):
            candidates.append(("patterns", "pat:strong_moon" if emotional_regulation >= 55 else "pat:afflicted_moon"))

    if "stability_index" in p or "stability_grade" in p:
        stability_index = _signal_numeric(stability.get("stability_index", signal_value))
        grade = str(stability.get("stability_grade") or stability.get("grade") or "").upper()
        if (isinstance(stability_index, float) and stability_index <= 45) or grade in {"D", "E", "F"}:
            candidates.append(("patterns", "pat:malefic_overload"))
            candidates.append(("lagna_lord", "ll:strategy:avoid_overextension"))
        else:
            candidates.append(("patterns", "pat:benefic_support"))
            candidates.append(("lagna_lord", "ll:strategy:structure_routine"))

    if "psychological_tension_axis" in p or "tension_index" in p:
        tension_value = _signal_numeric(signal_value)
        if tension_value is None and isinstance(signal_value, dict):
            tension_value = _signal_numeric(signal_value.get("score"))
        candidates.append(("patterns", "pat:afflicted_moon" if isinstance(tension_value, float) and tension_value >= 55 else "pat:strong_moon"))

    if "house_strengths" in p or "house_clusters" in p or "dominant_house_cluster" in p:
        dominant_house = None
        if isinstance(signal_value, dict) and signal_value:
            try:
                dominant_house = int(max(signal_value.items(), key=lambda item: float(item[1]))[0])
            except Exception:
                dominant_house = None
        if dominant_house is None and isinstance(house_strengths, dict) and house_strengths:
            try:
                dominant_house = int(max(house_strengths.items(), key=lambda item: float(item[1]))[0])
            except Exception:
                dominant_house = None
        if dominant_house in {1, 4, 7, 10}:
            candidates.append(("patterns", "pat:kendra_emphasis"))
        if dominant_house in {3, 6, 10, 11}:
            candidates.append(("patterns", "pat:upachaya_emphasis"))

    if "varga_alignment" in p:
        career = varga.get("career_alignment") if isinstance(varga.get("career_alignment"), dict) else {}
        overall = varga.get("overall_alignment") if isinstance(varga.get("overall_alignment"), dict) else {}
        career_score = _signal_numeric(career.get("score"))
        overall_score = _signal_numeric(overall.get("score"))
        if isinstance(career_score, float):
            candidates.append(("patterns", "pat:strong_10th_lord" if career_score >= 55 else "pat:weak_10th_lord"))
        if isinstance(overall_score, float):
            candidates.append(("patterns", "pat:strong_lagna_lord" if overall_score >= 55 else "pat:weak_lagna_lord"))

    if "probability_forecast" in p:
        burnout = _signal_numeric(forecast.get("burnout_2yr"))
        career_shift = _signal_numeric(forecast.get("career_shift_3yr"))
        marriage = _signal_numeric(forecast.get("marriage_5yr"))
        if isinstance(burnout, float):
            candidates.append(("lagna_lord", "ll:strategy:avoid_overextension" if burnout >= 65 else "ll:strategy:structure_routine"))
        if isinstance(career_shift, float):
            candidates.append(("lagna_lord", "ll:strategy:skill_compounding" if career_shift >= 55 else "ll:strategy:long_game"))
        if isinstance(marriage, float) and marriage >= 55:
            candidates.append(("lagna_lord", "ll:strategy:relationship_leverage"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for section, key in candidates:
        marker = f"{section}:{key}"
        if marker in seen:
            continue
        deduped.append((section, key))
        seen.add(marker)
    return deduped


def _interpretation_mapping_text(
    signal_path: str,
    signal_value: Any,
    structural_summary: dict[str, Any],
    mapping_audit: dict[str, Any] | None = None,
) -> str | None:
    candidates = _mapping_keys_for_signal(signal_path, signal_value, structural_summary)
    candidate_markers = [f"{section}:{key}" for section, key in candidates]
    chart_signature = structural_summary.get("chart_signature") if isinstance(structural_summary.get("chart_signature"), dict) else {}
    atomic_global_candidates = {
        "asc": chart_signature.get("ascendant_sign") or structural_summary.get("ascendant_sign") or structural_summary.get("asc_sign"),
        "sun": chart_signature.get("sun_sign") or structural_summary.get("sun_sign"),
        "moon": chart_signature.get("moon_sign") or structural_summary.get("moon_sign"),
    }
    key_used: str | None = None
    mapped_text: str | None = None

    for section, key in candidates:
        text = _interpretation_lookup(section, key)
        if isinstance(text, str) and text.strip():
            key_used = f"{section}:{key}"
            mapped_text = text.strip()
            break

    if isinstance(mapping_audit, dict):
        mapping_audit["total_signals_processed"] = int(mapping_audit.get("total_signals_processed", 0)) + 1
        if isinstance(mapped_text, str) and mapped_text:
            mapping_audit["mapping_hits"] = int(mapping_audit.get("mapping_hits", 0)) + 1
            used = mapping_audit.setdefault("used_mapping_keys", {})
            if isinstance(used, dict) and key_used:
                used[key_used] = int(used.get(key_used, 0)) + 1
        else:
            mapping_audit["mapping_misses"] = int(mapping_audit.get("mapping_misses", 0)) + 1

        atomic_usage = mapping_audit.setdefault("atomic_usage", {})
        if isinstance(atomic_usage, dict):
            atomic_candidates = {
                "ascendant": any(marker.startswith("atomic:asc:") for marker in candidate_markers),
                "sun": any(marker.startswith("atomic:ps:Sun:") for marker in candidate_markers),
                "moon": any(marker.startswith("atomic:ps:Moon:") for marker in candidate_markers),
            }
            for atomic_name, present in atomic_candidates.items():
                if not present:
                    continue
                bucket = atomic_usage.setdefault(atomic_name, {"hits": 0, "misses": 0})
                if not isinstance(bucket, dict):
                    continue
                if key_used and (
                    (atomic_name == "ascendant" and key_used.startswith("atomic:asc:"))
                    or (atomic_name == "sun" and key_used.startswith("atomic:ps:Sun:"))
                    or (atomic_name == "moon" and key_used.startswith("atomic:ps:Moon:"))
                ):
                    bucket["hits"] = int(bucket.get("hits", 0)) + 1
                else:
                    bucket["misses"] = int(bucket.get("misses", 0)) + 1

    if REPORT_MAPPING_DEBUG:
        logger.info(
            "mapping_debug signal_path=%s atomic_global_candidates=%s candidate_mapping_keys=%s mapping_hit=%s mapping_key_used=%s",
            signal_path,
            atomic_global_candidates,
            candidate_markers,
            "YES" if isinstance(mapped_text, str) and mapped_text else "NO",
            key_used or "",
        )

    return mapped_text


def _signal_focus_label_ko(signal_path: str) -> str:
    p = signal_path.lower()
    if "dominant_planet" in p or "planet_power_ranking" in p:
        return "지배 행성 축"
    if "personality_vector" in p:
        return "성향 벡터 축"
    if "stability" in p:
        return "안정성 축"
    if "tension" in p or "psychological_tension_axis" in p:
        return "긴장 축"
    if "house" in p:
        return "하우스 강도 축"
    if "varga" in p:
        return "분할차트 정렬 축"
    if "probability_forecast" in p:
        return "확률 전망 축"
    return "종합 구조 축"


def _risk_band(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if score >= 75:
        return "high"
    if score >= 55:
        return "elevated"
    if score >= 35:
        return "moderate"
    return "low"


def _interpret_signal_sentence(
    *,
    signal_path: str,
    signal_value: Any,
    chapter: str,
    ko_mode: bool,
    structural_summary: dict[str, Any] | None = None,
    mapping_audit: dict[str, Any] | None = None,
) -> tuple[str, str, str, str]:
    del chapter
    p = signal_path.lower()
    structural = structural_summary if isinstance(structural_summary, dict) else {}

    if ko_mode:
        selected_atomic_key: str | None = None
        atomic_text: dict[str, Any] | None = None
        for section, key in _mapping_keys_for_signal(signal_path, signal_value, structural):
            if section != "atomic":
                continue
            hit = key in INTERPRETATIONS_KR_ATOMIC
            if hit:
                ATOMIC_RUNTIME_AUDIT["atomic_lookup_hits"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_hits", 0)) + 1
            else:
                ATOMIC_RUNTIME_AUDIT["atomic_lookup_misses"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_lookup_misses", 0)) + 1
            logger.info("atomic_lookup key=%s hit=%s", key, hit)

            candidate = INTERPRETATIONS_KR_ATOMIC.get(key)
            if isinstance(candidate, dict) and "text" in candidate and isinstance(candidate.get("text"), str) and candidate.get("text", "").strip():
                selected_atomic_key = key
                atomic_text = candidate
                break

        if atomic_text and isinstance(atomic_text.get("text"), str):
            summary = str(atomic_text["text"]).strip()
            ATOMIC_RUNTIME_AUDIT["atomic_text_applied_count"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_text_applied_count", 0)) + 1
            logger.info(
                "atomic_text_applied key=%s length=%d",
                selected_atomic_key or "",
                len(summary),
            )
            if isinstance(mapping_audit, dict):
                mapping_audit["total_signals_processed"] = int(mapping_audit.get("total_signals_processed", 0)) + 1
                mapping_audit["mapping_hits"] = int(mapping_audit.get("mapping_hits", 0)) + 1
                used = mapping_audit.setdefault("used_mapping_keys", {})
                if isinstance(used, dict) and selected_atomic_key:
                    used[f"atomic:{selected_atomic_key}"] = int(used.get(f"atomic:{selected_atomic_key}", 0)) + 1
                atomic_usage = mapping_audit.setdefault("atomic_usage", {})
                if isinstance(atomic_usage, dict):
                    for atomic_name, prefix in (("ascendant", "asc:"), ("sun", "ps:Sun:"), ("moon", "ps:Moon:")):
                        bucket = atomic_usage.setdefault(atomic_name, {"hits": 0, "misses": 0})
                        if not isinstance(bucket, dict):
                            continue
                        if selected_atomic_key and selected_atomic_key.startswith(prefix):
                            bucket["hits"] = int(bucket.get("hits", 0)) + 1
                        else:
                            bucket["misses"] = int(bucket.get("misses", 0)) + 1

            return (summary, "", "", "")

        mapped_text = _interpretation_mapping_text(signal_path, signal_value, structural, mapping_audit=mapping_audit)
        if isinstance(mapped_text, str) and mapped_text.strip():
            return (mapped_text, "", "", "")

    if "dominant_planet" in p:
        planet = str(signal_value or "Moon")
        summary = _planet_meaning_text(planet, ko_mode)
        if ko_mode:
            return (
                summary,
                "지배 행성의 기질은 성향 벡터와 결합해 행동 표현의 강도를 조정하고 반복되는 선택 패턴을 만듭니다.",
                "강점은 실행 가속에 쓰되 과도한 발현이 충돌로 번지지 않도록 속도와 경계를 함께 관리해야 합니다.",
                "주요 결정은 지배 행성의 강점 영역에서 시작할수록 결과 일관성이 높아집니다.",
            )
        return (
            summary,
            "Dominant-planet temperament modulates personality expression and behavioral risk intensity across repeating patterns.",
            "Use this strength for momentum while controlling overexpression so drive does not turn into conflict.",
            "Execution consistency usually improves when major decisions start from dominant-planet strengths.",
        )

    if "tension_index" in p or "psychological_tension_axis.score" in p:
        raw = float(signal_value) * 100 if isinstance(signal_value, (int, float)) and float(signal_value) <= 1 else signal_value
        band = _risk_band(raw)
        if ko_mode:
            return (
                "긴장 지수가 높아 내적 갈등과 반응 과열 가능성이 커지며 판단 변동성이 증가합니다."
                if band in {"high", "elevated"}
                else "긴장 지수가 중간 이하라 정서 반응의 진폭이 비교적 안정적입니다.",
                "이 지표는 정서 처리 속도와 통제력의 균형을 보여주며 압박 구간에서 충돌 패턴의 반복 가능성을 시사합니다.",
                "고긴장 구간에서는 결정 속도를 낮추고 회복 루틴을 먼저 확보해 실행 강도를 높이는 것이 유리합니다.",
                "관계와 협상처럼 감정 비용이 큰 장면에서는 사전 정렬 시간이 성패를 좌우합니다.",
            )
        return (
            "Elevated tension indicates stronger internal conflict and higher decision volatility."
            if band in {"high", "elevated"}
            else "Moderate-to-low tension supports steadier emotional amplitude and judgment consistency.",
            "This metric reflects balance between emotional load and self-regulation under pressure.",
            "During high-tension windows, reduce decision velocity and restore regulation before escalation.",
            "Pre-alignment before negotiation or relational dialogue lowers avoidable emotional cost.",
        )

    if "stability_index" in p:
        band = _risk_band(signal_value)
        if ko_mode:
            return (
                "안정성 지수가 낮아 계획의 지속성과 성과의 일관성이 흔들릴 가능성이 큽니다."
                if band in {"high", "elevated"}
                else "안정성 지수가 확보되어 실행 리듬과 누적 성과의 가능성이 높아집니다.",
                "안정성은 행동 지속성과 회복 탄력의 결합 지표로, 변동기 판단 오차를 직접 좌우합니다.",
                "저안정 구간에서는 확장보다 리듬 복원과 누수 차단을 우선해야 변동 비용을 줄일 수 있습니다.",
                "수면, 일정, 실행 블록을 고정하는 루틴이 안정성과 회복력을 동시에 끌어올립니다.",
            )
        return (
            "Lower stability raises volatility and weakens continuity of execution."
            if band in {"high", "elevated"}
            else "Higher stability supports sustainable rhythm and cumulative outcomes.",
            "Stability combines persistence and recovery capacity under shifting pressure.",
            "When stability is weak, restore rhythm and reduce leakage before expansion.",
            "Consistent sleep, schedule, and execution blocks are primary stabilizers.",
        )

    if "saturn" in p:
        if ko_mode:
            return (
                "Saturn 영향이 강하면 규율과 책임감이 강화되지만 성과가 지연되어 체감되기 쉽습니다.",
                "이 구조에서는 단기 보상보다 장기 축적이 유리하고, 기준 미달 상태의 조기 확장은 역효과를 냅니다.",
                "검증된 절차와 품질 기준을 먼저 확보하면 지연 구간도 안정적으로 통과할 수 있습니다.",
                "속도보다 재현 가능한 루틴을 우선할수록 최종 성과가 안정적으로 커집니다.",
            )
        return (
            "Strong Saturn influence builds discipline and responsibility with delayed realization of results.",
            "This favors long-run accumulation and penalizes premature expansion below quality thresholds.",
            "Establish validated procedures and standards before scaling commitments.",
            "Prioritizing reproducible process over speed improves eventual outcome durability.",
        )

    if "varga_alignment" in p and isinstance(signal_value, (int, float)):
        score = float(signal_value)
        if ko_mode:
            return (
                "분할차트 정렬도가 높아 의사결정과 실제 결과 사이의 일치 가능성이 큽니다."
                if score >= 70
                else "분할차트 정렬도가 중간 이하이므로 전략과 환경 적합성의 미세 조정이 필요합니다.",
                "정렬 지표는 잠재력 자체보다 현재 맥락에서 얼마나 안정적으로 발현되는지를 보여줍니다.",
                "정렬이 낮은 영역은 속도보다 맥락 조정과 역할 재배치를 우선해야 손실을 줄일 수 있습니다.",
                "동일 노력이라도 정렬도가 높은 분야에서 더 낮은 비용으로 성과가 발생합니다.",
            )
        return (
            "High varga alignment increases convergence between decisions and lived outcomes."
            if score >= 70
            else "Mid/low varga alignment requires finer strategy-context calibration.",
            "Alignment indicates manifestation reliability under real-world constraints.",
            "For lower alignment domains, optimize context and role fit before acceleration.",
            "Equivalent effort tends to perform better in higher-alignment domains.",
        )

    if "probability_forecast" in p and isinstance(signal_value, (int, float)):
        prob = float(signal_value)
        if ko_mode:
            return (
                "해당 사건 확률이 높아 구조적 전환 경로가 현실화될 가능성이 큽니다."
                if prob >= 0.65
                else "확률이 중간 이하라면 급격한 전환보다 점진적 조정 전략이 더 합리적입니다.",
                "확률 지표는 예언이 아니라 현재 구조에서 어떤 경로가 더 쉽게 현실화되는지 보여주는 신호입니다.",
                "고확률 구간에서는 대비 시나리오를 먼저 배치해 전환 비용을 낮추는 것이 중요합니다.",
                "보수·기준·확장 시나리오를 분리하면 변동 구간 대응력이 높아집니다.",
            )
        return (
            "Elevated probability indicates stronger activation of a structural transition path."
            if prob >= 0.65
            else "With moderate/lower probability, gradual adjustment often outperforms abrupt shifts.",
            "This is a pathway-likelihood signal, not a deterministic prophecy.",
            "Pre-position contingency scenarios in high-probability windows to reduce transition cost.",
            "Separating conservative/base/expansion tracks improves resilience under uncertainty.",
        )

    if ko_mode:
        return (
            "현재 지표 조합은 축 간 일관성을 유지하면서 선택 우선순위를 재조정해야 함을 시사합니다.",
            "성향 벡터와 안정성·리스크를 함께 읽으면 과잉 반응 또는 회피 반응이 발생하는 구간을 더 정확히 포착할 수 있습니다.",
            "강점 축은 실행에 사용하고 약점 축은 보호 장치를 먼저 배치하는 방식이 손실을 줄입니다.",
            "결정 전 리듬을 표준화하면 동일 패턴의 반복 손실을 완화할 수 있습니다.",
        )
    return (
        "The current metric mix suggests reprioritizing choices while preserving coherence across core life axes.",
        "Cross-reading personality, stability, and risk signals helps anticipate overreaction or avoidance windows.",
        "Deploy strengths for execution while protecting weaker axes first to reduce avoidable downside.",
        "A standardized pre-decision routine lowers repeated losses from familiar pattern loops.",
    )


def _high_signal_forecast_line(signal_path: str, signal_value: Any, *, ko_mode: bool) -> str:
    path_lower = str(signal_path or "").lower()
    if "probability_forecast" not in path_lower:
        return ""
    if not isinstance(signal_value, (int, float)):
        return ""
    probability = float(signal_value)
    if probability < 0.65:
        return ""
    label = str(signal_path).split(".")[-1].replace("_", " ").strip()
    pct = int(round(probability * 100)) if probability <= 1 else int(round(probability))
    if ko_mode:
        return f"{label}: 고신호 확률 {pct}%"
    return f"{label}: high-signal likelihood {pct}%"

def _create_signal_fragment(
    *,
    chapter: str,
    index: int,
    signal_path: str,
    signal_value: Any,
    rule_id: str,
    language: str = "en",
    structural_summary: dict[str, Any] | None = None,
    mapping_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ko_mode = str(language or "").strip().lower().startswith("ko")
    structural = structural_summary if isinstance(structural_summary, dict) else {}
    atomic_key = _atomic_key_for_signal_path(signal_path, structural) if ko_mode else ""
    atomic_candidate = INTERPRETATIONS_KR_ATOMIC.get(atomic_key) if atomic_key else None
    atomic_text = (
        str(atomic_candidate.get("text")).strip()
        if isinstance(atomic_candidate, dict) and isinstance(atomic_candidate.get("text"), str) and atomic_candidate.get("text", "").strip()
        else ""
    )

    summary, analysis, implication, examples = _interpret_signal_sentence(
        signal_path=signal_path,
        signal_value=signal_value,
        chapter=chapter,
        ko_mode=ko_mode,
        structural_summary=structural,
        mapping_audit=mapping_audit,
    )
    source_type = "signal"
    should_use_atomic = ko_mode and atomic_text and str(signal_path).lower().startswith("atomic.")
    if should_use_atomic:
        summary = atomic_text
        analysis = ""
        implication = ""
        source_type = "atomic"
        ATOMIC_RUNTIME_AUDIT["atomic_text_applied_count"] = int(ATOMIC_RUNTIME_AUDIT.get("atomic_text_applied_count", 0)) + 1
        logger.info("atomic_text_applied key=%s length=%d", atomic_key, len(atomic_text))

    fragment = {
        "title": f"{chapter} - {'해석 블록' if ko_mode else 'Interpretive Block'} {index}",
        "summary": summary,
        "analysis": analysis,
        "implication": implication,
        "examples": examples,
        "_source": source_type,
    }
    key_forecast = _high_signal_forecast_line(signal_path, signal_value, ko_mode=ko_mode)
    if key_forecast:
        fragment["key_forecast"] = key_forecast
    trace = {
        "text": f"{summary} {analysis}",
        "source_signal": signal_path,
        "source_value": signal_value,
        "rule_id": rule_id,
    }
    return fragment, trace


def _build_deterministic_fallback_fragments(
    chapter: str,
    structural_summary: dict[str, Any],
    has_atomic_base: bool = False,
    mapping_audit: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signals = _collect_chapter_signals(chapter, structural_summary, limit=6)
    if not signals:
        signals = [("stability_metrics.stability_index", ((structural_summary.get("stability_metrics") or {}).get("stability_index", 0)))]

    fragments: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    max_fallback = 2 if has_atomic_base else 3
    for idx, (path, value) in enumerate(signals[:max_fallback], start=1):
        fragment, trace = _create_signal_fragment(
            chapter=chapter,
            index=idx,
            signal_path=path,
            signal_value=value,
            rule_id=f"fallback_{chapter.lower().replace(' ', '_')}_{idx}",
            language="ko" if _is_korean_language(structural_summary) else "en",
            structural_summary=structural_summary,
            mapping_audit=mapping_audit,
        )
        fragments.append(fragment)
        meta.append(
            {
                "id": f"fallback::{chapter}::{idx}",
                "priority": -500 + idx,
                "_intensity": 0.55,
                "_deterministic_trace": [trace],
            }
        )
    return fragments, meta


def _expand_chapter_depth(
    chapter: str,
    chapter_blocks: list[dict[str, Any]],
    chapter_meta: list[dict[str, Any]],
    structural_summary: dict[str, Any],
    mapping_audit: dict[str, Any] | None = None,
) -> None:
    ko_mode = _is_korean_language(structural_summary)
    target_chars = MIN_DEPTH_KO_CHARS
    target_words = MIN_DEPTH_EN_WORDS
    signals = _collect_chapter_signals(chapter, structural_summary, limit=40)
    if not signals:
        return

    existing_texts: set[str] = set()
    for fragment in chapter_blocks:
        if not isinstance(fragment, dict):
            continue
        fragment_text = " ".join(
            str(fragment.get(field, "")).strip()
            for field in ("title", "summary", "analysis", "implication", "examples")
        ).strip()
        if fragment_text:
            existing_texts.add(fragment_text)

    signal_cursor = 0
    guard = 0
    atomic_count = len([f for f in chapter_blocks if isinstance(f, dict) and f.get("_source") == "atomic"])
    if atomic_count > 0:
        atomic_indices = [i for i, f in enumerate(chapter_blocks) if isinstance(f, dict) and f.get("_source") == "atomic"]
        if not atomic_indices:
            return
        target_idx = atomic_indices[0]
        while guard < 80:
            guard += 1
            chars, words = _chapter_depth_stats(chapter_blocks)
            if ko_mode and chars >= target_chars:
                break
            if (not ko_mode) and words >= target_words:
                break

            path, value = signals[signal_cursor % len(signals)]
            rule_id = f"depth_guard_{chapter.lower().replace(' ', '_')}_{guard}"
            _, analysis_s, implication_s, examples_s = _interpret_signal_sentence(
                signal_path=path,
                signal_value=value,
                chapter=chapter,
                ko_mode=ko_mode,
                structural_summary=structural_summary,
                mapping_audit=mapping_audit,
            )
            fragment_text = " ".join(
                part.strip() for part in (analysis_s, implication_s, examples_s) if isinstance(part, str) and part.strip()
            )
            if fragment_text in existing_texts:
                signal_cursor += 1
                continue
            for field in ("analysis", "implication", "examples"):
                existing = chapter_blocks[target_idx].get(field, "")
                if not isinstance(existing, str):
                    existing = ""
                addition = analysis_s if field == "analysis" else implication_s if field == "implication" else examples_s
                chapter_blocks[target_idx][field] = (existing + "\n\n" + addition).strip()
            existing_texts.add(fragment_text)

            trace = {
                "text": f"{analysis_s}",
                "source_signal": path,
                "source_value": value,
                "rule_id": rule_id,
            }
            if target_idx < len(chapter_meta):
                traces = chapter_meta[target_idx].setdefault("_deterministic_trace", [])
                if isinstance(traces, list):
                    traces.append(trace)
            signal_cursor += 1
        return

    while guard < 80:
        guard += 1
        chars, words = _chapter_depth_stats(chapter_blocks)
        if ko_mode and chars >= target_chars:
            break
        if (not ko_mode) and words >= target_words:
            break

        if not chapter_blocks:
            fragment, trace = _create_signal_fragment(
                chapter=chapter,
                index=1,
                signal_path=signals[0][0],
                signal_value=signals[0][1],
                rule_id=f"depth_seed_{chapter.lower().replace(' ', '_')}",
                language="ko" if ko_mode else "en",
                structural_summary=structural_summary,
                mapping_audit=mapping_audit,
            )
            chapter_blocks.append(fragment)
            chapter_meta.append(
                {
                    "id": f"depth_seed::{chapter}",
                    "priority": -300,
                    "_intensity": 0.6,
                    "_deterministic_trace": [trace],
                }
            )
            signal_cursor = 1
            continue

        non_atomic_indices = [i for i, frag in enumerate(chapter_blocks) if isinstance(frag, dict) and frag.get("_source") != "atomic"]
        if non_atomic_indices:
            target_idx = min(non_atomic_indices, key=lambda i: _fragment_text_length(chapter_blocks[i])[0])
        else:
            path_seed, value_seed = signals[signal_cursor % len(signals)]
            fragment_seed, trace_seed = _create_signal_fragment(
                chapter=chapter,
                index=len(chapter_blocks) + 1,
                signal_path=path_seed,
                signal_value=value_seed,
                rule_id=f"depth_append_{chapter.lower().replace(' ', '_')}_{guard}",
                language="ko" if ko_mode else "en",
                structural_summary=structural_summary,
                mapping_audit=mapping_audit,
            )
            chapter_blocks.append(fragment_seed)
            chapter_meta.append(
                {
                    "id": f"depth_append::{chapter}::{guard}",
                    "priority": -250,
                    "_intensity": 0.6,
                    "_deterministic_trace": [trace_seed],
                }
            )
            signal_cursor += 1
            continue
        path, value = signals[signal_cursor % len(signals)]
        rule_id = f"depth_guard_{chapter.lower().replace(' ', '_')}_{guard}"
        summary_s, analysis_s, implication_s, examples_s = _interpret_signal_sentence(
            signal_path=path,
            signal_value=value,
            chapter=chapter,
            ko_mode=ko_mode,
            structural_summary=structural_summary,
            mapping_audit=mapping_audit,
        )
        fragment_text = " ".join(
            part.strip() for part in (summary_s, analysis_s, implication_s, examples_s) if isinstance(part, str) and part.strip()
        )
        if fragment_text in existing_texts:
            signal_cursor += 1
            continue
        for field in ("analysis", "implication", "examples"):
            existing = chapter_blocks[target_idx].get(field, "")
            if not isinstance(existing, str):
                existing = ""
            addition = analysis_s if field == "analysis" else implication_s if field == "implication" else examples_s
            chapter_blocks[target_idx][field] = (existing + "\n\n" + addition).strip()
        existing_texts.add(fragment_text)

        trace = {
            "text": f"{summary_s} {analysis_s}",
            "source_signal": path,
            "source_value": value,
            "rule_id": rule_id,
        }
        if target_idx < len(chapter_meta):
            traces = chapter_meta[target_idx].setdefault("_deterministic_trace", [])
            if isinstance(traces, list):
                traces.append(trace)
        signal_cursor += 1

def _has_manual_template_override() -> bool:
    cached_templates = _TEMPLATES_BY_LANG.get(_ACTIVE_LANGUAGE)
    cached_defaults = _DEFAULTS_BY_LANG.get(_ACTIVE_LANGUAGE)
    if cached_templates is None or cached_defaults is None:
        return False
    return (TEMPLATES is not cached_templates) or (DEFAULT_BLOCKS is not cached_defaults)


def build_semantic_signals(structural_summary: dict[str, Any]) -> dict[str, str]:
    """Build narrative modulation signals without changing engine outputs."""
    source = structural_summary if isinstance(structural_summary, dict) else {}
    vector = source.get("current_dasha_vector", {}) if isinstance(source.get("current_dasha_vector"), dict) else {}
    stability_metrics = source.get("stability_metrics", {}) if isinstance(source.get("stability_metrics"), dict) else {}

    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    activation = _safe_float(vector.get("opportunity_factor", 0.0), 0.0)
    risk = _safe_float(vector.get("risk_factor", 0.0), 0.0)
    stability = _safe_float(stability_metrics.get("stability_index", 50.0), 50.0)

    if activation >= 0.85:
        influence_band = "peak_alignment"
    elif activation >= 0.65:
        influence_band = "breakthrough_window"
    elif activation >= 0.45:
        influence_band = "high_activation"
    elif activation >= 0.25:
        influence_band = "activation"
    else:
        influence_band = "dormant"

    if risk > 0.6 and activation > 0.6:
        risk_pattern = "crucible_phase"
    elif risk > 0.6:
        risk_pattern = "stress_load"
    elif activation > 0.6:
        risk_pattern = "expansion_window"
    else:
        risk_pattern = "neutral_cycle"

    if stability >= 75:
        stability_band = "structural_strong"
    elif stability >= 55:
        stability_band = "structural_stable"
    elif stability >= 40:
        stability_band = "structural_fragile"
    else:
        stability_band = "structural_volatile"

    intensity_score = (
        activation * 0.5 +
        risk * 0.3 +
        (1.0 - stability / 100.0) * 0.2
    )

    if intensity_score >= 0.7:
        intensity_profile = "high_drama"
    elif intensity_score >= 0.5:
        intensity_profile = "elevated"
    else:
        intensity_profile = "measured"

    return {
        "influence_band": influence_band,
        "risk_pattern": risk_pattern,
        "stability_band": stability_band,
        "intensity_profile": intensity_profile,
    }


def select_template_blocks(structural_summary: dict[str, Any], language: str | None = None) -> dict[str, list[dict[str, Any]]]:
    if isinstance(language, str) and language.strip():
        _set_active_language(language)
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
    _apply_emotional_escalation(selected, structural_summary)
    _apply_recursive_correction(selected, structural_summary)
    _apply_psychological_echo(selected, structural_summary)
    _sort_selected_blocks(selected)

    return selected


def build_report_payload(rectified_structural_summary: dict[str, Any]) -> dict[str, Any]:
    explicit_language = False
    requested_language = "en"
    if isinstance(rectified_structural_summary, dict):
        raw_lang = rectified_structural_summary.get("language")
        if not isinstance(raw_lang, str):
            structural_candidate = rectified_structural_summary.get("structural_summary")
            if isinstance(structural_candidate, dict):
                raw_lang = structural_candidate.get("language")
        if isinstance(raw_lang, str) and raw_lang.strip():
            requested_language = raw_lang.strip().lower()
            explicit_language = True

    if explicit_language or not _has_manual_template_override():
        _set_active_language(requested_language)
    structural = rectified_structural_summary.get("structural_summary") if isinstance(rectified_structural_summary, dict) else None
    if not isinstance(structural, dict):
        structural = rectified_structural_summary if isinstance(rectified_structural_summary, dict) else {}
    if "language" not in structural and isinstance(requested_language, str):
        structural = {**structural, "language": requested_language}

    raw_blocks = select_template_blocks(structural, language=requested_language if explicit_language else None)
    mapping_audit = _new_mapping_audit()
    atomic_interpretations = _get_atomic_chart_interpretations(structural)
    chart_signature = structural.get("chart_signature") if isinstance(structural.get("chart_signature"), dict) else {}
    mapping_audit["atomic_inputs_available"] = {
        "ascendant": bool(structural.get("ascendant_sign") or structural.get("asc_sign") or chart_signature.get("ascendant_sign")),
        "sun": bool(structural.get("sun_sign") or chart_signature.get("sun_sign")),
        "moon": bool(structural.get("moon_sign") or chart_signature.get("moon_sign")),
    }

    chapter_blocks: dict[str, list[dict[str, Any]]] = {}
    chapter_meta: dict[str, list[dict[str, Any]]] = {}
    chapter_limits: dict[str, int] = {}
    chapter_spikes: dict[str, list[str]] = {}
    atomic_fragments_per_chapter: dict[str, int] = {chapter: 0 for chapter in REPORT_CHAPTERS}

    def _build_atomic_anchor_fragment(chapter_name: str, key_label: str, text: str, index_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        integrated = _integrate_atomic_with_signals(text, structural)
        return (
            {
                "title": f"{chapter_name} - 해석 블록 {index_seed}",
                "summary": integrated,
                "analysis": "",
                "implication": "",
                "examples": "",
                "_source": "atomic",
            },
            {
                "id": f"atomic::{chapter_name}::{key_label}",
                "priority": -999,
                "_intensity": 0.95,
                "_deterministic_trace": [
                    {
                        "text": text,
                        "source_signal": f"atomic.{key_label}",
                        "source_value": key_label,
                        "rule_id": f"atomic_anchor_{chapter_name.lower().replace(' ', '_')}_{key_label}",
                    }
                ],
            },
        )

    def _chapter_atomic_anchors(chapter_name: str) -> list[tuple[str, str]]:
        asc_text = atomic_interpretations.get("asc", "")
        sun_text = atomic_interpretations.get("sun", "")
        moon_text = atomic_interpretations.get("moon", "")
        chunks = [c for c in (asc_text, sun_text, moon_text) if isinstance(c, str) and c.strip()]
        if not chunks:
            return []
        if chapter_name == "Life Timeline Interpretation":
            return [("asc_sun_moon", "\n\n".join(chunks))]
        chapter_key_map = {
            "Executive Summary": "asc",
            "Personality Vector": "moon",
            "Career & Success": "sun",
            "Love & Relationships": "moon",
            "Psychological Architecture": "moon",
            "Behavioral Risks": "moon",
            "Stability Metrics": "asc",
            "Confidence & Forecast": "sun",
            "Final Summary": "asc",
            "Remedies & Program": "asc",
            "Karmic Patterns": "asc",
            "Purushartha Profile": "asc",
            "Health & Body Patterns": "asc",
            "Appendix (Optional)": "asc",
        }
        preferred_key = chapter_key_map.get(chapter_name, "asc")
        preferred_text = atomic_interpretations.get(preferred_key, "")
        if isinstance(preferred_text, str) and preferred_text.strip():
            return [(preferred_key, preferred_text)]
        return [("asc_sun_moon", "\n\n".join(chunks))]

    for chapter in REPORT_CHAPTERS:
        blocks = raw_blocks.get(chapter, [])
        use_fallback_builders = not blocks

        for block in blocks:
            if "_intensity" not in block:
                block["_intensity"] = compute_block_intensity(block, structural)

        spike_texts: list[str] = []
        for block in blocks:
            spike_data = block.get("insight_spike")
            if not isinstance(spike_data, dict):
                continue

            text = spike_data.get("text")
            min_intensity = spike_data.get("min_intensity")
            if not isinstance(text, str) or not isinstance(min_intensity, (int, float)):
                continue
            if not 0.0 <= float(min_intensity) <= 1.0:
                continue
            if block.get("_intensity", 0) >= float(min_intensity):
                spike_texts.append(text)

        spike_texts = list(dict.fromkeys(spike_texts))
        chapter_spikes[chapter] = spike_texts
        chapter_limit = _dynamic_chapter_limit(structural, spike_count=len(spike_texts))
        chapter_limits[chapter] = chapter_limit

        chapter_payload: list[dict[str, Any]] = []
        chapter_payload_meta: list[dict[str, Any]] = []

        anchors = _chapter_atomic_anchors(chapter)
        for idx_anchor, (anchor_key, anchor_text) in enumerate(anchors, start=1):
            if not isinstance(anchor_text, str) or not anchor_text.strip():
                continue
            anchor_fragment, anchor_meta = _build_atomic_anchor_fragment(chapter, anchor_key, anchor_text.strip(), idx_anchor)
            chapter_payload.append(anchor_fragment)
            chapter_payload_meta.append(anchor_meta)
            atomic_fragments_per_chapter[chapter] = int(atomic_fragments_per_chapter.get(chapter, 0)) + 1

        has_atomic = len(chapter_payload) > 0 and any(isinstance(f, dict) and f.get("_source") == "atomic" for f in chapter_payload)

        if use_fallback_builders and not has_atomic:
            fallback_payload, fallback_meta = _build_deterministic_fallback_fragments(
                chapter,
                structural,
                has_atomic_base=False,
                mapping_audit=mapping_audit,
            )
            remaining = max(0, chapter_limit - len(chapter_payload))
            chapter_payload.extend(fallback_payload[:remaining])
            chapter_payload_meta.extend(fallback_meta[:remaining])

        if blocks:
            for block in blocks:
                if len(chapter_payload) >= chapter_limit:
                    break
                intensity = block.get("_intensity", 0)
                payload_block = _render_payload_fragment(block, chapter, intensity)

                if payload_block:
                    payload_block["_source"] = payload_block.get("_source", "signal")
                    chapter_payload.append(payload_block)
                    chapter_payload_meta.append(block)

        chapter_blocks[chapter] = chapter_payload
        chapter_meta[chapter] = chapter_payload_meta

    _inject_shadbala_insight(structural, chapter_blocks, chapter_meta, chapter_limits)
    _inject_choice_forks(structural, chapter_blocks, chapter_meta, chapter_limits)
    _inject_scenario_compression(structural, chapter_blocks, chapter_meta, chapter_limits)

    # TODO(report-engine): Atomic dominance lock is intentionally disabled pending product decision.
    # Atomic dominance lock: where atomic fragment exists, remove non-atomic fragments.
    # for chapter in REPORT_CHAPTERS:
    #     frags = chapter_blocks.get(chapter, [])
    #     metas = chapter_meta.get(chapter, [])
    #     if not isinstance(frags, list) or not isinstance(metas, list):
    #         continue
    #     if not any(isinstance(f, dict) and f.get("_source") == "atomic" for f in frags):
    #         continue
    #     filtered_frags: list[dict[str, Any]] = []
    #     filtered_meta: list[dict[str, Any]] = []
    #     for idx, frag in enumerate(frags):
    #         if not isinstance(frag, dict):
    #             continue
    #         if frag.get("_source") != "atomic":
    #             continue
    #         filtered_frags.append(frag)
    #         filtered_meta.append(metas[idx] if idx < len(metas) else {})
    #     chapter_blocks[chapter] = filtered_frags
    #     chapter_meta[chapter] = filtered_meta

    for chapter in REPORT_CHAPTERS:
        _expand_chapter_depth(
            chapter,
            chapter_blocks.get(chapter, []),
            chapter_meta.get(chapter, []),
            structural,
            mapping_audit=mapping_audit,
        )

    final_chapter_blocks: dict[str, list[dict[str, Any]]] = {}
    for chapter in REPORT_CHAPTERS:
        spike_fragments = [{"spike_text": spike_text} for spike_text in chapter_spikes.get(chapter, [])]
        content_fragments = chapter_blocks.get(chapter, [])
        final_chapter_blocks[chapter] = content_fragments + spike_fragments

    def _strip_internal_fields(chapter_blocks_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        sanitized: dict[str, list[dict[str, Any]]] = {}
        if not isinstance(chapter_blocks_payload, dict):
            return sanitized
        for chapter_name, fragments in chapter_blocks_payload.items():
            if not isinstance(fragments, list):
                sanitized[chapter_name] = []
                continue
            cleaned_fragments: list[dict[str, Any]] = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                cleaned_fragment = {
                    key: value
                    for key, value in fragment.items()
                    if isinstance(key, str) and not key.startswith("_")
                }
                cleaned_fragments.append(cleaned_fragment)
            sanitized[chapter_name] = cleaned_fragments
        return sanitized

    sanitized_chapter_blocks = _strip_internal_fields(final_chapter_blocks)

    total = int(mapping_audit.get("total_signals_processed", 0))
    hits = int(mapping_audit.get("mapping_hits", 0))
    misses = int(mapping_audit.get("mapping_misses", 0))
    hit_rate = (hits / total) if total > 0 else 0.0
    atomic_usage = mapping_audit.get("atomic_usage", {})
    atomic_inputs_available = mapping_audit.get("atomic_inputs_available", {})
    if REPORT_MAPPING_DEBUG:
        logger.info(
            "mapping_summary total_signals_processed=%s mapping_hits=%s mapping_misses=%s hit_rate=%.4f atomic_inputs_available=%s atomic_usage=%s",
            total,
            hits,
            misses,
            hit_rate,
            atomic_inputs_available,
            atomic_usage,
        )
    # Always emit summary once so production logs can verify mapping effectiveness.
    logger.info(
        "mapping_effectiveness total=%s hits=%s misses=%s hit_rate=%.4f",
        total,
        hits,
        misses,
        hit_rate,
    )
    atomic_fragments_count = sum(int(v) for v in atomic_fragments_per_chapter.values())
    total_content_fragments = 0
    generic_fragments_count = 0
    for chapter in REPORT_CHAPTERS:
        for fragment in final_chapter_blocks.get(chapter, []):
            if not isinstance(fragment, dict) or "spike_text" in fragment:
                continue
            total_content_fragments += 1
            if fragment.get("_source") != "atomic":
                generic_fragments_count += 1
    atomic_fragment_ratio = (atomic_fragments_count / total_content_fragments) if total_content_fragments else 0.0
    generic_fragment_ratio = (generic_fragments_count / total_content_fragments) if total_content_fragments else 0.0
    atomic_dominance_verified = atomic_fragments_count >= generic_fragments_count if total_content_fragments else False
    logger.info(
        "atomic_injected_into_chapters=%s atomic_fragments_count=%s atomic_fragments_per_chapter=%s",
        "True" if atomic_fragments_count > 0 else "False",
        atomic_fragments_count,
        atomic_fragments_per_chapter,
    )
    logger.info(
        "atomic_dominance_verified=%s atomic_fragment_ratio=%.4f generic_fragment_ratio=%.4f",
        "True" if atomic_dominance_verified else "False",
        atomic_fragment_ratio,
        generic_fragment_ratio,
    )

    return {"chapter_blocks": sanitized_chapter_blocks}


def build_gpt_user_content(payload: dict[str, Any]) -> str:
    return "<BEGIN STRUCTURED BLOCKS>\n" + json.dumps(payload.get("chapter_blocks", {}), ensure_ascii=False, indent=2) + "\n<END STRUCTURED BLOCKS>"


