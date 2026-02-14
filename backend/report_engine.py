"""Deterministic report block selector and GPT payload builder."""

from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("report_engine")

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


def _load_template_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _load_templates() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    base = Path(__file__).resolve().parent / "report_templates"
    templates: list[dict[str, Any]] = []
    for filename in _TEMPLATE_FILES:
        for block in _load_template_file(base / filename):
            chapter = block.get("chapter")
            if chapter not in REPORT_CHAPTERS:
                logger.warning("Template block has unknown chapter '%s': id=%s", chapter, block.get("id"))
            templates.append(block)

    defaults: dict[str, list[dict[str, Any]]] = {chapter: [] for chapter in REPORT_CHAPTERS}
    for block in _load_template_file(base / _DEFAULT_TEMPLATE_FILE):
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


def _ensure_loaded() -> None:
    global TEMPLATES, DEFAULT_BLOCKS
    if not TEMPLATES and not DEFAULT_BLOCKS:
        TEMPLATES, DEFAULT_BLOCKS = _load_templates()
        _validate_rule_chapters()


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

    if intensity >= 0.85:
        include_fields = [
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
            "choice_fork",
            "predictive_compression",
        ]
    elif intensity >= 0.75:
        include_fields = [
            "title",
            "summary",
            "analysis",
            "implication",
            "examples",
            "choice_fork",
            "predictive_compression",
        ]
    elif intensity >= 0.5:
        include_fields = ["title", "summary", "analysis", "implication", "predictive_compression"]
    else:
        include_fields = ["title", "summary", "analysis", "predictive_compression"]

    payload_block: dict[str, Any] = {}
    for field in include_fields:
        if field == "title":
            payload_block[field] = str(content.get(field, chapter))
            continue
        if field not in content:
            continue

        value = content.get(field)
        if field == "choice_fork":
            if intensity >= 0.75 and isinstance(value, dict):
                payload_block["choice_fork"] = value
        elif field == "predictive_compression":
            if isinstance(value, dict):
                payload_block["predictive_compression"] = value
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
        "Shadbala 근사 지표와 Avastha 상태를 함께 본 핵심 판독입니다:\n"
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
                    "강한 행성은 다샤/고차라 타이밍의 실행축으로 쓰고, 약한 행성이 지배하는 영역은 "
                    "루틴과 회복력을 먼저 세운 뒤 확장하는 것이 안전합니다."
                ),
                "examples": (
                    "근거 태그는 Directional Strength, Exalted, Own Sign, Combust, Retrograde를 사용하며, "
                    "해석은 정량 점수보다 행성 간 상대 우선순위에 중점을 둡니다."
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
                "summary": f"핵심 행성 축은 {top_planets_text}로 수렴합니다.",
                "analysis": (
                    "상위 행성은 현실 실행력과 회복 탄력의 중심이며, 약한 행성 영역은 과속 시 변동성이 확대됩니다. "
                    "이번 리포트의 권고는 이 강약 구조를 기준으로 배열되어야 정확도가 높습니다."
                ),
                "implication": "의사결정은 강한 축에 맞추고, 약한 축은 단계적 보정으로 접근하는 것이 손실을 줄입니다.",
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
                "summary": "처방 우선순위는 약한 행성의 안정화, 강한 행성의 과부하 방지 순으로 잡습니다.",
                "analysis": (
                    "약한 행성은 수면/루틴/행동 반복의 기본기부터 보강하고, 강한 행성은 과도한 책임 집중을 분산해야 "
                    "전체 차트의 균형이 유지됩니다."
                ),
                "examples": "실행 계획은 2주 단위 점검으로 시작하고, 강약 밴드 변화에 따라 강도를 조정합니다.",
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
    _apply_emotional_escalation(selected, structural_summary)
    _apply_recursive_correction(selected, structural_summary)
    _apply_psychological_echo(selected, structural_summary)
    _sort_selected_blocks(selected)

    return selected


def build_report_payload(rectified_structural_summary: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    structural = rectified_structural_summary.get("structural_summary") if isinstance(rectified_structural_summary, dict) else None
    if not isinstance(structural, dict):
        structural = rectified_structural_summary if isinstance(rectified_structural_summary, dict) else {}

    raw_blocks = select_template_blocks(structural)

    chapter_blocks: dict[str, list[dict[str, Any]]] = {}
    chapter_meta: dict[str, list[dict[str, Any]]] = {}
    chapter_limits: dict[str, int] = {}
    chapter_spikes: dict[str, list[str]] = {}

    for chapter in REPORT_CHAPTERS:
        blocks = raw_blocks.get(chapter, [])
        if not blocks:
            blocks = DEFAULT_BLOCKS.get(chapter, [])

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
        chapter_limit = max(0, 5 - len(spike_texts))
        chapter_limits[chapter] = chapter_limit

        chapter_payload: list[dict[str, Any]] = []
        chapter_payload_meta: list[dict[str, Any]] = []
        for block in blocks:
            if len(chapter_payload) >= chapter_limit:
                break
            intensity = block.get("_intensity", 0)
            payload_block = _render_payload_fragment(block, chapter, intensity)

            if payload_block:
                chapter_payload.append(payload_block)
                chapter_payload_meta.append(block)

        chapter_blocks[chapter] = chapter_payload
        chapter_meta[chapter] = chapter_payload_meta

    _inject_shadbala_insight(structural, chapter_blocks, chapter_meta, chapter_limits)
    _inject_choice_forks(structural, chapter_blocks, chapter_meta, chapter_limits)
    _inject_scenario_compression(structural, chapter_blocks, chapter_meta, chapter_limits)

    final_chapter_blocks: dict[str, list[dict[str, Any]]] = {}
    for chapter in REPORT_CHAPTERS:
        spike_fragments = [{"spike_text": spike_text} for spike_text in chapter_spikes.get(chapter, [])]
        content_fragments = chapter_blocks.get(chapter, [])[: max(0, 5 - len(spike_fragments))]
        final_chapter_blocks[chapter] = (spike_fragments + content_fragments)[:5]

    return {"chapter_blocks": final_chapter_blocks}


def build_gpt_user_content(payload: dict[str, Any]) -> str:
    return "<BEGIN STRUCTURED BLOCKS>\n" + json.dumps(payload.get("chapter_blocks", {}), ensure_ascii=False, indent=2) + "\n<END STRUCTURED BLOCKS>"
