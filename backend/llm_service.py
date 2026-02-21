import os
import logging
import re
from typing import Any, Optional

from backend.report_engine import _get_atomic_chart_interpretations, build_semantic_signals

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
logger = logging.getLogger("vedic_ai")


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
    stability = structural_summary.get("stability_metrics", {}) if isinstance(structural_summary, dict) else {}
    purpose = structural_summary.get("life_purpose_vector", {}) if isinstance(structural_summary, dict) else {}
    forecast = structural_summary.get("probability_forecast", {}) if isinstance(structural_summary, dict) else {}
    tension_axis = structural_summary.get("psychological_tension_axis") if isinstance(structural_summary, dict) else None

    dominant = purpose.get("dominant_planet", "N/A")
    stability_index = stability.get("stability_index", "N/A")

    if isinstance(tension_axis, list) and tension_axis:
        tension_text = " ↔ ".join(str(x) for x in tension_axis)
    else:
        tension_text = "None"

    try:
        career_shift = float(forecast.get("career_shift_3yr", 0))
        career_shift_pct = f"{int(career_shift * 100)}%"
    except Exception:
        career_shift_pct = "N/A"

    try:
        burnout = float(forecast.get("burnout_2yr", 0))
        burnout_pct = f"{int(burnout * 100)}%"
    except Exception:
        burnout_pct = "N/A"

    return f"""
[Structural Executive Overview]

Dominant Force: {dominant}
Stability Index: {stability_index}
Core Tension Axis: {tension_text}
Career Shift Probability (3yr): {career_shift_pct}
Burnout Risk (2yr): {burnout_pct}

Use this overview as the primary narrative anchor.
"""


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
    low_density = (heading_count < 6) or (text_length < 3500) or (structural_ref_count < 3)

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
    atomic_interpretations = _get_atomic_chart_interpretations(structural_summary if isinstance(structural_summary, dict) else {})
    system_message = "Follow the user prompt exactly."
    user_message = build_llm_structural_prompt(
        structural_payload,
        language=language,
        atomic_interpretations=atomic_interpretations,
        chapter_blocks=validated,
        semantic_signals=semantic_signals,
        narrative_mode=narrative_mode,
        executive_summary=executive_summary,
    )
    chapter_blocks_hash = compute_hash_fn(validated)
    selected_model = str(model or OPENAI_MODEL).strip() or OPENAI_MODEL
    candidate_models = candidate_models_fn(selected_model)
    last_error: Optional[Exception] = None

    for candidate_model in candidate_models:
        payload = build_payload_fn(
            model=candidate_model,
            system_message=system_message,
            user_message=user_message,
            max_completion_tokens=max_tokens,
        )
        try:
            logger.info(
                "LLM API call started request_id=%s selected_model=%s chapter_blocks_hash=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
            )
            response = await async_client.chat.completions.create(**payload)
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
            response_text = normalize_paragraphs_fn(response_text, max_chars=300)
            audit_report = audit_llm_output(response_text, structural_summary)
            logger.info("[LLM AUDIT] score=%s flags=%s", audit_report.get("overall_score"), audit_report.get("flags"))
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
            return response_text
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
) -> str:
    import json

    atomic = atomic_interpretations if isinstance(atomic_interpretations, dict) else {}
    signals = dict(semantic_signals) if isinstance(semantic_signals, dict) else {}
    mode = str(narrative_mode).strip() if isinstance(narrative_mode, str) and narrative_mode.strip() else "measured_growth"
    overview = executive_summary if isinstance(executive_summary, str) else ""
    asc_text = str(atomic.get("asc", "")).strip()
    sun_text = str(atomic.get("sun", "")).strip()
    moon_text = str(atomic.get("moon", "")).strip()
    blocks_json = json.dumps(chapter_blocks, indent=2, ensure_ascii=False) if chapter_blocks else "{}"
    source = structural_summary if isinstance(structural_summary, dict) else {}
    current_vector = source.get("current_dasha_vector") or {}
    if not isinstance(current_vector, dict):
        current_vector = {}
    psych_axis = source.get("psychological_tension_axis")
    if isinstance(psych_axis, list):
        psych_axis_norm = " ↔ ".join(str(x) for x in psych_axis if str(x).strip())
    else:
        psych_axis_norm = psych_axis
    dominant_axis = (
        current_vector.get("dominant_axis")
        or source.get("dominant_axis")
        or psych_axis_norm
    )
    compressed_signals = {
        "dominant_axis": dominant_axis,
        "current_theme": current_vector.get("current_theme"),
        "risk_signal": signals.get("risk_pattern") or signals.get("risk_band") or None,
        "influence_signal": signals.get("influence_band") or signals.get("activation_band") or None,
        "stability_signal": signals.get("stability_band") or signals.get("stability_profile") or None,
    }

    return f"""
STRATEGIC ASTROLOGICAL COACH SYSTEM

VOICE & TONE

### PERSONA LOCK
당신은 감성 위로형 상담사가 아니다.
당신은 카르마 구조를 분석하고
현실적인 선택 전략을 제시하는 전략 코치형 점성가다.

특징:
- 구조를 먼저 설명하고 감정을 해석한다.
- 위로보다 방향 제시를 우선한다.
- 단정하지 않되, 회피하지 않는다.
- "괜찮아요"식 모호한 위로를 피한다.
- 실행 가능한 문장으로 마무리한다.

각 챕터 내 기본 흐름:
1) 구조 설명
2) 현재 패턴 진단
3) 리스크 또는 기회 명확화
4) 선택 전략 제시

Role:
You diagnose structural forces and provide strategic direction.
You are analytical, composed, and decisive.

Communication Rules:
- Move from STRUCTURE -> INTERPRETATION -> STRATEGY.
- Use probability-based language.
- Avoid mystical, fatalistic, or dramatic phrasing.
- Emotional validation allowed (max 2 sentences per section).
- Never explain scoring mechanics.

Tone:
Clear, precise, forward-looking.
Avoid repetition and abstract philosophy.

CALIBRATION RULES

High opportunity -> emphasize upward momentum.
High risk -> include firm caution.
Low stability -> emphasize restructuring.
Translate numeric signals into strategic language.
Do not expose internal scoring mechanics.
Structural indices may be mentioned once in interpretive form only.

Structural Anchor Enforcement:
- Each major chapter must explicitly reference at least one of:
  - dominant axis
  - structural stability
  - current activation theme
- Structural terms such as "구조", "흐름", "에너지 축", "주기", "패턴" must appear naturally throughout.
- Avoid generic motivational language disconnected from structural signals.

Risk Acknowledgment Rule:
If risk signal indicates elevated pressure,
include at least one direct cautionary sentence
using words like "주의", "경고", "압박", or "긴장".
Do not soften structural risk.

Stability Reference Rule:
Mention structural stability or foundational strength at least once
in interpretive form (not raw numeric exposure).

Avoid repeating identical structural phrasing across chapters.
Do not reuse identical emotional framing across chapters.
Do not repeat identical explanatory paragraphs.
Emotional modulation must vary by domain context.

Narrative Mode:
{mode}

Emotional Derivation & Tone Modulation (Internal Only):
- Before writing the report, internally derive an emotional_state from structural signals.
- Never print emotional_state as JSON or explicit labels.
- Never print numeric values for this layer.
- If a required structural key is missing, assume neutral mid-level conditions.
- Never fail or output placeholder explanations due to missing keys.

Fallback-safe derivation rules:
- If burnout_risk exists and is high -> core_tone = fatigued
- If risk_factor exists and high and stability_index exists and low -> core_tone = pressured
- If opportunity_factor exists and high and risk_factor low -> core_tone = expansive
- If stability_index exists and high -> confidence_pattern = grounded
- If stability_index exists and low -> confidence_pattern = fragile
- If psychological_tension_axis exists and risk elevated -> core_tone = conflicted
- If none is clearly dominant -> core_tone = steady

Tone mapping (compact):
- fatigued: slower rhythm, recovery emphasis, softer warnings
- pressured: shorter sentences, direct warnings, decision urgency
- expansive: longer sentences, opportunity framing, future orientation
- conflicted: contrast structures, dual-perspective phrasing
- steady: balanced pacing, moderate urgency, grounded reassurance

Emotional coherence rule:
- Maintain emotional coherence across the full report.
- Each chapter must apply emotional tone differently by domain context.
- Career: drive or pressure.
- Relationships: vulnerability or guardedness.
- Finance: risk tolerance or caution.
- Health: energy flow or depletion.

GLOBAL NARRATIVE CONTINUITY RULE:
- The selected Narrative Mode must remain consistent across all chapters.
- Tone may evolve logically, but must never invert.
- Do not describe pressure_window as peaceful.
- Do not describe expansion_window as stagnant.
- Emotional intensity must align with Narrative Mode throughout the report.

### STRUCTURAL ECHO REINFORCEMENT
각 챕터마다 최소 한 문장은 반드시
이 구조가 당신의 현재 흐름을 만든다 와 같은 방식으로
구조 -> 삶의 결과를 직접 연결하는 문장을 포함할 것.
구조적 요인이 실제 삶의 선택, 패턴, 결과로 이어진다는 점을 분명히 드러낼 것.

### CONTROLLED WARNING INJECTION
위험도가 낮더라도, 전체 보고서 중 최소 1회는
단호하고 직설적인 경고 문장을 포함할 것.
예:
이 부분을 방치하면 같은 패턴이 반복된다.
지금 정리하지 않으면 나중에 더 큰 비용을 치르게 된다.
예시 문구는 그대로 복사하지 말고 의미만 유지해 변형하여 사용할 것.
과장 없이, 회피하지 않는 어조로 작성할 것.

### LENGTH FLOOR REINFORCEMENT
총 분량은 최소 4,500자 이상을 목표로 작성할 것.
다음 핵심 챕터는 각각 최소 700자 이상:
- 커리어 및 사회적 방향
- 재정 및 돈의 흐름
- 관계 및 파트너십

요약 금지.
압축 서술 금지.
핵심 문단 생략 금지.

Structural Executive Overview:
{overview}

Chapter Signal Emphasis Guide:
- Psychological chapters -> tension axis and personality signals.
- Career chapter -> career shift likelihood and directional strategy.
- Stability chapter -> structural stability implications.
- Finance chapter -> financial instability risk and control strategy.
- Growth chapter -> long-term trajectory and execution priorities.

Compressed Structural Signals (JSON):
{json.dumps(compressed_signals, indent=2, ensure_ascii=False)}

Core Chart Identity:
Ascendant: {asc_text}
Sun: {sun_text}
Moon: {moon_text}

Chapter Blocks (JSON):
{blocks_json}

Write a complete 15-chapter report in Korean.
Each chapter must be concrete, non-repetitive, and strategically actionable.
"""

