import os
import logging
from typing import Any, Optional

from backend.report_engine import _get_atomic_chart_interpretations
from backend.main import (
    build_ai_psychological_input,
    _validate_deterministic_llm_blocks,
    compute_chapter_blocks_hash,
    _candidate_openai_models,
    _build_openai_payload,
    _emit_llm_audit_event,
    _normalize_long_paragraphs,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
logger = logging.getLogger("vedic_ai")

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
) -> str:
    if async_client is None:
        raise RuntimeError("OpenAI client not initialized")

    validated = _validate_deterministic_llm_blocks(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    structural_payload = build_ai_psychological_input({"structural_summary": structural_summary})
    atomic_interpretations = _get_atomic_chart_interpretations(structural_summary if isinstance(structural_summary, dict) else {})
    system_message = "Follow the user prompt exactly."
    user_message = build_llm_structural_prompt(
        structural_payload,
        language=language,
        atomic_interpretations=atomic_interpretations,
        chapter_blocks=validated,
    )
    chapter_blocks_hash = compute_chapter_blocks_hash(validated)
    selected_model = str(model or OPENAI_MODEL).strip() or OPENAI_MODEL
    candidate_models = _candidate_openai_models(selected_model)
    last_error: Optional[Exception] = None

    for candidate_model in candidate_models:
        payload = _build_openai_payload(
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
            response_text = _normalize_long_paragraphs(response_text, max_chars=300)
            model_used = f"openai/{candidate_model}"
            _emit_llm_audit_event(
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
) -> str:
    import json

    lang_instruction = (
        "MUST be written in highly natural, professional, and empathetic Korean (Hangul)."
        if language == "ko"
        else "MUST be written in fluent English."
    )

    atomic = atomic_interpretations if isinstance(atomic_interpretations, dict) else {}
    asc_text = str(atomic.get("asc", "")).strip()
    sun_text = str(atomic.get("sun", "")).strip()
    moon_text = str(atomic.get("moon", "")).strip()
    blocks_json = json.dumps(chapter_blocks, indent=2, ensure_ascii=False) if chapter_blocks else "{}"

    return f"""
You are a warm, highly intuitive, and world-class expert Vedic astrologer.
Your task is to synthesize the provided structural data into a beautifully written, easy-to-understand, and deeply
personal multidimensional narrative report that serves as a 'Strategic Decision-Making Tool'.

CRITICAL COMMERCIAL GUIDELINES (STRICTLY ENFORCED):
1. PREDICTIVE STRENGTH & TIMEFRAMES (CRITICAL):
   - You MUST write strong, structured predictions using specific timeframes: Short-term (1~2 years) and Mid-term (3~5
 years).
   - Use strong directional language (e.g., "Your growth momentum strengthens over the next 12-18 months.", "Career transition pressure is likely to increase before stabilizing.").
   - Provide conditional strategies (e.g., "If execution speed outpaces planning quality, decision fatigue rises; add weekly review checkpoints.").
   - DO NOT use generic safe statements (e.g., "There are both opportunities and risks ahead." -> BAN).
   - DO NOT predict fatal events or exact extreme dates. Focus on flow, trends, and risk management.

2. CONTENT RATIO:
   - 30% Core Nature / Psychology (Keep concise).
   - 40% Future Flow & Predictions (Life Timeline, Career, Love must be the longest chapters).
   - 20% Strategy & Actionable advice.
   - 10% Risk Management.

3. NO JARGON & NO RAW METRICS:
   - DO NOT use terms like 'varga_alignment', 'shadbala', 'lagna_lord', 'purushartha'.
   - DO NOT output raw grades, letters, or scores (NEVER write "stability grade D" or "score 45").
   - Use structural framing such as "From the chart structure", "Your baseline tendency suggests", "In social environments, this pattern unfolds as".

4. ABSOLUTELY NO REPETITIVE CLOSINGS OR CHATBOT TONE:
   - NEVER write repetitive boilerplate closings such as "This chapter concludes here" or "To summarize".
   - Do NOT summarize at the end of every chapter. End each paragraph naturally and definitively.
5. OUTPUT FORMAT CONTRACT (STRICT):
   - Every chapter MUST begin with a `##` markdown heading.
   - High-signal predictions MUST be prefixed with `**[KEY]**`.
   - Risk signals MUST be prefixed with `**[WARNING]**`.
   - Action items MUST be prefixed with `**[STRATEGY]**`.
   - Each paragraph MUST NOT exceed 4 sentences.
   - Closing boilerplate phrases (e.g., "This chapter concludes here") are strictly forbidden.
   - {lang_instruction}

Draft Narrative Blocks (Expand and synthesize these):
{blocks_json}

Atomic Interpretations (Core Identity Base):
Ascendant: {asc_text}
Sun: {sun_text}
Moon: {moon_text}

Structural Signals (Underlying Data):
{json.dumps(structural_summary, indent=2, ensure_ascii=False)}

Now generate the complete 15-chapter organic narrative report following the constraints exactly.
"""
