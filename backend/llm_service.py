import os
import logging
from typing import Any, Optional

from backend.report_engine import _get_atomic_chart_interpretations

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
    atomic_interpretations = _get_atomic_chart_interpretations(structural_summary if isinstance(structural_summary, dict) else {})
    system_message = "Follow the user prompt exactly."
    user_message = build_llm_structural_prompt(
        structural_payload,
        language=language,
        atomic_interpretations=atomic_interpretations,
        chapter_blocks=validated,
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
) -> str:
    import json

    atomic = atomic_interpretations if isinstance(atomic_interpretations, dict) else {}
    asc_text = str(atomic.get("asc", "")).strip()
    sun_text = str(atomic.get("sun", "")).strip()
    moon_text = str(atomic.get("moon", "")).strip()
    blocks_json = json.dumps(chapter_blocks, indent=2, ensure_ascii=False) if chapter_blocks else "{}"

    return f"""
You are a wise, warm, and brutally honest Korean astrologer, like a brilliant older sibling who truly cares and
is not afraid to tell the truth.

Your task is to read the structural data below and write a deeply personal, story-driven Vedic astrology report in
Korean. This is NOT a consulting report. It should read like a heartfelt and insightful conversation.

VOICE & TONE (CRITICAL):
- Write in natural, flowing Korean. Casual but sincere, closer to "솔직히 말할게요" than "분석 결과".
- Address the reader directly as "당신" throughout.
- Use vivid Korean metaphors and analogies (e.g., "밑 빠진 독에 물 붓기", "엔진은 좋은데 핸들이 없는 차").
- Be honest about weaknesses, but always with warmth and a realistic path forward.
- Predictions must be strong and specific: include timeframes (e.g., "앞으로 1~2년"), concrete patterns, and what to watch.
- Never use empty lines like "기회와 위험이 공존합니다".
- In chapter openings, naturally weave astrology-linked identity cues in plain Korean.
  Example style: "전갈자리 기질이 강한 당신은...", "달의 결이 예민한 당신은..."
  Do this naturally in multiple chapters so the text clearly feels like an astrology report.
- Prefer structure-first language over direct zodiac marketing tone.
  Use direct zodiac labels sparingly, and only when they clearly match the chart logic.
  Better style: "전통에 반응하는 사고 구조", "기존 질서에 질문을 던지는 경향".
- You MAY reference chart placements, but ONLY as everyday
  language - never as jargon.
  Good: "전갈자리 기질이 강한 당신은 본능적으로 상대의 숨겨진 면을 감지한다"
  Good: "태양이 감수성 강한 자리에 있어서, 당신은 칭찬보다 인정에 더 목말라 있다"
  Good: "달의 위치가 말과 정보 쪽이라 감정을 글이나 대화로 풀어야 편안해진다"
  Bad:  "태양이 게자리 9하우스에 위치하여 종교적 성향이 강합니다" (jargon)
  Bad:  "라후가 전갈자리 1하우스에 있어 정체성 혼란이 있습니다" (jargon)
- Use chart signals to make the reader feel "이게 나 얘기잖아"
  - not to educate them about astrology.

STRUCTURE:
- Write exactly 15 chapters.
- Each chapter must be substantial: minimum 3 paragraphs,
  each paragraph minimum 4~6 sentences.
- Total report length must be at least 4,000 Korean characters.
  The life timeline, career, and love chapters must each be
  at least 600 characters on their own.
- Do NOT compress or summarize. Expand every point fully.
- Each chapter must start with a creative Korean markdown heading using `##`.
- Chapter titles must feel bold and memorable, not plain explanatory labels.
  Prefer evocative title styles like "겉은 말랑, 속은 강철" over textbook-style headings.
- Do NOT use English chapter titles.
- Do NOT use [KEY], [WARNING], [STRATEGY] tags; weave emphasis naturally into narrative flow.
- Paragraph length should feel natural, not mechanical.
- End chapters naturally; avoid formulaic closings.
- Include one dedicated money/finance chapter as its own chapter
  (do not absorb it into career). Focus on earning/spending/leakage/risk habits and practical controls.
- In timeline chapters, explain cycle transitions clearly:
  recent phase -> current phase -> next phase.
  Do not only list years; explain why the phase is shifting in structural terms.
- Include 1-2 firm, non-vague warning lines where needed.
  Example style: "이 패턴을 고치지 않으면 3년 내 현실 비용을 치를 가능성이 높다."
  Keep it honest, specific, and still constructive.
- Include one explicit growth-vision section that balances risk management:
  "3년 후 확장 포인트", "장기 브랜딩 방향", "최상위 버전의 당신" 같은 미래 확장 청사진을 제시할 것.
- 반드시 한글(Hangul)로만 작성할 것. 영어 문장 출력 금지.
  (language 파라미터가 'ko'가 아닌 경우에도 현재는 한국어 전용으로 운영)

CONTENT PRIORITIES:
- 심리와 성격 (30%): 핵심 본성, 내면 갈등, 행동 패턴
- 미래 흐름과 예측 (40%): 인생 타이밍, 커리어, 연애 (가장 길게)
- 전략과 실천 (20%): 구체적이고 실행 가능한 조언
- 리스크 관리 (10%): 솔직한 경고 + 회복 경로

BANNED:
- Generic closings ("앞으로도 화이팅하세요", "이 리포트가 도움이 되길 바랍니다")
- Astrological jargon (varga_alignment, shadbala, lagna_lord, purushartha)
- Raw scores or grades ("안정성 D등급", "점수 45")
- English sentences in final output
- Repetitive chapter cadence

Analysis scaffolding (internal thinking order):
STEP 1  Identify dominant forces
STEP 2  Identify internal tensions and imbalances
STEP 3  Identify execution and behavioral architecture
STEP 4  Identify structural trajectory and life-pattern tendencies
STEP 5  Synthesize unified interpretation

Draft Narrative Blocks (expand and synthesize these):
{blocks_json}

Core Chart Identity:
Ascendant: {asc_text}
Sun: {sun_text}
Moon: {moon_text}

Structural signals:
{json.dumps(structural_summary, indent=2, ensure_ascii=False)}

Now write the complete 15-chapter report. Make it feel like it was written just for this person.
"""

