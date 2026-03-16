from __future__ import annotations

from backend.llm_service import _PREMIUM_12_KEYS, build_llm_structural_prompt


def _chapter_blocks_fixture() -> dict[str, list[dict[str, str]]]:
    return {
        key: [
            {
                "title": key,
                "summary": f"{key} summary",
                "analysis": f"{key} analysis",
                "implication": f"{key} implication",
                "examples": f"{key} examples",
            }
        ]
        for key in _PREMIUM_12_KEYS
    }


def test_longform_prompt_uses_json_only_input_and_omits_draft_blocks() -> None:
    prompt = build_llm_structural_prompt(
        {},
        language="ko",
        chapter_blocks=_chapter_blocks_fixture(),
        pre_llm_input_mode_override="json_only",
        suppress_draft_narrative_blocks=True,
        suppress_chapter_draft_text=True,
        route_profile="life_cycle_longform_v1",
    )

    assert "Draft Narrative Blocks (Sanitized):\n(omitted by render_profile override)" in prompt
    assert "Chapter Draft (KO, prompt-serialized):\n(omitted by render_profile override)" in prompt
    assert "### SANITIZED_DRAFT_READING" not in prompt
    assert "[life_cycle_longform_v1 전용 규칙]" in prompt
    assert "영어 행성명(Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn) 직접 노출 금지." in prompt
