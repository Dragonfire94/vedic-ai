from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest

from backend.llm_service import (
    _PRE_LLM_SECTION_DRAFT,
    _PRE_LLM_SECTION_JSON,
    _write_llm_prompt_snapshot_if_enabled,
    build_llm_structural_prompt,
    sanitize_prompt_text_last_mile,
)
from backend.report_config import PREMIUM_12_CHAPTER_ORDER

_WHOLE_PROMPT_BANNED_RE = re.compile(
    r"시데리얼|항성황도|라히리|아얀암샤|아얀암사|아야남사|"
    r"\bsidereal\b|\blahiri\b|\bayanamsa\b|"
    r"(?<![A-Za-z])shadbala(?![A-Za-z])|(?<![A-Za-z])avastha(?![A-Za-z])|Śadbala|Avasthā",
    re.IGNORECASE,
)


def _chapter_blocks_fixture() -> dict[str, list[dict[str, str]]]:
    blocks: dict[str, list[dict[str, str]]] = {}
    for chapter in PREMIUM_12_CHAPTER_ORDER:
        blocks[chapter] = [
            {
                "title": f"{chapter} - 해석 블록 1",
                "summary": "시데리얼(항성황도), 라히리 기준의 처녀자리 상승 문장",
                "analysis": "sidereal Lahiri token",
                "implication": "핵심 함의",
                "examples": "적용 예시",
            }
        ]
    return blocks


def _build_prompt(monkeypatch: pytest.MonkeyPatch, *, mode: str) -> str:
    monkeypatch.setenv("PRE_LLM_INPUT_MODE", mode)
    monkeypatch.setenv("PRE_LLM_FAILFAST", "1")
    return build_llm_structural_prompt(
        structural_summary={"signal": "test"},
        language="ko",
        chapter_blocks=_chapter_blocks_fixture(),
        semantic_signals={},
        narrative_mode="measured_growth",
        dasha_context={},
    )


def _extract_section_content(prompt: str, section_header: str) -> str:
    marker = f"{section_header}\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start = start + len(marker)
    next_header = prompt.find("\n### SANITIZED_", start)
    if next_header < 0:
        # Fall back to the legacy chapter block marker near the end of prompt.
        next_header = prompt.find("\nChapter Blocks (JSON):", start)
    if next_header < 0:
        next_header = len(prompt)
    return prompt[start:next_header]


def test_prompt_contains_sanitized_sections_in_both_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build_prompt(monkeypatch, mode="both")
    final_prompt = sanitize_prompt_text_last_mile(prompt)
    json_section = _extract_section_content(prompt, _PRE_LLM_SECTION_JSON)
    draft_section = _extract_section_content(prompt, _PRE_LLM_SECTION_DRAFT)
    assert _PRE_LLM_SECTION_JSON in prompt
    assert _PRE_LLM_SECTION_DRAFT in prompt
    assert "````json" in json_section
    assert "````md" in draft_section
    combined = f"{json_section}\n{draft_section}"
    assert "시데리얼" not in combined
    assert "항성황도" not in combined
    assert "라히리" not in combined
    assert "sidereal" not in combined.lower()
    assert "<!-- chapter_key:" not in combined
    assert "# 1." not in combined
    assert "해석 블록" not in combined
    assert "````json" in final_prompt
    assert "````md" in final_prompt
    assert _WHOLE_PROMPT_BANNED_RE.search(final_prompt) is None
    assert sanitize_prompt_text_last_mile(final_prompt) == final_prompt


def test_prompt_mode_json_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build_prompt(monkeypatch, mode="json_only")
    assert _PRE_LLM_SECTION_JSON in prompt
    assert _PRE_LLM_SECTION_DRAFT not in prompt


def test_prompt_mode_draft_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build_prompt(monkeypatch, mode="draft_only")
    assert _PRE_LLM_SECTION_DRAFT in prompt
    assert _PRE_LLM_SECTION_JSON not in prompt


def test_failfast_raises_when_selected_mode_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRE_LLM_INPUT_MODE", "draft_only")
    monkeypatch.setenv("PRE_LLM_FAILFAST", "1")
    monkeypatch.setattr("backend.llm_service.render_chapter_blocks_pre_llm", lambda _payload: "")
    with pytest.raises(RuntimeError, match="pre_llm_draft_section_empty"):
        build_llm_structural_prompt(
            structural_summary={"signal": "test"},
            language="ko",
            chapter_blocks=_chapter_blocks_fixture(),
            semantic_signals={},
            narrative_mode="measured_growth",
            dasha_context={},
        )


def test_json_section_is_not_empty_object_when_canonical_keys_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build_prompt(monkeypatch, mode="json_only")
    marker = f"{_PRE_LLM_SECTION_JSON}\n````json\n"
    start = prompt.find(marker)
    assert start >= 0
    end = prompt.find("\n````", start + len(marker))
    assert end > start
    json_payload = prompt[start + len(marker):end]
    parsed = json.loads(json_payload)
    assert isinstance(parsed, dict)
    assert "Executive Diagnosis" in parsed
    assert parsed != {}


def test_prompt_snapshot_write_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = _build_prompt(monkeypatch, mode="both")
    final_prompt = sanitize_prompt_text_last_mile(prompt)
    out_path = Path("logs") / f"test_llm_prompt_pre_call_{uuid.uuid4().hex}.txt"
    monkeypatch.setenv("LLM_WRITE_PROMPT_SNAPSHOT", "1")
    monkeypatch.setenv("LLM_PROMPT_SNAPSHOT_PATH", str(out_path))
    _write_llm_prompt_snapshot_if_enabled(final_prompt)
    assert out_path.exists()
    saved = out_path.read_text(encoding="utf-8")
    assert saved == final_prompt
    assert _PRE_LLM_SECTION_JSON in saved
    assert _WHOLE_PROMPT_BANNED_RE.search(saved) is None
    out_path.unlink(missing_ok=True)
