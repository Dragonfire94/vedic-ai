from __future__ import annotations

from backend.commercial_surface_renderer import render_commercial_markdown_from_chapter_blocks


def test_render_commercial_markdown_from_chapter_blocks_deterministic() -> None:
    chapter_blocks = {
        "Current Phase": [
            {
                "title": "Current Phase - 해석 블록 1",
                "summary": "요약 A",
                "analysis": "분석 A",
                "implication": "함의 A",
                "examples": "예시 A",
            }
        ],
        "Executive Diagnosis": [
            {
                "title": "핵심 진단",
                "summary": "요약 B",
                "analysis": "분석 B",
                "implication": "함의 B",
                "examples": "예시 B",
            }
        ],
    }
    out1 = render_commercial_markdown_from_chapter_blocks(chapter_blocks)
    out2 = render_commercial_markdown_from_chapter_blocks(chapter_blocks)

    assert out1 == out2
    assert "## [Executive Diagnosis] Executive Diagnosis" in out1
    assert "## [Current Phase] 현재 흐름의 국면" in out1
    assert out1.index("## [Executive Diagnosis]") < out1.index("## [Current Phase]")
    assert "해석 블록 1" not in out1
    assert "<!-- chapter_key:" not in out1
    assert "# 1." not in out1
    assert "### Action Steps" in out1


def test_renderer_applies_action_steps_only_to_core_7_chapters() -> None:
    chapter_blocks = {
        "Executive Diagnosis": [
            {"summary": "요약 문장", "analysis": "분석 문장", "implication": "함의 문장", "examples": "- 예시 하나"},
        ],
        "Current Phase": [
            {"summary": "요약 문장", "analysis": "분석 문장", "implication": "함의 문장", "examples": "- 예시 하나"},
        ],
    }
    out = render_commercial_markdown_from_chapter_blocks(chapter_blocks)
    executive_start = out.find("## [Executive Diagnosis]")
    current_start = out.find("## [Current Phase]")
    executive_section = out[executive_start:current_start]
    current_section = out[current_start:]
    assert "### Action Steps" not in executive_section
    assert "### Action Steps" in current_section


def test_renderer_ensures_minimum_two_body_paragraphs_per_chapter() -> None:
    chapter_blocks = {
        "Current Phase": [
            {"summary": "짧은 문장", "analysis": "", "implication": "", "examples": ""},
        ]
    }
    out = render_commercial_markdown_from_chapter_blocks(chapter_blocks)
    section = out.split("## [Current Phase] 현재 흐름의 국면", 1)[1]
    paragraphs = [p for p in section.split("\n\n") if p.strip() and not p.strip().startswith("###") and not p.strip().startswith("- ")]
    assert len(paragraphs) >= 2
