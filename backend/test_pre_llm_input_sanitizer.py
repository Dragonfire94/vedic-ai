from __future__ import annotations

import json
from pathlib import Path

from backend.pre_llm_input_sanitizer import (
    render_chapter_blocks_pre_llm,
    sanitize_chapter_blocks_for_llm,
)
from backend.report_config import PREMIUM_12_CHAPTER_ORDER


def _empty_blocks() -> dict[str, list[dict]]:
    return {chapter: [] for chapter in PREMIUM_12_CHAPTER_ORDER}


def test_sanitize_removes_sidereal_lahiri_tokens() -> None:
    blocks = _empty_blocks()
    blocks["Executive Diagnosis"] = [
        {
            "title": "Executive Diagnosis - 해석 블록 1",
            "summary": "시데리얼(항성황도), 라히리 기준의 처녀자리 상승 흐름입니다.",
            "analysis": "sidereal Lahiri ayanamsa token test",
            "implication": "아얀암샤 기준 문구",
            "examples": "라히리기준의 처녀자리 상승",
        }
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    block = out["Executive Diagnosis"][0]
    merged = " ".join(str(block.get(k, "")) for k in ("title", "summary", "analysis", "implication", "examples"))

    assert block.get("title", "") == ""
    assert "처녀자리 라그나(Lagna)" in merged
    assert "시데리얼" not in merged
    assert "항성황도" not in merged
    assert "라히리" not in merged
    assert "sidereal" not in merged.lower()
    assert "ayanamsa" not in merged.lower()


def test_sanitize_dedupes_composite_block_content() -> None:
    blocks = _empty_blocks()
    template = {
        "title": "Current Phase - 해석 블록 1",
        "summary": "중복 문장 테스트",
        "analysis": "중복 분석",
        "implication": "중복 함의",
        "examples": "중복 예시",
    }
    blocks["Current Phase"] = [dict(template), dict(template)]

    out = sanitize_chapter_blocks_for_llm(blocks)
    assert len(out["Current Phase"]) == 1


def test_render_pre_llm_markdown_has_clean_headings_and_no_legacy_noise() -> None:
    blocks = _empty_blocks()
    blocks["Executive Diagnosis"] = [
        {
            "title": "핵심 진단",
            "summary": "요약 문장",
            "analysis": "분석 문장",
            "implication": "함의 문장",
            "examples": "예시 문장",
        }
    ]
    blocks["Current Phase"] = [
        {
            "title": "Current Phase - 해석 블록 2",
            "summary": "현재 흐름 요약",
            "analysis": "",
            "implication": "",
            "examples": "",
        }
    ]

    sanitized = sanitize_chapter_blocks_for_llm(blocks)
    out = render_chapter_blocks_pre_llm(sanitized)

    assert "## [Executive Diagnosis]" in out
    assert "## [Current Phase]" in out
    assert "### 핵심 진단" in out
    assert "# 1." not in out
    assert "<!-- chapter_key:" not in out
    assert "해석 블록" not in out


def test_sanitize_is_idempotent() -> None:
    blocks = _empty_blocks()
    blocks["Mid-Term Direction"] = [
        {
            "title": "중기 흐름",
            "summary": "시데리얼(항성황도), 라히리 기준의 물병자리 상승",
            "analysis": "라히리 기준 문장",
            "implication": "sidereal token",
            "examples": "",
        }
    ]

    once = sanitize_chapter_blocks_for_llm(blocks)
    twice = sanitize_chapter_blocks_for_llm(once)
    assert once == twice


def test_sanitize_removes_leading_comma_after_clause_deletion() -> None:
    blocks = _empty_blocks()
    blocks["Executive Diagnosis"] = [
        {
            "title": "핵심 진단",
            "summary": ", 달이 물고기자리에 있을 때 감정의 결이 깊어집니다.",
            "analysis": "",
            "implication": "",
            "examples": "",
        }
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    summary = out["Executive Diagnosis"][0]["summary"]
    assert summary.startswith("달이 물고기자리에 있을 때")
    assert not summary.startswith(",")

    twice = sanitize_chapter_blocks_for_llm(out)
    assert out == twice


def test_subset_prefix_dedupe_keeps_longer_block_only() -> None:
    blocks = _empty_blocks()
    short_summary = ("이 문장은 중기 흐름을 설명하는 핵심 문단입니다. " * 14).strip()
    long_summary = (
        short_summary
        + " 추가 문단으로 실행 순서와 리스크 관리 포인트를 더 구체적으로 설명합니다. "
        + "세부 단계와 우선순위를 포함해 실제 적용 맥락을 보강합니다."
    )
    blocks["Mid-Term Direction"] = [
        {
            "title": "중기 흐름 요약",
            "summary": short_summary,
            "analysis": "",
            "implication": "",
            "examples": "",
        },
        {
            "title": "중기 흐름 확장",
            "summary": long_summary,
            "analysis": "",
            "implication": "",
            "examples": "",
        },
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    assert len(out["Mid-Term Direction"]) == 1
    assert "추가 문단으로 실행 순서" in out["Mid-Term Direction"][0]["summary"]


def test_subset_dedupe_does_not_drop_short_overlap_blocks() -> None:
    blocks = _empty_blocks()
    overlap = "공통 시작 문장입니다."
    blocks["Current Phase"] = [
        {
            "title": "흐름 A",
            "summary": overlap + " " + ("A 경로 문장 " * 10),
            "analysis": "",
            "implication": "",
            "examples": "",
        },
        {
            "title": "흐름 B",
            "summary": overlap + " " + ("B 경로 문장 " * 10),
            "analysis": "",
            "implication": "",
            "examples": "",
        },
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    assert len(out["Current Phase"]) == 2


def test_sanitize_rewrites_shadbala_avastha_heading_and_terms() -> None:
    blocks = _empty_blocks()
    blocks["Final Integration"] = [
        {
            "title": "Shadbala & Avastha Snapshot",
            "summary": "Remedy Priority by Shadbala를 기반으로 Avastha를 함께 봅니다.",
            "analysis": "Shadbala는 강약에 대한 기술용어입니다.",
            "implication": "",
            "examples": "",
        }
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    block = out["Final Integration"][0]
    merged = " ".join(str(block.get(k, "")) for k in ("title", "summary", "analysis", "implication", "examples"))
    assert block.get("title") == "강약 스냅샷"
    assert "보완 우선순위" in merged
    assert "shadbala" not in merged.lower()
    assert "avastha" not in merged.lower()


def test_summary_only_containment_dedupe_drops_later_subset_block() -> None:
    blocks = _empty_blocks()
    long_summary = (
        "이 문단은 중기 방향의 핵심 흐름과 실행 순서, 리스크 조정 포인트를 함께 설명합니다. "
        "우선순위를 한 번에 늘리기보다 단계적으로 줄여야 한다는 점을 강조하며, "
        "관계/커리어/에너지 리듬 사이의 연결을 같은 맥락에서 묶어 해석합니다. "
        "실행 타이밍을 좁혀서 적용하면 변동성을 낮추는 데 도움이 됩니다."
    )
    long_summary = f"{long_summary} {long_summary}"
    short_subset = long_summary[:220]
    blocks["Mid-Term Direction"] = [
        {
            "title": "확장 블록",
            "summary": long_summary,
            "analysis": "",
            "implication": "",
            "examples": "",
        },
        {
            "title": "요약 블록",
            "summary": short_subset,
            "analysis": "",
            "implication": "",
            "examples": "",
        },
    ]

    out = sanitize_chapter_blocks_for_llm(blocks)
    assert len(out["Mid-Term Direction"]) == 1
    assert out["Mid-Term Direction"][0]["title"] == "확장 블록"


def test_fixture_sample_rewrites_shadbala_snapshot() -> None:
    fixture_path = Path(__file__).resolve().parent / "tests" / "fixtures" / "chapter_blocks_pre_llm_sample.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    out = sanitize_chapter_blocks_for_llm(payload)
    block = out["Final Integration"][0]
    assert block.get("title") == "강약 스냅샷"
