from __future__ import annotations

import sys
from pathlib import Path

from backend.main import _finalize_ai_reading_result
from backend.report_config import PREMIUM_12_CHAPTER_ORDER

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cheap_validation_gate import _select_commercial_scan_surface  # noqa: E402


def _chapter_blocks_fixture() -> dict[str, list[dict[str, str]]]:
    blocks: dict[str, list[dict[str, str]]] = {}
    for chapter in PREMIUM_12_CHAPTER_ORDER:
        blocks[chapter] = [
            {
                "title": f"{chapter} - 해석 블록 1",
                "summary": "시데리얼(항성황도), 라히리 기준의 처녀자리 상승이라는 표현이 섞여 있습니다.",
                "analysis": "분석 문장",
                "implication": "함의 문장",
                "examples": "예시 문장",
            }
        ]
    return blocks


def test_finalize_always_populates_polished_reading_without_llm() -> None:
    result = {
        "reading": "## Executive Diagnosis\n\n<!-- chapter_key: Executive Diagnosis -->\n\nExecutive Diagnosis - 해석 블록 1\n",
        "polished_reading": None,
        "chapter_blocks": _chapter_blocks_fixture(),
        "summary": {"language": "ko"},
        "debug_info": {},
    }
    out = _finalize_ai_reading_result(
        result,
        chart_context_min={},
        include_debug_payload=True,
        pipeline_version="chapter_blocks_v2",
        production_mode=False,
    )
    polished = out.get("polished_reading")
    assert isinstance(polished, str) and polished.strip()
    assert "시데리얼" not in polished
    assert "항성황도" not in polished
    assert "라히리" not in polished
    assert "<!-- chapter_key:" not in polished
    assert "해석 블록" not in polished
    assert isinstance(out.get("vedic_technical_data"), dict)
    assert isinstance(out.get("vedic_technical_reading"), str)


def test_scan_surface_prefers_polished_reading() -> None:
    text, source = _select_commercial_scan_surface(
        {
            "reading": "raw-reading",
            "polished_reading": "polished-reading",
        }
    )
    assert source == "polished_reading"
    assert text == "polished-reading"


def test_finalize_does_not_sanitize_technical_appendix_fields(monkeypatch) -> None:
    def _fake_build_vedic_technical_artifacts(_chart_context: dict, *, pipeline_version: str | None = None):
        return (
            {"availability": {"ok": True}, "pipeline_version": pipeline_version},
            "### Shadbala\n\nAvastha technical appendix should remain.",
        )

    monkeypatch.setattr("backend.main.build_vedic_technical_artifacts", _fake_build_vedic_technical_artifacts)

    result = {
        "reading": "## [Final Integration] Shadbala & Avastha Snapshot\n\n### Remedy Priority by Shadbala\n\n본문",
        "polished_reading": None,
        "summary": {"language": "ko"},
        "debug_info": {},
    }
    out = _finalize_ai_reading_result(
        result,
        chart_context_min={},
        include_debug_payload=True,
        pipeline_version="chapter_blocks_v2",
        production_mode=False,
    )

    polished = str(out.get("polished_reading") or "")
    technical = str(out.get("vedic_technical_reading") or "")
    assert "Shadbala" not in polished
    assert "Avastha" not in polished
    assert "강약 스냅샷" in polished
    assert "Shadbala" in technical
    assert "Avastha" in technical
