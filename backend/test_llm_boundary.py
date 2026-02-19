"""Boundary tests for deterministic LLM input contract."""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(
        SUN=0,
        MOON=1,
        MARS=2,
        MERCURY=3,
        JUPITER=4,
        VENUS=5,
        SATURN=6,
        MEAN_NODE=7,
        julday=lambda *args, **kwargs: 0.0,
    )
    sys.modules["swisseph"] = swe_stub

if "pytz" not in sys.modules:
    class _TZ:
        def utcoffset(self, dt):
            class _Offset:
                def total_seconds(self):
                    return 0
            return _Offset()

    sys.modules["pytz"] = types.SimpleNamespace(timezone=lambda name: _TZ(), utc=None)

if "reportlab" not in sys.modules:
    sys.modules["reportlab"] = types.ModuleType("reportlab")
    sys.modules["reportlab.lib"] = types.ModuleType("reportlab.lib")
    sys.modules["reportlab.lib.colors"] = types.SimpleNamespace(black=None)
    sys.modules["reportlab.lib.pagesizes"] = types.SimpleNamespace(A4=None)
    sys.modules["reportlab.lib.styles"] = types.SimpleNamespace(getSampleStyleSheet=lambda: None, ParagraphStyle=object)
    sys.modules["reportlab.lib.units"] = types.SimpleNamespace(cm=1)
    sys.modules["reportlab.lib.enums"] = types.SimpleNamespace(TA_CENTER=0, TA_LEFT=0, TA_JUSTIFY=0)
    sys.modules["reportlab.platypus"] = types.SimpleNamespace(SimpleDocTemplate=object, Paragraph=object, Spacer=object, Table=object, TableStyle=object, PageBreak=object, KeepTogether=object, Flowable=object)
    sys.modules["reportlab.pdfbase"] = types.ModuleType("reportlab.pdfbase")
    sys.modules["reportlab.pdfbase.pdfmetrics"] = types.SimpleNamespace(registerFont=lambda *args, **kwargs: None)
    sys.modules["reportlab.pdfbase.ttfonts"] = types.SimpleNamespace(TTFont=object)

if "timezonefinder" not in sys.modules:
    class _TimezoneFinder:
        def timezone_at(self, **kwargs):
            return "UTC"

    sys.modules["timezonefinder"] = types.SimpleNamespace(TimezoneFinder=_TimezoneFinder)

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(AsyncOpenAI=lambda *args, **kwargs: object())

from backend import main


class TestLLMBoundary(unittest.TestCase):
    def _valid_blocks(self) -> dict[str, list[dict[str, str]]]:
        return {
            chapter: [{"title": chapter, "summary": "s", "analysis": "a", "implication": "i"}]
            for chapter in main.REPORT_CHAPTERS
        }

    def test_valid_chapter_blocks_pass(self) -> None:
        blocks = self._valid_blocks()
        out = main._validate_deterministic_llm_blocks(blocks)
        self.assertEqual(set(out.keys()), set(main.REPORT_CHAPTERS))

    def test_rejects_unknown_fragment_keys(self) -> None:
        blocks = self._valid_blocks()
        blocks["Executive Summary"][0]["engine"] = "forbidden"
        with self.assertRaises(ValueError):
            main._validate_deterministic_llm_blocks(blocks)

    def test_rejects_missing_chapters(self) -> None:
        blocks = self._valid_blocks()
        blocks.pop("Appendix (Optional)")
        with self.assertRaises(ValueError):
            main._validate_deterministic_llm_blocks(blocks)

    def test_ai_input_requires_structured_markers(self) -> None:
        with self.assertRaises(ValueError):
            main._build_ai_input("raw non-structured prompt", language="ko")

    def test_build_context_from_payload_is_structured(self) -> None:
        payload = {"chapter_blocks": self._valid_blocks()}
        context, chapter_hash = main._build_llm_structured_context(payload)
        self.assertIn(main.STRUCTURED_BLOCKS_BEGIN_TAG, context)
        self.assertIn(main.STRUCTURED_BLOCKS_END_TAG, context)
        self.assertIsInstance(chapter_hash, str)
        self.assertEqual(len(chapter_hash), 64)


if __name__ == "__main__":
    unittest.main()
