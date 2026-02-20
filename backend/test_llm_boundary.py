"""Boundary tests for deterministic LLM input contract."""

from __future__ import annotations

import os
import sys
import types
import unittest
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def _missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


if _missing("dotenv"):
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if _missing("httpx"):
    class _Timeout:
        def __init__(self, *args, **kwargs):
            pass

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["httpx"] = types.SimpleNamespace(AsyncClient=_AsyncClient, Timeout=_Timeout)

if _missing("swisseph"):
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

if _missing("pytz"):
    class _TZ:
        def utcoffset(self, dt):
            class _Offset:
                def total_seconds(self):
                    return 0
            return _Offset()

    sys.modules["pytz"] = types.SimpleNamespace(timezone=lambda name: _TZ(), utc=None)

if _missing("fastapi"):
    fastapi_mod = types.ModuleType("fastapi")

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass
        def add_middleware(self, *args, **kwargs):
            return None
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn

    def _identity(default=None, **kwargs):
        return default

    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Query = _identity
    fastapi_mod.Response = object
    fastapi_mod.HTTPException = Exception
    fastapi_mod.Body = _identity
    fastapi_mod.Request = object
    fastapi_mod.Header = _identity
    sys.modules["fastapi"] = fastapi_mod
    sys.modules["fastapi.middleware"] = types.ModuleType("fastapi.middleware")
    cors_mod = types.ModuleType("fastapi.middleware.cors")
    cors_mod.CORSMiddleware = object
    sys.modules["fastapi.middleware.cors"] = cors_mod

if _missing("pydantic"):
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.AliasChoices = lambda *args, **kwargs: None
    pydantic_mod.BaseModel = object
    pydantic_mod.ConfigDict = dict
    pydantic_mod.Field = lambda *args, **kwargs: None
    pydantic_mod.model_validator = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["pydantic"] = pydantic_mod

if _missing("reportlab"):
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

if _missing("timezonefinder"):
    class _TimezoneFinder:
        def timezone_at(self, **kwargs):
            return "UTC"

    sys.modules["timezonefinder"] = types.SimpleNamespace(TimezoneFinder=_TimezoneFinder)

if _missing("openai"):
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


    def test_accepts_key_forecast_fragment_key(self) -> None:
        blocks = self._valid_blocks()
        blocks["Confidence & Forecast"][0]["key_forecast"] = "career shift: high-signal likelihood 72%"
        out = main._validate_deterministic_llm_blocks(blocks)
        self.assertIn("key_forecast", out["Confidence & Forecast"][0])

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


    def test_pdf_render_includes_key_forecast_flowables(self) -> None:
        class _FakeParagraph:
            def __init__(self, text, style=None):
                self.text = str(text)
            def getPlainText(self):
                return self.text

        class _FakeTable:
            def __init__(self, data, colWidths=None, rowHeights=None):
                self.data = data
            def setStyle(self, style):
                return None

        class _FakeSpacer:
            def __init__(self, *args, **kwargs):
                pass

        class _FakeStyle:
            def __init__(self, font_size=10, leading=12):
                self.fontSize = font_size
                self.leading = leading

        original_paragraph = main.Paragraph
        original_table = main.Table
        original_spacer = main.Spacer
        original_pagestyle = main.ParagraphStyle
        original_colors = main.colors
        original_a4 = main.A4
        original_tablestyle = main.TableStyle
        try:
            main.Paragraph = _FakeParagraph
            main.Table = _FakeTable
            main.Spacer = _FakeSpacer
            main.ParagraphStyle = lambda *args, **kwargs: _FakeStyle(font_size=14, leading=20)
            main.colors = types.SimpleNamespace(HexColor=lambda value: value, white="white")
            main.A4 = (595.0, 842.0)
            main.TableStyle = lambda *args, **kwargs: None

            styles = {
                "SummaryLead": _FakeStyle(font_size=12, leading=16),
                "Subtitle": _FakeStyle(),
                "ChapterTitle": _FakeStyle(),
                "InsightSpike": _FakeStyle(),
                "Body": _FakeStyle(),
                "Small": _FakeStyle(),
                "TableHeaderCell": _FakeStyle(),
                "TableBodyCell": _FakeStyle(),
            }
            config = {"colors": {}, "page": {}, "chapters": {}}
            payload = {
                "chapter_blocks": {
                    "Confidence & Forecast": [
                        {"title": "Forecast", "key_forecast": "career shift: high-signal likelihood 74%"}
                    ]
                }
            }
            flowables = main.render_report_payload_to_pdf(payload, styles, config)
            paragraph_texts = [f.getPlainText() for f in flowables if hasattr(f, "getPlainText")]
            self.assertTrue(any("Forecast Snapshot" in text for text in paragraph_texts))
            self.assertTrue(any("career shift: high-signal likelihood 74%" in text for text in paragraph_texts))
        finally:
            main.Paragraph = original_paragraph
            main.Table = original_table
            main.Spacer = original_spacer
            main.ParagraphStyle = original_pagestyle
            main.colors = original_colors
            main.A4 = original_a4
            main.TableStyle = original_tablestyle


if __name__ == "__main__":
    unittest.main()
