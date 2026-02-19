"""Tests for LLM token and generation parameter guardrails."""

from __future__ import annotations

import os
import re
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
from fastapi import HTTPException


class TestLLMTokenLimits(unittest.TestCase):
    def test_defined_limits(self) -> None:
        self.assertEqual(main.AI_MAX_TOKENS_AI_READING, 1500)
        self.assertEqual(main.AI_MAX_TOKENS_PDF, 2000)
        self.assertEqual(main.AI_MAX_TOKENS_HARD_LIMIT, 3000)

    def test_resolve_llm_max_tokens_rejects_above_hard_limit(self) -> None:
        with self.assertRaises(HTTPException):
            main._resolve_llm_max_tokens(3001, main.AI_MAX_TOKENS_AI_READING)

    def test_openai_payload_generation_params_are_fixed(self) -> None:
        payload = main._build_openai_payload(
            model="openai/gpt-4o-mini",
            system_message="s",
            user_message="u",
            max_tokens=1500,
        )
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["frequency_penalty"], 0)
        self.assertEqual(payload["presence_penalty"], 0)
        self.assertEqual(payload["max_tokens"], 1500)

    def test_source_has_no_max_tokens_literal_above_hard_limit(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), "main.py"), "r", encoding="utf-8") as handle:
            source = handle.read()
        values = [int(v) for v in re.findall(r"max_tokens\"?\s*[:=]\s*(\d+)", source)]
        self.assertTrue(all(v <= main.AI_MAX_TOKENS_HARD_LIMIT for v in values))


if __name__ == "__main__":
    unittest.main()
