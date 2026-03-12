"""Tests for current LLM token and payload guardrails."""

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
from fastapi import HTTPException
from starlette.requests import Request


class TestLLMTokenLimits(unittest.TestCase):
    def test_defined_limits(self) -> None:
        self.assertEqual(main.AI_MAX_TOKENS_AI_READING, 18000)
        self.assertEqual(main.AI_MAX_TOKENS_PDF, 8000)
        self.assertEqual(main.AI_MAX_TOKENS_HARD_LIMIT, 22000)

    def test_resolve_llm_max_tokens_rejects_above_hard_limit(self) -> None:
        with self.assertRaises(HTTPException):
            main._resolve_llm_max_tokens(
                main.AI_MAX_TOKENS_HARD_LIMIT + 1,
                main.AI_MAX_TOKENS_AI_READING,
            )

    def test_resolve_llm_max_tokens_falls_back_to_default_for_invalid_values(self) -> None:
        self.assertEqual(
            main._resolve_llm_max_tokens(0, main.AI_MAX_TOKENS_AI_READING),
            main.AI_MAX_TOKENS_AI_READING,
        )
        self.assertEqual(
            main._resolve_llm_max_tokens("not-a-number", main.AI_MAX_TOKENS_PDF),
            main.AI_MAX_TOKENS_PDF,
        )

    def test_life_cycle_product_token_policy_uses_default_without_query_override(self) -> None:
        request = Request({"type": "http", "query_string": b""})
        self.assertEqual(
            main._resolve_product_llm_max_tokens(
                request=request,
                product_type="life_cycle",
                raw_value=main.AI_MAX_TOKENS_AI_READING,
            ),
            main.LIFE_CYCLE_AI_MAX_TOKENS_DEFAULT,
        )

    def test_life_cycle_product_token_policy_rejects_above_product_cap(self) -> None:
        request = Request({"type": "http", "query_string": b"llm_max_tokens=9001"})
        with self.assertRaises(HTTPException):
            main._resolve_product_llm_max_tokens(
                request=request,
                product_type="life_cycle",
                raw_value=main.LIFE_CYCLE_AI_MAX_TOKENS_HARD_CAP + 1,
            )

    def test_openai_payload_uses_max_completion_tokens(self) -> None:
        payload = main._build_openai_payload(
            model="openai/gpt-4o-mini",
            system_message="s",
            user_message="u",
            max_completion_tokens=1500,
        )
        self.assertEqual(payload["model"], "openai/gpt-4o-mini")
        self.assertEqual(payload["max_completion_tokens"], 1500)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["messages"][0]["content"], "s")
        self.assertEqual(payload["messages"][1]["content"], "u")


if __name__ == "__main__":
    unittest.main()
