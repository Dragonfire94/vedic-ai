"""Tests for LLM audit hashing and log payload safety."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import patch

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


class TestLLMAudit(unittest.TestCase):
    def test_hashing_is_stable_for_identical_inputs(self) -> None:
        payload = {"year": 1990, "month": 1, "day": 1, "lat": 37.0, "lon": 127.0}
        h1 = main._sha256_hex(payload)
        h2 = main._sha256_hex(dict(payload))
        self.assertEqual(h1, h2)

    def test_hashing_changes_when_blocks_change(self) -> None:
        blocks_a = {"Executive Summary": [{"title": "A"}]}
        blocks_b = {"Executive Summary": [{"title": "B"}]}
        h1 = main._sha256_hex(blocks_a)
        h2 = main._sha256_hex(blocks_b)
        self.assertNotEqual(h1, h2)

    @patch("backend.main.llm_audit_logger.info")
    def test_audit_log_contains_no_raw_content(self, mock_info) -> None:
        secret = "very-sensitive-narrative-text"
        event = main._emit_llm_audit_event(
            request_id="req-1",
            chart_hash=main._sha256_hex({"chart": "x"}),
            chapter_blocks_hash=main._sha256_hex({"blocks": secret}),
            model_used="openai/gpt-4o",
            endpoint="/ai_reading",
        )
        self.assertIn("request_id", event)
        self.assertIn("chart_hash", event)
        self.assertIn("chapter_blocks_hash", event)
        self.assertIn("timestamp_utc", event)
        self.assertIn("model_used", event)
        self.assertIn("endpoint", event)
        logged_line = mock_info.call_args[0][0]
        parsed = json.loads(logged_line)
        self.assertEqual(set(parsed.keys()), {"request_id", "chart_hash", "chapter_blocks_hash", "timestamp_utc", "model_used", "endpoint"})
        self.assertNotIn(secret, logged_line)


if __name__ == "__main__":
    unittest.main()
