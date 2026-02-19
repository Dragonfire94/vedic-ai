import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from reportlab.platypus import Spacer
from starlette.requests import Request

import backend.main as main


def _request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/pdf",
        "raw_path": b"/pdf",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _chart_fixture() -> dict:
    return {
        "planets": {
            "Sun": {
                "rasi": {"name_kr": "양", "name": "Aries"},
                "house": 1,
                "nakshatra": {"name": "Ashwini"},
                "features": {"dignity": "strong"},
            }
        },
        "houses": {"ascendant": {"rasi": {"name": "Aries"}}},
        "vargas": {},
    }


class TestPdfNarrativeSelection(unittest.TestCase):
    def test_pdf_prefers_polished_reading_when_cached_for_same_hash(self):
        ai_reading = {
            "chapter_blocks_hash": "hash-polished",
            "chapter_blocks": {"Executive Summary": [{"title": "Deep", "summary": "Deterministic deep summary"}]},
            "reading": "stale reading",
            "model": "gpt-4o-mini",
        }
        polished_text = "# 1. Executive Summary\nPolished narrative"

        with patch.object(main, "PDF_FEATURE_AVAILABLE", True), \
             patch.object(main, "get_chart", return_value=_chart_fixture()), \
             patch.object(main, "get_ai_reading", new=AsyncMock(return_value=ai_reading)), \
             patch.object(main, "SouthIndianChart", side_effect=lambda *args, **kwargs: Spacer(1, 1)), \
             patch.object(main, "cache") as cache_mock, \
             patch.object(main, "parse_markdown_to_flowables", return_value=[Spacer(1, 1)]) as parse_mock, \
             patch.object(main, "render_report_payload_to_pdf", return_value=[Spacer(1, 1)]) as render_mock, \
             patch.object(main, "build_report_payload", new=Mock(side_effect=AssertionError("should not rebuild payload"))), \
             patch.object(main, "build_structural_summary", new=Mock(side_effect=AssertionError("should not rebuild summary"))):
            cache_mock.get.side_effect = lambda key: polished_text if key == main._polished_output_cache_key("hash-polished", "ko") else None
            response = asyncio.run(
                main.generate_pdf(
                    request=_request(),
                    year=1990,
                    month=1,
                    day=1,
                    hour=12.0,
                    lat=37.5665,
                    lon=126.9780,
                    include_d9=0,
                    include_ai=1,
                    language="ko",
                    analysis_mode="pro",
                    detail_level="full",
                )
            )

        self.assertEqual(response.media_type, "application/pdf")
        self.assertGreater(len(response.body), 0)
        parse_mock.assert_called_once_with(polished_text, unittest.mock.ANY)
        render_mock.assert_not_called()

    def test_pdf_uses_deep_deterministic_chapter_blocks_when_no_polished_cache(self):
        deep_payload = {
            "Executive Summary": [{"title": "Deep", "summary": "Deterministic deep summary from chapter blocks"}],
            "Final Summary": [{"title": "Deep Final", "analysis": "Long deterministic analysis"}],
        }
        ai_reading = {
            "chapter_blocks_hash": "hash-deterministic",
            "chapter_blocks": deep_payload,
            "reading": "fallback reading",
            "model": "gpt-4o-mini",
        }

        with patch.object(main, "PDF_FEATURE_AVAILABLE", True), \
             patch.object(main, "get_chart", return_value=_chart_fixture()), \
             patch.object(main, "get_ai_reading", new=AsyncMock(return_value=ai_reading)), \
             patch.object(main, "SouthIndianChart", side_effect=lambda *args, **kwargs: Spacer(1, 1)), \
             patch.object(main, "cache") as cache_mock, \
             patch.object(main, "parse_markdown_to_flowables", return_value=[Spacer(1, 1)]) as parse_mock, \
             patch.object(main, "render_report_payload_to_pdf", return_value=[Spacer(1, 1)]) as render_mock, \
             patch.object(main, "build_report_payload", new=Mock(side_effect=AssertionError("should not rebuild payload"))), \
             patch.object(main, "build_structural_summary", new=Mock(side_effect=AssertionError("should not rebuild summary"))):
            cache_mock.get.return_value = None
            response = asyncio.run(
                main.generate_pdf(
                    request=_request(),
                    year=1990,
                    month=1,
                    day=1,
                    hour=12.0,
                    lat=37.5665,
                    lon=126.9780,
                    include_d9=0,
                    include_ai=1,
                    language="ko",
                    analysis_mode="pro",
                    detail_level="full",
                )
            )

        self.assertEqual(response.media_type, "application/pdf")
        self.assertGreater(len(response.body), 0)
        parse_mock.assert_not_called()
        render_mock.assert_called_once()
        rendered_payload = render_mock.call_args.args[0]
        self.assertEqual(rendered_payload.get("chapter_blocks"), deep_payload)


if __name__ == "__main__":
    unittest.main()
