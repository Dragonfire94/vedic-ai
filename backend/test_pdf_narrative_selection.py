import unittest
from unittest.mock import patch

import backend.main as main


class TestPdfNarrativeSelection(unittest.TestCase):
    def test_pdf_prefers_polished_reading_when_cached_for_same_hash(self):
        ai_reading = {
            "chapter_blocks_hash": "hash-polished",
            "chapter_blocks": {"Executive Diagnosis": [{"title": "Deep", "summary": "Deterministic deep summary"}]},
            "reading": "stale reading",
            "model": "gpt-4o-mini",
        }
        polished_text = "# 1. Executive Diagnosis\nPolished narrative"

        with patch.object(main, "load_polished_reading_from_cache", return_value=polished_text),              patch.object(main, "_reading_style_error_codes", return_value=[]):
            resolved = main._resolve_pdf_narrative_content(ai_reading, "ko")

        self.assertEqual(resolved["source"], "polished")
        self.assertEqual(resolved.get("text_source"), "polished")
        self.assertIn("Polished narrative", resolved["polished_text"])
        self.assertEqual(
            resolved["report_payload"].get("chapter_blocks"),
            ai_reading["chapter_blocks"],
        )

    def test_pdf_uses_deep_deterministic_chapter_blocks_when_no_polished_cache(self):
        deep_payload = {
            "Executive Diagnosis": [{"title": "Deep", "summary": "Deterministic deep summary from chapter blocks"}],
            "Final Integration": [{"title": "Deep Final", "analysis": "Long deterministic analysis"}],
        }
        ai_reading = {
            "chapter_blocks_hash": "hash-deterministic",
            "chapter_blocks": deep_payload,
            "reading": "fallback reading",
            "model": "gpt-4o-mini",
        }

        with patch.object(main, "load_polished_reading_from_cache", return_value=None):
            resolved = main._resolve_pdf_narrative_content(ai_reading, "ko")

        self.assertEqual(resolved["source"], "deterministic")
        self.assertIsNone(resolved["polished_text"])
        self.assertEqual(resolved["report_payload"].get("chapter_blocks"), deep_payload)


if __name__ == "__main__":
    unittest.main()
