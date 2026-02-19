import unittest

import backend.report_engine as report_engine


class TestAtomicPrimaryNarrative(unittest.TestCase):
    def test_executive_summary_primary_uses_atomic_text(self):
        structural_summary = {
            "language": "ko",
            "ascendant_sign": "Leo",
            "sun_sign": "Sagittarius",
            "moon_sign": "Gemini",
            "chart_signature": {
                "ascendant_sign": "Leo",
                "sun_sign": "Sagittarius",
                "moon_sign": "Gemini",
            },
        }
        payload = report_engine.build_report_payload({"structural_summary": structural_summary, "language": "ko"})
        executive = payload.get("chapter_blocks", {}).get("Executive Summary", [])
        self.assertTrue(executive)
        first_fragment = executive[0]
        self.assertIsInstance(first_fragment, dict)

        expected_atomic = report_engine._extract_interpretation_text(
            report_engine.INTERPRETATIONS_KR_ATOMIC.get("asc:Leo")
        )
        self.assertIsInstance(expected_atomic, str)
        self.assertTrue(expected_atomic.strip())
        self.assertEqual(first_fragment.get("summary"), expected_atomic.strip())

        generic_fallback_prefix = "현재 지표 조합은 핵심 축의 일관성을 유지하면서"
        self.assertFalse(str(first_fragment.get("summary", "")).startswith(generic_fallback_prefix))


if __name__ == "__main__":
    unittest.main()
