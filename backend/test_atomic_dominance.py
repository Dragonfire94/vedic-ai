import unittest

from backend.report_engine import REPORT_CHAPTERS, build_report_payload


class TestAtomicDominance(unittest.TestCase):
    def test_public_payload_hides_internal_source_markers_and_keeps_content(self):
        payload = build_report_payload(
            {
                "structural_summary": {
                    "language": "ko",
                    "ascendant_sign": "Leo",
                    "sun_sign": "Sagittarius",
                    "moon_sign": "Gemini",
                    "chart_signature": {
                        "ascendant_sign": "Leo",
                        "sun_sign": "Sagittarius",
                        "moon_sign": "Gemini",
                    },
                },
                "language": "ko",
            }
        )

        chapter_blocks = payload.get("chapter_blocks", {})
        for chapter in REPORT_CHAPTERS:
            fragments = chapter_blocks.get(chapter, [])
            content = [f for f in fragments if isinstance(f, dict) and "spike_text" not in f]
            self.assertTrue(content, msg=f"chapter has no content: {chapter}")
            for fragment in content:
                self.assertNotIn("_source", fragment)


if __name__ == "__main__":
    unittest.main()

