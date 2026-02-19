import unittest

from backend.report_engine import build_report_payload
import backend.main as main


class TestChapterBlocksSchemaCompliance(unittest.TestCase):
    def test_chapter_blocks_strip_internal_fields_and_pass_validator(self):
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
        self.assertIsInstance(chapter_blocks, dict)
        for fragments in chapter_blocks.values():
            self.assertIsInstance(fragments, list)
            for fragment in fragments:
                self.assertIsInstance(fragment, dict)
                for key in fragment.keys():
                    self.assertFalse(str(key).startswith("_"))

        validated = main._validate_deterministic_llm_blocks(chapter_blocks)
        self.assertIsInstance(validated, dict)


if __name__ == "__main__":
    unittest.main()

