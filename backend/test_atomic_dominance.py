import unittest

from backend.report_engine import REPORT_CHAPTERS, build_report_payload


class TestAtomicDominance(unittest.TestCase):
    def test_atomic_is_primary_and_not_outnumbered(self):
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
        atomic_count = 0
        generic_count = 0
        for chapter in REPORT_CHAPTERS:
            fragments = chapter_blocks.get(chapter, [])
            content = [f for f in fragments if isinstance(f, dict) and "spike_text" not in f]
            self.assertTrue(content, msg=f"chapter has no content: {chapter}")
            self.assertEqual(content[0].get("_source"), "atomic", msg=f"first fragment not atomic: {chapter}")
            for fragment in content:
                if fragment.get("_source") == "atomic":
                    atomic_count += 1
                else:
                    generic_count += 1

        self.assertGreaterEqual(atomic_count, generic_count)


if __name__ == "__main__":
    unittest.main()
