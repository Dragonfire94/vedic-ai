import unittest

from backend.report_engine import REPORT_CHAPTERS, build_report_payload


def _serialize_output(payload: dict) -> str:
    parts: list[str] = []
    chapter_blocks = payload.get("chapter_blocks", {})
    for chapter in REPORT_CHAPTERS:
        for fragment in chapter_blocks.get(chapter, []):
            if not isinstance(fragment, dict):
                continue
            for field in ("title", "summary", "analysis", "implication", "examples"):
                value = fragment.get(field)
                if isinstance(value, str):
                    parts.append(value)
    return "\n".join(parts)


class TestAtomicInterpretationUsage(unittest.TestCase):
    def test_atomic_interpretation_text_is_used(self):
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

        serialized_output = _serialize_output(payload)
        self.assertIn("시데리얼(항성황도), 라히리 기준", serialized_output)


if __name__ == "__main__":
    unittest.main()
