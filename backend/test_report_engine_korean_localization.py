import re
import unittest

import backend.report_engine as report_engine


class TestReportEngineKoreanLocalization(unittest.TestCase):
    def _structural(self) -> dict:
        return {
            "dominant_purushartha": "Dharma",
            "psychological_tension_axis": {"score": 83, "tension_level": "high"},
            "behavioral_risk_profile": {
                "primary_risk": "impulsivity",
                "impulsivity_risk": 76,
                "overcontrol_risk": 31,
            },
            "karmic_pattern_profile": {"primary_pattern": "correction"},
            "stability_metrics": {"grade": "C", "stability_index": 44, "tension_index": 0.66},
            "personality_vector": {"discipline_index": 81, "emotional_regulation": 58},
            "probability_forecast": {"career_shift_3yr": 0.63, "burnout_2yr": 0.71},
            "varga_alignment": {
                "career_alignment": {"score": 78, "level": "high"},
                "relationship_alignment": {"score": 52, "level": "moderate"},
            },
            "life_purpose_vector": {"dominant_planet": "Saturn", "primary_axis": "duty_axis"},
            "planet_power_ranking": ["Saturn", "Sun", "Moon"],
            "engine": {
                "influence_matrix": {"dominant_planet": "Saturn", "most_conflicted_axis": {"score": 83}},
                "house_clusters": {"dominant_house": 10},
            },
        }

    def _chapter_text(self, payload: dict, chapter: str) -> str:
        fragments = payload.get("chapter_blocks", {}).get(chapter, [])
        parts = []
        for f in fragments:
            if not isinstance(f, dict):
                continue
            for k in ("title", "summary", "analysis", "implication", "examples"):
                v = f.get(k)
                if isinstance(v, str):
                    parts.append(v)
        return "\n".join(parts)

    def test_korean_templates_selected_when_language_ko(self):
        payload = report_engine.build_report_payload(
            {"structural_summary": self._structural(), "language": "ko"}
        )
        text = self._chapter_text(payload, "Executive Summary")
        self.assertRegex(text, r"[가-힣]")
        self.assertNotIn("No dominant", text)

    def test_korean_is_deterministic_without_llm(self):
        payload_a = report_engine.build_report_payload(
            {"structural_summary": self._structural(), "language": "ko"}
        )
        payload_b = report_engine.build_report_payload(
            {"structural_summary": self._structural(), "language": "ko"}
        )
        self.assertEqual(payload_a, payload_b)

    def test_depth_structure_parity_between_ko_and_en(self):
        payload_en = report_engine.build_report_payload(
            {"structural_summary": self._structural(), "language": "en"}
        )
        payload_ko = report_engine.build_report_payload(
            {"structural_summary": self._structural(), "language": "ko"}
        )
        for chapter in report_engine.REPORT_CHAPTERS:
            en_count = len(payload_en["chapter_blocks"].get(chapter, []))
            ko_count = len(payload_ko["chapter_blocks"].get(chapter, []))
            self.assertEqual(en_count, ko_count, msg=f"Fragment count mismatch in {chapter}")


if __name__ == "__main__":
    unittest.main()
