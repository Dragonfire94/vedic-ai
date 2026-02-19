import ast
import unittest

import backend.report_engine as report_engine


def _serialize_chapter_blocks(payload: dict) -> str:
    parts: list[str] = []
    for chapter in report_engine.REPORT_CHAPTERS:
        for fragment in payload.get("chapter_blocks", {}).get(chapter, []):
            if not isinstance(fragment, dict):
                continue
            for field in ("title", "summary", "analysis", "implication", "examples"):
                value = fragment.get(field)
                if isinstance(value, str):
                    parts.append(value)
    return "\n".join(parts)


class TestInterpretationsKrIntegration(unittest.TestCase):
    def _structural_summary(self) -> dict:
        return {
            "language": "ko",
            "ascendant_sign": "Taurus",
            "sun_sign": "Capricorn",
            "moon_sign": "Leo",
            "chart_signature": {
                "ascendant_sign": "Taurus",
                "sun_sign": "Capricorn",
                "moon_sign": "Leo",
            },
            "planet_power_ranking": ["Mars", "Saturn", "Sun"],
            "life_purpose_vector": {
                "dominant_planet": "Mars",
                "primary_axis": "self_identity_axis",
                "dominant_purushartha": "Dharma",
            },
            "personality_vector": {
                "discipline_index": 71.0,
                "risk_appetite": 68.0,
                "emotional_regulation": 38.0,
                "authority_orientation": 73.0,
            },
            "stability_metrics": {
                "stability_index": 35.0,
                "stability_grade": "D",
                "tension_index": 0.71,
            },
            "psychological_tension_axis": {"score": 77, "axis": "venus_ketu"},
            "behavioral_risk_profile": {
                "authority_conflict_risk": 0.72,
                "emotional_volatility": 0.68,
                "burnout_risk": 0.61,
            },
            "varga_alignment": {
                "career_alignment": {"score": 0.32, "level": "low"},
                "overall_alignment": {"score": 0.41, "level": "low"},
                "relationship_alignment": {"score": 0.66, "level": "high"},
            },
            "probability_forecast": {
                "burnout_2yr": 0.74,
                "career_shift_3yr": 0.66,
                "marriage_5yr": 0.33,
            },
            "house_strengths": {"10": 9.1, "3": 8.4, "1": 7.2},
            "engine": {
                "house_clusters": {"cluster_scores": {"10": 9.1, "3": 8.4, "1": 7.2}},
                "influence_matrix": {"dominant_planet": "Mars", "most_conflicted_axis": ["Venus", "Ketu"]},
            },
        }

    def test_interpretations_library_loads(self):
        self.assertIsInstance(report_engine.INTERPRETATIONS_KR, dict)
        self.assertGreater(report_engine.INTERPRETATIONS_KR_ENTRIES_COUNT, 0)

    def test_interpretation_mapping_text_appears_in_chapter_blocks(self):
        mapped = report_engine._interpretation_lookup("atomic", "asc:Taurus")
        self.assertIsInstance(mapped, str)
        self.assertTrue(mapped.strip())
        payload = report_engine.build_report_payload({"structural_summary": self._structural_summary(), "language": "ko"})
        text = _serialize_chapter_blocks(payload)
        self.assertIn(mapped.strip(), text)

    def test_generic_fallback_text_is_reduced_when_library_enabled(self):
        structural = {"structural_summary": self._structural_summary(), "language": "ko"}
        mapped = report_engine._interpretation_lookup("atomic", "asc:Taurus")
        self.assertIsInstance(mapped, str)
        self.assertTrue(mapped.strip())

        original = report_engine.INTERPRETATIONS_KR
        original_atomic = report_engine.INTERPRETATIONS_KR_ATOMIC
        try:
            report_engine.INTERPRETATIONS_KR = original
            report_engine.INTERPRETATIONS_KR_ATOMIC = original_atomic
            enabled_payload = report_engine.build_report_payload(structural)
            enabled_count = _serialize_chapter_blocks(enabled_payload).count(mapped.strip())

            report_engine.INTERPRETATIONS_KR = {}
            report_engine.INTERPRETATIONS_KR_ATOMIC = {}
            disabled_payload = report_engine.build_report_payload(structural)
            disabled_count = _serialize_chapter_blocks(disabled_payload).count(mapped.strip())
        finally:
            report_engine.INTERPRETATIONS_KR = original
            report_engine.INTERPRETATIONS_KR_ATOMIC = original_atomic

        self.assertGreater(enabled_count, disabled_count)

    def test_atomic_global_candidates_are_injected_and_used_for_sun_moon(self):
        structural = self._structural_summary()
        keys = report_engine._mapping_keys_for_signal("stability_metrics.stability_index", 42.0, structural)
        markers = [f"{section}:{key}" for section, key in keys]
        self.assertIn("atomic:asc:Taurus", markers)
        self.assertIn("atomic:ps:Sun:Capricorn", markers)
        self.assertIn("atomic:ps:Moon:Leo", markers)

        with self.assertLogs("report_engine", level="INFO") as logs:
            report_engine.build_report_payload({"structural_summary": structural, "language": "ko"})

        summary_line = ""
        for line in logs.output:
            if "mapping_summary" in line and "atomic_usage=" in line:
                summary_line = line
        self.assertTrue(summary_line)
        atomic_usage_str = summary_line.split("atomic_usage=", 1)[1].strip()
        atomic_usage = ast.literal_eval(atomic_usage_str)
        self.assertGreater(int(atomic_usage["sun"]["hits"]), 0)
        self.assertGreater(int(atomic_usage["moon"]["hits"]), 0)


if __name__ == "__main__":
    unittest.main()
