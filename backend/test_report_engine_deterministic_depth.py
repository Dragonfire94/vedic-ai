import unittest

import backend.report_engine as report_engine


def _chapter_stats(chapter_blocks):
    chars = 0
    words = 0
    for fragment in chapter_blocks:
        if not isinstance(fragment, dict):
            continue
        for field in ("title", "summary", "analysis", "implication", "examples"):
            value = fragment.get(field)
            if isinstance(value, str):
                chars += len(value)
                words += len(value.split())
    return chars, words


class TestReportEngineDeterministicDepth(unittest.TestCase):
    def _rich_structural_summary(self, language: str = "ko"):
        return {
            "language": language,
            "planet_power_ranking": ["Saturn", "Sun", "Moon", "Mars"],
            "dominant_house_cluster": 10,
            "psychological_tension_axis": {"score": 78, "tension_level": "high", "axis": "identity_duty"},
            "life_purpose_vector": {
                "dominant_planet": "Saturn",
                "dominant_purushartha": "Artha",
                "primary_axis": "career_responsibility",
            },
            "purushartha_profile": {
                "dharma": 34.2,
                "artha": 61.5,
                "kama": 41.8,
                "moksha": 28.4,
                "dominant_purushartha": "Artha",
            },
            "behavioral_risk_profile": {
                "primary_risk": "impulsivity",
                "impulsivity_risk": 72.0,
                "overcontrol_risk": 42.0,
                "self_sabotage_risk": 5.8,
            },
            "interaction_risks": {
                "escalation_risk": 63.0,
                "decision_fatigue_risk": 52.0,
            },
            "enhanced_behavioral_risks": {
                "impulsivity_risk": 74.0,
                "overcontrol_risk": 44.0,
                "self_sabotage_risk": 6.1,
            },
            "stability_metrics": {
                "stability_index": 46.5,
                "stability_grade": "C",
                "grade": "C",
                "tension_index": 0.68,
            },
            "personality_vector": {
                "ego_power": 67.0,
                "emotional_regulation": 53.0,
                "authority_orientation": 74.0,
                "discipline_index": 81.0,
                "risk_appetite": 58.0,
            },
            "probability_forecast": {
                "career_shift_3yr": 0.62,
                "marriage_5yr": 0.58,
                "burnout_2yr": 0.71,
                "financial_instability_3yr": 0.49,
            },
            "karmic_pattern_profile": {
                "primary_pattern": "correction",
                "balanced_growth_pattern": 0.31,
                "identity_correction_pattern": 0.69,
            },
            "varga_alignment": {
                "relationship_alignment": {"score": 51.0, "level": "moderate"},
                "career_alignment": {"score": 79.0, "level": "high"},
                "creativity_progeny_alignment": {"score": 47.0, "level": "moderate"},
                "family_lineage_alignment": {"score": 55.0, "level": "moderate"},
                "overall_alignment": {"score": 58.0, "level": "moderate"},
                "available_vargas": ["d7", "d9", "d10", "d12"],
            },
            "engine": {
                "influence_matrix": {
                    "dominant_planet": "Saturn",
                    "saturn_conflict_score": 0.72,
                    "most_conflicted_axis": {"axis": "identity_duty", "score": 78},
                },
                "house_clusters": {"dominant_house": 10, "cluster_scores": {"10": 8.2, "6": 7.4}},
                "stability_metrics": {"stability_index": 46.5, "tension_index": 0.68},
                "personality_vector": {"discipline_index": 81.0, "risk_appetite": 58.0},
                "varga_alignment": {
                    "career_alignment": {"score": 79.0, "level": "high"},
                    "relationship_alignment": {"score": 51.0, "level": "moderate"},
                },
            },
        }

    def test_minimum_depth_guarantee_korean(self):
        payload = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="ko")})
        for chapter in report_engine.REPORT_CHAPTERS:
            chars, _ = _chapter_stats(payload["chapter_blocks"][chapter])
            self.assertGreaterEqual(chars, report_engine.MIN_DEPTH_KO_CHARS, msg=f"{chapter} below Korean depth target")

    def test_minimum_depth_guarantee_english(self):
        payload = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="en")})
        for chapter in report_engine.REPORT_CHAPTERS:
            _, words = _chapter_stats(payload["chapter_blocks"][chapter])
            self.assertGreaterEqual(words, report_engine.MIN_DEPTH_EN_WORDS, msg=f"{chapter} below English depth target")

    def test_no_placeholder_fallback_text(self):
        payload = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="ko")})
        serialized = str(payload)
        self.assertNotIn("No dominant", serialized)
        self.assertNotIn("default summary", serialized)

    def test_no_diagnostic_trace_language_in_user_facing_text(self):
        payload_ko = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="ko")})
        payload_en = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="en")})
        banned_phrases = [
            "결정론 구조 데이터만으로 생성되었습니다",
            "추적 규칙",
            "source path",
            "value =",
            "trace rule",
            "deterministic reference",
        ]
        for payload in (payload_ko, payload_en):
            serialized = str(payload).lower()
            for banned in banned_phrases:
                self.assertNotIn(banned.lower(), serialized)

    def test_interpretive_narrative_present(self):
        payload_ko = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="ko")})
        payload_en = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="en")})
        ko_text = str(payload_ko)
        en_text = str(payload_en)
        self.assertTrue(any(token in ko_text for token in ["추진력", "내적 갈등", "안정성", "규율"]))
        self.assertTrue(any(token in en_text for token in ["initiative", "internal conflict", "stability", "discipline"]))

    def test_deterministic_reproducibility(self):
        structural = {"structural_summary": self._rich_structural_summary(language="ko")}
        payload_a = report_engine.build_report_payload(structural)
        payload_b = report_engine.build_report_payload(structural)
        self.assertEqual(payload_a, payload_b)

    def test_trace_metadata_not_exposed_in_payload(self):
        payload = report_engine.build_report_payload({"structural_summary": self._rich_structural_summary(language="ko")})
        serialized = str(payload)
        self.assertNotIn("source_signal", serialized)
        self.assertNotIn("rule_id", serialized)


if __name__ == "__main__":
    unittest.main()
