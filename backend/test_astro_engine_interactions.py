import unittest

from astro_engine import (
    build_interaction_summary,
    build_structural_summary,
    calculate_interaction_risks,
    compute_varga_alignment,
    enhance_behavioral_risks_with_interactions,
)


class TestInteractionAmplifier(unittest.TestCase):
    def test_calculate_interaction_risks_rules(self):
        personality_vector = {
            "ego_power": 90.0,
            "emotional_regulation": 20.0,
            "authority_orientation": 90.0,
            "discipline_index": 40.0,
            "risk_appetite": 85.0,
        }
        influence_matrix = {"saturn_conflict_score": 70.0}
        behavioral_risks = {"emotional_volatility": 6.0}
        house_clusters = {"cluster_scores": {6: 10.0, 8: 8.0, 12: 5.0}}

        out = calculate_interaction_risks(
            personality_vector,
            influence_matrix,
            behavioral_risks,
            house_clusters,
        )

        self.assertEqual(out["narcissistic_instability"], 35.0)
        self.assertEqual(out["authority_breakdown_risk"], 25.0)
        self.assertAlmostEqual(out["impulsive_sabotage_risk"], 25.0)
        self.assertEqual(out["chronic_stress_amplification"], 14.5)
        self.assertEqual(out["emotional_oscillation"], 16.0)

    def test_enhance_behavioral_risks_with_interactions(self):
        base_risks = {
            "authority_conflict_risk": 3.0,
            "emotional_volatility": 4.0,
            "self_sabotage_risk": 2.0,
            "burnout_risk": 3.0,
        }
        interaction_risks = {
            "narcissistic_instability": 10.0,
            "authority_breakdown_risk": 8.0,
            "impulsive_sabotage_risk": 5.0,
            "chronic_stress_amplification": 6.0,
            "emotional_oscillation": 4.0,
        }

        out = enhance_behavioral_risks_with_interactions(base_risks, interaction_risks)

        self.assertEqual(out["authority_conflict_risk"], 6.0)
        self.assertEqual(out["emotional_volatility"], 8.0)
        self.assertEqual(out["self_sabotage_risk"], 6.0)
        self.assertEqual(out["burnout_risk"], 5.4)
        self.assertEqual(out["emotional_volatility_amplified"], 8.0)

    def test_build_interaction_summary(self):
        interaction_risks = {"narcissistic_instability": 1.0}
        enhanced_risks = {"authority_conflict_risk": 2.0}

        out = build_interaction_summary(interaction_risks, enhanced_risks)

        self.assertEqual(out["raw_interaction_risks"], interaction_risks)
        self.assertEqual(out["enhanced_behavioral_risks"], enhanced_risks)

    def test_compute_varga_alignment(self):
        chart_data = {
            "planets": {
                "Sun": {"rasi": {"name": "Aries"}},
                "Moon": {"rasi": {"name": "Taurus"}},
                "Mars": {"rasi": {"name": "Gemini"}},
                "Mercury": {"rasi": {"name": "Cancer"}},
                "Jupiter": {"rasi": {"name": "Leo"}},
                "Venus": {"rasi": {"name": "Virgo"}},
                "Saturn": {"rasi": {"name": "Libra"}},
            },
            "vargas": {
                "d10": {
                    "planets": {
                        "Sun": {"rasi": "Aries"},
                        "Moon": {"rasi": "Taurus"},
                        "Mars": {"rasi": "Scorpio"},
                        "Mercury": {"rasi": "Cancer"},
                        "Jupiter": {"rasi": "Leo"},
                        "Venus": {"rasi": "Aquarius"},
                        "Saturn": {"rasi": "Libra"},
                    }
                }
            },
        }
        out = compute_varga_alignment(chart_data)
        self.assertIn("career_alignment", out)
        self.assertEqual(out["career_alignment"]["level"], "high")
        self.assertIn("d10", out["available_vargas"])

    def test_build_structural_summary_contains_varga_alignment(self):
        chart_data = {
            "planets": {
                "Sun": {"house": 1, "rasi": {"name": "Aries"}, "features": {"combust": False, "retrograde": False}},
                "Moon": {"house": 2, "rasi": {"name": "Taurus"}, "features": {"combust": False, "retrograde": False}},
                "Mars": {"house": 3, "rasi": {"name": "Gemini"}, "features": {"combust": False, "retrograde": False}},
                "Mercury": {"house": 4, "rasi": {"name": "Cancer"}, "features": {"combust": False, "retrograde": False}},
                "Jupiter": {"house": 5, "rasi": {"name": "Leo"}, "features": {"combust": False, "retrograde": False}},
                "Venus": {"house": 6, "rasi": {"name": "Virgo"}, "features": {"combust": False, "retrograde": False}},
                "Saturn": {"house": 7, "rasi": {"name": "Libra"}, "features": {"combust": False, "retrograde": False}},
            },
            "houses": {
                "ascendant": {"rasi": {"name": "Aries"}},
                **{f"house_{idx}": {"rasi": "Aries"} for idx in range(1, 13)},
            },
            "vargas": {
                "d9": {"planets": {"Sun": {"rasi": "Aries"}}},
                "d10": {"planets": {"Sun": {"rasi": "Aries"}}},
            },
        }
        out = build_structural_summary(chart_data)
        self.assertIn("varga_alignment", out)
        self.assertIn("varga_alignment", out["engine"])


if __name__ == "__main__":
    unittest.main()
