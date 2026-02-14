import unittest

import backend.report_engine as report_engine


class TestReportEngineScenarioCompression(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS
        self.original_rules = report_engine.SCENARIO_COMPRESSION_RULES

        report_engine.TEMPLATES = [
            {
                "id": "structural_transition_window",
                "chapter": "Final Summary",
                "priority": 98,
                "content": {
                    "title": "Structural Transition Window (3–5 Years)",
                    "summary": "Multiple structural probability signals indicate a reconfiguration phase.",
                    "analysis": "Career and relational axes are simultaneously activated.",
                    "predictive_compression": {
                        "window": "Next 3–5 Years",
                        "dominant_theme": "Role Reallocation",
                    },
                },
            },
            {
                "id": "burnout_risk_window",
                "chapter": "Stability Metrics",
                "priority": 97,
                "content": {
                    "title": "Energy Collapse Risk Window",
                    "summary": "Burnout probability exceeds safe structural thresholds.",
                    "analysis": "Low stability index combined with elevated exhaustion probability.",
                    "predictive_compression": {
                        "window": "Next 2 Years",
                        "dominant_theme": "Energy Conservation Required",
                    },
                },
            },
        ]
        report_engine.DEFAULT_BLOCKS = {
            chapter: [
                {
                    "id": f"default_{chapter}",
                    "chapter": chapter,
                    "conditions": [],
                    "logic": "AND",
                    "priority": -999,
                    "content": {
                        "title": chapter,
                        "summary": "default summary",
                        "analysis": "default analysis",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }

        report_engine.DEFAULT_BLOCKS["Final Summary"].extend(
            [
                {
                    "id": f"overflow_final_{i}",
                    "chapter": "Final Summary",
                    "conditions": [{"field": "flags.overflow", "operator": "==", "value": True}],
                    "logic": "AND",
                    "priority": i,
                    "content": {
                        "title": f"Final overflow {i}",
                        "summary": "summary",
                        "analysis": "analysis",
                        "implication": "implication",
                    },
                }
                for i in range(1, 6)
            ]
        )

        report_engine.SCENARIO_COMPRESSION_RULES = [
            {
                "id": "structural_transition_window",
                "conditions": {
                    "probability_forecast.career_shift_3yr": {">=": 0.6},
                    "probability_forecast.marriage_5yr": {">=": 0.6},
                },
                "chapter": "Final Summary",
                "priority": 98,
            },
            {
                "id": "burnout_risk_window",
                "conditions": {
                    "probability_forecast.burnout_2yr": {">=": 0.7},
                    "stability_metrics.stability_index": {"<=": 50},
                },
                "chapter": "Stability Metrics",
                "priority": 97,
            },
        ]

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults
        report_engine.SCENARIO_COMPRESSION_RULES = self.original_rules

    def test_scenario_injected_when_conditions_met(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.8, "marriage_5yr": 0.7, "burnout_2yr": 0.8},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 40},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        self.assertTrue(any(b.get("title") == "Structural Transition Window (3–5 Years)" for b in final_summary))

    def test_scenario_not_injected_when_conditions_not_met(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.5, "marriage_5yr": 0.7},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 40},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        self.assertFalse(any(b.get("title") == "Structural Transition Window (3–5 Years)" for b in final_summary))

    def test_scenario_respects_intensity_gate(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.9, "marriage_5yr": 0.9},
                "psychological_tension_axis": {"score": 30},
                "behavioral_risk_profile": {},
                "stability_metrics": {"stability_index": 90},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        self.assertFalse(any(b.get("title") == "Structural Transition Window (3–5 Years)" for b in final_summary))

    def test_scenario_preserves_nested_structure(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.8, "marriage_5yr": 0.7},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        block = next(b for b in final_summary if b.get("title") == "Structural Transition Window (3–5 Years)")
        self.assertIsInstance(block.get("predictive_compression"), dict)
        self.assertIn("window", block["predictive_compression"])

    def test_scenario_replaces_lowest_priority_on_overflow(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.8, "marriage_5yr": 0.7},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95},
                "stability_metrics": {"stability_index": 10},
                "flags": {"overflow": True},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        titles = [b.get("title") for b in final_summary]
        self.assertEqual(len(final_summary), 5)
        self.assertIn("Structural Transition Window (3–5 Years)", titles)

    def test_no_duplicate_injection(self):
        payload = report_engine.build_report_payload(
            {
                "probability_forecast": {"career_shift_3yr": 0.8, "marriage_5yr": 0.7},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95},
                "stability_metrics": {"stability_index": 10},
            }
        )
        final_summary = payload["chapter_blocks"]["Final Summary"]
        matches = [b for b in final_summary if b.get("title") == "Structural Transition Window (3–5 Years)"]
        self.assertEqual(len(matches), 1)

    def test_backward_compatibility_without_probability(self):
        payload = report_engine.build_report_payload({"stability_metrics": {"stability_index": 40}})
        self.assertEqual(list(payload.keys()), ["chapter_blocks"])
        self.assertEqual(set(payload["chapter_blocks"].keys()), set(report_engine.REPORT_CHAPTERS))


if __name__ == "__main__":
    unittest.main()
