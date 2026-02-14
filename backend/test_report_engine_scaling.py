import unittest

import backend.report_engine as report_engine


class TestReportEngineScaling(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS

        report_engine.TEMPLATES = [
            {
                "id": "scaling_block",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.use_scaling", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100,
                "content": {
                    "title": "Scaling Block",
                    "summary": "Base summary",
                    "analysis": "Base analysis",
                    "implication": "Base implication",
                    "examples": "Base examples",
                },
                "scaling_variants": {
                    "moderate": {
                        "analysis_extension": "Moderate analysis extension.",
                        "implication_extension": "Moderate implication extension.",
                        "example_extension": "Moderate example extension.",
                    },
                    "high": {
                        "analysis_extension": "High analysis extension.",
                        "implication_extension": "High implication extension.",
                        "example_extension": "High example extension.",
                        "micro_scenario": "High micro scenario.",
                        "long_term_projection": "High long-term projection.",
                    },
                },
            }
        ] + [
            {
                "id": f"cap_block_{idx}",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "flags.cap", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100 - idx,
                "content": {
                    "title": f"Cap {idx}",
                    "summary": "S",
                    "analysis": "A",
                    "implication": "I",
                    "examples": "E",
                },
            }
            for idx in range(7)
        ]

        report_engine.DEFAULT_BLOCKS = {
            chapter: [
                {
                    "id": f"default_{chapter}",
                    "chapter": chapter,
                    "priority": -1,
                    "content": {
                        "title": chapter,
                        "summary": "default summary",
                        "analysis": "default analysis",
                        "implication": "default implication",
                        "examples": "default examples",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults

    def _payload_for_intensity(self, tension: int, risk: int, stability: int):
        return report_engine.build_report_payload(
            {
                "flags": {"use_scaling": True},
                "psychological_tension_axis": {"score": tension},
                "behavioral_risk_profile": {"impulsivity_risk": risk},
                "stability_metrics": {"stability_index": stability},
            }
        )

    def test_moderate_scaling_applied(self):
        payload = self._payload_for_intensity(tension=80, risk=80, stability=40)
        fragment = payload["chapter_blocks"]["Psychological Architecture"][0]

        self.assertIn("Moderate analysis extension.", fragment["analysis"])
        self.assertIn("Moderate implication extension.", fragment["implication"])
        self.assertNotIn("micro_scenario", fragment)

    def test_high_scaling_applied(self):
        payload = self._payload_for_intensity(tension=90, risk=90, stability=10)
        fragment = payload["chapter_blocks"]["Psychological Architecture"][0]

        self.assertIn("High analysis extension.", fragment["analysis"])
        self.assertIn("High implication extension.", fragment["implication"])
        self.assertIn("High example extension.", fragment["examples"])
        self.assertEqual(fragment["micro_scenario"], "High micro scenario.")
        self.assertEqual(fragment["long_term_projection"], "High long-term projection.")

    def test_no_scaling_when_intensity_low(self):
        payload = self._payload_for_intensity(tension=30, risk=30, stability=90)
        fragment = payload["chapter_blocks"]["Psychological Architecture"][0]

        self.assertNotIn("extension", fragment["analysis"])
        self.assertNotIn("extension", fragment.get("implication", ""))
        self.assertNotIn("micro_scenario", fragment)
        self.assertNotIn("long_term_projection", fragment)

    def test_micro_scenario_only_high(self):
        moderate = self._payload_for_intensity(tension=80, risk=80, stability=40)
        high = self._payload_for_intensity(tension=90, risk=90, stability=10)

        self.assertNotIn("micro_scenario", moderate["chapter_blocks"]["Psychological Architecture"][0])
        self.assertIn("micro_scenario", high["chapter_blocks"]["Psychological Architecture"][0])

    def test_long_term_projection_only_high(self):
        moderate = self._payload_for_intensity(tension=80, risk=80, stability=40)
        high = self._payload_for_intensity(tension=90, risk=90, stability=10)

        self.assertNotIn("long_term_projection", moderate["chapter_blocks"]["Psychological Architecture"][0])
        self.assertIn("long_term_projection", high["chapter_blocks"]["Psychological Architecture"][0])

    def test_no_fragment_count_increase(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"cap": True},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95},
                "stability_metrics": {"stability_index": 5},
            }
        )
        self.assertEqual(len(payload["chapter_blocks"]["Love & Relationships"]), 5)

    def test_backward_compatibility_without_scaling(self):
        report_engine.TEMPLATES = [
            {
                "id": "legacy_block",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.legacy", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100,
                "content": {
                    "title": "Legacy",
                    "summary": "Legacy summary",
                    "analysis": "Legacy analysis",
                    "implication": "Legacy implication",
                    "examples": "Legacy examples",
                },
            }
        ]
        payload = report_engine.build_report_payload(
            {
                "flags": {"legacy": True},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        fragment = payload["chapter_blocks"]["Psychological Architecture"][0]

        self.assertEqual(fragment["analysis"], "Legacy analysis")
        self.assertNotIn("micro_scenario", fragment)
        self.assertNotIn("long_term_projection", fragment)


if __name__ == "__main__":
    unittest.main()
