import unittest

import backend.report_engine as report_engine


class TestReportEngineInsightSpike(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS
        self.original_rules = report_engine.REINFORCE_RULES

        report_engine.TEMPLATES = [
            {
                "id": "high_spike_block",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.high", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100,
                "insight_spike": {"text": "Spike high.", "min_intensity": 0.8},
                "content": {"title": "High", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "low_spike_block",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.low", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 90,
                "insight_spike": {"text": "Spike low should not show.", "min_intensity": 0.8},
                "content": {"title": "Low", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "dup_spike_a",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "flags.dup", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 90,
                "insight_spike": {"text": "Duplicate spike", "min_intensity": 0.75},
                "content": {"title": "A", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "dup_spike_b",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "flags.dup", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 80,
                "insight_spike": {"text": "Duplicate spike", "min_intensity": 0.75},
                "content": {"title": "B", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "legacy_no_spike",
                "chapter": "Behavioral Risks",
                "conditions": [{"field": "flags.legacy", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 70,
                "content": {"title": "Legacy", "summary": "summary", "analysis": "analysis", "implication": "implication"},
            },
        ] + [
            {
                "id": f"cap_{idx}",
                "chapter": "Career & Success",
                "conditions": [{"field": "flags.cap", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100 - idx,
                "insight_spike": {"text": f"Spike {idx}", "min_intensity": 0.0},
                "content": {"title": f"Cap {idx}", "summary": "s", "analysis": "a", "implication": "i"},
            }
            for idx in range(6)
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
                        "implication": "default implication",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }
        report_engine.REINFORCE_RULES = []

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults
        report_engine.REINFORCE_RULES = self.original_rules

    def test_spike_injection_at_high_intensity(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"high": True},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95, "overcontrol_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        chapter = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertIn({"spike_text": "Spike high."}, chapter)

    def test_spike_not_injected_when_low_intensity(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"low": True},
                "psychological_tension_axis": {"score": 10},
                "behavioral_risk_profile": {"impulsivity_risk": 10, "overcontrol_risk": 10},
                "stability_metrics": {"stability_index": 95},
            }
        )
        chapter = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertFalse(any(fragment.get("spike_text") == "Spike low should not show." for fragment in chapter))

    def test_spike_position_at_top(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"high": True},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90, "overcontrol_risk": 80},
                "stability_metrics": {"stability_index": 20},
            }
        )
        chapter = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertEqual(chapter[0], {"spike_text": "Spike high."})
        self.assertIn("title", chapter[1])

    def test_spike_deduplication(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"dup": True},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 70, "overcontrol_risk": 70},
                "stability_metrics": {"stability_index": 20},
            }
        )
        chapter = payload["chapter_blocks"]["Love & Relationships"]
        spike_count = sum(1 for fragment in chapter if fragment.get("spike_text") == "Duplicate spike")
        self.assertEqual(spike_count, 1)

    def test_spike_respects_max_cap(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"cap": True},
                "psychological_tension_axis": {"score": 30},
                "behavioral_risk_profile": {"impulsivity_risk": 20, "overcontrol_risk": 20},
                "stability_metrics": {"stability_index": 90},
            }
        )
        chapter = payload["chapter_blocks"]["Career & Success"]
        self.assertLessEqual(len(chapter), 5)
        self.assertTrue(all("spike_text" in fragment for fragment in chapter))

    def test_backward_compatibility_without_spike(self):
        payload = report_engine.build_report_payload({"flags": {"legacy": True}})
        chapter = payload["chapter_blocks"]["Behavioral Risks"]
        self.assertEqual(chapter[0]["title"], "Legacy")
        self.assertIn("summary", chapter[0])
        self.assertIn("analysis", chapter[0])


if __name__ == "__main__":
    unittest.main()
