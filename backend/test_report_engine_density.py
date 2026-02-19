import unittest

import backend.report_engine as report_engine


class TestReportEngineDensity(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS
        self.original_rules = report_engine.REINFORCE_RULES

        report_engine.TEMPLATES = [
            {
                "id": "high_tension_risk_aggro",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.match", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 20,
                "chain_followups": ["followup_same_chapter"],
                "content": {
                    "title": "Aggro",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "followup_same_chapter",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 15,
                "chain_followups": ["high_tension_risk_aggro"],
                "content": {
                    "title": "Followup",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "high_stability_fall",
                "chapter": "Stability Metrics",
                "conditions": [{"field": "flags.match", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 19,
                "chain_followups": [],
                "content": {
                    "title": "Fall",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "tension_stability_interaction",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 30,
                "chain_followups": [],
                "content": {
                    "title": "Reinforced",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "career_conflict_karma",
                "chapter": "Career & Success",
                "conditions": [{"field": "flags.match", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 12,
                "chain_followups": [],
                "content": {
                    "title": "Career conflict",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "career_purushartha",
                "chapter": "Career & Success",
                "conditions": [{"field": "flags.match", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 11,
                "chain_followups": [],
                "content": {
                    "title": "Career purushartha",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "career_karma_pattern_reinforcement",
                "chapter": "Executive Summary",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 31,
                "chain_followups": [],
                "content": {
                    "title": "Career reinforced",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
        ] + [
            {
                "id": f"cap_block_{idx}",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "flags.cap", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100 - idx,
                "chain_followups": [],
                "content": {
                    "title": f"cap-{idx}",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            }
            for idx in range(7)
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
                        "examples": "",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }

        report_engine.REINFORCE_RULES = [
            {
                "if_block_ids": ["high_tension_risk_aggro", "high_stability_fall"],
                "add_block_id": "tension_stability_interaction",
                "target_chapter": "Psychological Architecture",
            },
            {
                "if_block_ids": ["career_conflict_karma", "career_purushartha"],
                "add_block_id": "career_karma_pattern_reinforcement",
                "target_chapter": "Executive Summary",
            },
        ]

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults
        report_engine.REINFORCE_RULES = self.original_rules

    def test_block_intensity_computation(self):
        intensity = report_engine.compute_block_intensity(
            {},
            {
                "psychological_tension_axis": 90,
                "behavioral_risk_profile": {"impulsivity_risk": 90, "overcontrol_risk": 30},
                "stability_metrics": {"stability_index": 20},
            },
        )
        self.assertAlmostEqual(intensity, (0.9 + 0.6 + 0.8) / 3.0)

    def test_chain_followup_inclusion(self):
        selected = report_engine.select_template_blocks({"flags": {"match": True}})
        ids = [b["id"] for b in selected["Psychological Architecture"]]
        self.assertIn("high_tension_risk_aggro", ids)
        self.assertIn("followup_same_chapter", ids)
        self.assertEqual(ids.count("high_tension_risk_aggro"), 1)

    def test_priority_intensity_sort(self):
        selected = report_engine.select_template_blocks(
            {
                "flags": {"match": True},
                "psychological_tension_axis": 95,
                "behavioral_risk_profile": {"impulsivity_risk": 95},
                "stability_metrics": {"stability_index": 5},
            }
        )
        blocks = selected["Psychological Architecture"]
        self.assertEqual(blocks[0]["id"], "tension_stability_interaction")

    def test_cross_chapter_reinforcement(self):
        selected = report_engine.select_template_blocks({"flags": {"match": True}})
        psych_ids = [b["id"] for b in selected["Psychological Architecture"]]
        exec_ids = [b["id"] for b in selected["Executive Summary"]]
        self.assertIn("tension_stability_interaction", psych_ids)
        self.assertIn("career_karma_pattern_reinforcement", exec_ids)

    def test_dynamic_field_selection(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"match": True},
                "psychological_tension_axis": 90,
                "behavioral_risk_profile": {"impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        strong = payload["chapter_blocks"]["Psychological Architecture"][0]
        self.assertIn("examples", strong)

        payload_low = report_engine.build_report_payload({"flags": {"match": True}, "stability_metrics": {"stability_index": 99}})
        low = payload_low["chapter_blocks"]["Psychological Architecture"][0]
        self.assertIn("title", low)
        self.assertIn("summary", low)

    def test_max_block_cap_after_expansion(self):
        payload = report_engine.build_report_payload({"flags": {"cap": True}})
        self.assertEqual(len(payload["chapter_blocks"]["Love & Relationships"]), 5)

    def test_examples_field_presence_by_intensity(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"match": True},
                "psychological_tension_axis": 100,
                "behavioral_risk_profile": {"impulsivity_risk": 100},
                "stability_metrics": {"stability_index": 0},
            }
        )
        self.assertIn("examples", payload["chapter_blocks"]["Psychological Architecture"][0])


if __name__ == "__main__":
    unittest.main()
