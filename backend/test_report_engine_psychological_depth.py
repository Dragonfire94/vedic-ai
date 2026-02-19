import unittest

import backend.report_engine as report_engine


class TestReportEnginePsychologicalDepth(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS
        self.original_rules = report_engine.REINFORCE_RULES

        report_engine.TEMPLATES = [
            {
                "id": "deep_payload_block",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.deep", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100,
                "chain_followups": [],
                "content": {
                    "title": "Deep",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                    "shadow_pattern": "shadow",
                    "defense_mechanism": "defense",
                    "emotional_trigger": "trigger",
                    "repetition_cycle": "cycle",
                    "integration_path": "integration",
                    "choice_fork": {"path_a": {"label": "A", "trajectory": "t1", "emotional_cost": "c1"}, "path_b": {"label": "B", "trajectory": "t2", "emotional_cost": "c2"}},
                },
            },
            {
                "id": "high_pressure_identity_fragmentation",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 95,
                "chain_followups": [],
                "content": {
                    "title": "HPIF",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                    "shadow_pattern": "shadow",
                    "defense_mechanism": "defense",
                    "emotional_trigger": "trigger",
                    "repetition_cycle": "cycle",
                    "integration_path": "integration",
                },
            },
            {
                "id": "high_pressure_identity_fragmentation",
                "chapter": "Final Summary",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 80,
                "chain_followups": [],
                "content": {
                    "title": "HPIF Summary",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "recursive_correction_loop",
                "chapter": "Life Timeline Interpretation",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 90,
                "chain_followups": [],
                "content": {
                    "title": "Recursive",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                    "shadow_pattern": "shadow",
                    "defense_mechanism": "defense",
                    "emotional_trigger": "trigger",
                    "repetition_cycle": "cycle",
                    "integration_path": "integration",
                    "choice_fork": {"path_a": {"label": "A", "trajectory": "t1", "emotional_cost": "c1"}, "path_b": {"label": "B", "trajectory": "t2", "emotional_cost": "c2"}},
                },
            },
            {
                "id": "legacy_block",
                "chapter": "Behavioral Risks",
                "conditions": [{"field": "flags.legacy", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 70,
                "chain_followups": [],
                "content": {
                    "title": "Legacy",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                },
            },
        ] + [
            {
                "id": f"cap_{idx}",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "flags.cap", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100 - idx,
                "chain_followups": [],
                "content": {
                    "title": f"Cap {idx}",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
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

    def test_shadow_fields_in_high_intensity(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"deep": True},
                "psychological_tension_axis": {"score": 85},
                "behavioral_risk_profile": {"impulsivity_risk": 80, "overcontrol_risk": 80},
                "stability_metrics": {"stability_index": 20},
            }
        )
        block = payload["chapter_blocks"]["Psychological Architecture"][0]
        self.assertIn("shadow_pattern", block)
        self.assertIn("defense_mechanism", block)
        self.assertIn("emotional_trigger", block)
        self.assertIn("repetition_cycle", block)
        self.assertIn("integration_path", block)

    def test_choice_fork_only_in_extreme_intensity(self):
        high = report_engine.build_report_payload(
            {
                "flags": {"deep": True},
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"impulsivity_risk": 90, "overcontrol_risk": 80},
                "stability_metrics": {"stability_index": 10},
            }
        )
        self.assertIn("choice_fork", high["chapter_blocks"]["Psychological Architecture"][0])

        moderate = report_engine.build_report_payload(
            {
                "flags": {"deep": True},
                "psychological_tension_axis": {"score": 80},
                "behavioral_risk_profile": {"impulsivity_risk": 70, "overcontrol_risk": 70},
                "stability_metrics": {"stability_index": 35},
            }
        )
        self.assertIn("choice_fork", moderate["chapter_blocks"]["Psychological Architecture"][0])

    def test_emotional_escalation_rule(self):
        selected = report_engine.select_template_blocks(
            {
                "psychological_tension_axis": {"score": 80},
                "stability_metrics": {"stability_index": 40},
            }
        )
        ids = [block["id"] for block in selected["Psychological Architecture"]]
        self.assertIn("high_pressure_identity_fragmentation", ids)

    def test_recursive_correction_injection(self):
        selected = report_engine.select_template_blocks(
            {
                "karmic_pattern_profile": {"primary_pattern": "correction"},
                "behavioral_risk_profile": {"primary_risk": "impulsivity"},
            }
        )
        ids = [block["id"] for block in selected["Life Timeline Interpretation"]]
        self.assertIn("recursive_correction_loop", ids)

    def test_psychological_echo(self):
        selected = report_engine.select_template_blocks(
            {
                "psychological_tension_axis": {"score": 90},
                "stability_metrics": {"stability_index": 30},
            }
        )
        final_ids = [block["id"] for block in selected["Final Summary"]]
        self.assertIn("high_pressure_identity_fragmentation", final_ids)

    def test_no_duplicate_injection(self):
        selected = report_engine.select_template_blocks(
            {
                "psychological_tension_axis": {"score": 95},
                "stability_metrics": {"stability_index": 25},
            }
        )
        psych_ids = [block["id"] for block in selected["Psychological Architecture"]]
        self.assertEqual(psych_ids.count("high_pressure_identity_fragmentation"), 1)

        payload = report_engine.build_report_payload({"flags": {"cap": True}})
        self.assertLessEqual(len(payload["chapter_blocks"]["Love & Relationships"]), 5)

    def test_backward_compatibility(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"legacy": True},
                "psychological_tension_axis": {"score": 10},
                "stability_metrics": {"stability_index": 95},
            }
        )
        block = payload["chapter_blocks"]["Behavioral Risks"][0]
        self.assertIn("title", block)
        self.assertIn("summary", block)
        self.assertIn("analysis", block)


if __name__ == "__main__":
    unittest.main()
