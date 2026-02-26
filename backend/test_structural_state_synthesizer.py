import copy
import unittest

from backend.astro_engine import synthesize_structural_state


class TestStructuralStateSynthesizer(unittest.TestCase):
    def test_missing_fields_fallback(self) -> None:
        out = synthesize_structural_state({})
        self.assertEqual(out["state_label"], "structural_equilibrium")
        self.assertGreaterEqual(out["state_intensity"], 0.0)
        self.assertLessEqual(out["state_intensity"], 1.0)

    def test_high_intensity_fragmented(self) -> None:
        summary = {
            "stability_metrics": {"stability_index": 10},
            "psychological_tension_axis": {"score": 90},
            "axis_coherence": {"axis_coherence_level": "fragmented"},
            "structural_saturation": {"value": 0.9},
            "sub_dasha_resonance": {"value": 0.1},
            "current_dasha_vector": {"sub_dasha_bias": {"volatility_shift": 0.15}},
        }
        out = synthesize_structural_state(summary)
        self.assertEqual(out["state_label"], "fragmented_high_density")

    def test_low_intensity_equilibrium(self) -> None:
        summary = {
            "stability_metrics": {"stability_index": 95},
            "psychological_tension_axis": {"score": 5},
            "axis_coherence": {"axis_coherence_level": "high"},
            "structural_saturation": {"value": 0.1},
            "sub_dasha_resonance": {"value": 0.9},
            "current_dasha_vector": {"sub_dasha_bias": {"volatility_shift": 0.0}},
        }
        out = synthesize_structural_state(summary)
        self.assertEqual(out["state_label"], "structural_equilibrium")

    def test_input_not_mutated(self) -> None:
        summary = {
            "stability_metrics": {"stability_index": 50},
            "psychological_tension_axis": {"score": 50},
            "axis_coherence": {"axis_coherence_level": "moderate"},
            "structural_saturation": {"value": 0.5},
            "sub_dasha_resonance": {"value": 0.5},
            "current_dasha_vector": {"sub_dasha_bias": {"volatility_shift": 0.0}},
        }
        before = copy.deepcopy(summary)
        _ = synthesize_structural_state(summary)
        self.assertEqual(summary, before)


if __name__ == "__main__":
    unittest.main()
