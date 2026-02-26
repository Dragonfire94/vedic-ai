import unittest

from backend.astro_engine import (
    calculate_axis_coherence_drift,
    calculate_structural_saturation,
    calculate_sub_dasha_resonance,
)


class TestStabilityMetrics(unittest.TestCase):
    def test_axis_coherence_high_when_identical(self) -> None:
        out = calculate_axis_coherence_drift(
            maha_axis="relationship_axis",
            sub_axis="relationship_axis",
            transit_axis="emotional",
        )
        self.assertEqual(out["axis_coherence_level"], "high")

    def test_axis_coherence_fragmented(self) -> None:
        out = calculate_axis_coherence_drift(
            maha_axis="self_identity_axis",
            sub_axis="relationship_axis",
            transit_axis="authority",
        )
        self.assertEqual(out["axis_coherence_level"], "fragmented")

    def test_axis_coherence_all_neutral(self) -> None:
        out = calculate_axis_coherence_drift(
            maha_axis="neutral",
            sub_axis="neutral",
            transit_axis="neutral",
        )
        self.assertEqual(out["axis_coherence_level"], "high")

    def test_structural_saturation_high(self) -> None:
        out = calculate_structural_saturation(
            influence_score=2.0,
            pressure_score=2.0,
            volatility_shift=0.15,
        )
        self.assertGreaterEqual(out["value"], 0.65)
        self.assertEqual(out["band"], "saturated")

    def test_sub_dasha_resonance_bands(self) -> None:
        harmonic = calculate_sub_dasha_resonance(0.9, 0.9)
        dissonant = calculate_sub_dasha_resonance(0.0, 1.0)
        self.assertGreaterEqual(harmonic["value"], 0.9)
        self.assertEqual(harmonic["band"], "harmonic")
        self.assertLessEqual(dissonant["value"], 0.4)
        self.assertEqual(dissonant["band"], "dissonant")


if __name__ == "__main__":
    unittest.main()
