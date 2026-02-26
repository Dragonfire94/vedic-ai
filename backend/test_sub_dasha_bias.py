import unittest

from backend.astro_engine import calculate_sub_dasha_bias


class TestSubDashaBias(unittest.TestCase):
    def test_strong_sub_over_weak_maha_increases_multiplier(self) -> None:
        strengths = {
            "Sun": {"score": 3.0},
            "Jupiter": {"score": 8.0},
        }
        out = calculate_sub_dasha_bias(
            planets={"Sun": {"house": 1}, "Jupiter": {"house": 5}},
            strengths=strengths,
            influence_matrix={"matrix": {}},
            house_clusters={},
            houses={"ascendant": {"rasi": {"name": "Aries"}}},
            mahadasha="Sun",
            sub_dasha="Jupiter",
        )
        self.assertGreater(out["bias_multiplier"], 1.0)

    def test_mutual_conflict_increases_volatility(self) -> None:
        out = calculate_sub_dasha_bias(
            planets={"Sun": {"house": 1}, "Mars": {"house": 7}},
            strengths={"Sun": {"score": 5.0}, "Mars": {"score": 5.0}},
            influence_matrix={"matrix": {}, "most_conflicted_axis": ["Sun", "Mars"]},
            house_clusters={},
            houses={"ascendant": {"rasi": {"name": "Aries"}}},
            mahadasha="Sun",
            sub_dasha="Mars",
        )
        self.assertGreater(out["volatility_shift"], 0.0)

    def test_yogakaraka_increases_multiplier(self) -> None:
        out = calculate_sub_dasha_bias(
            planets={"Saturn": {"house": 10}, "Moon": {"house": 4}},
            strengths={"Saturn": {"score": 6.0}, "Moon": {"score": 5.0}},
            influence_matrix={"matrix": {}},
            house_clusters={},
            houses={"ascendant": {"rasi": {"name": "Taurus"}}},
            mahadasha="Moon",
            sub_dasha="Saturn",
        )
        self.assertGreater(out["bias_multiplier"], 1.0)

    def test_clamp_enforced(self) -> None:
        out = calculate_sub_dasha_bias(
            planets={"Sun": {"house": 1}, "Jupiter": {"house": 1}},
            strengths={"Sun": {"score": 1.0}, "Jupiter": {"score": 10.0}},
            influence_matrix={"matrix": {("Jupiter", "Sun"): 1.0}, "most_conflicted_axis": ["Sun", "Jupiter"]},
            house_clusters={},
            houses={"ascendant": {"rasi": {"name": "Aries"}}},
            mahadasha="Sun",
            sub_dasha="Jupiter",
        )
        self.assertLessEqual(out["bias_multiplier"], 1.20)
        self.assertGreaterEqual(out["bias_multiplier"], 0.85)
        self.assertLessEqual(out["volatility_shift"], 0.15)
        self.assertGreaterEqual(out["volatility_shift"], -0.15)

    def test_coherence_index_bounds(self) -> None:
        out = calculate_sub_dasha_bias(
            planets={"Sun": {"house": 1}, "Mercury": {"house": 3}},
            strengths={"Sun": {"score": 5.0}, "Mercury": {"score": 5.0}},
            influence_matrix={"matrix": {}},
            house_clusters={},
            houses={"ascendant": {"rasi": {"name": "Aries"}}},
            mahadasha="Sun",
            sub_dasha="Mercury",
        )
        self.assertGreaterEqual(out["coherence_index"], 0.0)
        self.assertLessEqual(out["coherence_index"], 1.0)


if __name__ == "__main__":
    unittest.main()

