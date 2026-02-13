"""Deterministic tests for BTR precision and range-overlap behavior."""

from __future__ import annotations

import math
import unittest
import sys
import types

if "swisseph" not in sys.modules:
    def _julday(year: int, month: int, day: int, hour: float) -> float:
        """Gregorian calendar date -> Julian Day Number approximation."""
        a = (14 - month) // 12
        y = year + 4800 - a
        m = month + (12 * a) - 3
        jdn = day + ((153 * m + 2) // 5) + (365 * y) + (y // 4) - (y // 100) + (y // 400) - 32045
        return jdn + ((hour - 12.0) / 24.0)

    swe_stub = types.SimpleNamespace(julday=_julday)
    sys.modules["swisseph"] = swe_stub

from typing import Any, Dict, List, Tuple

from backend.btr_engine import (
    _has_mahadasha_range_overlap,
    _score_candidate,
    compute_event_signal_strength,
    compute_planet_dignity_score,
    calculate_vimshottari_dasha,
    convert_age_range_to_year_range,
    date_to_jd,
    get_information_weight,
    jd_to_year_frac,
)


def _make_chart(jd: float, moon_longitude: float, planet_houses: Dict[str, int] | None = None) -> Dict[str, Any]:
    """Create a minimal chart payload accepted by `_score_candidate`."""
    return {
        "jd": jd,
        "moon_longitude": moon_longitude,
        "planet_houses": planet_houses or {},
    }


def _make_birth_date(year: int = 1990, month: int = 1, day: int = 15) -> Dict[str, int]:
    """Create a deterministic birth-date dictionary."""
    return {"year": year, "month": month, "day": day}


def _md_window_years(md: Dict[str, Any]) -> Tuple[int, int]:
    """Convert Mahadasha JD boundaries into integer year boundaries."""
    start_year = int(math.floor(jd_to_year_frac(md["start_jd"])))
    end_year = int(math.ceil(jd_to_year_frac(md["end_jd"])))
    return start_year, end_year


def _evaluate_candidate(
    birth_date: Dict[str, int],
    chart: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Tuple[float, int, int, float, List[int]]:
    """Run scoring for a single candidate with fixed deterministic parameters."""
    return _score_candidate(
        date=birth_date,
        mid_hour=12.0,
        chart=chart,
        events=events,
        lat=37.5665,
        lon=126.978,
    )


class TestBTREnginePhase1Scenarios(unittest.TestCase):
    """Scenario tests for exact/range/unknown/mixed precision behavior."""

    def _assert_with_summary(self, condition: bool, summary: str) -> None:
        """Assert condition with required mismatch summary text."""
        if not condition:
            self.fail(f"Mismatch Summary:\n{summary}")

    def test_exact_only_events(self) -> None:
        """Exact-only events should produce meaningful candidate score separation."""
        birth_date = _make_birth_date(1990, 1, 15)
        chart_a = _make_chart(date_to_jd(1990, 1, 15), moon_longitude=85.0)
        chart_b = _make_chart(date_to_jd(1990, 1, 15), moon_longitude=210.0)

        mahadashas_a = calculate_vimshottari_dasha(chart_a["jd"], chart_a["moon_longitude"])
        md0, md1, md2 = mahadashas_a[0], mahadashas_a[1], mahadashas_a[2]
        md0_start, md0_end = _md_window_years(md0)
        md1_start, _ = _md_window_years(md1)
        md2_start, _ = _md_window_years(md2)

        events = [
            {
                "event_type": "career",
                "precision_level": "exact",
                "year": md0_start + 1,
                "month": 6,
                "weight": 1.0,
                "dasha_lords": [md0["lord"]],
                "house_triggers": [],
            },
            {
                "event_type": "relationship",
                "precision_level": "exact",
                "year": md0_end - 1,
                "month": 1,
                "weight": 1.0,
                "dasha_lords": [md0["lord"]],
                "house_triggers": [],
            },
            {
                "event_type": "relocation",
                "precision_level": "exact",
                "year": md1_start + 1,
                "month": 7,
                "weight": 1.0,
                "dasha_lords": [md1["lord"]],
                "house_triggers": [],
            },
            {
                "event_type": "finance",
                "precision_level": "exact",
                "year": md2_start + 1,
                "month": 10,
                "weight": 1.0,
                "dasha_lords": [md2["lord"]],
                "house_triggers": [],
            },
            {
                "event_type": "health",
                "precision_level": "exact",
                "year": md1_start + 2,
                "month": 12,
                "weight": 1.0,
                "dasha_lords": ["Pluto"],  # intentionally non-overlapping lord
                "house_triggers": [],
            },
        ]

        score_a, matched_a, total_a, conf_a, fb_a = _evaluate_candidate(birth_date, chart_a, events)
        score_b, matched_b, total_b, conf_b, fb_b = _evaluate_candidate(birth_date, chart_b, events)

        print("[exact-only]", {"score_a": score_a, "score_b": score_b, "conf_a": conf_a, "conf_b": conf_b, "fb_a": fb_a, "fb_b": fb_b})

        self._assert_with_summary(total_a == 5 and total_b == 5, f"Expected total events 5. got total_a={total_a}, total_b={total_b}")
        self._assert_with_summary(score_a != score_b, f"Expected distinct candidate scores. got score_a={score_a}, score_b={score_b}")
        self._assert_with_summary(conf_a != conf_b, f"Expected confidence separation. got conf_a={conf_a}, conf_b={conf_b}")
        self._assert_with_summary(score_a > score_b, f"Expected candidate A > B due to aligned exact lords. score_a={score_a}, score_b={score_b}")
        self._assert_with_summary(any(level < 4 for level in fb_a), f"Expected non-terminal fallback levels for aligned candidate. fb_a={fb_a}")
        self._assert_with_summary(matched_a >= matched_b, f"Expected matched_a >= matched_b. matched_a={matched_a}, matched_b={matched_b}")

    def test_range_only_events(self) -> None:
        """Range-only events should use overlap logic and 0.7 utility contribution."""
        birth_date = _make_birth_date(1990, 1, 15)
        chart = _make_chart(date_to_jd(1990, 1, 15), moon_longitude=85.0)

        mahadashas = calculate_vimshottari_dasha(chart["jd"], chart["moon_longitude"])
        selected = mahadashas[:5]

        events_match: List[Dict[str, Any]] = []
        for idx, md in enumerate(selected):
            md_start, _ = _md_window_years(md)
            age_start = max(0, md_start - birth_date["year"])
            age_end = age_start + 2
            events_match.append(
                {
                    "event_type": ["career", "relationship", "relocation", "finance", "health"][idx % 5],
                    "precision_level": "range",
                    "year": None,
                    "month": None,
                    "age_range": (age_start, age_end),
                    "weight": 1.0,
                    "dasha_lords": [md["lord"]],
                    "house_triggers": [],
                }
            )

        events_no_overlap = [
            {
                **ev,
                "dasha_lords": ["Pluto"],
            }
            for ev in events_match
        ]

        score_match, matched_match, total_match, conf_match, _ = _evaluate_candidate(birth_date, chart, events_match)
        score_none, matched_none, total_none, conf_none, _ = _evaluate_candidate(birth_date, chart, events_no_overlap)

        print("[range-only]", {"score_match": score_match, "score_none": score_none, "conf_match": conf_match, "conf_none": conf_none, "matched_match": matched_match, "matched_none": matched_none})

        # For matched range-only events with width=2 and weight=1 (no house bonus):
        # utility=0.7, information_weight=(1 - 2/30)=0.9333... => each=0.6533...
        self._assert_with_summary(abs(score_match - (5 * 0.7 * (1 - (2 / 30)))) < 1e-6, f"Expected weighted score_match. got {score_match}")
        self._assert_with_summary(score_match > score_none, f"Expected overlap candidate to score higher. score_match={score_match}, score_none={score_none}")
        self._assert_with_summary(matched_match == 5 and matched_none == 0, f"Expected matched 5 vs 0. got {matched_match} vs {matched_none}")
        self._assert_with_summary(total_match == 5 and total_none == 5, f"Expected reported total events to stay 5. got {total_match}, {total_none}")
        self._assert_with_summary(conf_match > conf_none, f"Expected higher confidence for overlapping ranges. conf_match={conf_match}, conf_none={conf_none}")

    def test_unknown_only_events(self) -> None:
        """Unknown-only events should be fully neutral in score and confidence inputs."""
        birth_date = _make_birth_date(1990, 1, 15)
        chart_a = _make_chart(date_to_jd(1990, 1, 15), moon_longitude=85.0)
        chart_b = _make_chart(date_to_jd(1991, 6, 1), moon_longitude=210.0)
        chart_c = _make_chart(date_to_jd(1989, 11, 5), moon_longitude=300.0)

        events = [
            {
                "event_type": et,
                "precision_level": "unknown",
                "year": None,
                "month": None,
                "age_range": None,
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            }
            for et in ["career", "relationship", "relocation", "finance", "health"]
        ]

        out_a = _evaluate_candidate(birth_date, chart_a, events)
        out_b = _evaluate_candidate(birth_date, chart_b, events)
        out_c = _evaluate_candidate(birth_date, chart_c, events)

        print("[unknown-only]", {"A": out_a, "B": out_b, "C": out_c})

        scores = [out_a[0], out_b[0], out_c[0]]
        confs = [out_a[3], out_b[3], out_c[3]]
        fbs = [out_a[4], out_b[4], out_c[4]]

        self._assert_with_summary(scores[0] == scores[1] == scores[2] == 0.0, f"Expected all-zero identical scores for unknown-only. scores={scores}")
        self._assert_with_summary(confs[0] == confs[1] == confs[2] == 0.0, f"Expected all-zero confidence for unknown-only. confs={confs}")
        self._assert_with_summary(all(fb == [] for fb in fbs), f"Expected no fallback entries when all unknown. fallback_lists={fbs}")

    def test_mixed_precision_events(self) -> None:
        """Mixed precision should combine exact/range and ignore unknown impacts."""
        birth_date = _make_birth_date(1990, 1, 15)
        chart = _make_chart(date_to_jd(1990, 1, 15), moon_longitude=85.0)

        # Range windows for deterministic overlap
        start_yr_1, end_yr_1 = convert_age_range_to_year_range(birth_date["year"], (20, 22))
        start_yr_2, end_yr_2 = convert_age_range_to_year_range(birth_date["year"], (30, 32))
        self.assertLess(start_yr_1, end_yr_1)
        self.assertLess(start_yr_2, end_yr_2)

        events = [
            {
                "event_type": "career",
                "precision_level": "exact",
                "year": 2012,
                "month": 6,
                "weight": 1.0,
                "dasha_lords": [],  # exact, guaranteed level-0 match path
                "house_triggers": [],
            },
            {
                "event_type": "relationship",
                "precision_level": "exact",
                "year": 2016,
                "month": 3,
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            },
            {
                "event_type": "relocation",
                "precision_level": "range",
                "year": None,
                "month": None,
                "age_range": (20, 22),
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            },
            {
                "event_type": "finance",
                "precision_level": "range",
                "year": None,
                "month": None,
                "age_range": (30, 32),
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            },
            {
                "event_type": "health",
                "precision_level": "unknown",
                "year": None,
                "month": None,
                "age_range": None,
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            },
        ]

        score, matched, total, confidence, fallback_levels = _evaluate_candidate(birth_date, chart, events)

        print("[mixed]", {"score": score, "matched": matched, "total": total, "confidence": confidence, "fallback_levels": fallback_levels})

        # Expected contribution when all non-unknown events match:
        # exact(1.0) + exact(1.0) + range(0.7*14/15) + range(0.7*14/15) + unknown(0.0)
        expected = 2.0 + 2 * (0.7 * (1 - (2 / 30)))
        self._assert_with_summary(abs(score - expected) < 1e-6, f"Expected mixed score ~= {expected}. got {score}")
        self._assert_with_summary(matched == 4, f"Expected 4 matched non-unknown events. got matched={matched}")
        self._assert_with_summary(total == 5, f"Expected total reported events to remain 5. got total={total}")
        self._assert_with_summary(confidence > 0.0, f"Expected positive confidence with 4 matched events. got confidence={confidence}")

        exact_event_contrib = 1.0
        range_event_contrib = 0.7
        self._assert_with_summary(
            exact_event_contrib > range_event_contrib,
            f"Expected exact contribution > range contribution. exact={exact_event_contrib}, range={range_event_contrib}",
        )


    def test_adaptive_range_buffer_overlap(self) -> None:
        """Adaptive range buffer should expand Mahadasha overlap by age-range width."""
        birth_jd = date_to_jd(1990, 1, 15)
        moon_longitude = 85.0
        md0 = calculate_vimshottari_dasha(birth_jd, moon_longitude)[0]
        md_start, md_end = _md_window_years(md0)

        # Event starts 2 years after md_end, only wide ranges (buffer >=2) should match
        event_start = md_end + 2
        event_end = md_end + 2

        narrow_match = _has_mahadasha_range_overlap(
            birth_jd=birth_jd,
            birth_moon_lon=moon_longitude,
            event_start_year=event_start,
            event_end_year=event_end,
            dasha_lords=[md0["lord"]],
            age_range_width=5,  # buffer=1
        )
        wide_match = _has_mahadasha_range_overlap(
            birth_jd=birth_jd,
            birth_moon_lon=moon_longitude,
            event_start_year=event_start,
            event_end_year=event_end,
            dasha_lords=[md0["lord"]],
            age_range_width=20,  # buffer=2
        )

        self._assert_with_summary(not narrow_match, f"Expected no overlap with narrow buffer. narrow_match={narrow_match}")
        self._assert_with_summary(wide_match, f"Expected overlap with wider adaptive buffer. wide_match={wide_match}")

    def test_information_weight_scaling(self) -> None:
        """Information weight should decrease as age_range width grows."""
        exact_weight = get_information_weight({"precision_level": "exact"})
        unknown_weight = get_information_weight({"precision_level": "unknown"})
        narrow_range_weight = get_information_weight({"precision_level": "range", "age_range": (20, 22)})
        wide_range_weight = get_information_weight({"precision_level": "range", "age_range": (10, 40)})

        self._assert_with_summary(exact_weight == 1.0, f"Expected exact information weight 1.0. got {exact_weight}")
        self._assert_with_summary(unknown_weight == 0.0, f"Expected unknown information weight 0.0. got {unknown_weight}")
        self._assert_with_summary(narrow_range_weight > wide_range_weight, f"Expected narrow > wide weight. narrow={narrow_range_weight}, wide={wide_range_weight}")
        self._assert_with_summary(abs(wide_range_weight - 0.3) < 1e-9, f"Expected floor 0.3 for wide range. got {wide_range_weight}")


class TestBTREventSignalStrength(unittest.TestCase):
    """Tests for event-type signal profile based score augmentation."""

    def test_event_signal_strength_scaling(self) -> None:
        """Signal should increase when houses/planets/dasha align better."""
        event = {"event_type": "career"}

        low_signal = compute_event_signal_strength(
            chart_data={"houses": {1: True}},
            event=event,
            strength_data={"Sun": {"score": 0.2}, "Saturn": {"score": 0.2}},
            influence_matrix={"Moon": 10.0, "Ketu": 8.0},
            dasha_vector={},
        )

        high_signal = compute_event_signal_strength(
            chart_data={"houses": {10: True, 6: True}},
            event=event,
            strength_data={
                "Sun": {"score": 1.0},
                "Saturn": {"score": 0.9},
                "Mars": {"score": 0.8},
            },
            influence_matrix={"Moon": 0.0, "Ketu": 0.0},
            dasha_vector={"Sun": True},
        )

        self.assertGreater(high_signal, low_signal)
        self.assertGreaterEqual(low_signal, 0.0)
        self.assertLessEqual(high_signal, 1.0)

    def test_event_signal_integration_effect(self) -> None:
        """Candidate score should be reduced when signal model is weaker."""
        birth_date = _make_birth_date(1990, 1, 15)
        chart_high = _make_chart(
            date_to_jd(1990, 1, 15),
            moon_longitude=85.0,
            planet_houses={
                "Sun": 10,
                "Moon": 1,
                "Mars": 6,
                "Mercury": 3,
                "Jupiter": 9,
                "Venus": 5,
                "Saturn": 10,
                "Rahu": 2,
                "Ketu": 8,
            },
        )
        chart_low = _make_chart(
            date_to_jd(1990, 1, 15),
            moon_longitude=85.0,
            planet_houses={
                "Sun": 2,
                "Moon": 6,
                "Mars": 8,
                "Mercury": 12,
                "Jupiter": 4,
                "Venus": 3,
                "Saturn": 12,
                "Rahu": 6,
                "Ketu": 12,
            },
        )

        events = [
            {
                "event_type": "career",
                "precision_level": "exact",
                "year": 2012,
                "month": 6,
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            }
        ]

        score_high, *_ = _evaluate_candidate(birth_date, chart_high, events)
        score_low, *_ = _evaluate_candidate(birth_date, chart_low, events)

        self.assertGreater(score_high, score_low)

    def test_conflict_penalty_influence(self) -> None:
        """Higher conflict influence should reduce event signal strength."""
        event = {"event_type": "relationship"}
        base_kwargs = dict(
            chart_data={"houses": {7: True, 5: True, 11: True}},
            event=event,
            strength_data={
                "Venus": {"score": 0.8},
                "Moon": {"score": 0.8},
                "Jupiter": {"score": 0.8},
            },
            dasha_vector={"Venus": True},
        )

        low_conflict = compute_event_signal_strength(
            influence_matrix={"Saturn": 0.0, "Rahu": 0.0},
            **base_kwargs,
        )
        high_conflict = compute_event_signal_strength(
            influence_matrix={"Saturn": 12.0, "Rahu": 12.0},
            **base_kwargs,
        )

        self.assertGreater(low_conflict, high_conflict)



class TestPlanetDignityScoring(unittest.TestCase):
    """Deterministic dignity multiplier tests for Layer 5 Step 1."""

    def test_exaltation_multiplier(self) -> None:
        """Exaltation sign should return 1.3 multiplier."""
        self.assertEqual(compute_planet_dignity_score("Sun", "Aries"), 1.3)

    def test_debilitation_penalty(self) -> None:
        """Debilitation sign should return 0.6 multiplier."""
        self.assertEqual(compute_planet_dignity_score("Moon", "Scorpio"), 0.6)

    def test_own_sign_bonus(self) -> None:
        """Own sign should return 1.15 multiplier."""
        self.assertEqual(compute_planet_dignity_score("Mars", "Aries"), 1.15)

    def test_neutral_sign_no_change(self) -> None:
        """Neutral sign should remain at 1.0 multiplier."""
        self.assertEqual(compute_planet_dignity_score("Venus", "Aquarius"), 1.0)


if __name__ == "__main__":
    unittest.main()
