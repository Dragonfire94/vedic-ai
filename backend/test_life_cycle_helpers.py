from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from backend.life_cycle_helpers import assign_life_stages, compute_life_highs_lows, compute_next_three_years, compute_repeat_patterns, compute_valid_until_lifecycle


def _row(idx: int) -> dict:
    return {
        "planet": f"P{idx}",
        "start_local": datetime(2000 + idx, 1, 1, tzinfo=dt_timezone.utc),
        "end_local": datetime(2000 + idx, 12, 31, tzinfo=dt_timezone.utc),
        "start_date": f"{2000 + idx}-01-01",
        "end_date": f"{2000 + idx}-12-31",
    }


def test_assign_life_stages_partitions_nine_rows_into_expected_groups() -> None:
    stages = assign_life_stages([_row(i) for i in range(9)])
    assert [stage["mahadasha_count"] for stage in stages] == [2, 2, 2, 3]
    assert [stage["label"] for stage in stages] == ["1단계", "2단계", "3단계", "4단계"]


def test_compute_valid_until_lifecycle_uses_next_transition_when_sooner() -> None:
    as_of_local = datetime(2026, 3, 11, tzinfo=dt_timezone.utc)
    next_local = datetime(2027, 1, 1, tzinfo=dt_timezone.utc)
    valid_until, fallback = compute_valid_until_lifecycle(as_of_local, next_local)
    assert valid_until == next_local
    assert fallback is False


def test_compute_valid_until_lifecycle_falls_back_to_three_year_cap() -> None:
    as_of_local = datetime(2026, 3, 11, tzinfo=dt_timezone.utc)
    valid_until, fallback = compute_valid_until_lifecycle(as_of_local, None)
    assert valid_until.date().isoformat() == "2029-03-11"
    assert fallback is True


def _scored_row(idx: int, *, planet: str, score: float) -> dict:
    return {
        "planet": planet,
        "planet_label": planet,
        "topic_label": f"{planet}-topic",
        "pressure_score": score,
        "start_date": f"200{idx}-01-01",
        "end_date": f"200{idx}-12-31",
        "low_window_action": f"{planet}-action",
    }


def test_compute_life_highs_lows_marks_all_transitions_medium_when_only_two_exist() -> None:
    out = compute_life_highs_lows(
        [
            _scored_row(1, planet="Moon", score=1.0),
            _scored_row(2, planet="Mercury", score=1.4),
            _scored_row(3, planet="Saturn", score=0.9),
        ]
    )
    assert [item["intensity"] for item in out["transitions"]] == ["중", "중"]
    assert out["small_transition_note_required"] is True


def test_compute_life_highs_lows_uses_strict_quantile_edges_for_three_transitions() -> None:
    out = compute_life_highs_lows(
        [
            _scored_row(1, planet="Moon", score=1.0),
            _scored_row(2, planet="Mercury", score=1.2),
            _scored_row(3, planet="Venus", score=1.2),
            _scored_row(4, planet="Saturn", score=1.8),
        ]
    )
    intensities = [item["intensity"] for item in out["transitions"]]
    assert intensities == ["중", "중", "하"]
    assert out["small_transition_note_required"] is True
    assert out["small_transition_note"] is not None


def test_compute_life_highs_lows_returns_ranked_highs_and_lows() -> None:
    out = compute_life_highs_lows(
        [
            _scored_row(1, planet="Saturn", score=0.7),
            _scored_row(2, planet="Moon", score=1.1),
            _scored_row(3, planet="Mercury", score=1.2),
            _scored_row(4, planet="Venus", score=1.7),
            _scored_row(5, planet="Jupiter", score=1.85),
        ]
    )
    assert [row["planet"] for row in out["highs"]] == ["Jupiter", "Venus", "Mercury"]
    assert [row["planet"] for row in out["lows"]] == ["Saturn", "Moon", "Mercury"]
    assert out["highs"][0]["window_label"] == "기회 창"
    assert out["lows"][0]["window_label"] == "주의 창"
    assert out["lows"][0]["care_action"] == "Saturn-action"


def test_compute_repeat_patterns_excludes_one_off_noise() -> None:
    out = compute_repeat_patterns(
        [
            _scored_row(1, planet="Venus", score=1.7),
            _scored_row(2, planet="Moon", score=1.1),
            _scored_row(3, planet="Jupiter", score=1.85),
            _scored_row(4, planet="Saturn", score=0.7),
        ]
    )
    assert "관계" in out
    assert [row["planet"] for row in out["관계"]["occurrences"]] == ["Venus", "Moon"]
    assert "돈·커리어" not in out
    assert "건강·에너지" not in out


def test_compute_repeat_patterns_keeps_domain_action_and_summary() -> None:
    out = compute_repeat_patterns(
        [
            _scored_row(1, planet="Mars", score=0.95),
            _scored_row(2, planet="Rahu", score=1.0),
            _scored_row(3, planet="Ketu", score=0.6),
        ]
    )
    assert "건강·에너지" in out
    assert out["건강·에너지"]["summary"]
    assert out["건강·에너지"]["action"]
    assert len(out["건강·에너지"]["occurrences"]) == 3


def _antardasha_row(year: int, *, mahadasha: str, bhukti: str, month: int = 1) -> dict:
    return {
        "mahadasha": mahadasha,
        "mahadasha_label": mahadasha,
        "bhukti": bhukti,
        "bhukti_label": bhukti,
        "topic_label": f"{bhukti}-topic",
        "start_local": datetime(year, month, 1, tzinfo=dt_timezone.utc),
        "end_local": datetime(year, min(month, 12), 28, tzinfo=dt_timezone.utc),
        "start_date": f"{year}-{month:02d}-01",
        "end_date": f"{year}-{month:02d}-28",
    }


def test_compute_next_three_years_limits_to_first_five_future_slots() -> None:
    as_of_local = datetime(2026, 3, 11, tzinfo=dt_timezone.utc)
    rows = [
        _antardasha_row(2026, mahadasha="Moon", bhukti="B0", month=4),
        _antardasha_row(2026, mahadasha="Moon", bhukti="B1", month=8),
        _antardasha_row(2027, mahadasha="Moon", bhukti="B2", month=2),
        _antardasha_row(2027, mahadasha="Moon", bhukti="B3", month=8),
        _antardasha_row(2028, mahadasha="Moon", bhukti="B4", month=2),
        _antardasha_row(2028, mahadasha="Moon", bhukti="B5", month=8),
    ]
    out = compute_next_three_years(
        rows,
        as_of_local=as_of_local,
        onboarding_goal="life_direction",
        concern_tokens=["우선순위"],
    )
    assert out["slot_count"] == 5
    assert [slot["bhukti"] for slot in out["slots"]] == ["B0", "B1", "B2", "B3", "B4"]
    assert out["has_slots"] is True


def test_compute_next_three_years_skips_past_rows_and_keeps_empty_closing_note() -> None:
    as_of_local = datetime(2026, 3, 11, tzinfo=dt_timezone.utc)
    out = compute_next_three_years(
        [
            _antardasha_row(2024, mahadasha="Moon", bhukti="Past"),
            _antardasha_row(2031, mahadasha="Moon", bhukti="Far"),
        ],
        as_of_local=as_of_local,
        onboarding_goal="career_money",
        concern_tokens=[],
    )
    assert out["slot_count"] == 0
    assert out["slots"] == []
    assert out["has_slots"] is False
    assert "업데이트된 지도를 확인해보세요" in out["closing_note"]


def test_compute_next_three_years_uses_concern_token_in_action() -> None:
    as_of_local = datetime(2026, 3, 11, tzinfo=dt_timezone.utc)
    out = compute_next_three_years(
        [_antardasha_row(2026, mahadasha="Moon", bhukti="Mercury", month=6)],
        as_of_local=as_of_local,
        onboarding_goal="relationship",
        concern_tokens=["전환 타이밍"],
    )
    assert out["slot_count"] == 1
    assert "전환 타이밍" in out["slots"][0]["action"]
