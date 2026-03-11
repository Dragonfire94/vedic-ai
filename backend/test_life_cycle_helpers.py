from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from backend.life_cycle_helpers import assign_life_stages, compute_valid_until_lifecycle


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
