"""Conservative stability audit for BTR engine.

This script is NOT for correctness testing only.
It checks for:

1. Score explosion
2. Confidence saturation
3. Over-matching due to wide ranges
4. Fallback escalation instability
5. Candidate separation robustness

Run:
python backend/test_btr_stability_audit.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if importlib.util.find_spec("swisseph") is None:
    analyze_birth_time: Optional[Callable[..., List[Dict[str, Any]]]] = None
else:
    from backend.btr_engine import analyze_birth_time


BIRTH_DATA: Dict[str, Any] = {
    "year": 1990,
    "month": 6,
    "day": 15,
    "hour": 12,
    "minute": 0,
    "latitude": 37.5665,
    "longitude": 126.9780,
    "timezone": 9,
}


def make_event(
    event_type: str,
    precision: str,
    year: Optional[int] = None,
    age_range: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Create minimal event payload compatible with BTR scoring."""
    return {
        "event_type": event_type,
        "precision_level": precision,
        "year": year,
        "month": None,
        "age_range": age_range,
        "weight": 1.0,
        "dasha_lords": [],
        "house_triggers": [],
    }


def scenario_exact_only() -> List[Dict[str, Any]]:
    return [
        make_event("career", "exact", 2015),
        make_event("relationship", "exact", 2018),
        make_event("health", "exact", 2020),
        make_event("relocation", "exact", 2012),
        make_event("finance", "exact", 2016),
    ]


def scenario_range_narrow() -> List[Dict[str, Any]]:
    return [
        make_event("career", "range", age_range=[24, 27]),
        make_event("relationship", "range", age_range=[28, 30]),
        make_event("health", "range", age_range=[29, 31]),
    ]


def scenario_range_wide() -> List[Dict[str, Any]]:
    return [
        make_event("career", "range", age_range=[20, 40]),
        make_event("relationship", "range", age_range=[18, 45]),
        make_event("health", "range", age_range=[15, 50]),
    ]


def scenario_mixed() -> List[Dict[str, Any]]:
    return [
        make_event("career", "exact", 2015),
        make_event("relationship", "range", age_range=[25, 35]),
        make_event("health", "unknown"),
        make_event("relocation", "range", age_range=[20, 22]),
    ]


def audit_scenario(name: str, events: List[Dict[str, Any]]) -> None:
    """Run one stability audit scenario and print conservative diagnostics."""
    print(f"\n--- Auditing: {name} ---")

    if analyze_birth_time is None:
        print("⚠️ Skipping audit: swisseph is not installed in this environment.")
        return

    candidates = analyze_birth_time(
        birth_date={
            "year": BIRTH_DATA["year"],
            "month": BIRTH_DATA["month"],
            "day": BIRTH_DATA["day"],
        },
        events=events,
        lat=BIRTH_DATA["latitude"],
        lon=BIRTH_DATA["longitude"],
        num_brackets=8,
        top_n=8,
    )

    scores = [c["score"] for c in candidates]
    confidences = [c["confidence"] for c in candidates]
    fallback_levels = [float(c.get("fallback_level", 0.0)) for c in candidates]

    if not scores:
        print("❌ No candidates returned.")
        return

    print("Score range:", min(scores), "→", max(scores))
    if max(scores) > 50:
        print("⚠️ Score explosion detected.")
    if min(scores) < -50:
        print("⚠️ Extreme negative score detected.")

    if len(scores) >= 2:
        ranked = sorted(scores, reverse=True)
        diff = ranked[0] - ranked[1]
        print("Top-2 score gap:", diff)
        if diff < 0.1:
            print("⚠️ Poor candidate separation.")

    print("Confidence range:", min(confidences), "→", max(confidences))
    if max(confidences) >= 0.99:
        print("⚠️ Confidence saturation near 1.0.")
    if min(confidences) < 0:
        print("⚠️ Negative confidence detected.")

    print("Fallback levels:", fallback_levels)
    if statistics.mean(fallback_levels) > 2:
        print("⚠️ Frequent fallback escalation detected.")

    print("✔ Audit complete.")


if __name__ == "__main__":
    audit_scenario("Exact Only", scenario_exact_only())
    audit_scenario("Range Narrow", scenario_range_narrow())
    audit_scenario("Range Wide", scenario_range_wide())
    audit_scenario("Mixed Precision", scenario_mixed())

    print("\n=== Conservative BTR Stability Audit Finished ===")
