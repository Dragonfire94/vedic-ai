"""Empirical tuning data analysis and safe weight adjustment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def analyze_tuning_data(filepath: str) -> dict:
    """
    Reads tuning_inputs.log (JSONL).
    Returns statistics per event_type:
        avg_gap
        avg_confidence
        event_count
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    accum: Dict[str, Dict[str, float]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            probs_raw = payload.get("probabilities", [])
            if isinstance(probs_raw, list):
                probs = [_safe_float(v) for v in probs_raw if isinstance(v, (int, float, str))]
            else:
                probs = []

            if probs:
                top = probs[0]
                second = probs[1] if len(probs) > 1 else 0.0
                separation_gap = top - second
            else:
                separation_gap = 0.0

            confidence = _safe_float(payload.get("confidence"), default=0.0)

            events = payload.get("events", [])
            if not isinstance(events, list):
                continue

            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = event.get("event_type")
                if not isinstance(event_type, str) or not event_type:
                    continue

                bucket = accum.setdefault(event_type, {"gap_sum": 0.0, "confidence_sum": 0.0, "event_count": 0.0})
                bucket["gap_sum"] += separation_gap
                bucket["confidence_sum"] += confidence
                bucket["event_count"] += 1.0

    stats: Dict[str, Dict[str, float]] = {}
    for event_type, bucket in accum.items():
        count = int(bucket["event_count"])
        if count <= 0:
            continue
        stats[event_type] = {
            "avg_gap": bucket["gap_sum"] / count,
            "avg_confidence": bucket["confidence_sum"] / count,
            "event_count": count,
        }

    return stats


def compute_weight_adjustments(stats: dict) -> dict:
    """Returns multiplicative adjustments for base_weight."""
    adjustments: Dict[str, float] = {}
    for event_type, row in stats.items():
        if not isinstance(event_type, str) or not isinstance(row, dict):
            continue

        avg_gap = _safe_float(row.get("avg_gap"), default=0.0)
        avg_conf = _safe_float(row.get("avg_confidence"), default=0.0)
        multiplier = 1.0

        if avg_gap > 1.5 and avg_conf > 0.7:
            multiplier = 1.05
        elif avg_gap < 0.5:
            multiplier = 0.95

        multiplier = max(0.9, min(1.1, multiplier))
        adjustments[event_type] = multiplier

    return adjustments


def apply_weight_adjustments(profile_path, adjustments):
    """
    Loads event_signal_profile.json,
    applies multiplier to base_weight,
    writes updated profile to new file:
    event_signal_profile_adjusted.json
    """
    source_path = Path(profile_path)
    profile = json.loads(source_path.read_text(encoding="utf-8"))

    if not isinstance(profile, dict):
        raise ValueError("event signal profile must be a JSON object")

    adjusted_profile: Dict[str, Dict[str, Any]] = json.loads(json.dumps(profile))
    for event_type, multiplier_raw in adjustments.items():
        if event_type not in adjusted_profile:
            continue
        values = adjusted_profile.get(event_type)
        if not isinstance(values, dict):
            continue

        multiplier = max(0.9, min(1.1, _safe_float(multiplier_raw, default=1.0)))
        base_weight = _safe_float(values.get("base_weight"), default=1.0)
        updated_weight = max(0.1, base_weight * multiplier)
        values["base_weight"] = round(updated_weight, 6)

    output_path = source_path.with_name("event_signal_profile_adjusted.json")
    output_path.write_text(json.dumps(adjusted_profile, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(output_path)
