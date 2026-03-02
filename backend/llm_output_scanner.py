from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from backend.vedic_lexicon import (
    QUARTER_HALF_PATTERN,
    YEAR_2026_PATTERN,
    YEAR_2027_PATTERN,
    YEAR_OTHER_PATTERN,
    YEAR_QUARTER_PATTERNS,
    extract_timing_map_span,
)

FORBIDDEN_PATTERNS = [
    re.compile(r"shadbala", re.IGNORECASE),
    re.compile(r"avastha", re.IGNORECASE),
    re.compile(r"evidence:", re.IGNORECASE),
    re.compile(r"strength axis", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
    re.compile(r"\bD\s*[-]?\s*(?:1|2|3|4|7|9|10|12|60)\b", re.IGNORECASE),
    re.compile(r"navamsa|navamsha|varga", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s*하우스\b"),
    re.compile(r"\b\d{1,2}(st|nd|rd|th)\s+house\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\s*°\b"),
    re.compile(r"\bbhava\b|\brashi\b|\bpada\b|\bayanamsa\b", re.IGNORECASE),
    YEAR_2026_PATTERN,
    YEAR_2027_PATTERN,
    YEAR_OTHER_PATTERN,
    QUARTER_HALF_PATTERN,
]
_YEAR_QUARTER_PATTERN_KEYS = {pattern.pattern for pattern in YEAR_QUARTER_PATTERNS}


def scan_forbidden_patterns(
    text: str,
    patterns: Iterable[re.Pattern] = FORBIDDEN_PATTERNS,
    *,
    allow_year_quarter_in_timing_map: bool = False,
) -> list[dict[str, str]]:
    raw_text = text or ""
    timing_span = extract_timing_map_span(raw_text) if allow_year_quarter_in_timing_map else None
    findings: list[dict[str, str]] = []
    for pattern in patterns:
        is_year_quarter_pattern = pattern.pattern in _YEAR_QUARTER_PATTERN_KEYS
        for match in pattern.finditer(raw_text):
            if (
                allow_year_quarter_in_timing_map
                and is_year_quarter_pattern
                and timing_span is not None
                and timing_span[0] <= match.start() < timing_span[1]
            ):
                continue
            start = max(0, match.start() - 40)
            end = min(len(raw_text), match.end() + 40)
            findings.append(
                {
                    "pattern": pattern.pattern,
                    "match": match.group(0),
                    "context": (raw_text[start:end] or "").replace("\n", " "),
                }
            )
    return findings


def scan_text_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    return scan_forbidden_patterns(text)
