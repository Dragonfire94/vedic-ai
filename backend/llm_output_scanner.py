from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

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
    re.compile(r"\b20\d{2}년\b"),
    re.compile(r"\b20\d{2}\b"),
    re.compile(r"(상반기|하반기|분기|\b[Qq][1-4]\b)"),
]


def scan_forbidden_patterns(
    text: str,
    patterns: Iterable[re.Pattern] = FORBIDDEN_PATTERNS,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            findings.append(
                {
                    "pattern": pattern.pattern,
                    "match": match.group(0),
                    "context": (text[start:end] or "").replace("\n", " "),
                }
            )
    return findings


def scan_text_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    return scan_forbidden_patterns(text)
