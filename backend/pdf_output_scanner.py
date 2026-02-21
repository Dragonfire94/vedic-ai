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
]


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def scan_forbidden_patterns(text: str, patterns: Iterable[re.Pattern] = FORBIDDEN_PATTERNS) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            start = max(0, match.start() - 32)
            end = min(len(text), match.end() + 32)
            findings.append(
                {
                    "pattern": pattern.pattern,
                    "match": match.group(0),
                    "context": text[start:end].replace("\n", " "),
                }
            )
    return findings

