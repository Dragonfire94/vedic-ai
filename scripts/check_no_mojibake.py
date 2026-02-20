#!/usr/bin/env python3
"""Fail when likely mojibake/corrupted Korean text is detected."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
EXCLUDE_DIRS = {".git", "node_modules", ".next", "__pycache__", ".venv"}
SELF_PATH = Path(__file__).resolve()

SUSPICIOUS_PATTERNS = [
    re.compile(r"\?{4,}"),
    re.compile("\uFFFD"),
    re.compile(r"\?ㅽ|\?덉|\?꾩"),
    re.compile(r"[留湲곕낯댁긽꾩쟾뒿]{2,}"),
]


def should_scan(path: Path) -> bool:
    if path.resolve() == SELF_PATH:
        return False
    if path.suffix.lower() not in INCLUDE_EXT:
        return False
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    return path.is_file()


def collect_targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(arg).resolve() for arg in argv]
    # Fallback scope for direct execution.
    return [ROOT / "backend" / "main.py"]


def main() -> int:
    failed = []
    for target in collect_targets(sys.argv[1:]):
        if target.is_dir():
            paths = [p for p in target.rglob("*") if should_scan(p)]
        else:
            paths = [target] if should_scan(target) else []

        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                failed.append((path, 0, "non-utf8 file"))
                continue

            for idx, line in enumerate(lines, start=1):
                for pattern in SUSPICIOUS_PATTERNS:
                    if pattern.search(line):
                        failed.append((path, idx, line.strip()))
                        break

    if failed:
        print("Detected suspicious mojibake/corrupted Korean text:")
        for path, line_no, line in failed:
            rel = path.relative_to(ROOT) if path.is_absolute() and str(path).startswith(str(ROOT)) else path
            print(f"- {rel}:{line_no}: {line}")
        return 1

    print("No suspicious mojibake/corrupted Korean text detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
