from __future__ import annotations

import re

PLAYBOOK_START_RE = re.compile(r"(?m)^[ \t]*#[ \t]*3개월 플레이북[ \t]*$")
PLAYBOOK_END_RE = re.compile(r"(?m)^[ \t]*(?:#[ \t]+|##[ \t]+\[)")

PLAYBOOK_LABELS: tuple[str, str, str] = ("이번 달", "다음 달", "그다음 달")

PLAYBOOK_LINE_CAUTION_PREFIX = "- 주의:"
PLAYBOOK_LINE_RULE_PREFIX = "- 규칙:"
PLAYBOOK_LINE_REASON_PREFIX = "- 이유:"
RULE_SEPARATOR = " / "

# Canonical action chapter key aliases (lowercased, normalized tokens).
# Matching must be exact on normalized key text (no partial contains).
ACTIONABLE_CHAPTER_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "Current Phase": (
        "current phase",
        "현재 흐름",
    ),
    "Career & Money": (
        "career & money",
        "career and money",
        "커리어 & 머니",
        "커리어와 돈",
    ),
    "Love & Relationship Patterns": (
        "love & relationship patterns",
        "love and relationship patterns",
        "사랑과 관계 패턴",
    ),
    "Health & Energy Rhythm": (
        "health & energy rhythm",
        "health and energy rhythm",
        "건강과 에너지 리듬",
    ),
    "Mid-Term Direction": (
        "mid-term direction",
        "mid term direction",
        "중기 방향",
    ),
    "Risk Management Points": (
        "risk management points",
        "리스크 관리 요점",
    ),
    "Growth Acceleration": (
        "growth acceleration",
        "성장 가속",
    ),
}
