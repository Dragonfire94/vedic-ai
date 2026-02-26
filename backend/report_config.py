"""Report presentation configuration constants."""

from __future__ import annotations

import re


CHAPTER_DISPLAY_NAME_KO = {
    "Executive Diagnosis": "Executive Diagnosis",
    "Current Phase": "현재 흐름의 국면",
    "Core Disposition": "당신의 핵심 기질",
    "Recurring Patterns": "반복되는 인생 패턴",
    "Emotional Fault Lines": "감정 구조와 무너지는 순간",
    "Career & Money": "커리어 & 돈 구조",
    "Love & Relationship Patterns": "연애 & 관계 패턴",
    "Health & Energy Rhythm": "건강 & 에너지 리듬",
    "Mid-Term Direction": "중기 흐름 방향",
    "Risk Management Points": "리스크 관리 포인트",
    "Growth Acceleration": "성장 가속 포인트",
    "Final Integration": "Final Integration",
    "Executive Summary": "Executive Diagnosis",
    "Purushartha Profile": "Executive Diagnosis",
    "Psychological Architecture": "당신의 핵심 기질",
    "Behavioral Risks": "감정 구조와 무너지는 순간",
    "Karmic Patterns": "반복되는 인생 패턴",
    "Stability Metrics": "리스크 관리 포인트",
    "Personality Vector": "성장 가속 포인트",
    "Life Timeline Interpretation": "현재 흐름의 국면",
    "Career & Success": "커리어 & 돈 구조",
    "Love & Relationships": "연애 & 관계 패턴",
    "Health & Body Patterns": "건강 & 에너지 리듬",
    "Confidence & Forecast": "중기 흐름 방향",
    "Remedies & Program": "성장 가속 포인트",
    "Final Summary": "Final Integration",
    "Appendix (Optional)": "Final Integration",
}

PREMIUM_12_CHAPTER_ORDER = [
    "Executive Diagnosis",
    "Current Phase",
    "Core Disposition",
    "Recurring Patterns",
    "Emotional Fault Lines",
    "Career & Money",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Mid-Term Direction",
    "Risk Management Points",
    "Growth Acceleration",
    "Final Integration",
]

PREMIUM_10_CHAPTER_ORDER = list(PREMIUM_12_CHAPTER_ORDER)

STRONG_META_LINE_PATTERNS = [
    re.compile(r"\bpredictive_compression\b", re.IGNORECASE),
    re.compile(r"\bchoice_fork\b", re.IGNORECASE),
    re.compile(r"\bstability_metrics\b", re.IGNORECASE),
    re.compile(r"\bpersonality_vector\b", re.IGNORECASE),
    re.compile(r"\bshadbala\b", re.IGNORECASE),
    re.compile(r"\bavastha\b", re.IGNORECASE),
    re.compile(r"^Evidence:\s*", re.IGNORECASE),
    re.compile(r"\bapproximate metrics\b", re.IGNORECASE),
    re.compile(r"\bstrength axis\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
]
MODERATE_META_LINE_PATTERNS: list[re.Pattern[str]] = []

CHAPTER_NARRATIVE_LINES_KO = {
    "Executive Diagnosis": [
        "지금 흐름을 한 문장으로 요약하면, 방향을 다시 고정해야 하는 시기입니다.",
        "핵심은 더 강하게 밀어붙이기보다, 흔들리는 기준을 먼저 세우는 것입니다.",
        "지금은 속도보다 일관성이 결과를 바꿉니다.",
    ],
    "Current Phase": [
        "시간이 앞으로 가면서, 초점이 바뀌는 구간입니다.",
        "예전 방식이 안 먹히는 건 실패가 아니라, 방식의 교체 신호일 수 있습니다.",
        "지금은 크게 뒤집기보다, 작은 전환을 여러 번 하는 편이 자연스럽습니다.",
        "천천히 바뀌어도 괜찮습니다.",
    ],
    "Core Disposition": [
        "당신은 마음이 한 번 움직이면 끝까지 가고 싶어 합니다.",
        "하지만 동시에, 틀에 갇히는 느낌이 들면 바로 빠져나오고 싶어집니다.",
        "그래서 자유와 안정 사이에서 줄다리기를 자주 합니다.",
        "둘 중 하나를 버리기보다, 상황마다 역할을 나누면 편해집니다.",
    ],
    "Recurring Patterns": [
        "비슷한 장면이 형태만 바꿔 다시 나타날 수 있습니다.",
        "그때마다 더 잘하려고 하기보다, 내가 자동으로 고르는 선택을 먼저 보는 게 중요합니다.",
        "패턴을 알아차리는 순간부터, 같은 일이 같은 결과로 가지 않습니다.",
        "이번에는 방향을 조금만 바꿔도 충분합니다.",
    ],
    "Emotional Fault Lines": [
        "당신이 무너질 때는 능력 부족이 아니라, 너무 오래 참았을 때입니다.",
        "참다가 한 번에 터지면, 회복보다 후회가 먼저 옵니다.",
        "그래서 괜찮은 척이 반복될수록 더 쉽게 지칩니다.",
        "미리 한 번씩 내려놓는 게, 오히려 오래 가게 합니다.",
    ],
    "Career & Money": [
        "일에서는 속도와 완성도 사이에서 늘 고민이 생깁니다.",
        "빨리 가면 마음이 닳고, 천천히 가면 불안이 올라올 수 있습니다.",
        "지금은 더 일하기보다, 덜 소모되는 방식으로 재배치하는 게 이득입니다.",
        "작게 시험하고, 잘 되는 걸 키우는 흐름이 맞습니다.",
    ],
    "Love & Relationship Patterns": [
        "관계에서는 마음이 깊은데, 표현은 오히려 조심스러울 수 있습니다.",
        "가까워질수록 확인이 필요해지고, 그게 피곤으로 바뀔 때가 있습니다.",
        "상대의 반응을 바꾸려 하기보다, 내가 편해지는 표현을 찾는 게 먼저입니다.",
        "한 번에 해결하려 하지 않아도 됩니다.",
    ],
    "Health & Energy Rhythm": [
        "에너지는 몰입할수록 소모가 빠른 편입니다.",
        "회복은 길게 쉬기보다, 짧고 규칙적인 리듬에서 더 잘 잡힙니다.",
        "무리한 일정이 겹치면 집중력보다 회복력이 먼저 흔들립니다.",
        "속도를 낮추는 순간에 몸이 먼저 안정됩니다.",
    ],
    "Mid-Term Direction": [
        "확신은 서서히 올라오는데, 중간에 스스로를 의심하는 파도가 한 번 낄 수 있습니다.",
        "그 순간 흔들린다고 해서 방향이 틀린 건 아닙니다.",
        "지금은 확신을 만들기보다, 확신이 유지되는 조건을 정하는 게 더 중요합니다.",
        "작게 확인하고 쌓아가면 흐름이 안정됩니다.",
    ],
    "Risk Management Points": [
        "버티는 힘은 있는데, 한 번 꺾이면 회복에 시간이 걸릴 수 있습니다.",
        "무리한 날의 여파가 길게 남는 편이라, 속도를 조절하는 게 실력입니다.",
        "지금은 꾸준함이 중요하되, 꾸준함을 강요하면 오히려 흐름이 깨집니다.",
        "작게 유지되는 루틴이 큰 결정을 지켜줍니다.",
    ],
    "Growth Acceleration": [
        "해결은 거창한 결심보다, 작은 조정에서 시작됩니다.",
        "지금은 마음을 다잡는 것보다, 마음이 편해지는 환경을 만드는 게 더 빠릅니다.",
        "딱 하나만 바꾼다면, 무리하는 순간을 조금 더 빨리 알아차리는 연습이 도움 됩니다.",
        "그것만으로도 리듬이 달라집니다.",
    ],
    "Final Integration": [
        "당신은 약해서 흔들리는 게 아니라, 너무 많은 걸 혼자 버티려 해서 흔들립니다.",
        "패턴을 알면, 같은 상황에서도 선택이 달라집니다.",
        "이번 흐름의 핵심은 더 하기가 아니라, 덜 닳기입니다.",
        "이제는 당신이 편해지는 방식으로 가도 됩니다.",
    ],
}
FALLBACK_NARRATIVE_LINES_KO = [
    "지금은 결론을 빨리 내리기보다, 한 번 더 확인하고 가는 편이 안전합니다.",
    "흐름이 흔들릴 수 있는 구간이라, 선택을 작게 쪼개면 훨씬 편해집니다.",
    "핵심은 더 강해지는 게 아니라, 덜 닳는 방식을 찾는 것입니다.",
]

FORBIDDEN_OUTPUT_REGEXES = [
    re.compile(r"\bshadbala\b", re.IGNORECASE),
    re.compile(r"\bavastha\b", re.IGNORECASE),
    re.compile(r"^Evidence:\s*", re.IGNORECASE),
    re.compile(r"\bapproximate metrics\b", re.IGNORECASE),
    re.compile(r"\bstrength axis\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
]

_STYLE_LABEL_PATTERNS = [
    re.compile(r"^(중심 주제|내적 줄다리기|전략 제안|요약|해석|전략|경고|리스크|기회)\s*:\s*", re.MULTILINE),
    re.compile(r"^([A-Za-z_]{3,20})\s*:\s*", re.MULTILINE),
]
_STYLE_EN_PREFIX_PATTERN = re.compile(r"\bChapter\s+\d+\b|\bExecutive\b\s*:|\bFinal\b\s*:|\bSummary\b\s*:", re.IGNORECASE)
_STYLE_PERCENT_PATTERN = re.compile(r"\b\d{1,3}%\b")
_STYLE_HARD_BAN_PATTERNS = {
    "구조": re.compile(r"구조"),
    "프로토콜": re.compile(r"프로토콜"),
    "교정": re.compile(r"교정"),
    "인덱스": re.compile(r"인덱스"),
    "축": re.compile(r"축"),
}
_STYLE_SOFT_BAN_PATTERNS = {
    "리스크": re.compile(r"리스크"),
    "지표": re.compile(r"지표"),
    "확률": re.compile(r"확률"),
    "필수": re.compile(r"필수"),
    "중요": re.compile(r"중요"),
    "전략": re.compile(r"전략"),
    "규율": re.compile(r"규율"),
    "계약": re.compile(r"계약"),
    "아키텍처": re.compile(r"아키텍처"),
    "마일스톤": re.compile(r"마일스톤"),
    "임계점": re.compile(r"임계점"),
    "검토": re.compile(r"검토"),
    "개입": re.compile(r"개입"),
}
_STYLE_HARD_BAN_REPLACEMENTS = {
    "구조": ["흐름", "패턴", "결"],
    "프로토콜": ["방식", "루틴", "순서"],
    "교정": ["정리", "가다듬기", "조율"],
    "인덱스": ["흐름", "상태", "결"],
    "축": ["중심 흐름", "핵심 방향", "결의 중심"],
}
_STYLE_SOFT_BAN_REPLACEMENTS = {
    "리스크": "부담",
    "지표": "흐름",
    "확률": "가능성",
    "필수": "먼저 챙겨야 하는",
    "중요": "눈여겨볼",
    "전략": "선택지",
    "규율": "리듬",
    "계약": "약속",
    "아키텍처": "흐름의 결",
    "마일스톤": "중간 점검 지점",
    "임계점": "버거워지는 순간",
    "검토": "다시 살펴보기",
    "개입": "손보기",
}
_STYLE_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+|(?<=…)\s+"
)
_STYLE_SOFT_DERIVED_PATTERNS = {
    "전략적": re.compile(r"전략적"),
    "필수적인": re.compile(r"필수적인"),
}
_STYLE_SOFT_DERIVED_REPLACEMENTS = {
    "전략적": "선택지 중심의",
    "필수적인": "먼저 챙겨야 하는",
}
_STYLE_DIRECTIVE_PATTERNS = {
    "필요합니다": re.compile(r"필요합니다"),
    "권장합니다": re.compile(r"권장합니다"),
    "검토하세요": re.compile(r"검토하세요"),
    "필수적입니다": re.compile(r"필수적입니다"),
}
_STYLE_HEADING_REWRITE_MAP = {
    "마음의 구조": "마음이 움직이는 방식",
    "관계 구조": "관계가 흔들리는 패턴",
    "성공 구조": "일에서 힘이 실리는 방식",
}
_STYLE_LINKER_PATTERNS = [
    re.compile(r"그러므로\s*"),
    re.compile(r"따라서\s*"),
    re.compile(r"이러한\s*"),
    re.compile(r"반면에\s*"),
    re.compile(r"즉\s*"),
    re.compile(r"또한\s*"),
]
