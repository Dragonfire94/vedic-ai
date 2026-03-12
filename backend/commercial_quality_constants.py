from __future__ import annotations

import re

# Canonical Dasha definition sentence used for deterministic insertion.
# This line MUST match DASHA_DEFINITION_RE so postprocess+gate counting is aligned.
DASHA_DEFINITION_CANONICAL_LINE = "다샤(Dasha)는 시기 흐름(인생의 큰 시즌)을 보여주는 장치입니다."

# Count only the definition template shape, not all "다샤(Dasha)" mentions.
DASHA_DEFINITION_RE = re.compile(
    r"다샤\s*[\(（]\s*Dasha\s*[\)）]\s*는\s*시기 흐름\(인생의 큰 시즌\)\s*을\s*보여주는\s*장치(?:입니다|입니다\.)",
    re.IGNORECASE,
)

# Nested duplication artifact seen in production outputs.
DASHA_NESTED_PHRASE_RE = re.compile(r"시기 흐름\(\s*시기 흐름\(", re.IGNORECASE)

# Non-nested but repetitive definition artifact inside the definition template span.
DASHA_DEFINITION_REDUNDANCY_RE = re.compile(
    r"시기 흐름\(인생의 큰 시즌\).{0,80}시기 흐름\(인생의 큰 시즌\)",
    re.IGNORECASE,
)

# Canonical duplicate sentence shape seen in production that should collapse to one canonical line.
DASHA_DEFINITION_REDUNDANT_TEMPLATE_RE = re.compile(
    r"시기 흐름\(인생의 큰 시즌\)\s*을\s*보여주는\s*다샤\s*[\(（]\s*Dasha\s*[\)）]\s*는\s*"
    r"시기 흐름\(인생의 큰 시즌\)\s*을\s*보여주는\s*장치(?:입니다|입니다\.)?",
    re.IGNORECASE,
)

# The shrink fallback MUST NOT match DASHA_DEFINITION_RE.
DASHA_DEFINITION_SHRINK_LINE = "(용어 설명은 상단 참조)"

# Actionability and contamination parsing contracts.
ACTION_BOUNDARY_BASE_WINDOW = 100
ACTION_BOUNDARY_MAX_WINDOW = 160
ACTION_BOUNDARY_EXPAND_MAX_LEN = 140

ACTIONABLE_HINT_RE = re.compile(
    r"("
    r"하세요|해보세요|유지하세요|"
    r"하기\b|고정하기\b|기록하기\b|보류하기\b|확인하기\b|"
    r"고정|보류|기록|확인|유지|루틴|정리|점검|"
    r"휴식|공유|완료|시작|관찰|수립|검토|작성|"
    r"권장|추천|해도 됩니다"
    r")",
    re.IGNORECASE,
)

# Actionability/explanatory bullet contracts for chapter-level deterministic rewrite.
ACTIONABLE_BULLET_RE = ACTIONABLE_HINT_RE
EXPLANATORY_BULLET_RE = re.compile(
    r"(?:입니다\.?|합니다\.?|됩니다\.?|수\s*있(?:습니다|다))",
    re.IGNORECASE,
)

# Chapter-specific commercial action toolkit (deterministic replacements).
CORE_ACTION_TOOLKIT: dict[str, list[str]] = {
    "Current Phase": [
        "오늘 10분: 가장 큰 결정 1개를 조건, 대안, 리스크 3줄로 분해하기",
        "결정 전 2분: 지금 당장 이유 1줄 쓰고 12시간 보류하기",
        "오늘 마감 전: 진행 중 선택 1개를 검증 체크리스트로 재확인하기",
    ],
    "Career & Money": [
        "계약 전 2분: 범위, 가격, 기한을 한 문장씩 문서화하기",
        "이번 주 15분: KPI 1개와 실패 조건 1개를 명시하고 공유하기",
        "결제 전 1분: 비용 항목 3개를 체크하고 승인하기",
    ],
    "Love & Relationship Patterns": [
        "메시지 보내기 전 12시간 보류 후 확인 질문 1개만 보내기",
        "오늘 10분: 원하는 것과 싫은 것 각각 3줄 기록하기",
        "갈등 대화 전 2분: 결론 대신 합의 문장 1줄 먼저 정하기",
    ],
    "Health & Energy Rhythm": [
        "오늘 20분: 수면, 식사, 운동 중 1개만 시간과 횟수로 고정하기",
        "피로 신호 감지 시 30분 회복 블록을 일정에 예약하기",
        "하루 종료 전 3분: 에너지 소모 원인 1개와 보완 행동 1개 기록하기",
    ],
    "Mid-Term Direction": [
        "이번 달 테마 1개를 정하고 캘린더에 2개 블록을 고정하기",
        "이번 주 1시간 파일럿 1개 실행 후 결과 1줄 기록하기",
        "확장 전 5분: 중단 기준 1개를 먼저 정하고 시작하기",
    ],
    "Risk Management Points": [
        "의심 제안은 24시간 보류 후 계약 3항(범위, 대가, 기한) 점검하기",
        "오늘 5분: 리스크 1개와 완화 행동 1개를 한 줄로 적기",
        "결정 직전 2분: 최악 시나리오 1개와 대응 1개를 확인하기",
    ],
    "Growth Acceleration": [
        "이번 주 1개 실험: 1시간 파일럿을 만들고 결과, 교훈 1줄 기록하기",
        "내일 10분: 반복 작업 1개를 자동화 후보로 목록화하기",
        "주간 점검 5분: 유지할 행동 1개와 버릴 행동 1개를 선택하기",
    ],
}

# Frequently repeated generic bullets that should be replaced in core chapters.
ACTION_STEPS_GENERIC_BANNED_FPS: set[str] = {
    "결정을 서두르지 말고 오늘의 우선순위 1개만 정해 마무리하세요",
    "수면 식사 일정 루틴을 먼저 고정해 리듬을 안정시키세요",
}

# FRONT summary boundary contract: only this block may be deduped in FRONT.
ONE_PAGE_SUMMARY_START_RE = re.compile(r"(?m)^#\s*한 장 요약\s*$")
TOP_LEVEL_H1_RE = re.compile(r"(?m)^#\s+")

TRAIL_CONNECTOR_RE = re.compile(r"\b(그리고|하지만|또한|다만)\b")
TRAIL_SENTENCE_END_RE = re.compile(r"(?:합니다\.?|됩니다\.?|입니다\.?|다\.)")
TRAIL_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
TRAIL_MIN_LEN = 50

# Inline action-chain detection contracts (single source).
INLINE_ACTION_CHAIN_LIST_START_RE = re.compile(r"^\s*(?:-\s+|•\s+|\d+\.\s+)")
INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE = re.compile(r"\s+-\s+")
INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE = re.compile(r"\s*;\s*")
INLINE_ACTION_CHAIN_SLASH_SPLIT_RE = re.compile(r"\s+/\s+")
INLINE_ACTION_CHAIN_EXPLANATION_RE = re.compile(r"(?:예:|예를 들어|즉|라는 뜻)")
ACTION_STEPS_INLINE_HEADING_RE = re.compile(r"^\s*###\s*Action Steps(?P<rest>\s+.+)$", re.IGNORECASE)
ACTION_STEPS_INLINE_SPLIT_RE = re.compile(r"(?:\s+-\s+|\s+/\s+|\s*;\s*|\s+\d+\.\s+)")
CODE_FENCE_TOGGLE_RE = re.compile(r"^\s*`{3,}")

# Residual sentence validity contract after inline-chain removal.
RESIDUAL_VALID_CHAR_RE = re.compile(r"[0-9A-Za-z가-힣]")
RESIDUAL_SENTENCE_END_RE = re.compile(r"(?:[.!?]|다\.|니다\.|요\.)")
RESIDUAL_MIN_CHARS = 8

# life_cycle target-stage transition ranking contract.
# Floor-index quantiles are used intentionally; do not interpolate.
TRANSITION_INTENSITY_THRESHOLDS: tuple[float, float] = (0.33, 0.67)

PLANET_DOMAIN_MAP: dict[str, list[str]] = {
    "관계": ["Venus", "Moon"],
    "돈·커리어": ["Jupiter", "Sun", "Mercury"],
    "건강·에너지": ["Mars", "Rahu", "Ketu"],
}
