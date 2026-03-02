from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

_FALLBACK_AS_OF_UTC = "1970-01-01T00:00:00Z"
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
_ALLOWLIST_HEADING_LINE_RE = re.compile(r"^\s*## \[")
_ALLOWLIST_ACTION_LINE_RE = re.compile(r"^\s*### Action Steps\s*$")
_FRONT_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]+")
_FRONT_INTERNAL_LABEL_RE = re.compile(r"\b(?:hook|risk_pack|front_summary|scene_examples|playbook_slots|current_phase|seven_day_system)\s*:", re.IGNORECASE)
_FRONT_OK_RE = re.compile(r"\bO\.?K\.?\b", re.IGNORECASE)
_FRONT_24H_RE = re.compile(r"\b24\s*h\b", re.IGNORECASE)
_FRONT_AB_RE = re.compile(r"\bA\s*/\s*B\b", re.IGNORECASE)

_CHAPTER_KEY_ALLOWLIST_TOKENS = {
    "executive",
    "diagnosis",
    "current",
    "phase",
    "core",
    "disposition",
    "recurring",
    "patterns",
    "emotional",
    "fault",
    "lines",
    "career",
    "money",
    "love",
    "relationship",
    "health",
    "energy",
    "rhythm",
    "mid",
    "term",
    "direction",
    "risk",
    "management",
    "points",
    "growth",
    "acceleration",
    "final",
    "integration",
    "action",
    "steps",
    "lagna",
    "dasha",
    "bhukti",
    "yoga",
    "karma",
}

_RISK_ONE_LINER = {
    "impulsivity": "급한 날에는 확인 절차가 빠지기 쉬워, 속도보다 검증 규칙을 먼저 붙여야 손실이 줄어듭니다.",
    "overcontrol": "완벽하게 맞추려는 압박이 커질수록 시작이 늦어지기 쉬워, 작은 실행 단위가 흐름을 살립니다.",
}

_VECTOR_MAP = {
    "skill_consolidation_phase": "실력을 결과로 연결하는 운영 순서를 먼저 고정하면 체감 성과가 빨라집니다.",
    "stability_oriented": "관계는 빠른 결론보다 합의 문장을 먼저 맞출 때 유지력이 커집니다.",
}
_THEME_CONSUMER_MAP = {
    "value alignment, attraction, and resources": "관계·돈·기준을 정렬하는 시즌",
    "value alignment": "관계·돈·기준을 정렬하는 시즌",
    "attraction and resources": "관계·돈·자원을 정리하는 시즌",
    "stabilization": "흐름을 안정화하는 시즌",
    "pressure-management": "압박을 줄이며 리듬을 복구하는 시즌",
    "restructure": "우선순위를 재정렬하는 시즌",
}

_TAG_MAP = {
    "emotional": "감정",
    "risk": "주의",
    "opportunity": "기회",
}

_PADDING_SLOT_LABELS = ("이번 달", "다음 달", "그다음 달")
_PADDING_SLOT_TAGS = ("감정", "주의", "기회")
_KEYWORD_BUCKETS = ("결정", "계약", "회복", "합의", "속도", "감정", "우선순위", "검증", "리듬", "관계", "돈", "일정")
_SELF_SABOTAGE_HIGH = 6.0
_EMOTIONAL_HIGH = 6.0
_AUTHORITY_HIGH = 6.0
_FINANCIAL_INSTABILITY_HIGH = 0.55
_VALID_SCENE_TRIGGERS = {
    "authority_friction",
    "impulsivity",
    "burnout",
    "self_sabotage",
    "overcontrol",
    "financial_instability",
    "emotional_volatility",
}

# Schema-locked MVP scene library: 20 items (career 8 / money_contract 6 / relationship 6)
_SCENE_LIBRARY: list[dict[str, str]] = [
    {
        "id": "career_01",
        "domain": "career",
        "trigger": "authority_friction",
        "scene_text": "역할과 책임이 애매한 프로젝트에서 내가 떠안고, 끝나고 나면 공은 다른 쪽으로 가는 장면이 반복되기 쉽습니다.",
        "alt_text": "기준이 흐린 업무에서 결국 내가 수습을 맡게 되고, 성과 인정은 엇갈리는 상황이 자주 생깁니다.",
    },
    {
        "id": "career_02",
        "domain": "career",
        "trigger": "authority_friction",
        "scene_text": "결정권자가 계속 바뀌어 지시가 뒤집히면서 내 일정만 무너지는 상황에서 스트레스가 크게 올라갑니다.",
        "alt_text": "윗선의 방향이 자주 바뀌는 환경에서는 계획이 깨지고 소모가 커지기 쉽습니다.",
    },
    {
        "id": "career_03",
        "domain": "career",
        "trigger": "impulsivity",
        "scene_text": "회의에서 바로 \"제가 할게요\"라고 받아놓고 나중에 범위가 커져 수습하는 흐름이 생기기 쉽습니다.",
        "alt_text": "일단 수락하고 달리다 보니 업무 범위가 불어나서 뒤늦게 정리 비용이 드는 경우가 있습니다.",
    },
    {
        "id": "career_04",
        "domain": "career",
        "trigger": "impulsivity",
        "scene_text": "좋아 보이는 기회가 뜨면 검증 없이 뛰어들었다가 중간에 우선순위가 꼬이는 장면이 나올 수 있습니다.",
        "alt_text": "시작이 빠른 만큼 확인이 늦어져서 진행 중 방향을 다시 잡느라 시간을 쓰게 되기도 합니다.",
    },
    {
        "id": "career_05",
        "domain": "career",
        "trigger": "burnout",
        "scene_text": "한동안 몰아치고 나서 어느 날 집중이 끊기며 작은 일도 버겁게 느껴지는 시기가 올 수 있습니다.",
        "alt_text": "강하게 달린 뒤 갑자기 동력이 떨어져서 일의 난이도가 체감상 크게 올라가는 때가 생깁니다.",
    },
    {
        "id": "career_06",
        "domain": "career",
        "trigger": "burnout",
        "scene_text": "무리해서 버티다가 몸이 먼저 멈추는 느낌이 오면 일정이 통째로 밀리는 경험으로 이어질 수 있습니다.",
        "alt_text": "회복을 미루고 밀어붙이면 컨디션이 한 번에 꺼지면서 계획이 크게 흔들릴 수 있습니다.",
    },
    {
        "id": "career_07",
        "domain": "career",
        "trigger": "self_sabotage",
        "scene_text": "완성도가 마음에 안 들어 공개를 미루다 타이밍을 놓치는 패턴이 나타날 수 있습니다.",
        "alt_text": "조금만 더 다듬겠다고 미루는 사이에 기회 창이 닫히는 경우가 생기기 쉽습니다.",
    },
    {
        "id": "career_08",
        "domain": "career",
        "trigger": "overcontrol",
        "scene_text": "세부를 다 잡으려다 시작이 늦어지고 결국 급하게 마감을 맞추는 흐름이 생기기 쉽습니다.",
        "alt_text": "처음부터 완벽을 만들려다 출발이 늦어져서 마지막에 속도로 때우는 일이 생길 수 있습니다.",
    },
    {
        "id": "money_01",
        "domain": "money_contract",
        "trigger": "impulsivity",
        "scene_text": "조건 확인 전에 결제나 계약부터 진행해 놓고 나중에 추가 비용을 발견하는 장면이 생길 수 있습니다.",
        "alt_text": "먼저 진행하고 나중에 약관을 보게 되면 예상 밖 지출로 기분이 상할 수 있습니다.",
    },
    {
        "id": "money_02",
        "domain": "money_contract",
        "trigger": "financial_instability",
        "scene_text": "수입이 들어와도 고정지출과 변동지출이 섞여 월말에 갑자기 불안해지는 패턴이 나타날 수 있습니다.",
        "alt_text": "돈이 새는 구멍이 보이지 않으면 어느 순간 잔고가 줄어드는 속도가 체감될 수 있습니다.",
    },
    {
        "id": "money_03",
        "domain": "money_contract",
        "trigger": "financial_instability",
        "scene_text": "작은 구독이나 소액 지출이 쌓여 어디서 새는지 모르는 상태가 길어질 수 있습니다.",
        "alt_text": "큰돈이 아니라 작은 비용들이 누적돼 지출 구조가 흐려지는 문제가 생길 수 있습니다.",
    },
    {
        "id": "money_04",
        "domain": "money_contract",
        "trigger": "self_sabotage",
        "scene_text": "가격이나 보상 이야기를 어색해해서 말을 못 하고 끝나고 나서 혼자 후회하는 상황이 생길 수 있습니다.",
        "alt_text": "조건을 분명히 묻지 못해 손해를 감수하고 나중에 아쉬움이 남는 경우가 있을 수 있습니다.",
    },
    {
        "id": "money_05",
        "domain": "money_contract",
        "trigger": "overcontrol",
        "scene_text": "모든 선택지를 완벽히 비교하려다 결정을 못 내리고 좋은 기회를 놓치는 흐름이 나올 수 있습니다.",
        "alt_text": "확신이 생길 때까지 미루다 보니 조건이 바뀌거나 제안이 사라지는 일이 생기기 쉽습니다.",
    },
    {
        "id": "money_06",
        "domain": "money_contract",
        "trigger": "emotional_volatility",
        "scene_text": "기분이 상한 상태에서 협상을 끊어버리면 나중에 다시 잡기 어려운 흐름이 생길 수 있습니다.",
        "alt_text": "감정이 올라온 순간 결론을 내리면 손익을 조정할 기회를 스스로 닫게 될 수 있습니다.",
    },
    {
        "id": "rel_01",
        "domain": "relationship",
        "trigger": "emotional_volatility",
        "scene_text": "상대의 한마디를 크게 해석해 메시지를 길게 보내고 다음 날 후회하는 장면이 생길 수 있습니다.",
        "alt_text": "감정이 올라오면 말이 길어져 오해를 키웠다고 느끼는 순간이 생길 수 있습니다.",
    },
    {
        "id": "rel_02",
        "domain": "relationship",
        "trigger": "impulsivity",
        "scene_text": "확인하기 전에 단정 짓고 말해 관계가 급격히 차가워지는 순간이 나타날 수 있습니다.",
        "alt_text": "사실 확인보다 결론이 먼저 나가면 작은 오해가 큰 거리로 이어질 수 있습니다.",
    },
    {
        "id": "rel_03",
        "domain": "relationship",
        "trigger": "self_sabotage",
        "scene_text": "가까워지면 기대가 커져 스스로 거리를 두고 상대가 헷갈려하는 패턴이 생길 수 있습니다.",
        "alt_text": "원하면서도 불안해져서 한 발 물러나 관계 흐름이 끊기는 일이 반복될 수 있습니다.",
    },
    {
        "id": "rel_04",
        "domain": "relationship",
        "trigger": "authority_friction",
        "scene_text": "관계에서 누가 맞는지로 싸움이 번지면 해결보다 자존심이 앞서는 상황이 생기기 쉽습니다.",
        "alt_text": "주도권 다툼으로 흐르면 본래 문제는 남고 감정만 커지는 패턴이 나타날 수 있습니다.",
    },
    {
        "id": "rel_05",
        "domain": "relationship",
        "trigger": "overcontrol",
        "scene_text": "상대의 반응을 미리 예측해 시나리오를 짜다 보면 자연스러운 대화가 어려워질 수 있습니다.",
        "alt_text": "머릿속에서 답을 먼저 정해두면 대화가 검사처럼 느껴져 관계 온도가 떨어질 수 있습니다.",
    },
    {
        "id": "rel_06",
        "domain": "relationship",
        "trigger": "burnout",
        "scene_text": "사람 만나는 일정이 몰리면 감정 에너지가 고갈돼 연락을 미루고 잠수처럼 보이는 시기가 생길 수 있습니다.",
        "alt_text": "관계가 겹치면 회복이 늦어져 답장을 미루게 되고 상대가 불안해하는 상황이 나올 수 있습니다.",
    },
]


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _parse_iso_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    norm = token[:-1] + "+00:00" if token.endswith("Z") else token
    try:
        parsed = datetime.fromisoformat(norm)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def resolve_as_of_utc(
    *,
    card_meta_as_of_utc: str | None = None,
    payload_as_of_utc: str | None = None,
    vedic_meta_as_of_utc: str | None = None,
) -> str:
    for candidate in (card_meta_as_of_utc, payload_as_of_utc, vedic_meta_as_of_utc):
        parsed = _parse_iso_utc(candidate)
        if parsed is not None:
            return _iso_z(parsed)
    return _FALLBACK_AS_OF_UTC


def apply_ascii_allowlist_guard_to_content(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    normalized = _normalize_newlines(text)
    out_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        if _ALLOWLIST_HEADING_LINE_RE.match(raw_line) or _ALLOWLIST_ACTION_LINE_RE.match(raw_line):
            out_lines.append(raw_line.rstrip())
            continue

        def _token_repl(match: re.Match[str]) -> str:
            token = match.group(0)
            return token if token.lower() in _CHAPTER_KEY_ALLOWLIST_TOKENS else ""

        guarded = _ASCII_TOKEN_RE.sub(_token_repl, raw_line)
        guarded = re.sub(r"[ \t]{2,}", " ", guarded).rstrip()
        out_lines.append(guarded)

    out = "\n".join(out_lines)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def front_text_sanitize_ko(text: str) -> str:
    if not isinstance(text, str):
        return ""
    out = _normalize_newlines(text)
    out = _FRONT_OK_RE.sub("확인", out)
    out = _FRONT_24H_RE.sub("24시간", out)
    out = _FRONT_AB_RE.sub("비교 실험", out)
    out = _FRONT_INTERNAL_LABEL_RE.sub("", out)
    out = _FRONT_ASCII_ALPHA_RE.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _sanitize_front_sentence_field(value: Any) -> str:
    return front_text_sanitize_ko(str(value or "")).strip()


def _map_theme_to_consumer_ko(theme: str) -> str:
    token = str(theme or "").strip()
    if not token:
        return "관계·돈·기준을 정렬하는 시즌"
    lowered = token.lower()
    for key, mapped in _THEME_CONSUMER_MAP.items():
        if key in lowered:
            return mapped
    return "관계·돈·기준을 정렬하는 시즌"


def _map_timing_tag(value: Any) -> str:
    token = str(value or "").strip().lower()
    for key, mapped in _TAG_MAP.items():
        if key in token:
            return mapped
    return "흐름 변화"


def _extract_timing_windows(dasha_context: dict[str, Any]) -> list[dict[str, Any]]:
    ctx = dasha_context if isinstance(dasha_context, dict) else {}
    timing_axis = ctx.get("timing_axis", {}) if isinstance(ctx.get("timing_axis"), dict) else {}
    windows = timing_axis.get("timing_windows")
    if not isinstance(windows, list):
        windows = ctx.get("timing_windows")
    if not isinstance(windows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        start_utc = item.get("start_utc") or item.get("start") or item.get("window_start")
        parsed = _parse_iso_utc(str(start_utc) if start_utc is not None else None)
        if parsed is None:
            continue
        note = item.get("note") or item.get("domain") or item.get("tag") or item.get("label")
        rows.append(
            {
                "start_dt": parsed,
                "start_utc": _iso_z(parsed),
                "tag": _map_timing_tag(note),
                "note": str(note or "흐름 변화").strip() or "흐름 변화",
            }
        )
    rows.sort(key=lambda row: row["start_dt"])
    return rows

def _build_timing_slots(*, as_of_utc: str, dasha_context: dict[str, Any]) -> list[dict[str, str]]:
    as_of_dt = _parse_iso_utc(as_of_utc) or _parse_iso_utc(_FALLBACK_AS_OF_UTC)
    timing_rows = _extract_timing_windows(dasha_context)
    future_rows = [row for row in timing_rows if row.get("start_dt") and row["start_dt"] >= as_of_dt]
    selected = future_rows[:3]

    slots: list[dict[str, str]] = []
    for idx in range(3):
        if idx < len(selected):
            row = selected[idx]
            slots.append(
                {
                    "label": _PADDING_SLOT_LABELS[idx],
                    "tag": str(row.get("tag") or "흐름 변화"),
                    "note": str(row.get("note") or "흐름 변화"),
                }
            )
        else:
            slots.append(
                {
                    "label": _PADDING_SLOT_LABELS[idx],
                    "tag": _PADDING_SLOT_TAGS[idx],
                    "note": "일반 가이드",
                }
            )
    return slots


def _normalize_risk_key(primary_risk: str) -> str:
    token = str(primary_risk or "").strip().lower()
    if token in {"impulsivity", "impulsivity_risk"}:
        return "impulsivity"
    if token in {"overcontrol", "overcontrol_risk"}:
        return "overcontrol"
    if token in {"burnout", "burnout_risk"}:
        return "burnout"
    if token in {"self_sabotage", "self_sabotage_risk"}:
        return "self_sabotage"
    if token in {"financial_instability", "financial_instability_3yr"}:
        return "financial_instability"
    if token in {"emotional_volatility", "emotion"}:
        return "emotional_volatility"
    if token in {"authority_conflict", "authority_conflict_risk"}:
        return "authority_friction"
    return token or "balanced"


def _to_10_scale(value: Any, default: float = 0.0) -> float:
    raw = _safe_float(value, default)
    if raw <= 1.0:
        return max(0.0, min(10.0, raw * 10.0))
    if raw <= 10.0:
        return max(0.0, min(10.0, raw))
    if raw <= 100.0:
        return max(0.0, min(10.0, raw / 10.0))
    return 10.0


def _extract_primary_risk(risks: dict[str, Any]) -> str:
    profile = risks if isinstance(risks, dict) else {}
    primary = _normalize_risk_key(str(profile.get("primary_risk") or ""))
    if primary and primary != "balanced":
        return primary
    impulsivity = _to_10_scale(profile.get("impulsivity_risk"), 0.0)
    overcontrol = _to_10_scale(profile.get("overcontrol_risk"), 0.0)
    if impulsivity >= overcontrol and impulsivity >= 6.0:
        return "impulsivity"
    if overcontrol > impulsivity and overcontrol >= 6.0:
        return "overcontrol"
    burnout = _to_10_scale(profile.get("burnout_risk"), 0.0)
    emotional = _to_10_scale(profile.get("emotional_volatility"), 0.0)
    self_sabotage = _to_10_scale(profile.get("self_sabotage_risk"), 0.0)
    if self_sabotage >= max(burnout, emotional, 6.0):
        return "self_sabotage"
    if emotional >= burnout:
        return "emotional_volatility"
    return "burnout"


def _current_phase_key(summary: dict[str, Any]) -> str:
    source = summary if isinstance(summary, dict) else {}
    structural_state = source.get("structural_state", {}) if isinstance(source.get("structural_state"), dict) else {}
    state_label = str(structural_state.get("state_label") or "").strip().lower()
    if "stabil" in state_label:
        return "stabilization"
    if "transition" in state_label or "shift" in state_label:
        return "transition"
    if "pressure" in state_label or "volatile" in state_label:
        return "pressure"
    return "stabilization"


def _pick_emotion_axis(risks: dict[str, Any]) -> tuple[str, str, str]:
    emotional = _safe_float(risks.get("emotional_volatility"), 0.0)
    burnout = _safe_float(risks.get("burnout_risk"), 0.0)
    if emotional >= burnout:
        return (
            "감정이 올라온 날",
            "반응 속도가 빨라져 말의 마찰 비용이 커질 수 있습니다.",
            "emotion_energy",
        )
    return (
        "회복 없이 버틴 주간",
        "집중력 저하로 일정 재정비 비용이 한 번에 커질 수 있습니다.",
        "emotion_energy",
    )


def _pick_relationship_axis(summary: dict[str, Any], risks: dict[str, Any]) -> tuple[str, str, str]:
    theme = str(summary.get("dominant_life_theme") or "").strip().lower()
    authority = _safe_float(risks.get("authority_conflict_risk"), 0.0)
    rel_break = _safe_float(risks.get("relationship_break_risk"), 0.0)
    if "authority" in theme or authority >= rel_break:
        return (
            "역할·권한이 애매한 상황",
            "책임 경계가 흐려져 관계와 일에서 동시 마찰 비용이 늘어납니다.",
            "relationship_authority",
        )
    return (
        "기대치 합의 없이 가까워질 때",
        "오해 누적으로 관계 피로와 정리 비용이 같이 커집니다.",
        "relationship_authority",
    )


def _build_pattern_items(summary: dict[str, Any], risks: dict[str, Any], primary_risk: str) -> list[dict[str, str]]:
    risk_map = {
        "impulsivity": ("피곤한 날", "결정 속도가 빨라져 조건 누락 비용이 커집니다."),
        "overcontrol": ("확신이 부족한 날", "결정을 늦춰 기회비용이 누적될 수 있습니다."),
        "burnout": ("과업이 겹친 주간", "회복 지연으로 다음 일정의 품질 비용이 커집니다."),
        "emotional_volatility": ("감정이 흔들린 날", "대화 톤이 거칠어져 관계 비용이 늘어납니다."),
        "authority_friction": ("권한이 모호한 회의", "책임 경계 충돌로 재작업 비용이 생깁니다."),
    }
    trigger, cost = risk_map.get(primary_risk, ("속도를 올린 날", "확인 누락으로 수습 비용이 커질 수 있습니다."))
    risk_item = {
        "category": "risk",
        "trigger": trigger,
        "cost": cost,
        "text": f"{trigger}에는 결정이 빨라져 {cost}",
    }

    em_trigger, em_cost, em_category = _pick_emotion_axis(risks)
    emotion_item = {
        "category": em_category,
        "trigger": em_trigger,
        "cost": em_cost,
        "text": f"{em_trigger}에는 {em_cost}",
    }

    rel_trigger, rel_cost, rel_category = _pick_relationship_axis(summary, risks)
    rel_item = {
        "category": rel_category,
        "trigger": rel_trigger,
        "cost": rel_cost,
        "text": f"{rel_trigger}에는 {rel_cost}",
    }
    return [risk_item, emotion_item, rel_item]


def _build_lever_items(summary: dict[str, Any], slots: list[dict[str, str]]) -> list[dict[str, str]]:
    source = summary if isinstance(summary, dict) else {}
    current_vector = source.get("current_dasha_vector", {}) if isinstance(source.get("current_dasha_vector"), dict) else {}
    current_theme_raw = str(current_vector.get("current_theme") or "").strip()
    current_theme = _map_theme_to_consumer_ko(current_theme_raw)
    season_item = {
        "category": "season",
        "text": _sanitize_front_sentence_field(
            f"지금 시즌 핵심은 {current_theme}이며, 이번 주에는 역할·기한을 한 줄로 먼저 고정합니다."
        ),
    }

    priority_tags = ("주의", "감정", "기회")
    timing_slot = None
    for tag in priority_tags:
        timing_slot = next((slot for slot in slots if str(slot.get("tag") or "") == tag), None)
        if timing_slot is not None:
            break
    if timing_slot is None and slots:
        timing_slot = slots[0]
    timing_label = str((timing_slot or {}).get("label") or "이번 달")
    timing_tag = str((timing_slot or {}).get("tag") or "주의")
    timing_item = {
        "category": "timing",
        "text": _sanitize_front_sentence_field(
            f"{timing_label}은 {timing_tag} 태그가 강해, 큰 결정을 하루 보류한 뒤 확인 질문 한 줄을 먼저 확인합니다."
        ),
    }

    career_vector = str(source.get("career_vector") or "").strip()
    relationship_vector = str(source.get("relationship_vector") or "").strip()
    vector_line = _VECTOR_MAP.get(career_vector) or _VECTOR_MAP.get(relationship_vector)
    if not vector_line:
        vector_line = "일과 관계 모두 합의 문장을 먼저 고정하면 유지 비용이 크게 줄어듭니다."
    vector_item = {
        "category": "vector",
        "text": _sanitize_front_sentence_field(
            f"{vector_line} 오늘은 우선순위 한 가지를 끝내는 절차를 먼저 실행합니다."
        ),
    }
    return [season_item, timing_item, vector_item]


def _extract_keyword(text: str) -> str:
    token = str(text or "")
    for keyword in _KEYWORD_BUCKETS:
        if keyword in token:
            return keyword
    return token[:6]


def _dedupe_lines(lines: list[str], *, target_count: int) -> list[str]:
    out: list[str] = []
    seen_text: set[str] = set()
    seen_start: set[str] = set()
    seen_keywords: set[str] = set()
    for raw in lines:
        item = str(raw or "").strip()
        if not item:
            continue
        norm = re.sub(r"\s+", " ", item)
        start_key = norm[:10]
        keyword = _extract_keyword(norm)
        if norm in seen_text:
            continue
        if start_key in seen_start:
            continue
        if keyword in seen_keywords:
            continue
        seen_text.add(norm)
        seen_start.add(start_key)
        seen_keywords.add(keyword)
        out.append(norm)
        if len(out) >= target_count:
            break
    return out

def _build_dont_do_lines(primary_risk: str) -> tuple[list[str], list[str]]:
    dont_pool_map = {
        "impulsivity": [
            "피곤한 날 큰 결론을 바로 내리지 말 것",
            "조건 확인 없이 속도로만 합의하지 말 것",
            "감정이 오른 상태에서 계약 문장을 확정하지 말 것",
            "한 번에 많은 일을 동시에 시작하지 말 것",
        ],
        "overcontrol": [
            "완벽한 답을 찾느라 실행을 무기한 미루지 말 것",
            "검토만 반복하고 첫 실행을 건너뛰지 말 것",
            "초기 오류를 두려워해 대화를 미루지 말 것",
            "하루 계획을 과도하게 세분화하지 말 것",
        ],
        "burnout": [
            "회복 없이 연속으로 강행하지 말 것",
            "일정 과부하를 의지로만 버티지 말 것",
            "피로한 날 중요한 대화를 몰아넣지 말 것",
            "휴식 시간을 뒤로 밀어두지 말 것",
        ],
    }
    do_pool_map = {
        "impulsivity": [
            "결정을 하루 보류하고 검증 질문 한 줄을 먼저 쓰기",
            "역할·돈·기한·기대치를 문장 하나로 고정하기",
            "우선순위 한 가지를 먼저 끝내고 다음 선택하기",
            "감정이 오른 날은 설명보다 확인 질문을 먼저 하기",
        ],
        "overcontrol": [
            "완성 전에 작은 실행 1건을 먼저 시작하기",
            "검토 시간을 20분으로 제한하고 바로 실행하기",
            "중요 대화는 초안을 먼저 공유하고 조정하기",
            "하루 목표를 한 문장으로 단순화하기",
        ],
        "burnout": [
            "수면·식사·일정 루틴 중 한 가지를 먼저 고정하기",
            "하루 두 번 5~10분 회복 루틴을 먼저 넣기",
            "일정 사이 회복 시간 1칸을 필수로 예약하기",
            "에너지 낮은 날엔 유지 업무부터 먼저 정리하기",
        ],
    }
    pool_key = primary_risk if primary_risk in dont_pool_map else "impulsivity"
    dont_lines = _dedupe_lines(dont_pool_map.get(pool_key, []), target_count=3)
    do_lines = _dedupe_lines(do_pool_map.get(pool_key, []), target_count=3)
    while len(dont_lines) < 3:
        dont_lines.append("감정이 오른 날 큰 결론을 바로 내리지 말 것")
        dont_lines = _dedupe_lines(dont_lines, target_count=3)
    while len(do_lines) < 3:
        do_lines.append("큰 결론은 24시간 보류 후 기준 문장을 먼저 확인하기")
        do_lines = _dedupe_lines(do_lines, target_count=3)
    return dont_lines[:3], do_lines[:3]


def _fit_text(text: str, *, min_len: int, max_len: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    if len(raw) <= max_len and len(raw) >= min_len:
        return raw

    compact = re.sub(r"\b(먼저|우선|지금은|일단|특히)\b", "", raw)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) <= max_len and len(compact) >= min_len:
        return compact

    replacements = {
        "검증 질문": "확인 질문",
        "우선순위": "순위",
        "기준 문장": "기준선",
        "회복 루틴": "회복",
        "일정": "스케줄",
        "정리합니다": "정리해요",
        "실행합니다": "실행해요",
    }
    shrunk = compact
    for src, dst in replacements.items():
        shrunk = shrunk.replace(src, dst)
    shrunk = re.sub(r"\s+", " ", shrunk).strip()
    if len(shrunk) <= max_len and len(shrunk) >= min_len:
        return shrunk

    clipped = shrunk[:max_len].rstrip(" ,.;:")
    clipped = re.sub(r"\s+", " ", clipped).strip()
    if len(clipped) < min_len:
        clipped = f"{clipped} 실행 기준을 먼저 고정합니다.".strip()
    return clipped[:max_len].rstrip(" ,.;:")


def _build_playbook_slots(slots: list[dict[str, str]], primary_risk: str) -> list[dict[str, Any]]:
    del primary_risk
    caution_map = {
        "감정": "감정이 먼저 올라오면 결론을 서두르기 쉬워 손실이 커질 수 있습니다.",
        "주의": "검증 없이 진행하면 작은 누락도 비용과 신뢰 손실로 번질 수 있습니다.",
        "기회": "한 번에 크게 확장하면 유지 비용이 늘어 기회가 약해질 수 있습니다.",
        "흐름 변화": "흐름이 바뀌는 구간에서는 기준 정리가 먼저 이득입니다.",
    }
    reason_map = {
        "감정": "감정 반응 속도가 빨라질수록 작은 오해가 커져 확인 절차가 손실을 줄입니다.",
        "주의": "주의 구간은 누락 비용이 커서 속도보다 검증이 결과를 지켜줍니다.",
        "기회": "기회 구간은 작은 테스트가 확장보다 성공 확률을 안정적으로 올립니다.",
        "흐름 변화": "흐름 변화 구간은 기준선 정리가 끝나야 다음 선택의 품질이 올라갑니다.",
    }
    rule_map = {
        "감정": ("확인 질문 1개 먼저 하기", "중요 대화 12시간 보류하기"),
        "주의": ("결정 24시간 보류하기", "조건 한 줄 문서화하기"),
        "기회": ("파일럿 1개 먼저 시작하기", "되는 것만 확대하기"),
        "흐름 변화": ("우선순위 1개 먼저 완료하기", "합의 문장 1줄 고정하기"),
    }

    out: list[dict[str, Any]] = []
    for idx, slot in enumerate(slots[:3]):
        tag = str(slot.get("tag") or "흐름 변화")
        caution = _sanitize_front_sentence_field(_fit_text(caution_map.get(tag, caution_map["흐름 변화"]), min_len=25, max_len=45))
        why = _sanitize_front_sentence_field(_fit_text(reason_map.get(tag, reason_map["흐름 변화"]), min_len=35, max_len=70))
        selected_rules = rule_map.get(tag, rule_map["흐름 변화"])
        rule_lines = [
            _sanitize_front_sentence_field(_fit_text(str(selected_rules[0]), min_len=15, max_len=35)),
            _sanitize_front_sentence_field(_fit_text(str(selected_rules[1]), min_len=15, max_len=35)),
        ]
        out.append(
            {
                "label": _PADDING_SLOT_LABELS[idx],
                "tag": tag,
                "caution": caution,
                "rules": rule_lines,
                "why": why,
            }
        )
    return out


def _domain_ko(domain: str) -> str:
    if domain == "career":
        return "일"
    if domain == "money_contract":
        return "돈"
    return "관계"


def _normalize_scene_trigger(value: str) -> str:
    trigger = _normalize_risk_key(value)
    return trigger if trigger in _VALID_SCENE_TRIGGERS else ""


def _resolve_domain_trigger(
    *,
    domain: str,
    summary: dict[str, Any],
    risk_profile: dict[str, Any],
    primary_risk: str,
) -> str:
    primary = _normalize_scene_trigger(primary_risk)
    forecast = summary.get("probability_forecast", {}) if isinstance(summary.get("probability_forecast"), dict) else {}
    financial = _safe_float(forecast.get("financial_instability_3yr"), -1.0)
    self_sabotage = _to_10_scale(risk_profile.get("self_sabotage_risk"), 0.0)
    emotional = _to_10_scale(risk_profile.get("emotional_volatility"), 0.0)
    authority = _to_10_scale(risk_profile.get("authority_conflict_risk"), 0.0)

    if domain == "career":
        return primary or "impulsivity"
    if domain == "money_contract":
        if financial >= _FINANCIAL_INSTABILITY_HIGH:
            return "financial_instability"
        return primary or "impulsivity"
    if domain == "relationship":
        if self_sabotage >= _SELF_SABOTAGE_HIGH:
            return "self_sabotage"
        if emotional >= _EMOTIONAL_HIGH:
            return "emotional_volatility"
        if authority >= _AUTHORITY_HIGH:
            return "authority_friction"
        return primary or "emotional_volatility"
    return primary or "impulsivity"


def _select_scene_examples(
    *,
    summary: dict[str, Any],
    risk_profile: dict[str, Any],
    primary_risk: str,
) -> list[dict[str, str]]:
    used_text: set[str] = set()
    out: list[dict[str, str]] = []

    for domain in ("career", "money_contract", "relationship"):
        resolved_trigger = _resolve_domain_trigger(
            domain=domain,
            summary=summary,
            risk_profile=risk_profile,
            primary_risk=primary_risk,
        )
        stage1 = [
            item for item in _SCENE_LIBRARY
            if item.get("domain") == domain
            and item.get("trigger") == resolved_trigger
        ]
        stage2 = [item for item in _SCENE_LIBRARY if item.get("domain") == domain]
        selected = None
        for pool in (stage1, stage2):
            for cand in pool:
                text = str(cand.get("scene_text") or "").strip()
                if not text or text in used_text:
                    continue
                selected = cand
                break
            if selected is not None:
                break
        if selected is None:
            continue
        used_text.add(str(selected.get("scene_text") or "").strip())
        out.append(
            {
                "id": str(selected.get("id") or "").strip(),
                "domain": domain,
                "domain_ko": _domain_ko(domain),
                "scene_text": _sanitize_front_sentence_field(selected.get("scene_text")),
                "alt_text": _sanitize_front_sentence_field(selected.get("alt_text")),
                "trigger": str(selected.get("trigger") or "").strip() or resolved_trigger,
            }
        )
    return out[:3]

def build_commercial_signal_card(
    *,
    structural_summary: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    card_meta_as_of_utc: str | None = None,
    payload_as_of_utc: str | None = None,
    vedic_meta_as_of_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = structural_summary if isinstance(structural_summary, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    as_of_utc = resolve_as_of_utc(
        card_meta_as_of_utc=card_meta_as_of_utc,
        payload_as_of_utc=payload_as_of_utc,
        vedic_meta_as_of_utc=vedic_meta_as_of_utc,
    )

    risk_profile = summary.get("behavioral_risk_profile", {}) if isinstance(summary.get("behavioral_risk_profile"), dict) else {}
    primary_risk = _extract_primary_risk(risk_profile)
    risk_line = _sanitize_front_sentence_field(_RISK_ONE_LINER.get(
        primary_risk,
        "지금은 속도를 올리기보다 리듬을 고정하는 선택이 전체 결과를 더 안정적으로 만듭니다.",
    ))
    career_vector = str(summary.get("career_vector") or "").strip()
    relationship_vector = str(summary.get("relationship_vector") or "").strip()

    timing_slots = _build_timing_slots(as_of_utc=as_of_utc, dasha_context=timing)
    pattern_items = _build_pattern_items(summary, risk_profile, primary_risk)
    lever_items = _build_lever_items(summary, timing_slots)
    dont_lines, do_lines = _build_dont_do_lines(primary_risk)
    pattern_items = [{**item, "text": _sanitize_front_sentence_field(item.get("text"))} for item in pattern_items]
    dont_lines = [_sanitize_front_sentence_field(item) for item in dont_lines]
    do_lines = [_sanitize_front_sentence_field(item) for item in do_lines]
    scenes = _select_scene_examples(
        summary=summary,
        risk_profile=risk_profile,
        primary_risk=primary_risk,
    )
    front_intro = [
        _sanitize_front_sentence_field("지금은 속도를 버리는 시기가 아니라, 속도에 규칙을 붙이면 손실이 크게 줄어드는 구간입니다."),
        _sanitize_front_sentence_field("핵심은 큰 결심보다 작은 규칙을 고정해 반복 손실을 끊는 데 있습니다."),
    ]
    front_closing = [
        _sanitize_front_sentence_field("지금 구간은 크게 벌리기보다 기준을 고정할수록 결과의 흔들림이 줄어드는 흐름입니다."),
    ]
    front_templates = [
        _sanitize_front_sentence_field("합의 1줄: 우리는 A를 B 범위로, C 기준까지, D 시점까지 진행합니다."),
        _sanitize_front_sentence_field("확인 질문 1개: 지금 내가 이해한 내용이 맞는지 한 번만 확인할게요."),
        _sanitize_front_sentence_field("24h 보류: 오늘은 결론을 보류하고 내일 10분만 다시 보고 결정합니다."),
    ]
    front_scene_details = [
        _sanitize_front_sentence_field("일에서는 역할 경계를 먼저 정리하면 수습 비용이 크게 줄어듭니다."),
        _sanitize_front_sentence_field("돈에서는 조건 확인을 먼저 두면 불필요한 누수를 막을 수 있습니다."),
        _sanitize_front_sentence_field("관계에서는 결론보다 확인 질문을 먼저 두면 오해가 줄어듭니다."),
    ]

    card_meta = {
        "as_of_utc": as_of_utc,
        "source_hash": None,
        "dasha_engine_profile": str(timing.get("dasha_engine_profile") or "").strip() or None,
        "ayanamsa_profile": str(timing.get("ayanamsa_profile") or "").strip() or None,
    }
    card_ko = {
        "front_intro": front_intro,
        "hook": [
            risk_line,
            _sanitize_front_sentence_field("지금은 크게 바꾸기보다 기준을 정리해 유지력을 올리는 편이 유리합니다."),
        ],
        "current_phase": {
            "headline": _sanitize_front_sentence_field("현재는 정리와 고정이 성과를 만드는 구간입니다."),
            "window_left_text": _sanitize_front_sentence_field("지금 구간의 핵심은 속도보다 일관성입니다."),
        },
        "scenes": [item.get("scene_text") for item in scenes if isinstance(item, dict)],
        "scene_examples": scenes,
        "brakes": list(do_lines),
        "front_scene_details": front_scene_details,
        "vectors": {
            "career_line": _sanitize_front_sentence_field(
                _VECTOR_MAP.get(career_vector, "일에서는 반복 가능한 운영 방식을 먼저 고정하는 전략이 효과적입니다.")
            ),
            "relationship_line": _sanitize_front_sentence_field(
                _VECTOR_MAP.get(relationship_vector, "관계에서는 빠른 답보다 기준을 맞추는 과정이 중요합니다.")
            ),
            "inner_outer_line": _sanitize_front_sentence_field("내면의 확신과 외부 실행 속도를 분리해서 관리하면 소모를 크게 줄일 수 있습니다."),
        },
        "three_month": timing_slots,
        "playbook_slots": _build_playbook_slots(timing_slots, primary_risk),
        "risk_pack": {
            "primary": risk_line,
            "primary_risk": primary_risk or "balanced",
        },
        "front_summary": {
            "patterns": pattern_items,
            "levers": lever_items,
            "dont": dont_lines,
            "do": do_lines,
        },
        "front_templates": front_templates,
        "front_closing": front_closing,
        "seven_day_system": {
            "items": [
                _sanitize_front_sentence_field("우선순위 1개 완료"),
                _sanitize_front_sentence_field("큰 결정 24시간 보류"),
                _sanitize_front_sentence_field("합의 문장 1줄 고정"),
                _sanitize_front_sentence_field("회복 루틴 2회(각 5~10분)"),
            ],
            "operating_steps": [
                _sanitize_front_sentence_field("아침 3분: 오늘 우선순위 1개를 먼저 정합니다."),
                _sanitize_front_sentence_field("점심 2분: 짧은 회복 루틴으로 리듬을 다시 맞춥니다."),
                _sanitize_front_sentence_field("저녁 5분: 보류할 결론과 합의 1줄을 정리합니다."),
            ],
            "cta": _sanitize_front_sentence_field("당신은 큰 결심보다 작은 규칙 고정이 운을 바꾸는 타입입니다."),
        },
    }
    return card_meta, card_ko


def render_signal_card_ko_for_prompt(card_ko: dict[str, Any]) -> str:
    card = card_ko if isinstance(card_ko, dict) else {}
    lines: list[str] = ["핵심 흐름"]
    for item in card.get("hook", []) if isinstance(card.get("hook"), list) else []:
        if isinstance(item, str) and item.strip():
            lines.append(f"- {item.strip()}")

    front_summary = card.get("front_summary", {}) if isinstance(card.get("front_summary"), dict) else {}
    patterns = front_summary.get("patterns", []) if isinstance(front_summary.get("patterns"), list) else []
    levers = front_summary.get("levers", []) if isinstance(front_summary.get("levers"), list) else []
    dont = front_summary.get("dont", []) if isinstance(front_summary.get("dont"), list) else []
    do = front_summary.get("do", []) if isinstance(front_summary.get("do"), list) else []

    if patterns:
        lines.append("")
        lines.append("핵심 패턴")
        for item in patterns[:3]:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")

    if levers:
        lines.append("")
        lines.append("시즌 레버")
        for item in levers[:3]:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")

    if dont or do:
        lines.append("")
        lines.append("금지 권장")
        for item in dont[:3]:
            text = str(item or "").strip()
            if text:
                lines.append(f"- 금지 {text}")
        for item in do[:3]:
            text = str(item or "").strip()
            if text:
                lines.append(f"- 권장 {text}")

    slots = card.get("playbook_slots", []) if isinstance(card.get("playbook_slots"), list) else []
    if slots:
        lines.append("")
        lines.append("3개월 플레이북")
        for slot in slots[:3]:
            if not isinstance(slot, dict):
                continue
            label = str(slot.get("label") or "").strip()
            caution = str(slot.get("caution") or "").strip()
            rules = slot.get("rules", []) if isinstance(slot.get("rules"), list) else []
            why = str(slot.get("why") or "").strip()
            if label:
                lines.append(f"- {label}")
            if caution:
                lines.append(f"  - 주의 {caution}")
            if rules:
                joined = " · ".join(str(r).strip() for r in rules[:2] if str(r).strip())
                if joined:
                    lines.append(f"  - 규칙 {joined}")
            if why:
                lines.append(f"  - 이유 {why}")

    scenes = card.get("scene_examples", []) if isinstance(card.get("scene_examples"), list) else []
    if scenes:
        lines.append("")
        lines.append("상황 예시")
        for row in scenes[:3]:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("domain_ko") or "").strip()
            text = str(row.get("scene_text") or "").strip()
            if text and domain:
                lines.append(f"- [{domain}] {text}")

    system = card.get("seven_day_system", {}) if isinstance(card.get("seven_day_system"), dict) else {}
    items = system.get("items", []) if isinstance(system.get("items"), list) else []
    if items:
        lines.append("")
        lines.append("7일 시스템")
        for item in items[:4]:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")
    cta = str(system.get("cta") or "").strip()
    if cta:
        lines.append(f"- {cta}")

    text = "\n".join(lines).strip()
    return apply_ascii_allowlist_guard_to_content(text)


def render_chapter_blocks_draft_md(draft_md: str) -> str:
    text = str(draft_md or "").strip()
    if not text:
        return ""
    return apply_ascii_allowlist_guard_to_content(text)
