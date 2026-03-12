from __future__ import annotations

from typing import Any

LIFE_CYCLE_LITE_H2_ORDER = [
    "## cover/meta",
    "## How to use 1p",
    "## 인생 구조 한 장 요약",
    "## 4단계 인생 구조",
    "## 현재 위치",
    "## 마하다샤 단계 목록",
    "## 방법론 카드",
    "## valid_until 설명",
    "## CTA-lite",
    "## 면책/윤리/데이터 보호",
]


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _render_stage_lines(payload: dict[str, Any]) -> list[str]:
    stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    out: list[str] = []
    if not stages:
        return ["- 아직 충분한 단계 데이터를 만들지 못했습니다. 현재 위치부터 다시 확인해 주세요."]
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        current_mark = " (현재 위치)" if stage.get("is_current") else ""
        out.append(
            f"- {_safe_text(stage.get('label'), '단계')}{current_mark}: {_safe_text(stage.get('start_date'), '?')} ~ {_safe_text(stage.get('end_date'), '?')} | {_safe_text(stage.get('dominant_planet_label'), '현재')} | {_safe_text(stage.get('summary_label'), '흐름 정리') }"
        )
    return out or ["- 단계 요약을 생성하지 못했습니다."]


def _render_mahadasha_lines(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("mahadasha_sequence") if isinstance(payload.get("mahadasha_sequence"), list) else []
    out: list[str] = []
    for row in rows[:9]:
        if not isinstance(row, dict):
            continue
        current_mark = " (현재)" if row.get("is_current") else ""
        out.append(
            f"- {_safe_text(row.get('start_date'), '?')} ~ {_safe_text(row.get('end_date'), '?')}: {_safe_text(row.get('planet_label'), '현재')} | {_safe_text(row.get('theme'), '흐름 설명')}{current_mark}"
        )
    return out or ["- 마하다샤 단계 목록을 아직 만들지 못했습니다."]


def render_life_cycle_sections(sections: list[tuple[str, list[str]]]) -> str:
    blocks: list[str] = []
    for heading, body_lines in sections:
        blocks.append("\n".join([heading, *body_lines]).strip())
    return "\n\n".join(blocks).strip() + "\n"


def build_life_cycle_lite_sections(payload: dict[str, Any]) -> list[tuple[str, list[str]]]:
    subject_name = _safe_text(payload.get("subject_name"), "당신")
    summary_hook = _safe_text(payload.get("summary_hook"), f"{subject_name}님의 인생 흐름을 큰 단계로 다시 읽는 보고서입니다.")
    current_stage = payload.get("current_stage") if isinstance(payload.get("current_stage"), dict) else {}
    current_mahadasha = payload.get("current_mahadasha") if isinstance(payload.get("current_mahadasha"), dict) else {}
    focus_tokens = payload.get("focus_tokens") if isinstance(payload.get("focus_tokens"), list) else []
    concern_tokens = payload.get("concern_tokens") if isinstance(payload.get("concern_tokens"), list) else []

    focus_line = ", ".join(str(token) for token in focus_tokens[:2]) if focus_tokens else payload.get("summary_target")
    concern_line = ", ".join(str(token) for token in concern_tokens[:3]) if concern_tokens else "현재 우선순위"
    valid_until = _safe_text(payload.get("valid_until"), "미정")
    next_mahadasha = _safe_text(payload.get("next_mahadasha_date"), "미정")
    fallback_text = (
        "다음 마하다샤 시작일이 불분명해 3년 기준의 보수적 유효기간을 사용했습니다."
        if payload.get("valid_until_fallback")
        else f"다음 큰 전환 시점은 {next_mahadasha} 전후로 읽히므로 그 전까지를 현재 해석의 유효기간으로 봅니다."
    )

    return [
        (
            "## cover/meta",
            [
                "- 보고서: Vedic Life Cycle Report",
                f"- 대상: {subject_name}",
                f"- 기준 시점(as_of_local): {_safe_text(payload.get('as_of_local_iso'), '미정')}",
                f"- 적용 범위: {_safe_text(payload.get('birth_year'), '?')} ~ {_safe_text(payload.get('horizon_end_year'), '?')}년",
                f"- 리포트 유효기간(valid_until): {valid_until}",
            ],
        ),
        (
            "## How to use 1p",
            [
                "- 1단계: 먼저 인생 구조 한 장 요약에서 지금 어디쯤 와 있는지 확인합니다.",
                "- 2단계: 4단계 인생 구조에서 큰 흐름을 보고, 현재 위치 섹션으로 다시 내려옵니다.",
                "- 3단계: 마하다샤 단계 목록에서 전환 시점과 반복되는 행성 톤을 비교합니다.",
                f"- 4단계: valid_until 전까지는 {focus_line or '핵심 주제'}에만 집중하고, 새 판단은 다음 갱신 시점에 다시 점검합니다.",
                f"- 복구 플랜: 내용이 너무 넓게 느껴지면 '{concern_line}' 한 가지 질문만 남기고 나머지는 보류합니다.",
                "오늘의 행동: 오늘 안에 가장 중요한 결정 1개를 적고, 지금 단계에서 필요한 기준 1줄만 남깁니다.",
            ],
        ),
        (
            "## 인생 구조 한 장 요약",
            [
                summary_hook,
                f"지금은 {_safe_text(current_stage.get('summary_label'), '흐름을 다시 정리하는 시기')}에 가깝고, 현재 마하다샤는 {_safe_text(current_mahadasha.get('planet_label'), '현재')} 톤으로 읽힙니다.",
                f"그래서 이 보고서는 '{focus_line or '삶의 큰 방향'}'을 급하게 넓히기보다, 현재 위치를 정확히 이해하는 데 초점을 둡니다.",
            ],
        ),
        ("## 4단계 인생 구조", _render_stage_lines(payload)),
        (
            "## 현재 위치",
            [
                f"- 현재 단계: {_safe_text(current_stage.get('label'), '미정')} | {_safe_text(current_stage.get('summary_label'), '흐름 재정리')}",
                f"- 현재 마하다샤: {_safe_text(current_mahadasha.get('planet_label'), '미정')}",
                f"- 다음 큰 전환 예상일: {next_mahadasha}",
                f"- 지금 필요한 태도: {_safe_text(current_mahadasha.get('theme'), '지금 단계의 기준을 다시 세우기')}",
            ],
        ),
        ("## 마하다샤 단계 목록", _render_mahadasha_lines(payload)),
        (
            "## 방법론 카드",
            [
                "- 이 리포트는 Vimshottari Dasha를 기반으로 인생의 큰 시즌을 읽습니다.",
                "- 차트의 모든 요소를 늘어놓기보다, 실제로 읽히는 흐름과 전환점 이해에 집중합니다.",
                "- 현재 위치와 다음 큰 전환을 먼저 보여주고, 그다음 행동으로 연결합니다.",
                "- 전문 용어는 최소화하고 소비자 언어로 다시 설명합니다.",
                "- 같은 데이터를 다시 보더라도 기준 시점(as_of)이 달라지면 해석의 무게중심이 달라질 수 있습니다.",
                "- 이 보고서는 확정 예언이 아니라 현재 시즌을 읽는 의사결정 보조 자료입니다.",
            ],
        ),
        (
            "## valid_until 설명",
            [
                f"이 리포트의 현재 해석 유효기간은 {valid_until}까지입니다.",
                fallback_text,
                "오늘의 행동: valid_until 전까지 유지할 기준 1개와 버릴 기준 1개를 메모해 두세요.",
            ],
        ),
        (
            "## CTA-lite",
            [
                "행동: 다음 갱신일이 가까워지기 전에 지금 기준과 실제 변화가 얼마나 맞았는지 3줄로 기록해 두세요.",
            ],
        ),
        (
            "## 면책/윤리/데이터 보호",
            [
                "- 이 문서는 의료, 법률, 투자 판단을 대신하지 않습니다.",
                "- 사람을 단정하거나 공포를 유도하는 표현은 사용하지 않습니다.",
                "- 점성 정보는 현재 흐름을 해석하는 참고 자료로만 사용합니다.",
                "- 민감한 개인 정보는 필요한 범위에서만 처리합니다.",
                "- 중요한 결정은 현실 정보와 함께 교차 확인해 주세요.",
            ],
        ),
    ]


def render_life_cycle_lite_markdown(payload: dict[str, Any]) -> str:
    return render_life_cycle_sections(build_life_cycle_lite_sections(payload))
