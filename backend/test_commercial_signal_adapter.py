from __future__ import annotations

from backend.commercial_signal_adapter import (
    _FALLBACK_AS_OF_UTC,
    _SCENE_LIBRARY,
    apply_ascii_allowlist_guard_to_content,
    build_commercial_signal_card,
    render_signal_card_ko_for_prompt,
    resolve_as_of_utc,
)


def test_resolve_as_of_utc_uses_fallback_when_missing() -> None:
    out = resolve_as_of_utc(
        card_meta_as_of_utc=None,
        payload_as_of_utc=None,
        vedic_meta_as_of_utc=None,
    )
    assert out == _FALLBACK_AS_OF_UTC


def test_ascii_allowlist_guard_is_case_insensitive_and_preserves_header_lines() -> None:
    src = """## [Current Phase] 현재 흐름

### Action Steps

- RANDOMTOKEN should be removed
- Dasha token should remain
"""
    out = apply_ascii_allowlist_guard_to_content(src)
    assert "## [Current Phase] 현재 흐름" in out
    assert "### Action Steps" in out
    assert "RANDOMTOKEN" not in out
    assert "Dasha" in out


def test_build_commercial_signal_card_filters_past_windows_and_pads_to_three_slots() -> None:
    card_meta, card_ko = build_commercial_signal_card(
        structural_summary={"behavioral_risk_profile": {"primary_risk": "impulsivity"}},
        dasha_context={
            "timing_axis": {
                "timing_windows": [
                    {"start_utc": "2026-02-01T00:00:00Z", "note": "risk"},
                    {"start_utc": "2026-03-10T00:00:00Z", "note": "emotional"},
                    {"start_utc": "2026-04-11T00:00:00Z", "note": "opportunity"},
                ]
            }
        },
        card_meta_as_of_utc="2026-03-01T00:00:00Z",
    )
    assert card_meta["as_of_utc"] == "2026-03-01T00:00:00Z"
    rows = card_ko.get("three_month")
    assert isinstance(rows, list)
    assert len(rows) == 3
    assert rows[0]["label"] == "이번 달"
    assert rows[1]["label"] == "다음 달"
    assert rows[2]["label"] == "그다음 달"


def test_render_signal_card_ko_for_prompt_avoids_key_label_output() -> None:
    card = {
        "hook": ["테스트 훅입니다."],
        "current_phase": {"headline": "현재 흐름", "window_left_text": "운영 규칙이 중요합니다."},
        "scenes": ["장면 하나", "장면 둘"],
        "brakes": ["규칙 하나", "규칙 둘", "규칙 셋"],
        "vectors": {},
        "three_month": [{"label": "이번 달", "tag": "감정", "note": "일반 가이드"}],
        "risk_pack": {},
    }
    out = render_signal_card_ko_for_prompt(card)
    assert "hook:" not in out
    assert "current_phase:" not in out
    assert "risk_pack:" not in out
    assert "핵심 흐름" in out


def test_scene_library_distribution_and_schema() -> None:
    assert len(_SCENE_LIBRARY) == 20
    domain_counts = {"career": 0, "money_contract": 0, "relationship": 0}
    ids: set[str] = set()
    for row in _SCENE_LIBRARY:
        assert isinstance(row.get("id"), str) and row["id"].strip()
        assert isinstance(row.get("domain"), str) and row["domain"] in domain_counts
        assert isinstance(row.get("trigger"), str) and row["trigger"].strip()
        assert isinstance(row.get("scene_text"), str) and row["scene_text"].strip()
        assert isinstance(row.get("alt_text"), str) and row["alt_text"].strip()
        assert row["id"] not in ids
        ids.add(row["id"])
        domain_counts[row["domain"]] += 1
    assert domain_counts["career"] == 8
    assert domain_counts["money_contract"] == 6
    assert domain_counts["relationship"] == 6


def test_money_domain_prioritizes_financial_instability_trigger() -> None:
    _meta, card_ko = build_commercial_signal_card(
        structural_summary={
            "behavioral_risk_profile": {"primary_risk": "impulsivity"},
            "probability_forecast": {"financial_instability_3yr": 0.8},
        },
        dasha_context={},
        card_meta_as_of_utc="2026-03-01T00:00:00Z",
    )
    scenes = card_ko.get("scene_examples", [])
    assert isinstance(scenes, list)
    money_scene = next((s for s in scenes if isinstance(s, dict) and s.get("domain") == "money_contract"), None)
    assert isinstance(money_scene, dict)
    assert money_scene.get("trigger") == "financial_instability"


def test_relationship_domain_prioritizes_self_sabotage_trigger() -> None:
    _meta, card_ko = build_commercial_signal_card(
        structural_summary={
            "behavioral_risk_profile": {
                "primary_risk": "impulsivity",
                "self_sabotage_risk": 8.0,
                "emotional_volatility": 2.0,
                "authority_conflict_risk": 2.0,
            }
        },
        dasha_context={},
        card_meta_as_of_utc="2026-03-01T00:00:00Z",
    )
    scenes = card_ko.get("scene_examples", [])
    assert isinstance(scenes, list)
    rel_scene = next((s for s in scenes if isinstance(s, dict) and s.get("domain") == "relationship"), None)
    assert isinstance(rel_scene, dict)
    assert rel_scene.get("trigger") == "self_sabotage"


def test_scene_selection_has_domain_fallback_when_trigger_unmatched() -> None:
    _meta, card_ko = build_commercial_signal_card(
        structural_summary={
            "behavioral_risk_profile": {"primary_risk": "financial_instability"},
            "probability_forecast": {"financial_instability_3yr": 0.1},
        },
        dasha_context={},
        card_meta_as_of_utc="2026-03-01T00:00:00Z",
    )
    scenes = card_ko.get("scene_examples", [])
    assert isinstance(scenes, list)
    domains = [s.get("domain") for s in scenes if isinstance(s, dict)]
    assert sorted(domains) == ["career", "money_contract", "relationship"]
    rel_scene = next((s for s in scenes if isinstance(s, dict) and s.get("domain") == "relationship"), None)
    assert isinstance(rel_scene, dict)
    assert rel_scene.get("trigger") != "financial_instability"
