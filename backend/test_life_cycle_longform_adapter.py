from __future__ import annotations

import backend.main as main_module
from backend.life_cycle_longform_adapter import build_life_cycle_longform_chapter_blocks
from backend.report_config import PREMIUM_12_CHAPTER_ORDER
from backend.test_life_cycle_target_contract import _target_payload


def test_build_life_cycle_longform_chapter_blocks_matches_premium_chapter_order() -> None:
    blocks = build_life_cycle_longform_chapter_blocks(_target_payload())
    assert list(blocks.keys()) == PREMIUM_12_CHAPTER_ORDER
    assert all(isinstance(blocks[chapter], list) and blocks[chapter] for chapter in PREMIUM_12_CHAPTER_ORDER)


def test_build_life_cycle_longform_chapter_blocks_stays_within_deterministic_fragment_contract() -> None:
    blocks = build_life_cycle_longform_chapter_blocks(_target_payload())
    validated = main_module._validate_deterministic_llm_blocks(blocks)

    assert validated["Executive Diagnosis"][0]["title"] == "지금 먼저 붙잡아야 할 질문"
    assert "민서님" in validated["Executive Diagnosis"][0]["summary"]
    assert "전환 타이밍과 우선순위" in validated["Career & Money"][0]["implication"]
    assert "브랜드 전략 업무" in validated["Career & Money"][0]["examples"]
    assert "유효기간" in validated["Final Integration"][0]["summary"] or "2028-03-11" in validated["Final Integration"][0]["summary"]
    assert "prototype" not in validated["Final Integration"][0]["implication"]
    assert all("seed" not in blocks[chapter][0].get("title", "") for chapter in PREMIUM_12_CHAPTER_ORDER)
