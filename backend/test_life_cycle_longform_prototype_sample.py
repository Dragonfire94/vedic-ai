from __future__ import annotations

import json
from pathlib import Path

import backend.main as main_module
import scripts.build_life_cycle_longform_prototype_sample as prototype


def _fake_chart(**_: object) -> dict:
    return {
        "input": {"year": 1990, "month": 1, "day": 1},
        "julian_day": 2447892.5,
        "planets": {"Moon": {"longitude": 123.45}},
        "meta": {
            "birth_jd": 2447892.5,
            "dasha_reference_jd": 2461110.5,
        },
    }


def test_generate_longform_sample_writes_response_and_reading(monkeypatch) -> None:
    out_dir = Path("logs") / "life_cycle_longform_prototype_test"
    monkeypatch.setattr(main_module, "get_chart", _fake_chart)

    result = prototype.generate_longform_sample(out_dir=out_dir, use_llm=False)

    response_path = Path(result["response_path"])
    reading_path = Path(result["reading_path"])
    assert response_path.exists()
    assert reading_path.exists()

    payload = json.loads(response_path.read_text(encoding="utf-8"))
    reading_text = reading_path.read_text(encoding="utf-8")
    assert payload["meta"]["render_profile"] == "life_cycle_longform_v1"
    assert payload["meta"]["generation_mode"] == "report_engine.life_cycle_adapter_v1"
    assert payload["meta"]["source_render_profile"] == "life_cycle_target_v1"
    assert payload["meta"]["longform_cutover_candidate"] is True
    assert payload["product_type"] == "life_cycle"
    assert "Executive Diagnosis" in reading_text
    assert "long-form seed" not in reading_text
    assert "prototype" not in reading_text
    assert "중장기 구간-03-16" not in reading_text
    assert "2029-03-16" in reading_text
    assert "이직 타이밍와 수입 안정" not in reading_text
    assert "이직 타이밍과 수입 안정" in reading_text
    assert "세부 테마(포커스)를 쪼개는 세부 테마" not in reading_text
