from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_module

OUT_DIR = ROOT / "PRD" / "release_evidence" / "v1_5_0"
SAMPLE_RESPONSE_PATH = OUT_DIR / "life_cycle_longform_sample_response.json"
SAMPLE_READING_PATH = OUT_DIR / "life_cycle_longform_sample_reading.md"


def build_sample_params(*, llm_max_tokens: int | None = None) -> dict[str, Any]:
    params = {
        "year": 1990,
        "month": 1,
        "day": 1,
        "hour": 12,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "use_cache": 0,
        "language": "ko",
        "product_type": "life_cycle",
        "render_profile": main_module.LIFE_CYCLE_LONGFORM_RENDER_PROFILE,
        "subject_name": "민서",
        "onboarding_goal": "career_money",
        "focus_tokens": "커리어,돈",
        "concern_tokens": "이직 타이밍,수입 안정",
        "occupation_context": "브랜드 전략 업무",
        "relationship_status": "싱글",
        "request_id": "life_cycle_longform_prototype",
    }
    if isinstance(llm_max_tokens, int) and llm_max_tokens > 0:
        params["llm_max_tokens"] = llm_max_tokens
    return params


def generate_longform_sample(*, out_dir: Path = OUT_DIR, use_llm: bool = False, llm_max_tokens: int | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / SAMPLE_RESPONSE_PATH.name
    reading_path = out_dir / SAMPLE_READING_PATH.name

    previous_async_client = getattr(main_module, "async_client", None)
    if not use_llm:
        main_module.async_client = None

    try:
        client = TestClient(main_module.app)
        response = client.get("/ai_reading", params=build_sample_params(llm_max_tokens=llm_max_tokens))
    finally:
        if not use_llm:
            main_module.async_client = previous_async_client

    response.raise_for_status()
    data = response.json()

    response_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reading_path.write_text(str(data.get("polished_reading") or data.get("reading") or ""), encoding="utf-8")

    return {
        "response_path": response_path,
        "reading_path": reading_path,
        "render_profile": data.get("meta", {}).get("render_profile"),
        "model": data.get("model"),
        "reading_chars": len(str(data.get("polished_reading") or data.get("reading") or "")),
        "used_llm": bool(use_llm),
        "llm_max_tokens": llm_max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first life_cycle long-form prototype sample.")
    parser.add_argument("--allow-llm", action="store_true", help="Allow live LLM refinement if the runtime client is configured.")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="Override llm_max_tokens for the prototype request.")
    args = parser.parse_args()

    result = generate_longform_sample(use_llm=bool(args.allow_llm), llm_max_tokens=args.llm_max_tokens)
    print("Life cycle long-form prototype sample generated")
    print(f"- response_path: {result['response_path']}")
    print(f"- reading_path: {result['reading_path']}")
    print(f"- render_profile: {result['render_profile']}")
    print(f"- model: {result['model']}")
    print(f"- reading_chars: {result['reading_chars']}")
    print(f"- allow_llm: {result['used_llm']}")
    if result["llm_max_tokens"] is not None:
        print(f"- llm_max_tokens: {result['llm_max_tokens']}")


if __name__ == "__main__":
    main()
