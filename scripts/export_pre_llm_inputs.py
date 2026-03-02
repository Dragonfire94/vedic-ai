from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.astro_engine import build_structural_summary
from backend.golden_sample_runner import _candidate_metrics, generate_golden_charts, select_profiles
from backend.llm_service import _derive_narrative_mode, build_llm_structural_prompt, sanitize_prompt_text_last_mile
from backend.pre_llm_input_sanitizer import (
    render_chapter_blocks_pre_llm,
    sanitize_chapter_blocks_for_llm,
)
from backend.report_engine import build_dasha_narrative_context, build_report_payload, build_semantic_signals
from backend.main import get_chart


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for payload in generate_golden_charts():
        chart = get_chart(
            year=payload["year"],
            month=payload["month"],
            day=payload["day"],
            hour=payload["hour"],
            lat=payload["lat"],
            lon=payload["lon"],
            house_system=payload["house_system"],
            include_nodes=payload["include_nodes"],
            include_d9=payload["include_d9"],
            include_vargas=payload["include_vargas"],
            gender=payload["gender"],
            timezone=payload["timezone"],
        )
        structural_summary = build_structural_summary(chart, analysis_mode=payload["analysis_mode"])
        candidates.append(_candidate_metrics(payload, chart, structural_summary))
    return candidates


def _resolve_candidate(*, profile: str, seed: int | None) -> tuple[str, dict[str, Any]]:
    candidates = _build_candidates()
    if seed is not None:
        for candidate in candidates:
            if int(candidate.get("seed", -1)) == seed:
                return f"seed_{seed}", candidate
        raise ValueError(f"seed={seed} 후보를 찾지 못했습니다.")

    selected = dict(select_profiles(candidates))
    if profile not in selected:
        available = ", ".join(sorted(selected.keys()))
        raise ValueError(f"profile='{profile}'를 찾지 못했습니다. available=[{available}]")
    return profile, selected[profile]


def _token_count(text: str, token: str) -> int:
    return text.lower().count(token.lower())


def _regex_count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pre-LLM sanitized input artifacts without LLM calls.")
    parser.add_argument("--profile", type=str, default="most_balanced", help="golden profile name")
    parser.add_argument("--seed", type=int, default=None, help="direct seed override (0..49)")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="output directory. default: logs/pre_llm_inputs_<timestamp>",
    )
    args = parser.parse_args()

    profile_name, candidate = _resolve_candidate(profile=args.profile, seed=args.seed)
    payload = candidate["input"]
    structural_summary = candidate["structural_summary"]

    report_payload = build_report_payload({"structural_summary": structural_summary, "language": args.language})
    chapter_blocks_raw = report_payload.get("chapter_blocks", {}) if isinstance(report_payload, dict) else {}
    chapter_blocks_sanitized = sanitize_chapter_blocks_for_llm(chapter_blocks_raw if isinstance(chapter_blocks_raw, dict) else {})
    deterministic_pre_llm = render_chapter_blocks_pre_llm(chapter_blocks_sanitized)

    narrative_mode = _derive_narrative_mode(structural_summary if isinstance(structural_summary, dict) else {})
    semantic_signals = build_semantic_signals(structural_summary if isinstance(structural_summary, dict) else {})
    dasha_context = build_dasha_narrative_context(structural_summary if isinstance(structural_summary, dict) else {})

    llm_prompt_pre_call_raw = build_llm_structural_prompt(
        structural_summary=structural_summary if isinstance(structural_summary, dict) else {},
        language=args.language,
        chapter_blocks=chapter_blocks_raw if isinstance(chapter_blocks_raw, dict) else {},
        semantic_signals=semantic_signals if isinstance(semantic_signals, dict) else {},
        narrative_mode=narrative_mode,
        dasha_context=dasha_context if isinstance(dasha_context, dict) else {},
    )
    llm_prompt_pre_call = sanitize_prompt_text_last_mile(llm_prompt_pre_call_raw)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir.strip():
        out_dir = Path(args.output_dir.strip())
    else:
        out_dir = Path(f"logs/pre_llm_inputs_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(out_dir / "input_payload.json", payload)
    _json_dump(out_dir / "structural_summary.json", structural_summary)
    _json_dump(out_dir / "chapter_blocks.raw.json", chapter_blocks_raw)
    _json_dump(out_dir / "chapter_blocks.sanitized.json", chapter_blocks_sanitized)
    (out_dir / "deterministic_reading_pre_llm.md").write_text(deterministic_pre_llm, encoding="utf-8")
    (out_dir / "llm_prompt_pre_call.txt").write_text(llm_prompt_pre_call, encoding="utf-8")

    whole_prompt_surface = llm_prompt_pre_call
    summary = {
        "profile_name": profile_name,
        "seed": candidate.get("seed"),
        "output_dir": str(out_dir),
        "language": args.language,
        "llm_called": False,
        "artifacts": {
            "input_payload": str(out_dir / "input_payload.json"),
            "structural_summary": str(out_dir / "structural_summary.json"),
            "chapter_blocks_raw": str(out_dir / "chapter_blocks.raw.json"),
            "chapter_blocks_sanitized": str(out_dir / "chapter_blocks.sanitized.json"),
            "deterministic_reading_pre_llm": str(out_dir / "deterministic_reading_pre_llm.md"),
            "llm_prompt_pre_call": str(out_dir / "llm_prompt_pre_call.txt"),
        },
        "hygiene_counts": {
            "sidereal": _regex_count(whole_prompt_surface, r"\bsidereal\b") + _token_count(whole_prompt_surface, "시데리얼"),
            "lahiri": _regex_count(whole_prompt_surface, r"\blahiri\b") + _token_count(whole_prompt_surface, "라히리"),
            "ayanamsa": _regex_count(whole_prompt_surface, r"\bayanamsa\b") + _token_count(whole_prompt_surface, "아얀암"),
            "shadbala": _regex_count(whole_prompt_surface, r"(?<![A-Za-z])shadbala(?![A-Za-z])") + _token_count(whole_prompt_surface, "Śadbala"),
            "avastha": _regex_count(whole_prompt_surface, r"(?<![A-Za-z])avastha(?![A-Za-z])") + _token_count(whole_prompt_surface, "Avasthā"),
            "chapter_key_comment": _token_count(whole_prompt_surface, "<!-- chapter_key:"),
            "numbered_h1": sum(
                1
                for line in whole_prompt_surface.splitlines()
                if line.strip().startswith("# ") and ". " in line
            ),
            "placeholder_title": _token_count(whole_prompt_surface, "해석 블록"),
        },
    }
    _json_dump(out_dir / "pre_llm_export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
