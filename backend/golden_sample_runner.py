from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from backend.astro_engine import (
    PLANET_ORDER,
    SIGN_LORDS,
    _init_debug_metrics,
    build_influence_matrix,
    build_structural_summary,
    calculate_planet_strength,
    compute_house_clusters,
    detect_yogas,
    summarize_debug_metrics,
    summarize_dasha_timeline,
)
from backend.llm_service import (
    _derive_narrative_mode,
    audit_llm_output,
    refine_reading_with_llm,
)
from backend.main import (
    _build_openai_payload,
    _candidate_openai_models,
    _emit_llm_audit_event,
    _normalize_long_paragraphs,
    _validate_deterministic_llm_blocks,
    async_client,
    build_ai_psychological_input,
    compute_chapter_blocks_hash,
    get_chart,
)
from backend.report_engine import build_report_payload, build_semantic_signals

OUTPUT_ROOT = Path("logs/golden_samples")
logger = __import__("logging").getLogger("golden_sample_runner")
LOCATION_POOL = [
    {"name": "seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "new_york", "lat": 40.7128, "lon": -74.0060},
    {"name": "london", "lat": 51.5074, "lon": -0.1278},
    {"name": "sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "sao_paulo", "lat": -23.5505, "lon": -46.6333},
    {"name": "cairo", "lat": 30.0444, "lon": 31.2357},
    {"name": "paris", "lat": 48.8566, "lon": 2.3522},
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_golden_charts() -> list[dict[str, Any]]:
    """Generate 50 deterministic chart input payloads."""
    charts: list[dict[str, Any]] = []
    for seed in range(50):
        rng = Random(seed)
        loc = LOCATION_POOL[seed % len(LOCATION_POOL)]
        payload = {
            "seed": seed,
            "label": f"candidate_{seed:02d}",
            "year": 1970 + (seed % 36),
            "month": (seed % 12) + 1,
            "day": (seed % 28) + 1,
            "hour": round((seed % 24) + (rng.choice([0.0, 0.25, 0.5, 0.75])), 2),
            "lat": loc["lat"],
            "lon": loc["lon"],
            "timezone": None,
            "house_system": "W",
            "include_nodes": 1,
            "include_d9": 1,
            "include_vargas": "",
            "gender": "male" if seed % 2 == 0 else "female",
            "location_name": loc["name"],
            "analysis_mode": "standard",
            "language": "ko",
        }
        charts.append(payload)
    return charts


def _run_debug_snapshot(
    planets: dict[str, Any],
    houses: dict[str, Any],
    strength: dict[str, Any],
    yogas: list[dict[str, Any]],
    influence_data: dict[str, Any],
    house_clusters: dict[str, Any],
    stability_index: float,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], float]:
    debug = _init_debug_metrics()
    for dasha in PLANET_ORDER:
        summarize_dasha_timeline(
            planets=planets,
            strength_data=strength,
            yogas=yogas,
            influence_matrix=influence_data,
            house_clusters=house_clusters,
            houses=houses,
            current_dasha=dasha,
            stability_index=stability_index,
            debug_metrics=debug,
        )
    aggregated = summarize_debug_metrics(debug)

    house_pressures = debug.get("house_pressure", [])
    dusthana_pressure = 0.0
    for idx, dasha in enumerate(PLANET_ORDER):
        house = ((planets.get(dasha, {}) or {}).get("house"))
        hp = house_pressures[idx] if idx < len(house_pressures) else 0.0
        if house in {6, 8, 12}:
            dusthana_pressure = max(dusthana_pressure, _safe_float(hp, 0.0))
    return debug, aggregated, float(dusthana_pressure)


def _lagna_lord_proxy(houses: dict[str, Any], strength: dict[str, Any]) -> tuple[float, bool]:
    asc_sign = (((houses.get("ascendant") or {}).get("rasi") or {}).get("name"))
    lagna_lord = SIGN_LORDS.get(asc_sign) if isinstance(asc_sign, str) else None
    lagna_total_raw = ((strength.get(lagna_lord or "", {}) or {}).get("shadbala") or {}).get("total")
    lagna_total = _safe_float(lagna_total_raw, 0.0)
    totals = [
        _safe_float((((strength.get(p, {}) or {}).get("shadbala") or {}).get("total")), 0.0)
        for p in PLANET_ORDER
    ]
    sorted_totals = sorted(totals, reverse=True)
    top_k = max(1, int(len(sorted_totals) * 0.2 + 0.9999))
    threshold = sorted_totals[top_k - 1] if sorted_totals else 0.0
    is_top20 = lagna_total >= threshold
    return lagna_total, is_top20


def _mean_shadbala(strength: dict[str, Any]) -> float:
    vals = [
        _safe_float((((strength.get(p, {}) or {}).get("shadbala") or {}).get("total")), 0.0)
        for p in PLANET_ORDER
    ]
    return float(mean(vals)) if vals else 0.0


def _candidate_metrics(payload: dict[str, Any], chart: dict[str, Any], structural_summary: dict[str, Any]) -> dict[str, Any]:
    planets = chart.get("planets", {}) if isinstance(chart.get("planets"), dict) else {}
    houses = chart.get("houses", {}) if isinstance(chart.get("houses"), dict) else {}
    strength = calculate_planet_strength(planets, houses)
    yogas = detect_yogas(planets, houses)
    influence_data = build_influence_matrix(planets, strength, yogas)
    house_clusters = compute_house_clusters(planets, houses, strength, yogas)
    stability_index = _safe_float(
        ((structural_summary.get("stability_metrics") or {}).get("stability_index")),
        50.0,
    )
    debug_raw, debug_agg, dusthana_pressure = _run_debug_snapshot(
        planets, houses, strength, yogas, influence_data, house_clusters, stability_index
    )

    lagna_total, lagna_top20 = _lagna_lord_proxy(houses, strength)
    mean_shadbala = _mean_shadbala(strength)

    return {
        "seed": payload["seed"],
        "input": payload,
        "chart": chart,
        "structural_summary": structural_summary,
        "debug_metrics_raw": debug_raw,
        "debug_metrics_agg": debug_agg,
        "stability_index": stability_index,
        "risk_factor": _safe_float((debug_agg.get("risk_factor") or {}).get("mean"), 0.0),
        "opportunity_factor": _safe_float((debug_agg.get("opportunity_factor") or {}).get("mean"), 0.0),
        "influence_score": _safe_float((debug_agg.get("influence_score") or {}).get("mean"), 0.0),
        "dominant_score": _safe_float((debug_agg.get("dominant_score") or {}).get("mean"), 0.0),
        "dusthana_pressure": dusthana_pressure,
        "lagna_lord_total": lagna_total,
        "lagna_lord_top20": lagna_top20,
        "mean_shadbala": mean_shadbala,
    }


def _select_first(
    candidates: list[dict[str, Any]],
    used: set[int],
    key_fn,
    reverse: bool = True,
    predicate=None,
) -> dict[str, Any] | None:
    pool = [c for c in candidates if c["seed"] not in used and (predicate(c) if predicate else True)]
    if not pool:
        return None
    ordered = sorted(pool, key=lambda c: (key_fn(c), -c["seed"]), reverse=reverse)
    selected = ordered[0]
    used.add(selected["seed"])
    return selected


def _balance_rank(candidates: list[dict[str, Any]], c: dict[str, Any]) -> float:
    fields = ["stability_index", "risk_factor", "opportunity_factor", "influence_score"]
    deviation = 0.0
    for field in fields:
        ordered = sorted(_safe_float(x[field], 0.0) for x in candidates)
        n = len(ordered)
        if n <= 1:
            continue
        val = _safe_float(c[field], 0.0)
        pos = ordered.index(min(ordered, key=lambda x: abs(x - val)))
        percentile = pos / (n - 1)
        deviation += abs(percentile - 0.5)
    return deviation


def select_profiles(candidates: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    used: set[int] = set()
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, selected: dict[str, Any] | None) -> None:
        if selected is not None:
            out.append((name, selected))

    add("highest_stability", _select_first(candidates, used, lambda c: c["stability_index"], reverse=True))
    add("lowest_stability", _select_first(candidates, used, lambda c: c["stability_index"], reverse=False))
    add("highest_opportunity", _select_first(candidates, used, lambda c: c["opportunity_factor"], reverse=True))
    add("highest_risk", _select_first(candidates, used, lambda c: c["risk_factor"], reverse=True))
    add("highest_influence", _select_first(candidates, used, lambda c: c["influence_score"], reverse=True))
    add("lowest_influence", _select_first(candidates, used, lambda c: c["influence_score"], reverse=False))
    add("highest_dominant_score", _select_first(candidates, used, lambda c: c["dominant_score"], reverse=True))
    add("dusthana_pressure", _select_first(candidates, used, lambda c: c["dusthana_pressure"], reverse=True))

    add(
        "lagna_lord_proxy",
        _select_first(
            candidates,
            used,
            key_fn=lambda c: c["lagna_lord_total"],
            reverse=True,
            predicate=lambda c: bool(c["lagna_lord_top20"]),
        ) or _select_first(candidates, used, lambda c: c["lagna_lord_total"], reverse=True),
    )
    add("weak_overall_shadbala", _select_first(candidates, used, lambda c: c["mean_shadbala"], reverse=False))
    add("strong_overall_shadbala", _select_first(candidates, used, lambda c: c["mean_shadbala"], reverse=True))
    add("most_balanced", _select_first(candidates, used, lambda c: _balance_rank(candidates, c), reverse=False))
    return out


async def _run_llm_for_profile(
    profile_name: str,
    candidate: dict[str, Any],
    *,
    max_tokens: int = 2000,
) -> dict[str, Any]:
    structural_summary = candidate["structural_summary"]
    narrative_mode = _derive_narrative_mode(structural_summary)
    semantic_signals = build_semantic_signals(structural_summary)

    report_payload = build_report_payload({"structural_summary": structural_summary, "language": "ko"})
    chapter_blocks = report_payload.get("chapter_blocks", {})
    llm_status = "OK"
    ai_text = ""
    audit_report: dict[str, Any] = {}
    llm_error = None

    try:
        if async_client is None:
            raise RuntimeError("OpenAI client unavailable")
        logger.debug("[OPENAI CALL] seed=%s profile=%s", candidate["seed"], profile_name)
        try:
            ai_text = await asyncio.wait_for(
                refine_reading_with_llm(
                    async_client=async_client,
                    validate_blocks_fn=_validate_deterministic_llm_blocks,
                    build_ai_input_fn=build_ai_psychological_input,
                    candidate_models_fn=_candidate_openai_models,
                    build_payload_fn=_build_openai_payload,
                    emit_audit_fn=_emit_llm_audit_event,
                    normalize_paragraphs_fn=_normalize_long_paragraphs,
                    compute_hash_fn=compute_chapter_blocks_hash,
                    chapter_blocks=chapter_blocks,
                    structural_summary=structural_summary,
                    language="ko",
                    request_id=f"golden_{candidate['seed']:02d}_{profile_name}",
                    chart_hash=f"golden_seed_{candidate['seed']:02d}",
                    endpoint="golden_sample_runner",
                    max_tokens=max_tokens,
                ),
                timeout=90,
            )
        except asyncio.TimeoutError:
            print(f"[PROFILE TIMEOUT] profile={profile_name}")
            return {
                "profile_name": profile_name,
                "llm_status": "TIMEOUT",
                "ai_text": "",
                "audit_report": {
                    "structural_integrity_score": 0,
                    "tone_alignment_score": 0,
                    "boilerplate_score": 0,
                    "density_score": 0,
                    "repetition_score": 0,
                    "overall_score": 0,
                    "flags": {
                        "missing_anchor": True,
                        "tone_inconsistency": True,
                        "boilerplate_detected": False,
                        "low_density": True,
                        "repetition_issue": False,
                    },
                    "error": "PROFILE_TIMEOUT_90S",
                },
                "narrative_mode": narrative_mode,
                "semantic_signals": semantic_signals,
                "llm_error": "PROFILE_TIMEOUT_90S",
            }
        audit_report = audit_llm_output(ai_text, structural_summary)
    except Exception as exc:
        llm_status = "FAILED_NO_LLM"
        llm_error = str(exc)
        audit_report = {
            "structural_integrity_score": 0,
            "tone_alignment_score": 0,
            "boilerplate_score": 0,
            "density_score": 0,
            "repetition_score": 0,
            "overall_score": 0,
            "flags": {
                "missing_anchor": True,
                "tone_inconsistency": True,
                "boilerplate_detected": False,
                "low_density": True,
                "repetition_issue": False,
            },
            "error": llm_error,
        }

    return {
        "profile_name": profile_name,
        "llm_status": llm_status,
        "ai_text": ai_text,
        "audit_report": audit_report,
        "narrative_mode": narrative_mode,
        "semantic_signals": semantic_signals,
        "llm_error": llm_error,
    }


def _store_profile_output(index: int, profile_name: str, candidate: dict[str, Any], llm_result: dict[str, Any]) -> dict[str, Any]:
    profile_dir = OUTPUT_ROOT / f"{index:02d}_{profile_name}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(profile_dir / "structural_summary.json", candidate["structural_summary"])
    _json_dump(
        profile_dir / "debug_metrics.json",
        {
            "raw": candidate["debug_metrics_raw"],
            "aggregated": candidate["debug_metrics_agg"],
            "dusthana_pressure": candidate["dusthana_pressure"],
            "lagna_lord_total": candidate["lagna_lord_total"],
            "lagna_lord_top20": candidate["lagna_lord_top20"],
            "mean_shadbala": candidate["mean_shadbala"],
        },
    )
    _json_dump(profile_dir / "audit.json", llm_result["audit_report"])

    if llm_result["llm_status"] == "OK":
        ai_payload = {
            "profile_name": profile_name,
            "seed": candidate["seed"],
            "narrative_mode": llm_result["narrative_mode"],
            "semantic_signals": llm_result["semantic_signals"],
            "ai_text": llm_result["ai_text"],
        }
        _json_dump(profile_dir / "ai_reading.json", ai_payload)
        (profile_dir / "ai_reading.txt").write_text(llm_result["ai_text"], encoding="utf-8")

    return {
        "profile_name": profile_name,
        "stability_index": round(candidate["stability_index"], 4),
        "risk_factor": round(candidate["risk_factor"], 4),
        "opportunity_factor": round(candidate["opportunity_factor"], 4),
        "influence_score": round(candidate["influence_score"], 4),
        "dominant_score": round(candidate["dominant_score"], 4),
        "mean_shadbala": round(candidate["mean_shadbala"], 4),
        "audit_score": int((llm_result.get("audit_report") or {}).get("overall_score", 0)),
        "narrative_mode": llm_result.get("narrative_mode"),
        "llm_status": llm_result["llm_status"],
    }


async def run_golden_sample_generation(mode: str = "structural") -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    inputs = generate_golden_charts()
    candidates: list[dict[str, Any]] = []
    generation_failures = 0

    for payload in inputs:
        try:
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
        except Exception as exc:
            generation_failures += 1
            print(f"[CANDIDATE ERROR] seed={payload['seed']} error={exc}")
            continue

    if not candidates:
        print("[FATAL] No structural candidates generated.")
        return 1

    selected = select_profiles(candidates)
    if not selected:
        print("[FATAL] No profiles selected.")
        return 1

    summary_index: list[dict[str, Any]] = []
    llm_ok_count = 0
    for idx, (profile_name, candidate) in enumerate(selected, start=1):
        try:
            if mode == "full":
                logger.debug("[LLM START] profile=%s", profile_name)
                llm_result = await _run_llm_for_profile(profile_name, candidate)
                logger.debug("[LLM END] profile=%s", profile_name)
            else:
                llm_result = {
                    "profile_name": profile_name,
                    "llm_status": "SKIPPED_STRUCTURAL_ONLY",
                    "ai_text": "",
                    "audit_report": {
                        "structural_integrity_score": 0,
                        "tone_alignment_score": 0,
                        "boilerplate_score": 0,
                        "density_score": 0,
                        "repetition_score": 0,
                        "overall_score": 0,
                        "flags": {},
                    },
                    "narrative_mode": None,
                    "semantic_signals": {},
                    "llm_error": None,
                }
            if llm_result["llm_status"] == "OK":
                llm_ok_count += 1
            summary_row = _store_profile_output(idx, profile_name, candidate, llm_result)
            summary_index.append(summary_row)
        except Exception as exc:
            print(f"[PROFILE ERROR] profile={profile_name} seed={candidate['seed']} error={exc}")
            summary_index.append(
                {
                    "profile_name": profile_name,
                    "stability_index": round(candidate["stability_index"], 4),
                    "risk_factor": round(candidate["risk_factor"], 4),
                    "opportunity_factor": round(candidate["opportunity_factor"], 4),
                    "influence_score": round(candidate["influence_score"], 4),
                    "dominant_score": round(candidate["dominant_score"], 4),
                    "mean_shadbala": round(candidate["mean_shadbala"], 4),
                    "audit_score": 0,
                    "narrative_mode": None,
                    "llm_status": "FAILED_NO_LLM",
                }
            )
            continue

    _json_dump(OUTPUT_ROOT / "summary_index.json", summary_index)

    print("\n===== GOLDEN SAMPLE SUMMARY =====")
    for row in summary_index:
        print(
            f"{row['profile_name']:<26} "
            f"stability={row['stability_index']:<6} "
            f"risk={row['risk_factor']:<6} "
            f"opp={row['opportunity_factor']:<6} "
            f"influence={row['influence_score']:<6} "
            f"audit={row['audit_score']:<4} "
            f"mode={row['narrative_mode']} "
            f"llm={row['llm_status']}"
        )

    print(f"\nCandidates generated: {len(candidates)} / 50 (failures={generation_failures})")
    print(f"Profiles selected: {len(summary_index)}")
    print(f"LLM success count: {llm_ok_count}")
    print(f"Execution mode: {mode}")
    print(f"Output path: {OUTPUT_ROOT.resolve()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["structural", "full"],
        default="structural",
        help="Execution mode for golden sample runner",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_golden_sample_generation(mode=args.mode)))
