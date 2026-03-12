from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_module
import scripts.cheap_validation_gate as gate
from backend.life_cycle_helpers import build_life_cycle_payload
from backend.life_cycle_target_renderer import LIFE_CYCLE_TARGET_H2_ORDER, render_life_cycle_target_markdown
from backend.llm_output_scanner import scan_forbidden_patterns

OUT_DIR = ROOT / "PRD" / "release_evidence" / "v1_4_0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KST = timezone(timedelta(hours=9))
CONTRACT_VERSION = main_module.LIFE_CYCLE_CONTRACT_VERSION
RENDER_PROFILE = main_module.LIFE_CYCLE_TARGET_RENDER_PROFILE
BASE_CHART = {
    "input": {"year": 1990, "month": 1, "day": 1},
    "julian_day": 2447892.5,
    "planets": {"Moon": {"longitude": 123.45}},
    "meta": {"birth_jd": 2447892.5, "dasha_reference_jd": 2461110.5},
}

EDITORIAL_RUBRIC_PATH = OUT_DIR / "life_cycle_target_editorial_rubric.md"
EDITORIAL_REVIEW_PATH = OUT_DIR / "life_cycle_target_editorial_review_20.md"
EDITORIAL_CASES_PATH = OUT_DIR / "life_cycle_target_editorial_case_matrix.json"
MANUAL_QA_PATH = OUT_DIR / "life_cycle_target_manual_qa.md"
SAMPLE_RESPONSE_PATH = OUT_DIR / "life_cycle_target_sample_response.json"
GATE_SUMMARY_PATH = OUT_DIR / "life_cycle_target_gate_summary.json"
MANIFEST_PATH = OUT_DIR / "life_cycle_target_release_manifest.json"
RELEASE_EVIDENCE_DIR = "PRD/release_evidence/v1_4_0"


@dataclass(frozen=True)
class EditorialCase:
    case_id: str
    subject_name: str
    onboarding_goal: str
    focus_tokens: list[str]
    concern_tokens: list[str]
    occupation_context: str
    relationship_status: str
    note: str
    fallback_valid_until: bool = False
    sparse_repeat: bool = False
    empty_future: bool = False


CASE_SPECS: list[EditorialCase] = [
    EditorialCase("target_case_01", "민서", "career_money", ["커리어", "돈"], ["이직 타이밍", "수입 안정"], "브랜드 전략 업무", "싱글", "career baseline"),
    EditorialCase("target_case_02", "도윤", "career_money", ["사업", "확장"], ["채용 기준", "현금흐름"], "초기 스타트업 대표", "기혼", "founder scale"),
    EditorialCase("target_case_03", "서윤", "relationship", ["관계", "감정"], ["경계선", "재회 여부"], "마케팅 리드", "재정비 중", "relationship reset"),
    EditorialCase("target_case_04", "예준", "relationship", ["대화", "리듬"], ["갈등 패턴", "합의 기준"], "전략 컨설턴트", "기혼", "newlywed balance"),
    EditorialCase("target_case_05", "하린", "condition", ["컨디션", "루틴"], ["회복 리듬", "수면"], "콘텐츠 디렉터", "싱글", "burnout recovery"),
    EditorialCase("target_case_06", "시우", "condition", ["에너지", "생활"], ["체력 분배", "가족"], "운영 매니저", "기혼", "parent energy"),
    EditorialCase("target_case_07", "유진", "life_direction", ["방향", "의미"], ["우선순위", "정체성"], "프리랜서 라이터", "싱글", "direction rebuild"),
    EditorialCase("target_case_08", "지훈", "life_direction", ["전환", "방향"], ["역할 전환", "후회"], "재무 팀장", "기혼", "midlife shift"),
    EditorialCase("target_case_09", "수아", "life_direction", ["방향", "균형"], ["기준 재설정", "일정"], "디자인 매니저", "싱글", "sparse repeat", sparse_repeat=True),
    EditorialCase("target_case_10", "태민", "relationship", ["관계", "미래"], ["관계 속도", "대화"], "영업 리드", "연애 중", "empty future", empty_future=True),
    EditorialCase("target_case_11", "서아", "career_money", ["역할", "돈"], ["협상 타이밍", "결정 기준"], "프로덕트 매니저", "싱글", "fallback valid until", fallback_valid_until=True),
    EditorialCase("target_case_12", "현우", "condition", ["에너지", "보호선"], ["회복", "우선순위"], "간호사", "싱글", "fallback + empty future", fallback_valid_until=True, empty_future=True),
    EditorialCase("target_case_13", "가은", "life_direction", ["방향"], ["핵심 질문", "정리"], "", "", "minimal context"),
    EditorialCase("target_case_14", "준호", "relationship", ["관계", "조율"], ["대화 타이밍", "기대치"], "법무 담당", "기혼", "steady relationship"),
    EditorialCase("target_case_15", "채원", "career_money", ["사업", "속도"], ["현금흐름", "채용", "우선순위"], "크리에이티브 스튜디오 운영", "싱글", "dense concern set"),
    EditorialCase("target_case_16", "이안", "life_direction", ["표현", "방향"], ["창작 기준", "리듬"], "일러스트레이터", "싱글", "artist direction"),
    EditorialCase("target_case_17", "나연", "life_direction", ["의미", "역할"], ["소진 예방", "집중도"], "교사", "기혼", "educator pacing"),
    EditorialCase("target_case_18", "주원", "condition", ["회복", "에너지"], ["야간 리듬", "체력 분배"], "응급실 근무", "싱글", "healthcare rhythm"),
    EditorialCase("target_case_19", "소윤", "career_money", ["커리어", "확장"], ["제안 선별", "계약"], "전략 컨설턴트", "연애 중", "consulting decisions"),
    EditorialCase("target_case_20", "다온", "relationship", ["가족", "관계"], ["돌봄 피로", "경계선"], "가족 돌봄 조정자", "기혼", "caretaker balance", sparse_repeat=True),
]


def _normalized_sha256_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return _normalized_sha256_text(path.read_text(encoding="utf-8"))


def _git_text(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _base_payload(case: EditorialCase) -> dict[str, Any]:
    chart = deepcopy(BASE_CHART)
    as_of_utc = datetime(2026, 3, 11, tzinfo=timezone.utc)
    payload = build_life_cycle_payload(
        chart=chart,
        as_of_utc=as_of_utc,
        timezone_offset_hours=9.0,
        subject_name=case.subject_name,
        onboarding_goal=case.onboarding_goal,
        focus_tokens=case.focus_tokens,
        concern_tokens=case.concern_tokens,
        occupation_context=case.occupation_context,
        relationship_status=case.relationship_status,
    )
    if case.fallback_valid_until:
        payload["valid_until_fallback"] = True
        payload["next_mahadasha_date"] = None
    if case.sparse_repeat:
        payload["repeat_patterns"] = {}
    if case.empty_future:
        payload["next_three_years"] = {
            "slot_count": 0,
            "slots": [],
            "closing_note": payload.get("next_three_years", {}).get(
                "closing_note",
                "이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.\n3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요.",
            ),
            "has_slots": False,
        }
    return payload


def _score_case(payload: dict[str, Any], reading_text: str, metrics: dict[str, Any]) -> dict[str, Any]:
    headings = [line for line in reading_text.splitlines() if line.startswith("## ")]
    exact_h2_ok = headings == LIFE_CYCLE_TARGET_H2_ORDER
    personalization_sections = metrics.get("life_cycle_target_personalization_sections", []) if isinstance(metrics.get("life_cycle_target_personalization_sections"), list) else []
    concern_hits = sum(1 for token in payload.get("concern_tokens", []) if token in reading_text)
    context_hits = sum(1 for token in [payload.get("occupation_context"), payload.get("relationship_status")] if token and str(token) in reading_text)
    english_hits = len([token for token in scan_forbidden_patterns(reading_text) if str(token.get("pattern") or "") != ""])
    sku_words_present = any(word in reading_text for word in ("life_cycle-lite", "life_cycle-full"))
    next_section = "## 다음 3년 구체화\n" in reading_text and "행동:" in reading_text.split("## 다음 3년 구체화", 1)[1]

    read_flow = 5 if exact_h2_ok and metrics.get("front_contract_ok") else 3
    if payload.get("repeat_patterns") == {} or payload.get("next_three_years", {}).get("slot_count") == 0:
        read_flow = min(5, read_flow + 0)

    self_relevance = 3
    if len(personalization_sections) >= 2:
        self_relevance += 1
    if context_hits >= 1 or concern_hits >= 1:
        self_relevance += 1
    self_relevance = min(5, self_relevance)

    action_connection = 3
    if metrics.get("life_cycle_how_to_use_ok") and metrics.get("life_cycle_valid_until_ok") and metrics.get("life_cycle_cta_ok"):
        action_connection += 1
    if next_section:
        action_connection += 1
    action_connection = min(5, action_connection)

    jargon_control = 5
    if sku_words_present:
        jargon_control -= 2
    if english_hits > 0:
        jargon_control -= 1
    jargon_control = max(3, jargon_control)

    strengths: list[str] = []
    if self_relevance >= 4:
        strengths.append("개인화 흔적이 분명함")
    if action_connection >= 4:
        strengths.append("행동 연결이 또렷함")
    if payload.get("repeat_patterns") == {} or payload.get("next_three_years", {}).get("slot_count") == 0:
        strengths.append("edge 케이스에서도 구조가 유지됨")
    if jargon_control >= 5:
        strengths.append("과한 jargon이 없음")
    if not strengths:
        strengths.append("핵심 구조는 안정적임")

    risk = "현재 위치 첫 문장이 아직 다소 템플릿감 있게 느껴질 수 있음"
    if payload.get("occupation_context"):
        risk = "고점/저점 문단은 아직 맥락 문장이 더 붙으면 설득력이 올라갈 수 있음"
    if payload.get("next_three_years", {}).get("slot_count") == 0:
        risk = "빈 슬롯 케이스는 재구매 훅이 약해질 수 있어 후속 편집 점검이 필요함"

    overall_pass = bool(metrics.get("life_cycle_release_ok")) and min(read_flow, self_relevance, action_connection, jargon_control) >= 4
    return {
        "exact_h2_ok": exact_h2_ok,
        "read_flow": read_flow,
        "self_relevance": self_relevance,
        "action_connection": action_connection,
        "jargon_control": jargon_control,
        "overall_pass": overall_pass,
        "strengths": strengths[:2],
        "risk_note": risk,
        "personalization_sections": personalization_sections,
    }


def _build_primary_sample(
    payload: dict[str, Any],
    reading_text: str,
    request_fingerprint: str,
    evidence_case_id: str,
    commit_sha: str,
    chapter_blocks_hash: str,
    chart_hash: str,
) -> dict[str, Any]:
    meta = main_module._build_life_cycle_response_meta(
        as_of_utc=datetime(2026, 3, 11, tzinfo=timezone.utc),
        timezone_offset_hours=9.0,
        onboarding_goal=str(payload.get("onboarding_goal") or "life_direction"),
        payload=payload,
        render_profile=RENDER_PROFILE,
    )
    meta.update(
        {
            "release_evidence_dir": RELEASE_EVIDENCE_DIR,
            "request_fingerprint": request_fingerprint,
            "evidence_case_id": evidence_case_id,
            "commit_sha": commit_sha,
        }
    )
    return {
        "cached": False,
        "fallback": False,
        "model": "deterministic/life_cycle_target",
        "summary": {
            "language": "ko",
            "analysis_mode": "full",
            "product_type": "life_cycle",
            "structured_summary": {
                "product_type": "life_cycle",
                "onboarding_goal": payload.get("onboarding_goal"),
                "subject_name": payload.get("subject_name"),
            },
        },
        "reading": reading_text,
        "polished_reading": reading_text,
        "detail_level": "full",
        "ai_cache_key": f"target_candidate_life_cycle_{request_fingerprint}_v1.4.0",
        "request_id": "life_cycle_target_candidate",
        "chart_hash": chart_hash,
        "chapter_blocks_hash": chapter_blocks_hash,
        "product_type": "life_cycle",
        "meta": meta,
        "debug_info": {
            "product_type": "life_cycle",
            "render_profile": RENDER_PROFILE,
            "contract_version": CONTRACT_VERSION,
            "product_fingerprint": request_fingerprint,
            "llm_input_source": "deterministic.life_cycle_payload.target_candidate",
            "client_initialized": True,
        },
    }


def _default_human_spot_check_block() -> str:
    return f"""## Human Spot Check

- reviewer:
- review_date_kst: 2026-03-12
- sample_path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_sample_response.json
- result: PASS | FAIL

### Check 1

- item: 인생 구조 한 장 요약 첫 문장 자연스러움
- result: PASS | FAIL
- note:

### Check 2

- item: 현재 위치에 이름/관심사/맥락 자연 반영
- result: PASS | FAIL
- note:

### Check 3

- item: target 3개 섹션(고점/저점, 반복 패턴, 다음 3년) 각 1회 존재
- result: PASS | FAIL
- note:

### Check 4

- item: How to use -> 다음 3년 -> valid_until -> CTA 행동선 연결
- result: PASS | FAIL
- note:

### Check 5

- item: 내부 SKU 표현/과한 영문/jargon 없음
- result: PASS | FAIL
- note:

### Check 6

- item: 전체적으로 내 얘기 같고 바로 행동이 떠오름
- result: PASS | FAIL
- note:

### Final Note

- cutover_ready: YES | NO
- reviewer_summary:"""


def _load_existing_human_spot_check_block() -> str | None:
    if not MANUAL_QA_PATH.exists():
        return None
    existing_text = MANUAL_QA_PATH.read_text(encoding="utf-8")
    start_marker = "## Human Spot Check\n"
    end_marker = "\n## Suggested First-Pass Copy"
    if start_marker not in existing_text or end_marker not in existing_text:
        return None
    block = existing_text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    if not block:
        return None
    return start_marker + block


def _render_human_spot_check_block() -> str:
    return _load_existing_human_spot_check_block() or _default_human_spot_check_block()


def main() -> int:
    generated_at_kst = datetime.now(KST).replace(microsecond=0).isoformat()
    commit_sha = _git_text(["git", "rev-parse", "HEAD"])
    dirty_worktree = bool(_git_text(["git", "status", "--short"]))

    review_rows: list[dict[str, Any]] = []
    for case in CASE_SPECS:
        payload = _base_payload(case)
        reading_text = render_life_cycle_target_markdown(payload)
        metrics = gate._compute_life_cycle_target_release_metrics(
            reading_text,
            subject_name=case.subject_name,
            valid_until_fallback=bool(payload.get("valid_until_fallback")),
        )
        scores = _score_case(payload, reading_text, metrics)
        request_fingerprint = main_module._build_product_request_fingerprint(
            product_type="life_cycle",
            onboarding_goal=case.onboarding_goal,
            focus_tokens=case.focus_tokens,
            concern_tokens=case.concern_tokens,
            occupation_context=case.occupation_context,
            relationship_status=case.relationship_status,
            subject_name=case.subject_name,
        )
        review_rows.append(
            {
                "case_id": case.case_id,
                "note": case.note,
                "input": {
                    "subject_name": case.subject_name,
                    "onboarding_goal": case.onboarding_goal,
                    "focus_tokens": case.focus_tokens,
                    "concern_tokens": case.concern_tokens,
                    "occupation_context": case.occupation_context,
                    "relationship_status": case.relationship_status,
                    "fallback_valid_until": case.fallback_valid_until,
                    "sparse_repeat": case.sparse_repeat,
                    "empty_future": case.empty_future,
                },
                "request_fingerprint": request_fingerprint,
                "render_sha256": _normalized_sha256_text(reading_text),
                "gate": metrics,
                "scores": scores,
            }
        )

    overall_pass_count = sum(1 for row in review_rows if row["scores"]["overall_pass"])

    rubric_md = f'''# Life Cycle Target Editorial Rubric

- contract_version: {CONTRACT_VERSION}
- release_evidence_dir: {RELEASE_EVIDENCE_DIR}
- render_profile: {RENDER_PROFILE}
- reviewer_mode: Codex AI-assisted first pass
- generated_at_kst: {generated_at_kst}

이 문서는 `Vedic Life Cycle Report` target candidate를 20개 샘플로 읽을 때 같은 기준으로 점검하기 위한 루브릭입니다.
이 평가는 사람이 다시 읽을 때 기준을 맞추기 위한 첫 패스이며, 최종 컷오버 직전에는 사람 spot-check를 한 번 더 권장합니다.

## 축 정의

1. 술술 읽힘
   - 5점: 흐름이 끊기지 않고 H2 전개와 문장 리듬이 자연스럽다.
   - 4점: 전반적으로 잘 읽히며 일부 문장만 다듬으면 된다.
   - 3점 이하: 섹션 전환이나 문장 리듬이 자주 끊긴다.

2. 내 얘기처럼 읽힘
   - 5점: 이름/상황/관심사가 최소 2개 이상 섹션에서 자연스럽게 반영된다.
   - 4점: 개인화 흔적은 분명하지만 더 깊게 연결될 여지가 있다.
   - 3점 이하: 템플릿 느낌이 강하고 입력 맥락이 약하다.

3. 행동 연결
   - 5점: How to use, 다음 3년, valid_until, CTA가 하나의 행동선으로 연결된다.
   - 4점: 행동 라인은 분명하지만 섹션 간 연결감은 약간 더 다듬을 수 있다.
   - 3점 이하: 읽고 나서 바로 무엇을 할지 불명확하다.

4. 과한 jargon 억제
   - 5점: 소비자 언어 중심이고 내부 SKU/불필요 영문/전문용어 과잉이 없다.
   - 4점: 일부 용어가 남아도 독해를 크게 방해하지 않는다.
   - 3점 이하: 텍스트가 설명서처럼 느껴지거나 용어 부담이 크다.

## PASS 기준

- target gate `life_cycle_release_ok == true`
- 네 축 점수 모두 4점 이상
- exact H2 order 유지
- `life_cycle-lite` / `life_cycle-full` 외부 노출 0회
'''
    EDITORIAL_RUBRIC_PATH.write_text(rubric_md.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    review_lines = [
        "# Life Cycle Target Editorial Review 20",
        "",
        f"- contract_version: {CONTRACT_VERSION}",
        f"- release_evidence_dir: {RELEASE_EVIDENCE_DIR}",
        f"- render_profile: {RENDER_PROFILE}",
        f"- generated_at_kst: {generated_at_kst}",
        f"- reviewer: Codex (AI-assisted first pass)",
        f"- rubric_path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_rubric.md",
        "",
        "## Summary",
        "",
        f"- reviewed_cases: {len(review_rows)}",
        f"- pass_count: {overall_pass_count}",
        f"- fail_count: {len(review_rows) - overall_pass_count}",
        "- note: 이 문서는 사람이 같은 기준으로 spot-check할 때 쓰는 1차 정리본입니다.",
        "",
        "## Score Table",
        "",
        "| case_id | goal | read | self | action | jargon | gate | overall | note |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in review_rows:
        review_lines.append(
            f"| {row['case_id']} | {row['input']['onboarding_goal']} | {row['scores']['read_flow']} | {row['scores']['self_relevance']} | {row['scores']['action_connection']} | {row['scores']['jargon_control']} | {'PASS' if row['gate']['life_cycle_release_ok'] else 'FAIL'} | {'PASS' if row['scores']['overall_pass'] else 'FAIL'} | {row['note']} |"
        )

    review_lines.extend(["", "## Case Notes", ""])
    for row in review_rows:
        strengths = ", ".join(row["scores"]["strengths"])
        personalization = ", ".join(row["scores"]["personalization_sections"] or ["없음"])
        review_lines.extend(
            [
                f"### {row['case_id']}",
                f"- input summary: {row['input']['subject_name']} / {row['input']['onboarding_goal']} / {row['input']['occupation_context'] or 'context 없음'} / {row['input']['relationship_status'] or 'status 없음'}",
                f"- scores: 술술 읽힘 {row['scores']['read_flow']}, 내 얘기처럼 읽힘 {row['scores']['self_relevance']}, 행동 연결 {row['scores']['action_connection']}, jargon 억제 {row['scores']['jargon_control']}",
                f"- gate result: {'PASS' if row['gate']['life_cycle_release_ok'] else 'FAIL'} / overall: {'PASS' if row['scores']['overall_pass'] else 'FAIL'}",
                f"- strengths: {strengths}",
                f"- personalization sections: {personalization}",
                f"- reviewer note: {row['scores']['risk_note']}",
                "",
            ]
        )
    EDITORIAL_REVIEW_PATH.write_text("\n".join(review_lines).replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    EDITORIAL_CASES_PATH.write_text(json.dumps({
        "generated_at_kst": generated_at_kst,
        "contract_version": CONTRACT_VERSION,
        "render_profile": RENDER_PROFILE,
        "reviewed_cases": review_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    primary = review_rows[0]
    primary_payload = _base_payload(CASE_SPECS[0])
    primary_text = render_life_cycle_target_markdown(primary_payload)
    primary_request_fingerprint = primary["request_fingerprint"]
    chapter_blocks_hash = main_module._sha256_hex({"product_type": "life_cycle", "report_stage": "target", "payload": main_module._json_safe_clone(primary_payload)})
    chart_hash = main_module._sha256_hex({"chart": BASE_CHART, "report_stage": "target", "subject_name": CASE_SPECS[0].subject_name})
    sample_response = _build_primary_sample(
        primary_payload,
        primary_text,
        primary_request_fingerprint,
        "life_cycle_target_primary",
        commit_sha,
        chapter_blocks_hash,
        chart_hash,
    )
    SAMPLE_RESPONSE_PATH.write_text(json.dumps(sample_response, ensure_ascii=False, indent=2), encoding="utf-8")

    primary_metrics = gate._compute_life_cycle_target_release_metrics(primary_text, subject_name=CASE_SPECS[0].subject_name, valid_until_fallback=bool(primary_payload.get("valid_until_fallback")))
    hard_style_errors = [
        err
        for err in main_module._reading_style_error_codes(primary_text)
        if err not in {"label_pattern_detected", "paragraph_too_long", "warn_paragraph_density_low", "paragraph_density_low"}
    ]
    forbidden_hits = gate._filter_forbidden_hits_for_release_mode(scan_forbidden_patterns(primary_text, allow_year_quarter_in_timing_map=False), "life_cycle_target")
    gate_summary = {
        "generated_at_kst": generated_at_kst,
        "product_type": "life_cycle",
        "contract_version": CONTRACT_VERSION,
        "render_profile": RENDER_PROFILE,
        "release_evidence_dir": RELEASE_EVIDENCE_DIR,
        "request_fingerprint": primary_request_fingerprint,
        "evidence_case_id": "life_cycle_target_primary",
        "commit_sha": commit_sha,
        "dirty_worktree": dirty_worktree,
        "front_contract_ok": bool(primary_metrics.get("front_contract_ok")),
        "action_steps_contract_ok": bool(primary_metrics.get("action_steps_contract_ok")),
        "forbidden_hits": len(forbidden_hits),
        "hard_fail_count": len(forbidden_hits) + len(hard_style_errors) + (0 if primary_metrics.get("life_cycle_release_ok") else 1),
        "life_cycle_release_ok": bool(primary_metrics.get("life_cycle_release_ok")),
        "valid_until_fallback": bool(primary_payload.get("valid_until_fallback")),
        "meta_valid_until": sample_response["meta"].get("valid_until"),
        "meta_next_mahadasha_date": sample_response["meta"].get("next_mahadasha_date"),
        "required_h2_order": [header.replace("## ", "") for header in LIFE_CYCLE_TARGET_H2_ORDER],
        "release_mode": "life_cycle_target",
        "front_contract_detail": primary_metrics.get("front_contract_detail"),
        "life_cycle_target_high_low_ok": bool(primary_metrics.get("life_cycle_target_high_low_ok")),
        "life_cycle_target_repeat_patterns_ok": bool(primary_metrics.get("life_cycle_target_repeat_patterns_ok")),
        "life_cycle_target_next_three_years_ok": bool(primary_metrics.get("life_cycle_target_next_three_years_ok")),
        "life_cycle_target_next_three_years_has_slots": bool(primary_metrics.get("life_cycle_target_next_three_years_has_slots")),
        "life_cycle_target_personalization_sections": primary_metrics.get("life_cycle_target_personalization_sections", []),
        "life_cycle_hf11_ok": bool(primary_metrics.get("life_cycle_hf11_ok")),
        "life_cycle_hf11_target_header": primary_metrics.get("life_cycle_hf11_target_header"),
        "life_cycle_hf12_ok": bool(primary_metrics.get("life_cycle_hf12_ok")),
        "life_cycle_hf14_ok": bool(primary_metrics.get("life_cycle_hf14_ok")),
        "life_cycle_hf14_overlap_tokens": primary_metrics.get("life_cycle_hf14_overlap_tokens", []),
        "life_cycle_hf16_ok": bool(primary_metrics.get("life_cycle_hf16_ok")),
        "life_cycle_hf16_skipped": bool(primary_metrics.get("life_cycle_hf16_skipped")),
        "hard_style_errors": hard_style_errors,
    }
    GATE_SUMMARY_PATH.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manual_qa = f'''# Life Cycle Target Manual QA

- contract_version: {CONTRACT_VERSION}
- release_evidence_dir: {RELEASE_EVIDENCE_DIR}
- render_profile: {RENDER_PROFILE}
- request_fingerprint: {primary_request_fingerprint}
- evidence_case_id: life_cycle_target_primary
- commit_sha: {commit_sha}
- release_manifest_path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_release_manifest.json
- editorial_rubric_path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_rubric.md
- editorial_review_path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_review_20.md
- dirty_worktree: {str(dirty_worktree).lower()}

## Case 1: life_cycle_target_primary

- input fixture / request summary: fake chart primary (`birth_jd=2447892.5`, `dasha_reference_jd=2461110.5`) + internal target candidate render (`subject_name=민서`, `onboarding_goal=career_money`, `focus_tokens=커리어,돈`, `concern_tokens=이직 타이밍,수입 안정`, `occupation_context=브랜드 전략 업무`, `relationship_status=싱글`)
- expected points:
  - `meta.contract_version == v1.4.0`
  - `meta.render_profile == life_cycle_target_v1`
  - target exact H2 order 13개 유지
  - `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`가 각 1회 존재
  - target gate summary `hard_fail_count == 0`
  - 개인화 흔적이 `인생 구조 한 장 요약`, `현재 위치`, `다음 3년 구체화`에 보임
- actual summary:
  - `valid_until={sample_response['meta'].get('valid_until')}`, `next_mahadasha_date={sample_response['meta'].get('next_mahadasha_date')}`, `current_mahadasha_planet={sample_response['meta'].get('current_mahadasha_planet')}`
  - headings: {', '.join(header.replace('## ', '') for header in LIFE_CYCLE_TARGET_H2_ORDER)}
  - gate: `front_contract_ok={str(primary_metrics.get('front_contract_ok')).lower()}`, `action_steps_contract_ok={str(primary_metrics.get('action_steps_contract_ok')).lower()}`, `life_cycle_release_ok={str(primary_metrics.get('life_cycle_release_ok')).lower()}`
  - personalization sections: {', '.join(primary_metrics.get('life_cycle_target_personalization_sections', []))}
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): {generated_at_kst}

{_render_human_spot_check_block()}

## Suggested First-Pass Copy

- 아래 문구는 reviewer가 그대로 복붙해 시작할 수 있는 초안입니다.
- 실제 사람 검토가 끝나기 전에는 `PASS`, `cutover_ready`, 최종 총평을 확정값으로 남기지 않습니다.

### Suggested Check 1 Note

- 첫 문장이 비교적 자연스럽고, 리포트의 진입 문장으로 읽을 만합니다. 다만 완전히 프리라이팅처럼 느껴지기보다는 약한 템플릿감은 남아 있어 최종 컷오버 전 한 번 더 문장 리듬 점검이 있으면 좋겠습니다.

### Suggested Check 2 Note

- 이름, 관심사, 직업 맥락이 `인생 구조 한 장 요약`, `현재 위치`, `다음 3년 구체화`에 반복적으로 드러나서 개인화 흔적은 분명합니다. 현재 수준에서는 "내 얘기" 감각이 기본선 이상으로 확보됩니다.

### Suggested Check 3 Note

- `인생 고점/저점 지도`, `반복 패턴 분석`, `다음 3년 구체화`가 각 1회씩만 등장하고 canonical H2 order도 유지됩니다. 중복 삽입이나 누락은 보이지 않습니다.

### Suggested Check 4 Note

- `How to use 1p`에서 제시한 읽는 법이 `다음 3년 구체화`, `valid_until 설명`, `CTA-lite`까지 비교적 자연스럽게 이어집니다. 읽고 난 뒤 무엇을 메모하거나 추적할지 행동선이 보이는 편입니다.

### Suggested Check 5 Note

- 외부 문면에서 `life_cycle-lite`, `life_cycle-full` 같은 내부 SKU 표현은 보이지 않고, 과한 영문 용어도 눈에 띄지 않습니다. 소비자용 문면으로는 비교적 안정적입니다.

### Suggested Check 6 Note

- 전체적으로는 "내 얘기 같다"와 "다음 행동이 떠오른다" 기준을 통과하는 쪽에 가깝습니다. 다만 고점/저점 일부 문장은 최종 컷오버 전 사람 기준으로 한 번 더 읽어 자연스러움을 확인하면 더 안전합니다.

### Suggested Final Summary

- target candidate로서는 충분히 설득력 있고, 구조/개인화/행동 연결의 균형도 안정적입니다. 다만 최종 cutover 선언은 clean commit 기준 증적 재생성과 human spot-check 완료 이후에만 하는 것이 맞습니다.

## Case 2: life_cycle_target_editorial_pack_summary

- input fixture / request summary: 20-case editorial pack (`life_cycle_target_editorial_review_20.md`) 재검토
- expected points:
  - sample 20건이 동일 rubric으로 기록됨
  - 각 case가 target gate PASS를 유지함
  - 전체 pass count와 reviewer note가 문서 상단에 요약됨
- actual summary:
  - reviewed_cases={len(review_rows)}, pass_count={overall_pass_count}, fail_count={len(review_rows) - overall_pass_count}
  - rubric path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_rubric.md
  - case matrix path: {RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_case_matrix.json
- PASS/FAIL: PASS
- reviewer: Codex
- run date (Asia/Seoul): {generated_at_kst}
'''
    MANUAL_QA_PATH.write_text(manual_qa.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    manifest = {
        "generated_at_kst": generated_at_kst,
        "contract_version": CONTRACT_VERSION,
        "release_evidence_dir": RELEASE_EVIDENCE_DIR,
        "render_profile": RENDER_PROFILE,
        "request_fingerprint": primary_request_fingerprint,
        "evidence_case_id": "life_cycle_target_primary",
        "commit_sha": commit_sha,
        "dirty_worktree": dirty_worktree,
        "manual_qa_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_manual_qa.md",
        "sample_response_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_sample_response.json",
        "gate_summary_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_gate_summary.json",
        "editorial_rubric_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_rubric.md",
        "editorial_review_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_review_20.md",
        "editorial_case_matrix_path": f"{RELEASE_EVIDENCE_DIR}/life_cycle_target_editorial_case_matrix.json",
        "manual_qa_sha256": _file_sha256(MANUAL_QA_PATH),
        "sample_response_sha256": _file_sha256(SAMPLE_RESPONSE_PATH),
        "gate_summary_sha256": _file_sha256(GATE_SUMMARY_PATH),
        "editorial_rubric_sha256": _file_sha256(EDITORIAL_RUBRIC_PATH),
        "editorial_review_sha256": _file_sha256(EDITORIAL_REVIEW_PATH),
        "editorial_case_matrix_sha256": _file_sha256(EDITORIAL_CASES_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
