from __future__ import annotations

import sys
import os
import ast
import subprocess
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.cheap_validation_gate as gate
from scripts.cheap_validation_gate import _compute_actionable_bullet_coverage  # noqa: E402
from backend.commercial_surface_renderer import prepend_front_modules, render_fallback_front_modules


def test_actionable_coverage_does_not_count_plain_bullets_without_action_steps() -> None:
    text = """## [Current Phase] Current Phase

본문 문장입니다.
- 헤더 없는 불릿입니다.
"""
    out = _compute_actionable_bullet_coverage(text)
    assert out["covered"] == 0
    assert "Current Phase" in out["missing"]


def test_actionable_coverage_requires_two_bullets_under_action_steps() -> None:
    one_bullet = """## [Current Phase] Current Phase

### Action Steps

- 하나만 있는 불릿입니다.
"""
    out_one = _compute_actionable_bullet_coverage(one_bullet)
    assert out_one["covered"] == 0

    two_bullets = """## [Current Phase] Current Phase

### Action Steps

- 첫 번째 실행 문장입니다.
- 두 번째 실행 문장입니다.
"""
    out_two = _compute_actionable_bullet_coverage(two_bullets)
    assert out_two["covered"] == 1
    assert "Current Phase" not in out_two["missing"]


def _candidate_fixture() -> dict:
    return {
        "profile_name": "unit_profile",
        "input": {
            "year": 1978,
            "month": 9,
            "day": 17,
            "hour": 20.75,
            "lat": 19.076,
            "lon": 72.8777,
            "house_system": "W",
            "include_nodes": 1,
            "include_d9": 1,
            "include_vargas": "",
            "gender": "male",
            "analysis_mode": "standard",
        },
    }


def _commercial_surface_with_action_steps(*, crlf: bool = False) -> str:
    chapters = [
        "Current Phase",
        "Career & Money",
        "Love & Relationship Patterns",
        "Health & Energy Rhythm",
        "Mid-Term Direction",
        "Risk Management Points",
        "Growth Acceleration",
    ]
    sep = "\r\n" if crlf else "\n"
    chunks: list[str] = []
    for chapter in chapters:
        chunks.append(
            sep.join(
                [
                    f"## [{chapter}] {chapter}",
                    "",
                    "설명 문장입니다. 필요합니다.",
                    "",
                    "### Action Steps",
                    "",
                    "- 첫 번째 실행 문장입니다.",
                    "- 두 번째 실행 문장입니다.",
                ]
            )
        )
    return (sep + sep).join(chunks)


def _commercial_surface_with_inline_action_steps_heading(*, crlf: bool = False) -> str:
    chapters = [
        "Current Phase",
        "Career & Money",
        "Love & Relationship Patterns",
        "Health & Energy Rhythm",
        "Mid-Term Direction",
        "Risk Management Points",
        "Growth Acceleration",
    ]
    sep = "\r\n" if crlf else "\n"
    chunks: list[str] = []
    for chapter in chapters:
        chunks.append(
            sep.join(
                [
                    f"## [{chapter}] {chapter}",
                    "",
                    "설명 문장입니다. 필요합니다.",
                    "",
                    "### Action Steps - 첫 번째 실행 문장입니다. - 두 번째 실행 문장입니다.",
                ]
            )
        )
    return (sep + sep).join(chunks)


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _FakeTestClient:
    def __init__(self, _app, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, _url, params=None, timeout=None):
        del params, timeout
        return _FakeResponse(self._payload)


def _make_local_test_out_dir() -> Path:
    out_dir = Path("logs") / f"cheap_gate_test_{uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_truepath_scan_surface_alignment_uses_payload_surface_and_scored_priority(monkeypatch) -> None:
    polished = _commercial_surface_with_action_steps(crlf=True)
    payload = {
        "reading": "## [Current Phase] fallback\n\n본문",
        "polished_reading": polished,
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True
    assert out["scan_surface_source"] == "polished_reading"
    assert out["scored_surface_name"] == "polished_reading"
    assert out["actionable_bullet_coverage"]["covered"] == 7
    assert out["postprocess_applied"] == (out["scan_surface_sha256"] != out["post_sha256"])
    scan_path = Path(out["reading_scan_surface_path"])
    post_path = Path(out["reading_post_remediation_path"])
    scored_path = Path(out["scored_surface_path"])
    scan_text = scan_path.read_text(encoding="utf-8")
    post_text = post_path.read_text(encoding="utf-8")
    scored_text = scored_path.read_text(encoding="utf-8")
    assert out["scan_surface_sha256"] == gate._sha256_text(scan_text)
    assert out["post_sha256"] == gate._sha256_text(post_text)
    assert out["scored_surface_sha256"] == gate._sha256_text(scored_text)
    assert out["scan_surface_file_sha256"] == gate._sha256_bytes(scan_path.read_bytes())
    assert out["post_file_sha256"] == gate._sha256_bytes(post_path.read_bytes())
    assert out["scored_surface_file_sha256"] == gate._sha256_bytes(scored_path.read_bytes())
    assert "필요합니다." in scan_text


def test_truepath_scan_surface_alignment_falls_back_to_reading_when_polished_missing(monkeypatch) -> None:
    reading = _commercial_surface_with_action_steps(crlf=False)
    payload = {
        "reading": reading,
        "polished_reading": "",
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True
    assert out["scan_surface_source"] == "reading"
    assert out["scored_surface_name"] == "reading_post_remediation"
    assert out["actionable_bullet_coverage"]["covered"] == 7
    assert isinstance(out.get("front_english_token_hits"), dict)
    assert isinstance(out.get("front_markdown_integrity"), dict)
    assert "front_contract_ok" in out
    assert "front_contract_detail" in out
    assert "front_contract_fallback_applied" in out
    assert "front_playbook_repaired" in out
    assert "front_playbook_section_found" in out
    assert "front_playbook_slot_contract_ok_count" in out


def test_is_nonempty_text_uses_strip_semantics() -> None:
    assert gate._is_nonempty_text(None) is False
    assert gate._is_nonempty_text("") is False
    assert gate._is_nonempty_text("   \n") is False
    assert gate._is_nonempty_text("내용") is True


def test_truepath_metrics_are_computed_on_scored_surface_not_scan(monkeypatch) -> None:
    polished = _commercial_surface_with_inline_action_steps_heading(crlf=False)
    payload = {
        "reading": "## [Current Phase] fallback\n\n본문",
        "polished_reading": polished,
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True
    assert out["scored_surface_name"] == "polished_reading"
    scan_text = Path(out["reading_scan_surface_path"]).read_text(encoding="utf-8")
    scored_text = Path(out["scored_surface_path"]).read_text(encoding="utf-8")
    assert "### Action Steps -" in scan_text
    assert "### Action Steps -" not in scored_text
    assert int(out["action_steps_inline_heading_violations"]) == 0


def test_truepath_surface_files_are_lf_only_and_byte_hash_matches_summary(monkeypatch) -> None:
    polished = _commercial_surface_with_action_steps(crlf=True)
    payload = {
        "reading": "## [Current Phase] fallback\r\n\r\n본문",
        "polished_reading": polished,
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True

    scan_path = Path(out["reading_scan_surface_path"])
    post_path = Path(out["reading_post_remediation_path"])
    scored_path = Path(out["scored_surface_path"])
    for p in (scan_path, post_path, scored_path):
        data = p.read_bytes()
        assert b"\r\n" not in data
        assert b"\r" not in data

    assert gate._sha256_bytes(scan_path.read_bytes()) == out["scan_surface_sha256"]
    assert gate._sha256_bytes(post_path.read_bytes()) == out["post_sha256"]
    assert gate._sha256_bytes(scored_path.read_bytes()) == out["scored_surface_sha256"]


def test_truepath_surface_files_do_not_use_path_write_text(monkeypatch) -> None:
    polished = _commercial_surface_with_action_steps(crlf=False)
    payload = {
        "reading": "## [Current Phase] fallback\n\n본문",
        "polished_reading": polished,
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    forbidden_surface_names = {
        "reading_scan_surface.md",
        "reading_post_remediation.md",
        "scored_surface.md",
    }
    original_write_text = Path.write_text

    def _guarded_write_text(self: Path, *args, **kwargs):
        if self.name in forbidden_surface_names:
            raise AssertionError(f"surface file must use _write_text_lf, got Path.write_text: {self}")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _guarded_write_text)
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True


def test_front_english_token_hits_scopes_to_front_only() -> None:
    with_front = """<!-- FRONT_START -->
# 한 장 요약

한글 문장만 있습니다.
<!-- FRONT_END -->

<!-- CHAPTERS_START -->
## [Current Phase] Current Phase

This english token should not be counted because it is in chapters.
<!-- CHAPTERS_END -->
"""
    hits = gate._compute_front_english_token_hits(with_front)
    assert hits["count"] == 0
    assert hits["ok"] is True

    with_front_english = """<!-- FRONT_START -->
# 한 장 요약

This token is in front.
<!-- FRONT_END -->

<!-- CHAPTERS_START -->
## [Current Phase] Current Phase

본문
<!-- CHAPTERS_END -->
"""
    hits2 = gate._compute_front_english_token_hits(with_front_english)
    assert hits2["count"] > 0
    assert hits2["ok"] is False


def test_truepath_reports_front_contract_and_fallback_applied(monkeypatch) -> None:
    chapters = _commercial_surface_with_action_steps(crlf=False)
    front = render_fallback_front_modules()
    payload = {
        "reading": prepend_front_modules(chapters, front),
        "polished_reading": "",
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")

    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True
    assert out["front_contract_ok"] is True
    assert out["front_contract_fallback_applied"] is True
    assert out["front_markdown_integrity"]["ok"] is True
    assert out["front_playbook_section_found"] is True
    assert out["front_playbook_slot_contract_ok_count"] == 3
    assert out["front_playbook_repaired"] is False
    assert "chapters_boundary_fallback_used" in out
    assert "front_byte_equal_after_b_pass" in out
    assert "front_end_offset_stable" in out
    assert "action_steps_contract_ok" in out
    assert "action_steps_block_count_violations" in out
    assert "action_steps_inline_heading_violations" in out
    assert "action_steps_inline_repairs" in out
    assert "action_steps_contaminated_line_violations" in out
    assert "action_steps_non_actionable_line_violations" in out
    assert "header_structure_violations" in out
    assert "definition_dedup_removed_count" in out
    assert "definition_dasha_occurrences_after" in out
    assert "inline_action_chain_violations" in out
    assert "inline_action_chain_migrations" in out
    assert "inline_action_chain_overflow_summaries" in out
    assert "dasha_definition_nested_pattern_violations" in out
    assert "dasha_definition_nested_pattern_repairs" in out
    assert "dasha_definition_redundancy_violations" in out
    assert "dasha_definition_redundancy_repairs" in out
    assert "dasha_insert_retry_blocked" in out
    assert "one_page_summary_dedup_repairs" in out
    assert "commercial_quality_metrics_valid" in out


def test_front_contract_uses_strip_surface_playbook_section() -> None:
    text = """# 한 장 요약

핵심 패턴 3개
- a
- b
- c

금지·권장 3개
- a

상황 예시
- [일] 예시
- [돈] 예시

미니 템플릿 3개
- t1
- t2
- t3

# 3개월 플레이북

이번 달
- 주의: A
- 규칙: B / C
- 이유: D

다음 달
- 주의: E
- 규칙: F / G
- 이유: H

그다음 달
- 주의: I
- 규칙: J / K
- 이유: L

# 7일 시스템
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
운영법(하루 10분)
"""
    contract = gate._compute_front_contract(text)
    assert contract["section_found"] is True
    assert contract["slots_ok"] is True
    assert contract["slot_ok_count"] == 3


def test_front_contract_section_not_found_when_playbook_missing() -> None:
    text = """# 한 장 요약

본문

# 7일 시스템
- [ ] 1
"""
    contract = gate._compute_front_contract(text)
    assert contract["section_found"] is False
    assert contract["slots_ok"] is False


def test_action_steps_contract_metrics_detects_synthetic_heading_and_inline_bullets() -> None:
    text = """## [Current Phase] Action Steps

본문

### Action Steps

- 항목 하나 - 항목 둘 - 항목 셋
"""
    metrics = gate._compute_action_steps_contract_metrics(text)
    struct_metrics = gate._compute_structure_and_definition_metrics(text)
    assert metrics["action_steps_contract_ok"] is False
    assert metrics["action_steps_inline_repairs"] >= 1
    assert struct_metrics["header_structure_violations"] >= 1


def test_inline_action_chain_violations_detected_outside_action_steps() -> None:
    text = """## [Core Disposition] 핵심 기질

- 큰 결정 전 -> 24시간 보류하기 - 합의 전 -> 기대 한 줄로 고정하기 - 회복 루틴 -> 5분 실행하기
"""
    assert gate._compute_inline_action_chain_violations(text) >= 1


def test_inline_action_chain_violations_ignore_overflow_summary_line() -> None:
    text = """## [Core Disposition] 핵심 기질

추가 제안: 큰 결정 보류; 합의 문장 한 줄 고정; 회복 루틴 유지
"""
    assert gate._compute_inline_action_chain_violations(text) == 0


def test_action_steps_inline_heading_violations_detected() -> None:
    text = """## [Current Phase] 현재 흐름

### Action Steps - 큰 결정 보류 - 합의 문장 고정
"""
    metrics = gate._compute_action_steps_contract_metrics(text)
    assert int(metrics["action_steps_inline_heading_violations"]) >= 1


def test_truepath_quality_metrics_invalid_when_chapter_boundary_missing(monkeypatch) -> None:
    payload = {
        "reading": "# 한 장 요약\n\n본문만 있고 챕터 헤더가 없습니다.",
        "polished_reading": "",
        "fallback": False,
        "ai_cache_key": "cache-key",
        "chapter_blocks_hash": "hash",
        "audit": {"overall_score": 88, "flags": {}},
        "debug_info": {"model_used": "gpt-5-mini"},
    }
    monkeypatch.setattr(gate, "OUT_DIR", _make_local_test_out_dir())
    monkeypatch.setattr(gate, "TestClient", lambda app: _FakeTestClient(app, payload))
    monkeypatch.setenv("PDF_DISABLED", "1")
    out = gate._run_single_true_path(_candidate_fixture(), strict_vedic=True)
    assert out["ok"] is True
    assert out["chapters_boundary_fallback_used"] is True
    assert out["commercial_quality_metrics_valid"] is False
    assert int(out["definition_dasha_occurrences_after"]) == 0


def test_gate_script_runs_without_pythonpath_env() -> None:
    env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "scripts/cheap_validation_gate.py", "--help"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_gate_does_not_redefine_key_quality_constants() -> None:
    source_path = Path("scripts/cheap_validation_gate.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    key_names = {
        "ACTIONABLE_HINT_RE",
        "INLINE_ACTION_CHAIN_LIST_START_RE",
        "INLINE_ACTION_CHAIN_HYPHEN_SPLIT_RE",
        "INLINE_ACTION_CHAIN_SEMICOLON_SPLIT_RE",
        "INLINE_ACTION_CHAIN_SLASH_SPLIT_RE",
        "INLINE_ACTION_CHAIN_EXPLANATION_RE",
    }

    for name in key_names:
        assert name in source

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in key_names:
                    raise AssertionError(f"gate must import constant instead of redefining: {target.id}")
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in key_names:
            raise AssertionError(f"gate must not re-annotate constant: {node.target.id}")
