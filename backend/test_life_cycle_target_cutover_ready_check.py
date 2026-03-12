from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import scripts.check_life_cycle_target_cutover_ready as cutover_check


def _make_case_matrix() -> dict:
    return {
        "reviewed_cases": [
            {
                "case_id": f"case_{idx:02d}",
                "scores": {"overall_pass": True},
                "gate": {"life_cycle_release_ok": True},
            }
            for idx in range(1, 21)
        ]
    }


def _write_target_evidence_bundle(base_dir: Path, *, placeholders: bool) -> cutover_check.TargetEvidencePaths:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = cutover_check.TargetEvidencePaths.from_evidence_dir(base_dir)
    release_dir = "tmp/life_cycle_target"
    request_fingerprint = "abc123fingerprint"
    evidence_case_id = "life_cycle_target_primary"
    commit_sha = "deadbeefcafebabe"

    reviewer = "" if placeholders else "human-reviewer"
    result = "PASS | FAIL" if placeholders else "PASS"
    cutover_ready = "YES | NO" if placeholders else "YES"
    reviewer_summary = "" if placeholders else "human spot-check completed"
    check_result = "PASS | FAIL" if placeholders else "PASS"

    manual_qa = f'''# Life Cycle Target Manual QA

- contract_version: v1.4.0
- release_evidence_dir: {release_dir}
- render_profile: life_cycle_target_v1
- request_fingerprint: {request_fingerprint}
- evidence_case_id: {evidence_case_id}
- commit_sha: {commit_sha}
- release_manifest_path: {release_dir}/life_cycle_target_release_manifest.json

## Human Spot Check

- reviewer: {reviewer}
- review_date_kst: 2026-03-12
- sample_path: {release_dir}/life_cycle_target_sample_response.json
- result: {result}

### Check 1
- result: {check_result}

### Check 2
- result: {check_result}

### Check 3
- result: {check_result}

### Check 4
- result: {check_result}

### Check 5
- result: {check_result}

### Check 6
- result: {check_result}

### Final Note
- cutover_ready: {cutover_ready}
- reviewer_summary: {reviewer_summary}

## Suggested First-Pass Copy
'''
    paths.manual_qa_path.write_text(manual_qa, encoding="utf-8", newline="\n")

    sample = {
        "meta": {
            "contract_version": "v1.4.0",
            "render_profile": "life_cycle_target_v1",
            "release_evidence_dir": release_dir,
            "request_fingerprint": request_fingerprint,
            "evidence_case_id": evidence_case_id,
            "commit_sha": commit_sha,
        }
    }
    paths.sample_response_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_summary = {
        "contract_version": "v1.4.0",
        "render_profile": "life_cycle_target_v1",
        "release_evidence_dir": release_dir,
        "request_fingerprint": request_fingerprint,
        "evidence_case_id": evidence_case_id,
        "commit_sha": commit_sha,
        "release_mode": "life_cycle_target",
        "front_contract_ok": True,
        "action_steps_contract_ok": True,
        "life_cycle_release_ok": True,
        "life_cycle_target_high_low_ok": True,
        "life_cycle_target_repeat_patterns_ok": True,
        "life_cycle_target_next_three_years_ok": True,
        "hard_fail_count": 0,
        "dirty_worktree": False,
    }
    paths.gate_summary_path.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    paths.editorial_rubric_path.write_text("# rubric\n", encoding="utf-8", newline="\n")
    paths.editorial_review_path.write_text("# review\n", encoding="utf-8", newline="\n")
    paths.editorial_case_matrix_path.write_text(json.dumps(_make_case_matrix(), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "contract_version": "v1.4.0",
        "release_evidence_dir": release_dir,
        "render_profile": "life_cycle_target_v1",
        "request_fingerprint": request_fingerprint,
        "evidence_case_id": evidence_case_id,
        "commit_sha": commit_sha,
        "dirty_worktree": False,
        "manual_qa_path": f"{release_dir}/life_cycle_target_manual_qa.md",
        "sample_response_path": f"{release_dir}/life_cycle_target_sample_response.json",
        "gate_summary_path": f"{release_dir}/life_cycle_target_gate_summary.json",
        "editorial_rubric_path": f"{release_dir}/life_cycle_target_editorial_rubric.md",
        "editorial_review_path": f"{release_dir}/life_cycle_target_editorial_review_20.md",
        "editorial_case_matrix_path": f"{release_dir}/life_cycle_target_editorial_case_matrix.json",
        "manual_qa_sha256": cutover_check._file_sha256(paths.manual_qa_path),
        "sample_response_sha256": cutover_check._file_sha256(paths.sample_response_path),
        "gate_summary_sha256": cutover_check._file_sha256(paths.gate_summary_path),
        "editorial_rubric_sha256": cutover_check._file_sha256(paths.editorial_rubric_path),
        "editorial_review_sha256": cutover_check._file_sha256(paths.editorial_review_path),
        "editorial_case_matrix_sha256": cutover_check._file_sha256(paths.editorial_case_matrix_path),
    }
    paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def test_collect_cutover_readiness_failures_passes_for_clean_completed_bundle() -> None:
    base_dir = Path("logs") / f"target_cutover_ready_ok_{uuid4().hex}"
    paths = _write_target_evidence_bundle(base_dir, placeholders=False)

    failures = cutover_check.collect_cutover_readiness_failures(
        paths=paths,
        expected_release_evidence_dir="tmp/life_cycle_target",
        git_status_text="",
    )

    assert failures == []


def test_collect_cutover_readiness_failures_blocks_placeholder_human_spot_check() -> None:
    base_dir = Path("logs") / f"target_cutover_ready_placeholder_{uuid4().hex}"
    paths = _write_target_evidence_bundle(base_dir, placeholders=True)

    failures = cutover_check.collect_cutover_readiness_failures(
        paths=paths,
        expected_release_evidence_dir="tmp/life_cycle_target",
        git_status_text="",
    )

    assert any(failure == "Human spot-check reviewer is not filled in." for failure in failures)
    assert any(failure == "Human spot-check result is not finalized to PASS." for failure in failures)
    assert any(failure == "Human spot-check cutover_ready is not finalized to YES." for failure in failures)
