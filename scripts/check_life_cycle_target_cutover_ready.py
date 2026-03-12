from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_EVIDENCE_DIR_REL = "PRD/release_evidence/v1_4_0"
DEFAULT_TARGET_RENDER_PROFILE = "life_cycle_target_v1"
DEFAULT_CONTRACT_VERSION = "v1.4.0"


@dataclass(frozen=True)
class TargetEvidencePaths:
    evidence_dir: Path
    manual_qa_path: Path
    sample_response_path: Path
    gate_summary_path: Path
    manifest_path: Path
    editorial_rubric_path: Path
    editorial_review_path: Path
    editorial_case_matrix_path: Path

    @classmethod
    def from_evidence_dir(cls, evidence_dir: Path) -> "TargetEvidencePaths":
        return cls(
            evidence_dir=evidence_dir,
            manual_qa_path=evidence_dir / "life_cycle_target_manual_qa.md",
            sample_response_path=evidence_dir / "life_cycle_target_sample_response.json",
            gate_summary_path=evidence_dir / "life_cycle_target_gate_summary.json",
            manifest_path=evidence_dir / "life_cycle_target_release_manifest.json",
            editorial_rubric_path=evidence_dir / "life_cycle_target_editorial_rubric.md",
            editorial_review_path=evidence_dir / "life_cycle_target_editorial_review_20.md",
            editorial_case_matrix_path=evidence_dir / "life_cycle_target_editorial_case_matrix.json",
        )


@dataclass(frozen=True)
class HumanSpotCheckStatus:
    reviewer: str
    review_date_kst: str
    sample_path: str
    result: str
    check_results: list[str]
    cutover_ready: str
    reviewer_summary: str


def _normalized_sha256_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return _normalized_sha256_text(path.read_text(encoding="utf-8"))


def _git_status_short(root: Path) -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_manual_meta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        key = key.strip()
        if key in {
            "contract_version",
            "release_evidence_dir",
            "render_profile",
            "request_fingerprint",
            "evidence_case_id",
            "commit_sha",
            "release_manifest_path",
        }:
            out[key] = value.strip()
    return out


def _extract_human_spot_check_status(text: str) -> HumanSpotCheckStatus | None:
    start_marker = "## Human Spot Check\n"
    if start_marker not in text:
        return None

    section = text.split(start_marker, 1)[1]
    if "\n## Suggested First-Pass Copy" in section:
        section = section.split("\n## Suggested First-Pass Copy", 1)[0]

    reviewer = ""
    review_date_kst = ""
    sample_path = ""
    result = ""
    cutover_ready = ""
    reviewer_summary = ""
    check_results: list[str] = []
    current_mode = "top"

    for raw_line in section.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("### Check "):
            current_mode = "check"
            continue
        if stripped == "### Final Note":
            current_mode = "final"
            continue
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        key = key.strip()
        value = value.strip()
        if current_mode == "top":
            if key == "reviewer":
                reviewer = value
            elif key == "review_date_kst":
                review_date_kst = value
            elif key == "sample_path":
                sample_path = value
            elif key == "result":
                result = value
        elif current_mode == "check":
            if key == "result":
                check_results.append(value)
        elif current_mode == "final":
            if key == "cutover_ready":
                cutover_ready = value
            elif key == "reviewer_summary":
                reviewer_summary = value

    return HumanSpotCheckStatus(
        reviewer=reviewer,
        review_date_kst=review_date_kst,
        sample_path=sample_path,
        result=result,
        check_results=check_results,
        cutover_ready=cutover_ready,
        reviewer_summary=reviewer_summary,
    )


def _is_placeholder(value: str) -> bool:
    normalized = str(value or "").strip()
    return (not normalized) or ("|" in normalized)


def collect_cutover_readiness_failures(
    *,
    paths: TargetEvidencePaths,
    expected_release_evidence_dir: str = DEFAULT_RELEASE_EVIDENCE_DIR_REL,
    expected_contract_version: str = DEFAULT_CONTRACT_VERSION,
    expected_render_profile: str = DEFAULT_TARGET_RENDER_PROFILE,
    git_status_text: str | None = None,
    root: Path = ROOT,
) -> list[str]:
    failures: list[str] = []

    if git_status_text is None:
        git_status_text = _git_status_short(root)
    if str(git_status_text or "").strip():
        failures.append("Git worktree is not clean. Regenerate target evidence only after tracked/untracked changes are committed or intentionally cleared.")

    required_paths = {
        "manual QA": paths.manual_qa_path,
        "sample response": paths.sample_response_path,
        "gate summary": paths.gate_summary_path,
        "release manifest": paths.manifest_path,
        "editorial rubric": paths.editorial_rubric_path,
        "editorial review": paths.editorial_review_path,
        "editorial case matrix": paths.editorial_case_matrix_path,
    }
    missing_paths: list[str] = []
    for label, path in required_paths.items():
        if not path.exists():
            missing_paths.append(f"Missing {label} artifact: {path}")
    failures.extend(missing_paths)
    if missing_paths:
        return failures

    manual_text = paths.manual_qa_path.read_text(encoding="utf-8")
    manual_meta = _parse_manual_meta(manual_text)
    human_spot_check = _extract_human_spot_check_status(manual_text)
    sample = _load_json(paths.sample_response_path)
    gate_summary = _load_json(paths.gate_summary_path)
    manifest = _load_json(paths.manifest_path)
    editorial_case_matrix = _load_json(paths.editorial_case_matrix_path)

    sample_meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    reviewed_cases = editorial_case_matrix.get("reviewed_cases", []) if isinstance(editorial_case_matrix.get("reviewed_cases"), list) else []

    if manifest.get("contract_version") != expected_contract_version:
        failures.append("Target manifest contract_version does not match the expected v1.4.0 cutover contract.")
    if manifest.get("render_profile") != expected_render_profile:
        failures.append("Target manifest render_profile does not match life_cycle_target_v1.")
    if manifest.get("release_evidence_dir") != expected_release_evidence_dir:
        failures.append("Target manifest release_evidence_dir does not match the expected evidence directory.")

    for key in ("contract_version", "release_evidence_dir", "render_profile", "request_fingerprint", "evidence_case_id", "commit_sha"):
        manual_value = manual_meta.get(key)
        manifest_value = str(manifest.get(key) or "").strip()
        if manual_value != manifest_value:
            failures.append(f"Manual QA {key} does not match manifest.")

    if manual_meta.get("release_manifest_path") != f"{expected_release_evidence_dir}/life_cycle_target_release_manifest.json":
        failures.append("Manual QA release_manifest_path does not point to the target manifest.")

    for key in ("contract_version", "render_profile", "release_evidence_dir", "request_fingerprint", "evidence_case_id", "commit_sha"):
        sample_value = str(sample_meta.get(key) or "").strip()
        manifest_value = str(manifest.get(key) or "").strip()
        if sample_value != manifest_value:
            failures.append(f"Target sample meta {key} does not match manifest.")
        gate_value = str(gate_summary.get(key) or "").strip()
        if gate_value != manifest_value:
            failures.append(f"Target gate summary {key} does not match manifest.")

    expected_paths = {
        "manual_qa_path": f"{expected_release_evidence_dir}/life_cycle_target_manual_qa.md",
        "sample_response_path": f"{expected_release_evidence_dir}/life_cycle_target_sample_response.json",
        "gate_summary_path": f"{expected_release_evidence_dir}/life_cycle_target_gate_summary.json",
        "editorial_rubric_path": f"{expected_release_evidence_dir}/life_cycle_target_editorial_rubric.md",
        "editorial_review_path": f"{expected_release_evidence_dir}/life_cycle_target_editorial_review_20.md",
        "editorial_case_matrix_path": f"{expected_release_evidence_dir}/life_cycle_target_editorial_case_matrix.json",
    }
    for key, expected_value in expected_paths.items():
        if str(manifest.get(key) or "").strip() != expected_value:
            failures.append(f"Target manifest {key} does not match the expected artifact path.")

    expected_hashes = {
        "manual_qa_sha256": _file_sha256(paths.manual_qa_path),
        "sample_response_sha256": _file_sha256(paths.sample_response_path),
        "gate_summary_sha256": _file_sha256(paths.gate_summary_path),
        "editorial_rubric_sha256": _file_sha256(paths.editorial_rubric_path),
        "editorial_review_sha256": _file_sha256(paths.editorial_review_path),
        "editorial_case_matrix_sha256": _file_sha256(paths.editorial_case_matrix_path),
    }
    for key, expected_value in expected_hashes.items():
        if str(manifest.get(key) or "").strip() != expected_value:
            failures.append(f"Target manifest {key} does not match the current file SHA-256.")

    required_gate_flags = {
        "front_contract_ok": True,
        "action_steps_contract_ok": True,
        "life_cycle_release_ok": True,
        "life_cycle_target_high_low_ok": True,
        "life_cycle_target_repeat_patterns_ok": True,
        "life_cycle_target_next_three_years_ok": True,
    }
    for key, expected_value in required_gate_flags.items():
        if bool(gate_summary.get(key)) is not expected_value:
            failures.append(f"Target gate summary {key} is not {expected_value}.")
    if int(gate_summary.get("hard_fail_count") or 0) != 0:
        failures.append("Target gate summary hard_fail_count is not 0.")
    if str(gate_summary.get("release_mode") or "").strip() != "life_cycle_target":
        failures.append("Target gate summary release_mode is not life_cycle_target.")
    if bool(gate_summary.get("dirty_worktree")):
        failures.append("Target gate summary was generated from a dirty worktree.")
    if bool(manifest.get("dirty_worktree")):
        failures.append("Target manifest was generated from a dirty worktree.")

    if len(reviewed_cases) != 20:
        failures.append("Editorial case matrix does not contain exactly 20 reviewed cases.")
    else:
        for row in reviewed_cases:
            scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
            gate = row.get("gate", {}) if isinstance(row.get("gate"), dict) else {}
            if not bool(scores.get("overall_pass")):
                failures.append(f"Editorial case {row.get('case_id')} did not pass the overall review gate.")
                break
            if not bool(gate.get("life_cycle_release_ok")):
                failures.append(f"Editorial case {row.get('case_id')} did not pass the target release gate.")
                break

    if human_spot_check is None:
        failures.append("Human Spot Check section is missing from target manual QA.")
    else:
        if _is_placeholder(human_spot_check.reviewer):
            failures.append("Human spot-check reviewer is not filled in.")
        if _is_placeholder(human_spot_check.review_date_kst):
            failures.append("Human spot-check review_date_kst is not filled in.")
        if human_spot_check.sample_path != f"{expected_release_evidence_dir}/life_cycle_target_sample_response.json":
            failures.append("Human spot-check sample_path does not point to the target sample response.")
        if human_spot_check.result != "PASS":
            failures.append("Human spot-check result is not finalized to PASS.")
        if len(human_spot_check.check_results) != 6 or any(result != "PASS" for result in human_spot_check.check_results):
            failures.append("Human spot-check per-check results are not all finalized to PASS.")
        if human_spot_check.cutover_ready != "YES":
            failures.append("Human spot-check cutover_ready is not finalized to YES.")
        if _is_placeholder(human_spot_check.reviewer_summary):
            failures.append("Human spot-check reviewer_summary is not filled in.")

    return failures


def main() -> int:
    paths = TargetEvidencePaths.from_evidence_dir(ROOT / DEFAULT_RELEASE_EVIDENCE_DIR_REL)
    failures = collect_cutover_readiness_failures(paths=paths)
    if failures:
        print("Life cycle target cutover readiness: NOT READY")
        for idx, failure in enumerate(failures, start=1):
            print(f"{idx}. {failure}")
        return 1

    print("Life cycle target cutover readiness: READY")
    print(f"- evidence_dir: {DEFAULT_RELEASE_EVIDENCE_DIR_REL}")
    print(f"- render_profile: {DEFAULT_TARGET_RENDER_PROFILE}")
    print("- gate_summary: PASS")
    print("- human_spot_check: PASS")
    print("- git_worktree: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
