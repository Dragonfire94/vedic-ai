from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "PRD" / "release_evidence" / "v1_4_0"
MANUAL_QA_PATH = EVIDENCE_DIR / "life_cycle_target_manual_qa.md"
SAMPLE_RESPONSE_PATH = EVIDENCE_DIR / "life_cycle_target_sample_response.json"
GATE_SUMMARY_PATH = EVIDENCE_DIR / "life_cycle_target_gate_summary.json"
MANIFEST_PATH = EVIDENCE_DIR / "life_cycle_target_release_manifest.json"


def _normalized_sha256_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return _normalized_sha256_text(path.read_text(encoding="utf-8"))


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


def test_life_cycle_target_release_manifest_matches_evidence_hashes_and_identity() -> None:
    manual_meta = _parse_manual_meta(MANUAL_QA_PATH.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE_RESPONSE_PATH.read_text(encoding="utf-8"))
    gate_summary = json.loads(GATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    sample_meta = sample["meta"]

    assert manifest["contract_version"] == "v1.4.0"
    assert manifest["render_profile"] == "life_cycle_target_v1"
    assert manifest["release_evidence_dir"] == "PRD/release_evidence/v1_4_0"

    assert manual_meta["contract_version"] == manifest["contract_version"]
    assert manual_meta["release_evidence_dir"] == manifest["release_evidence_dir"]
    assert manual_meta["render_profile"] == manifest["render_profile"]
    assert manual_meta["request_fingerprint"] == manifest["request_fingerprint"]
    assert manual_meta["evidence_case_id"] == manifest["evidence_case_id"]
    assert manual_meta["commit_sha"] == manifest["commit_sha"]
    assert manual_meta["release_manifest_path"] == "PRD/release_evidence/v1_4_0/life_cycle_target_release_manifest.json"

    assert sample_meta["contract_version"] == manifest["contract_version"]
    assert sample_meta["render_profile"] == manifest["render_profile"]
    assert sample_meta["release_evidence_dir"] == manifest["release_evidence_dir"]
    assert sample_meta["request_fingerprint"] == manifest["request_fingerprint"]
    assert sample_meta["evidence_case_id"] == manifest["evidence_case_id"]
    assert sample_meta["commit_sha"] == manifest["commit_sha"]

    assert gate_summary["contract_version"] == manifest["contract_version"]
    assert gate_summary["render_profile"] == manifest["render_profile"]
    assert gate_summary["release_evidence_dir"] == manifest["release_evidence_dir"]
    assert gate_summary["request_fingerprint"] == manifest["request_fingerprint"]
    assert gate_summary["evidence_case_id"] == manifest["evidence_case_id"]
    assert gate_summary["commit_sha"] == manifest["commit_sha"]

    assert manifest["manual_qa_path"] == "PRD/release_evidence/v1_4_0/life_cycle_target_manual_qa.md"
    assert manifest["sample_response_path"] == "PRD/release_evidence/v1_4_0/life_cycle_target_sample_response.json"
    assert manifest["gate_summary_path"] == "PRD/release_evidence/v1_4_0/life_cycle_target_gate_summary.json"

    assert manifest["manual_qa_sha256"] == _file_sha256(MANUAL_QA_PATH)
    assert manifest["sample_response_sha256"] == _file_sha256(SAMPLE_RESPONSE_PATH)
    assert manifest["gate_summary_sha256"] == _file_sha256(GATE_SUMMARY_PATH)
