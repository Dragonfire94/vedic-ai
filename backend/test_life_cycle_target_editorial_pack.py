from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import scripts.build_life_cycle_target_editorial_pack as pack


def test_build_primary_sample_carries_release_identity_into_meta() -> None:
    payload = pack._base_payload(pack.CASE_SPECS[0])
    reading_text = pack.render_life_cycle_target_markdown(payload)

    sample = pack._build_primary_sample(
        payload,
        reading_text,
        request_fingerprint="target-fingerprint",
        evidence_case_id="target-case",
        commit_sha="deadbeef",
        chapter_blocks_hash="chapter-hash",
        chart_hash="chart-hash",
    )

    meta = sample["meta"]
    assert meta["contract_version"] == pack.CONTRACT_VERSION
    assert meta["render_profile"] == pack.RENDER_PROFILE
    assert meta["release_evidence_dir"] == pack.RELEASE_EVIDENCE_DIR
    assert meta["request_fingerprint"] == "target-fingerprint"
    assert meta["evidence_case_id"] == "target-case"
    assert meta["commit_sha"] == "deadbeef"
    assert sample["chapter_blocks_hash"] == "chapter-hash"
    assert sample["chart_hash"] == "chart-hash"


def test_target_editorial_pack_main_keeps_sample_gate_manifest_identity_aligned(monkeypatch) -> None:
    out_dir = Path("logs") / f"life_cycle_target_editorial_pack_test_{uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pack, "OUT_DIR", out_dir)
    monkeypatch.setattr(pack, "EDITORIAL_RUBRIC_PATH", out_dir / "life_cycle_target_editorial_rubric.md")
    monkeypatch.setattr(pack, "EDITORIAL_REVIEW_PATH", out_dir / "life_cycle_target_editorial_review_20.md")
    monkeypatch.setattr(pack, "EDITORIAL_CASES_PATH", out_dir / "life_cycle_target_editorial_case_matrix.json")
    monkeypatch.setattr(pack, "MANUAL_QA_PATH", out_dir / "life_cycle_target_manual_qa.md")
    monkeypatch.setattr(pack, "SAMPLE_RESPONSE_PATH", out_dir / "life_cycle_target_sample_response.json")
    monkeypatch.setattr(pack, "GATE_SUMMARY_PATH", out_dir / "life_cycle_target_gate_summary.json")
    monkeypatch.setattr(pack, "MANIFEST_PATH", out_dir / "life_cycle_target_release_manifest.json")
    monkeypatch.setattr(pack, "RELEASE_EVIDENCE_DIR", "tmp/life_cycle_target")

    def _fake_git_text(args: list[str]) -> str:
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return "deadbeefcafebabe"
        if args[:2] == ["git", "status"]:
            return ""
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(pack, "_git_text", _fake_git_text)

    assert pack.main() == 0

    sample = json.loads(pack.SAMPLE_RESPONSE_PATH.read_text(encoding="utf-8"))
    gate_summary = json.loads(pack.GATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(pack.MANIFEST_PATH.read_text(encoding="utf-8"))
    manual_qa = pack.MANUAL_QA_PATH.read_text(encoding="utf-8")

    meta = sample["meta"]
    assert meta["contract_version"] == gate_summary["contract_version"] == manifest["contract_version"]
    assert meta["render_profile"] == gate_summary["render_profile"] == manifest["render_profile"]
    assert meta["request_fingerprint"] == gate_summary["request_fingerprint"] == manifest["request_fingerprint"]
    assert meta["evidence_case_id"] == gate_summary["evidence_case_id"] == manifest["evidence_case_id"]
    assert meta["commit_sha"] == gate_summary["commit_sha"] == manifest["commit_sha"]
    assert meta["release_evidence_dir"] == manifest["release_evidence_dir"] == "tmp/life_cycle_target"
    assert manifest["sample_response_sha256"] == pack._file_sha256(pack.SAMPLE_RESPONSE_PATH)
    assert "## Human Spot Check" in manual_qa
    assert "## Suggested First-Pass Copy" in manual_qa
