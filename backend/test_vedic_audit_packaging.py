from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_vedic_audit_package.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_vedic_audit_result.py"
SCHEMA_PATH = ROOT / "backend" / "audit_templates" / "vedic_technical_audit_output_schema_v1.json"
SAMPLE_PASS_PATH = ROOT / "backend" / "audit_templates" / "examples" / "audit_result_sample_pass.json"
TEST_WORKDIR_BASE = ROOT / "logs" / "qa_phase11_run7_20260222_005336" / "pytest_audit_packaging"


def _case_dir(case_name: str) -> Path:
    path = TEST_WORKDIR_BASE / f"{case_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_py(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=False)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_normalized_text(text: str) -> str:
    return _sha256_text(_normalize_newlines(text))


def _sha256_canonical_json(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


def _base_ai_reading_payload(*, include_polished: bool = True) -> dict:
    payload = {
        "reading": "## [Executive Diagnosis]\n\nfallback reading",
        "vedic_technical_data": {
            "availability": {"ok": False, "reason": "partial_data", "missing_fields": ["dashas.timeline"]},
            "meta": {"generated_utc": "2026-03-01T00:00:00Z", "pipeline_version": "x"},
            "calculation_settings": {"ayanamsa": "Lahiri"},
            "rasi_D1": {"lagna": {"sign": "Aries"}, "planets": []},
            "varga": {"D9_navamsa": {"planets": []}, "D10_dashamsa": {"planets": []}},
            "dashas": {"system": "Vimshottari", "current": {"mahadasha": None, "bhukti": None}, "timeline": []},
            "transits": {"timing_map": []},
            "yogas": [],
            "shadbala": {"summary": None, "details": {}},
        },
        "vedic_technical_reading": "technical block with ```inline code``` marker",
    }
    if include_polished:
        payload["polished_reading"] = "## [Executive Diagnosis]\n\npolished reading"
    return payload


def _valid_audit_result() -> dict:
    return {
        "verdict": {
            "technical_ok": True,
            "commercial_consistent_with_technical": True,
            "confidence": 0.84,
            "blocking_issues_count": 0,
        },
        "availability": {
            "ok": False,
            "reason": "partial_data",
            "missing_fields": ["dashas.timeline"],
            "audit_limitations": ["timeline unavailable"],
        },
        "technical_findings": {
            "critical": [
                {
                    "issue": "sample",
                    "evidence_paths": ["vedic_technical_data.dashas.timeline"],
                    "why_it_matters": "x",
                    "suggested_fix": "y",
                }
            ],
            "minor": [
                {
                    "issue": "sample2",
                    "evidence_paths": ["vedic_technical_data.transits.timing_map"],
                    "note": "z",
                }
            ],
        },
        "commercial_contradictions": [],
        "commercial_minimal_edits": [],
        "notes_for_engineers": ["fill timeline upstream"],
    }


def test_build_package_outputs_files_and_manifest() -> None:
    tmp_path = _case_dir("outputs_manifest")
    src = tmp_path / "input.json"
    outdir = tmp_path / "out"
    src.write_text(json.dumps(_base_ai_reading_payload(include_polished=True), ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(BUILD_SCRIPT, "--input", str(src), "--outdir", str(outdir))
    assert res.returncode == 0, res.stdout + res.stderr

    payload_path = outdir / "audit_input_payload.json"
    prompt_path = outdir / "audit_prompt_filled.md"
    manifest_path = outdir / "audit_manifest.json"
    assert payload_path.exists()
    assert prompt_path.exists()
    assert manifest_path.exists()

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")

    assert payload["commercial_source"] == "polished_reading"
    assert manifest["commercial_source"] == "polished_reading"
    for key in (
        "template_file",
        "template_version",
        "schema_file",
        "schema_version",
        "payload_sha256",
        "commercial_sha256",
        "technical_data_sha256",
        "technical_md_sha256",
        "created_utc",
        "input_path",
    ):
        assert key in manifest
    assert manifest["commercial_sha256"] == _sha256_normalized_text(payload["commercial_reading"])
    assert manifest["technical_md_sha256"] == _sha256_normalized_text(payload["vedic_technical_reading"])
    assert manifest["technical_data_sha256"] == _sha256_canonical_json(payload["vedic_technical_data"])
    assert "````json" in prompt
    assert "````md" in prompt
    assert "telemetry: payload_chars=" in res.stdout


def test_build_package_fallbacks_to_reading_source() -> None:
    tmp_path = _case_dir("fallback_reading")
    src = tmp_path / "input_reading_only.json"
    outdir = tmp_path / "out_reading_only"
    src.write_text(json.dumps(_base_ai_reading_payload(include_polished=False), ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(BUILD_SCRIPT, "--input", str(src), "--outdir", str(outdir))
    assert res.returncode == 0, res.stdout + res.stderr
    payload = json.loads((outdir / "audit_input_payload.json").read_text(encoding="utf-8"))
    assert payload["commercial_source"] == "reading"


def test_build_package_requires_vedic_technical_data() -> None:
    tmp_path = _case_dir("missing_technical_data")
    src = tmp_path / "invalid.json"
    outdir = tmp_path / "out_invalid"
    src.write_text(json.dumps({"reading": "only reading"}, ensure_ascii=False), encoding="utf-8")

    res = _run_py(BUILD_SCRIPT, "--input", str(src), "--outdir", str(outdir))
    assert res.returncode == 1
    assert "missing_required_field: vedic_technical_data" in (res.stdout + res.stderr)


def test_build_package_supports_truncate_and_data_only() -> None:
    tmp_path = _case_dir("truncate_data_only")
    src = tmp_path / "input_long.json"
    outdir = tmp_path / "out_long"
    payload = _base_ai_reading_payload(include_polished=True)
    payload["vedic_technical_reading"] = "A" * 200
    src.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    res_trunc = _run_py(
        BUILD_SCRIPT,
        "--input",
        str(src),
        "--outdir",
        str(outdir),
        "--truncate-technical-md",
        "50",
    )
    assert res_trunc.returncode == 0, res_trunc.stdout + res_trunc.stderr
    manifest = json.loads((outdir / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["technical_reading_truncated"] is True

    outdir_data_only = tmp_path / "out_data_only"
    res_data_only = _run_py(
        BUILD_SCRIPT,
        "--input",
        str(src),
        "--outdir",
        str(outdir_data_only),
        "--prefer-data-only",
    )
    assert res_data_only.returncode == 0, res_data_only.stdout + res_data_only.stderr
    payload2 = json.loads((outdir_data_only / "audit_input_payload.json").read_text(encoding="utf-8"))
    manifest2 = json.loads((outdir_data_only / "audit_manifest.json").read_text(encoding="utf-8"))
    assert payload2["vedic_technical_reading"] == ""
    assert manifest2["stats"]["technical_reading_included"] is False
    assert manifest2["options"]["prefer_data_only"] is True
    assert manifest2["technical_md_sha256"] == _sha256_normalized_text("")


def test_build_package_hash_normalization_for_crlf_vs_lf() -> None:
    tmp_path = _case_dir("hash_newline_normalization")
    outdir_crlf = tmp_path / "out_crlf"
    outdir_lf = tmp_path / "out_lf"
    base = _base_ai_reading_payload(include_polished=False)

    src_crlf = tmp_path / "input_crlf.json"
    payload_crlf = dict(base)
    payload_crlf["reading"] = "line1\r\nline2"
    src_crlf.write_text(json.dumps(payload_crlf, ensure_ascii=False), encoding="utf-8")
    res_crlf = _run_py(BUILD_SCRIPT, "--input", str(src_crlf), "--outdir", str(outdir_crlf))
    assert res_crlf.returncode == 0, res_crlf.stdout + res_crlf.stderr

    src_lf = tmp_path / "input_lf.json"
    payload_lf = dict(base)
    payload_lf["reading"] = "line1\nline2"
    src_lf.write_text(json.dumps(payload_lf, ensure_ascii=False), encoding="utf-8")
    res_lf = _run_py(BUILD_SCRIPT, "--input", str(src_lf), "--outdir", str(outdir_lf))
    assert res_lf.returncode == 0, res_lf.stdout + res_lf.stderr

    manifest_crlf = json.loads((outdir_crlf / "audit_manifest.json").read_text(encoding="utf-8"))
    manifest_lf = json.loads((outdir_lf / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest_crlf["commercial_sha256"] == manifest_lf["commercial_sha256"]


def test_validate_script_accepts_valid_result() -> None:
    tmp_path = _case_dir("validate_pass")
    result_path = tmp_path / "audit_result.json"
    result_path.write_text(json.dumps(_valid_audit_result(), ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(VALIDATE_SCRIPT, "--result", str(result_path), "--schema", str(SCHEMA_PATH))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VALID" in res.stdout


def test_validate_script_rejects_invalid_confidence() -> None:
    tmp_path = _case_dir("validate_confidence_fail")
    result = _valid_audit_result()
    result["verdict"]["confidence"] = 1.5
    result_path = tmp_path / "invalid_confidence.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(VALIDATE_SCRIPT, "--result", str(result_path), "--schema", str(SCHEMA_PATH))
    assert res.returncode == 1
    assert "confidence" in (res.stdout + res.stderr)


def test_validate_script_rejects_negative_blocking_count() -> None:
    tmp_path = _case_dir("validate_blocking_fail")
    result = _valid_audit_result()
    result["verdict"]["blocking_issues_count"] = -1
    result_path = tmp_path / "invalid_blocking.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(VALIDATE_SCRIPT, "--result", str(result_path), "--schema", str(SCHEMA_PATH))
    assert res.returncode == 1
    assert "blocking_issues_count" in (res.stdout + res.stderr)


def test_validate_script_rejects_missing_technical_path_prefix() -> None:
    tmp_path = _case_dir("validate_prefix_fail")
    result = _valid_audit_result()
    result["technical_findings"]["critical"][0]["evidence_paths"] = ["wrong.path"]
    result_path = tmp_path / "invalid_path_prefix.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    res = _run_py(VALIDATE_SCRIPT, "--result", str(result_path), "--schema", str(SCHEMA_PATH))
    assert res.returncode == 1
    out = res.stdout + res.stderr
    assert "technical_findings.critical[0].evidence_paths" in out
    assert "evidence_paths" in out


def test_validate_script_accepts_sample_pass_file() -> None:
    assert SAMPLE_PASS_PATH.exists()
    res = _run_py(VALIDATE_SCRIPT, "--result", str(SAMPLE_PASS_PATH), "--schema", str(SCHEMA_PATH))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VALID" in res.stdout
