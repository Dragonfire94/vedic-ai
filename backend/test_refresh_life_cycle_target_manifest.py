from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from scripts.check_life_cycle_target_cutover_ready import TargetEvidencePaths, _file_sha256
from scripts.refresh_life_cycle_target_manifest import refresh_target_manifest


def test_refresh_target_manifest_updates_hashes_after_manual_qa_change() -> None:
    base_dir = Path("logs") / f"target_manifest_refresh_{uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = TargetEvidencePaths.from_evidence_dir(base_dir)

    paths.manual_qa_path.write_text("manual-v1\n", encoding="utf-8", newline="\n")
    paths.sample_response_path.write_text('{"meta": {}}\n', encoding="utf-8", newline="\n")
    paths.gate_summary_path.write_text('{"life_cycle_release_ok": true}\n', encoding="utf-8", newline="\n")
    paths.editorial_rubric_path.write_text("rubric\n", encoding="utf-8", newline="\n")
    paths.editorial_review_path.write_text("review\n", encoding="utf-8", newline="\n")
    paths.editorial_case_matrix_path.write_text('{"reviewed_cases": []}\n', encoding="utf-8", newline="\n")
    paths.manifest_path.write_text(
        json.dumps(
            {
                "request_fingerprint": "fingerprint",
                "manual_qa_sha256": "stale",
                "sample_response_sha256": "stale",
                "gate_summary_sha256": "stale",
                "editorial_rubric_sha256": "stale",
                "editorial_review_sha256": "stale",
                "editorial_case_matrix_sha256": "stale",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    paths.manual_qa_path.write_text("manual-v2\n", encoding="utf-8", newline="\n")
    manifest = refresh_target_manifest(paths)

    assert manifest["manual_qa_sha256"] == _file_sha256(paths.manual_qa_path)
    assert manifest["sample_response_sha256"] == _file_sha256(paths.sample_response_path)
    assert manifest["gate_summary_sha256"] == _file_sha256(paths.gate_summary_path)
    assert manifest["editorial_rubric_sha256"] == _file_sha256(paths.editorial_rubric_path)
    assert manifest["editorial_review_sha256"] == _file_sha256(paths.editorial_review_path)
    assert manifest["editorial_case_matrix_sha256"] == _file_sha256(paths.editorial_case_matrix_path)
    assert manifest["manual_qa_path"].endswith("life_cycle_target_manual_qa.md")
    assert manifest["sample_response_path"].endswith("life_cycle_target_sample_response.json")
