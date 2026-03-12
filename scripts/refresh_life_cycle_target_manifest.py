from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_life_cycle_target_cutover_ready import (
    DEFAULT_RELEASE_EVIDENCE_DIR_REL,
    TargetEvidencePaths,
    _file_sha256,
)




def _manifest_path_value(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")

def refresh_target_manifest(paths: TargetEvidencePaths) -> dict[str, Any]:
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "manual_qa_path": _manifest_path_value(paths.manual_qa_path),
            "sample_response_path": _manifest_path_value(paths.sample_response_path),
            "gate_summary_path": _manifest_path_value(paths.gate_summary_path),
            "editorial_rubric_path": _manifest_path_value(paths.editorial_rubric_path),
            "editorial_review_path": _manifest_path_value(paths.editorial_review_path),
            "editorial_case_matrix_path": _manifest_path_value(paths.editorial_case_matrix_path),
            "manual_qa_sha256": _file_sha256(paths.manual_qa_path),
            "sample_response_sha256": _file_sha256(paths.sample_response_path),
            "gate_summary_sha256": _file_sha256(paths.gate_summary_path),
            "editorial_rubric_sha256": _file_sha256(paths.editorial_rubric_path),
            "editorial_review_sha256": _file_sha256(paths.editorial_review_path),
            "editorial_case_matrix_sha256": _file_sha256(paths.editorial_case_matrix_path),
        }
    )
    paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    paths = TargetEvidencePaths.from_evidence_dir(ROOT / DEFAULT_RELEASE_EVIDENCE_DIR_REL)
    manifest = refresh_target_manifest(paths)
    print("Life cycle target manifest refreshed")
    print(f"- manifest_path: {paths.manifest_path}")
    print(f"- request_fingerprint: {manifest.get('request_fingerprint')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
